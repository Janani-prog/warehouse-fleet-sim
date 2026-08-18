"""Benchmark: classical vs. learned traffic avoidance across full
multi-robot simulator runs on held-out seeds.

Metrics per run: near-miss count, throughput (orders completed), mean
detour (extra ticks per completed order beyond its A* optimal path length),
and how often the obstacle-aware replan fallback had to fire (a proxy for
how much congestion the avoidance policy alone couldn't resolve).

    python -m ml.benchmark_traffic --model data/models/traffic_mlp.pt \
        --seeds 2000-2019 --out data/results/traffic_benchmark.csv
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from scipy import stats

from ml.astar import astar
from ml.learned_traffic import load_model, make_learned_avoidance
from ml.traffic import resolve as classical_avoidance
from sim.world import World


def run_scenario(seed: int, avoidance, num_robots: int, order_rate: float, ticks: int) -> dict:
    world = World(seed=seed, num_robots=num_robots, order_rate=order_rate, avoidance=avoidance)
    world.run(ticks)

    completed = [o for o in world.orders.values() if o.status.value == "completed"]
    near_miss_total = sum(row["near_miss_count"] for row in world.telemetry.tick_rows)
    replans = sum(1 for e in world.telemetry.event_rows if e["type"] == "replanned_around_blockage")

    # detour = actual pickup->dropoff ticks minus that leg's A* optimal
    # length. (The assign->pickup leg isn't included: the robot serving an
    # order isn't logged at assignment time, so its start position for that
    # leg isn't recoverable from telemetry alone.)
    detour_ticks = []
    for o in completed:
        optimal = astar(world.warehouse, o.origin, o.destination)
        if optimal is None:
            continue
        actual = o.dropoff_tick - o.pickup_tick
        detour_ticks.append(actual - (len(optimal) - 1))

    return {
        "seed": seed,
        "total_orders": len(world.orders),
        "completed": len(completed),
        "completion_rate": len(completed) / len(world.orders) if world.orders else float("nan"),
        "near_miss_count": near_miss_total,
        "replans_around_blockage": replans,
        "mean_detour_ticks": sum(detour_ticks) / len(detour_ticks) if detour_ticks else float("nan"),
    }


def run_benchmark(seeds: list[int], model_path: str, num_robots: int, order_rate: float, ticks: int) -> pd.DataFrame:
    model = load_model(model_path)
    learned = make_learned_avoidance(model)

    rows = []
    for seed in seeds:
        classical_result = run_scenario(seed, classical_avoidance, num_robots, order_rate, ticks)
        learned_result = run_scenario(seed, learned, num_robots, order_rate, ticks)
        for policy, result in (("classical", classical_result), ("learned", learned_result)):
            rows.append({"policy": policy, **result})
    return pd.DataFrame(rows)


def paired_test_summary(df: pd.DataFrame) -> pd.DataFrame:
    """Same-seed paired comparison (classical vs learned ran on identical
    seeds), Wilcoxon signed-rank per metric - the same paired-trial spirit
    the causal eval harness (M9) will use for the full closed loop."""
    classical = df[df.policy == "classical"].sort_values("seed").reset_index(drop=True)
    learned = df[df.policy == "learned"].sort_values("seed").reset_index(drop=True)
    rows = []
    for col in ["completion_rate", "near_miss_count", "mean_detour_ticks"]:
        c, l = classical[col], learned[col]
        stat, p = stats.wilcoxon(c, l)
        rows.append(
            {
                "metric": col,
                "mean_classical": c.mean(),
                "mean_learned": l.mean(),
                "mean_diff_learned_minus_classical": (l - c).mean(),
                "wilcoxon_statistic": stat,
                "p_value": p,
            }
        )
    return pd.DataFrame(rows)


def plot_summary(df: pd.DataFrame, out_path: Path) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(11, 3.5))
    metrics = ["completion_rate", "near_miss_count", "mean_detour_ticks"]
    titles = ["Completion rate", "Near-miss count / run", "Mean detour (ticks)"]
    for ax, metric, title in zip(axes, metrics, titles):
        data = [df[df.policy == p][metric].dropna() for p in ("classical", "learned")]
        ax.boxplot(data, tick_labels=["Classical", "Learned"])
        ax.set_title(title)
    fig.tight_layout()
    fig.savefig(out_path)


def _parse_seeds(spec: str) -> list[int]:
    if "-" in spec:
        lo, hi = spec.split("-")
        return list(range(int(lo), int(hi) + 1))
    return [int(s) for s in spec.split(",")]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=str, default="data/models/traffic_mlp.pt")
    parser.add_argument("--seeds", type=str, default="2000-2019")
    parser.add_argument("--robots", type=int, default=10)
    parser.add_argument("--order-rate", type=float, default=0.2)
    parser.add_argument("--ticks", type=int, default=400)
    parser.add_argument("--out", type=str, default="data/results/traffic_benchmark.csv")
    args = parser.parse_args()

    seeds = _parse_seeds(args.seeds)
    df = run_benchmark(seeds, args.model, args.robots, args.order_rate, args.ticks)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)
    plot_summary(df, out_path.with_suffix(".png"))

    paired = paired_test_summary(df)
    paired_path = out_path.with_name(out_path.stem + "_paired_stats.csv")
    paired.to_csv(paired_path, index=False)

    print(f"Saved per-run results to {out_path}")
    print(f"Saved summary chart to {out_path.with_suffix('.png')}")
    print(f"Saved paired Wilcoxon stats to {paired_path}")
    print()
    print(df.groupby("policy")[["completion_rate", "near_miss_count", "replans_around_blockage", "mean_detour_ticks"]].mean())
    print()
    print(paired.to_string(index=False))


if __name__ == "__main__":
    main()
