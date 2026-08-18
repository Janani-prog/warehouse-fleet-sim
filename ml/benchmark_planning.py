"""Benchmark: classical A* vs. imitation-learned planner on held-out trips.

Reports path length ratio (learned / A*, successes only), planning latency,
and success rate for both. Reproducible from a single command with a fixed
seed (the held-out dataset itself is generated with its own seed, separate
from the training set, in generate_planning_dataset.py).

    python -m ml.benchmark_planning --dataset data/datasets/astar_test.parquet \
        --model data/models/planner_mlp.pt --out data/results/planning_benchmark.csv
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from ml.astar import astar
from ml.learned_planner import load_model, rollout
from sim.grid import generate_default_layout


def run_benchmark(dataset_path: str, model_path: str, max_steps: int = 150) -> pd.DataFrame:
    warehouse = generate_default_layout()
    model = load_model(model_path)
    df = pd.read_parquet(dataset_path)

    rows = []
    for _, row in df.iterrows():
        start = (int(row["start_x"]), int(row["start_y"]))
        goal = (int(row["goal_x"]), int(row["goal_y"]))

        t0 = time.perf_counter()
        a_path = astar(warehouse, start, goal)
        a_latency = time.perf_counter() - t0
        a_success = a_path is not None
        a_len = len(a_path) if a_success else None

        t0 = time.perf_counter()
        l_path, l_success = rollout(model, warehouse, start, goal, max_steps=max_steps)
        l_latency = time.perf_counter() - t0
        l_len = len(l_path) if l_success else None

        rows.append(
            {
                "trip_id": row["trip_id"],
                "astar_success": a_success,
                "astar_path_len": a_len,
                "astar_latency_s": a_latency,
                "learned_success": l_success,
                "learned_path_len": l_len,
                "learned_latency_s": l_latency,
                "length_ratio": (l_len / a_len) if (a_success and l_success) else None,
            }
        )
    return pd.DataFrame(rows)


def summarize(results: pd.DataFrame) -> dict:
    both_success = results.dropna(subset=["length_ratio"])
    return {
        "n_trials": len(results),
        "astar_success_rate": results["astar_success"].mean(),
        "learned_success_rate": results["learned_success"].mean(),
        "astar_mean_latency_ms": results["astar_latency_s"].mean() * 1000,
        "learned_mean_latency_ms": results["learned_latency_s"].mean() * 1000,
        "mean_length_ratio_learned_over_astar": both_success["length_ratio"].mean(),
        "n_both_succeeded": len(both_success),
    }


def plot_summary(results: pd.DataFrame, summary: dict, out_path: Path) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(11, 3.5))

    axes[0].bar(["A*", "Learned"], [summary["astar_success_rate"], summary["learned_success_rate"]])
    axes[0].set_title("Success rate")
    axes[0].set_ylim(0, 1.05)

    axes[1].bar(["A*", "Learned"], [summary["astar_mean_latency_ms"], summary["learned_mean_latency_ms"]])
    axes[1].set_title("Mean planning latency (ms)")

    both = results.dropna(subset=["length_ratio"])
    axes[2].hist(both["length_ratio"], bins=15)
    axes[2].axvline(1.0, color="black", linestyle="--", linewidth=1)
    axes[2].set_title("Learned / A* path length ratio")

    fig.tight_layout()
    fig.savefig(out_path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=str, default="data/datasets/astar_test.parquet")
    parser.add_argument("--model", type=str, default="data/models/planner_mlp.pt")
    parser.add_argument("--out", type=str, default="data/results/planning_benchmark.csv")
    parser.add_argument("--max-steps", type=int, default=150)
    args = parser.parse_args()

    results = run_benchmark(args.dataset, args.model, max_steps=args.max_steps)
    summary = summarize(results)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    results.to_csv(out_path, index=False)
    plot_summary(results, summary, out_path.with_suffix(".png"))

    print(f"Saved per-trial results to {out_path}")
    print(f"Saved summary chart to {out_path.with_suffix('.png')}")
    print()
    for k, v in summary.items():
        print(f"{k}: {v}")


if __name__ == "__main__":
    main()
