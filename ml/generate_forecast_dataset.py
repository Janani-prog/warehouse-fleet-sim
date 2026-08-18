"""Generate a windowed (last WINDOW ticks -> anomaly in next HORIZON ticks)
dataset for the anomaly forecaster, from a deliberate mix of healthy and
overloaded (robots, order_rate) configs - a forecaster trained only on the
tuned "healthy demo" config would rarely see a real anomaly to learn from.

    python -m ml.generate_forecast_dataset --seed 0 --episodes-per-config 4 \
        --ticks 700 --out data/datasets/forecast_train.parquet
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from ml.anomaly_labels import is_collision_risk_event, is_congestion_event
from ml.forecast_features import episode_feature_matrix
from sim.world import World

WINDOW = 30
HORIZON = 10

# deliberate mix: several comfortably-provisioned configs, several
# undersized/overloaded ones, spanning the range explored in CLAUDE.md's
# threshold-calibration sweep.
EPISODE_CONFIGS = [
    (8, 0.15),
    (10, 0.15),
    (6, 0.1),
    (10, 0.2),
    (6, 0.25),
    (5, 0.3),
    (8, 0.3),
    (10, 0.35),
    (4, 0.15),
]


def build_windows(feature_matrix: np.ndarray) -> list[dict]:
    active_orders = feature_matrix[:, 0]
    near_miss = feature_matrix[:, 1]
    T = len(feature_matrix)

    rows = []
    for t in range(WINDOW - 1, T - HORIZON):
        window = feature_matrix[t - WINDOW + 1 : t + 1]
        future_active = active_orders[t + 1 : t + 1 + HORIZON]
        future_nm = near_miss[t + 1 : t + 1 + HORIZON]
        label_congestion = float(any(is_congestion_event(int(v)) for v in future_active))
        label_collision = float(any(is_collision_risk_event(int(v)) for v in future_nm))
        rows.append(
            {
                "tick": t,
                "window": window.flatten().tolist(),
                "label_congestion": label_congestion,
                "label_collision": label_collision,
            }
        )
    return rows


def collect_episode(seed: int, num_robots: int, order_rate: float, ticks: int) -> list[dict]:
    world = World(seed=seed, num_robots=num_robots, order_rate=order_rate)
    world.run(ticks)
    feature_matrix = episode_feature_matrix(world.telemetry)
    return build_windows(feature_matrix)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--episodes-per-config", type=int, default=4)
    parser.add_argument("--ticks", type=int, default=700)
    parser.add_argument("--out", type=str, default="data/datasets/forecast_train.parquet")
    args = parser.parse_args()

    all_rows: list[dict] = []
    episode_id = 0
    for num_robots, order_rate in EPISODE_CONFIGS:
        for i in range(args.episodes_per_config):
            episode_seed = args.seed * 100_000 + episode_id
            rows = collect_episode(episode_seed, num_robots, order_rate, args.ticks)
            for r in rows:
                r["episode_id"] = episode_id
                r["num_robots"] = num_robots
                r["order_rate"] = order_rate
            all_rows.extend(rows)
            episode_id += 1

    df = pd.DataFrame(all_rows)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out_path, index=False)

    n_cong = int(df["label_congestion"].sum())
    n_coll = int(df["label_collision"].sum())
    print(
        f"Saved {len(df)} windows from {episode_id} episodes to {out_path} "
        f"(congestion positive: {n_cong} [{n_cong / len(df):.1%}], "
        f"collision-risk positive: {n_coll} [{n_coll / len(df):.1%}])"
    )


if __name__ == "__main__":
    main()
