"""Generate a labeled (local context, proceed/yield) dataset for the learned
avoidance policy by running the simulator under classical avoidance across
many episodes and recording every conflict-arbitration decision.

    python -m ml.generate_traffic_dataset --seed 0 --episodes 12 --ticks 400 \
        --out data/datasets/traffic_train.parquet
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from ml.traffic_features import encode
from sim.world import World


def collect_episode(seed: int, ticks: int, num_robots: int, order_rate: float) -> list[dict]:
    rows: list[dict] = []

    def on_move(positions, desired, final):
        others_by_robot = {
            rid: {p for other_rid, p in positions.items() if other_rid != rid}
            for rid in positions
        }
        for rid, want in desired.items():
            if want == positions[rid]:
                continue  # robot had no move to make; not an avoidance decision
            feat = encode(world.warehouse, positions[rid], want, others_by_robot[rid])
            label = 1.0 if final[rid] == want else 0.0
            rows.append({"features": feat.tolist(), "label": label})

    world = World(seed=seed, num_robots=num_robots, order_rate=order_rate)
    world.on_move = on_move
    world.run(ticks)
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--episodes", type=int, default=12)
    parser.add_argument("--ticks", type=int, default=400)
    parser.add_argument("--robots", type=int, default=10)
    parser.add_argument("--order-rate", type=float, default=0.2)
    parser.add_argument("--out", type=str, default="data/datasets/traffic_train.parquet")
    args = parser.parse_args()

    all_rows: list[dict] = []
    for i in range(args.episodes):
        episode_seed = args.seed * 10_000 + i
        all_rows.extend(collect_episode(episode_seed, args.ticks, args.robots, args.order_rate))

    df = pd.DataFrame(all_rows)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out_path, index=False)

    n_proceed = int(df["label"].sum())
    print(f"Saved {len(df)} examples ({n_proceed} proceed / {len(df) - n_proceed} yield) to {out_path}")


if __name__ == "__main__":
    main()
