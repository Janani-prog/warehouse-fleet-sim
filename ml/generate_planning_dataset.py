"""Generate an A*-solved (start, goal, path) dataset for imitation learning.

Samples realistic warehouse trips (pickup<->dropoff, robot-position-like free
cells) and solves each with A*, so the learned planner trains on the same
kind of journeys robots actually make in the simulator.

    python -m ml.generate_planning_dataset --seed 0 --n 500 --out data/datasets/astar_train.parquet
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from ml.astar import astar
from sim.grid import Warehouse, generate_default_layout


def _free_cells(warehouse: Warehouse) -> list[tuple[int, int]]:
    return [
        (x, y)
        for x in range(warehouse.width)
        for y in range(warehouse.height)
        if warehouse.is_free(x, y)
    ]


def sample_trip(
    warehouse: Warehouse, free_cells: list[tuple[int, int]], rng: np.random.Generator
) -> tuple[tuple[int, int], tuple[int, int]]:
    """Mimic the three kinds of trips a robot actually makes: pickup point to
    dropoff, dropoff to pickup point (returning for the next order), or a
    general free-cell pair (covers off-route positions)."""
    kind = rng.integers(3)
    if kind == 0:
        start = warehouse.pickup_points[rng.integers(len(warehouse.pickup_points))]
        goal = warehouse.dropoff_points[rng.integers(len(warehouse.dropoff_points))]
    elif kind == 1:
        start = warehouse.dropoff_points[rng.integers(len(warehouse.dropoff_points))]
        goal = warehouse.pickup_points[rng.integers(len(warehouse.pickup_points))]
    else:
        start = free_cells[rng.integers(len(free_cells))]
        goal = free_cells[rng.integers(len(free_cells))]
    return start, goal


def generate(warehouse: Warehouse, seed: int, n: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    free_cells = _free_cells(warehouse)

    rows = []
    trip_id = 0
    attempts = 0
    while trip_id < n and attempts < n * 5:
        attempts += 1
        start, goal = sample_trip(warehouse, free_cells, rng)
        if start == goal:
            continue
        path = astar(warehouse, start, goal)
        if path is None:
            continue
        rows.append(
            {
                "trip_id": trip_id,
                "start_x": start[0],
                "start_y": start[1],
                "goal_x": goal[0],
                "goal_y": goal[1],
                "path": [list(c) for c in path],
                "path_len": len(path),
            }
        )
        trip_id += 1

    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--n", type=int, default=500)
    parser.add_argument("--out", type=str, default="data/datasets/astar_train.parquet")
    args = parser.parse_args()

    warehouse = generate_default_layout()
    df = generate(warehouse, args.seed, args.n)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out_path, index=False)
    print(f"Saved {len(df)} A*-solved trips to {out_path}")


if __name__ == "__main__":
    main()
