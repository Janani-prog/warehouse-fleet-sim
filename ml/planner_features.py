"""Feature encoding shared by the learned planner's training and inference:
a local occupancy window around the current cell plus a normalized goal
delta. Kept small and local so the policy network trains fast on CPU."""

from __future__ import annotations

import numpy as np

from sim.grid import Warehouse

WINDOW_RADIUS = 2  # 5x5 window
ACTIONS: list[tuple[int, int]] = [(1, 0), (-1, 0), (0, 1), (0, -1)]  # right, left, down, up
FEATURE_DIM = (2 * WINDOW_RADIUS + 1) ** 2 + 2


def local_window(warehouse: Warehouse, pos: tuple[int, int]) -> np.ndarray:
    """Flattened (2r+1)x(2r+1) occupancy window centered on pos.
    1.0 = blocked (rack or out of bounds), 0.0 = free."""
    x, y = pos
    cells = []
    for dy in range(-WINDOW_RADIUS, WINDOW_RADIUS + 1):
        for dx in range(-WINDOW_RADIUS, WINDOW_RADIUS + 1):
            cells.append(0.0 if warehouse.is_free(x + dx, y + dy) else 1.0)
    return np.array(cells, dtype=np.float32)


def goal_delta(pos: tuple[int, int], goal: tuple[int, int], warehouse: Warehouse) -> np.ndarray:
    dx = (goal[0] - pos[0]) / warehouse.width
    dy = (goal[1] - pos[1]) / warehouse.height
    return np.array([dx, dy], dtype=np.float32)


def encode(warehouse: Warehouse, pos: tuple[int, int], goal: tuple[int, int]) -> np.ndarray:
    return np.concatenate([local_window(warehouse, pos), goal_delta(pos, goal, warehouse)])


def action_index(delta: tuple[int, int]) -> int:
    return ACTIONS.index(delta)
