"""Feature encoding for the learned avoidance policy: a local occupancy
window that distinguishes racks from robots (unlike the path planner's
window, which only needs to see static obstacles), plus the direction the
robot is trying to move in this tick."""

from __future__ import annotations

import numpy as np

from ml.planner_features import ACTIONS
from sim.grid import Warehouse

WINDOW_RADIUS = 2  # 5x5 window, matching the planner's for consistency
FEATURE_DIM = (2 * WINDOW_RADIUS + 1) ** 2 + len(ACTIONS)

FREE = 0.0
ROBOT = 0.5
BLOCKED = 1.0  # rack or out of bounds


def local_traffic_window(
    warehouse: Warehouse,
    pos: tuple[int, int],
    other_robot_positions: set[tuple[int, int]],
) -> np.ndarray:
    x, y = pos
    cells = []
    for dy in range(-WINDOW_RADIUS, WINDOW_RADIUS + 1):
        for dx in range(-WINDOW_RADIUS, WINDOW_RADIUS + 1):
            cell = (x + dx, y + dy)
            if not warehouse.is_free(*cell):
                cells.append(BLOCKED)
            elif cell in other_robot_positions:
                cells.append(ROBOT)
            else:
                cells.append(FREE)
    return np.array(cells, dtype=np.float32)


def desired_direction_onehot(pos: tuple[int, int], desired_cell: tuple[int, int]) -> np.ndarray:
    delta = (desired_cell[0] - pos[0], desired_cell[1] - pos[1])
    onehot = np.zeros(len(ACTIONS), dtype=np.float32)
    if delta in ACTIONS:
        onehot[ACTIONS.index(delta)] = 1.0
    return onehot


def encode(
    warehouse: Warehouse,
    pos: tuple[int, int],
    desired_cell: tuple[int, int],
    other_robot_positions: set[tuple[int, int]],
) -> np.ndarray:
    return np.concatenate(
        [
            local_traffic_window(warehouse, pos, other_robot_positions),
            desired_direction_onehot(pos, desired_cell),
        ]
    )
