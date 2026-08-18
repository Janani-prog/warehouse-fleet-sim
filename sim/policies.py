"""M1 placeholder movement: greedy random walk toward a target cell.

This stands in for real path planning until M2 (A* / learned planner) and
real collision avoidance until M3 (ORCA / learned policy) are wired in. It
exists so the simulator produces meaningful telemetry (orders actually get
delivered, queues actually drain) without depending on either.
"""

from __future__ import annotations

import numpy as np

from sim.grid import Warehouse

GREEDY_BIAS = 0.8


def step_toward(
    pos: tuple[int, int],
    target: tuple[int, int] | None,
    warehouse: Warehouse,
    occupied: set[tuple[int, int]],
    rng: np.random.Generator,
) -> tuple[int, int]:
    """One grid step from pos, biased toward target if given, avoiding
    currently-occupied cells where possible. Falls back to staying put."""
    candidates = [c for c in warehouse.neighbors(*pos) if c not in occupied]
    if not candidates:
        return pos

    if target is not None and target != pos and rng.random() < GREEDY_BIAS:
        dx = target[0] - pos[0]
        dy = target[1] - pos[1]
        preferred = [c for c in candidates if abs(c[0] - target[0]) + abs(c[1] - target[1]) < abs(dx) + abs(dy)]
        if preferred:
            return tuple(preferred[rng.integers(len(preferred))])

    return tuple(candidates[rng.integers(len(candidates))])
