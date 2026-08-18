"""Per-tick feature vector for the anomaly forecaster: a compact summary of
fleet state, cheap to compute live from the same telemetry the simulator
already produces (no new instrumentation needed - see the M3 status note in
CLAUDE.md)."""

from __future__ import annotations

import numpy as np

FEATURE_NAMES = [
    "active_orders",
    "near_miss_count",
    "max_queue_depth",
    "mean_queue_depth",
    "max_robot_density",
    "num_blocked_robots",
]
FEATURE_DIM = len(FEATURE_NAMES)


def tick_feature_vector(
    active_orders: int,
    near_miss_count: int,
    zone_queue_depths: list[int],
    zone_robot_densities: list[int],
    num_blocked_robots: int,
) -> np.ndarray:
    max_queue = max(zone_queue_depths) if zone_queue_depths else 0
    mean_queue = (sum(zone_queue_depths) / len(zone_queue_depths)) if zone_queue_depths else 0.0
    max_density = max(zone_robot_densities) if zone_robot_densities else 0
    return np.array(
        [active_orders, near_miss_count, max_queue, mean_queue, max_density, num_blocked_robots],
        dtype=np.float32,
    )


def episode_feature_matrix(telemetry) -> np.ndarray:
    """Build the full [num_ticks, FEATURE_DIM] matrix from a completed
    TelemetryLogger's in-memory rows (sim.telemetry.TelemetryLogger)."""
    zones_by_tick: dict[int, list[dict]] = {}
    for row in telemetry.zone_rows:
        zones_by_tick.setdefault(row["tick"], []).append(row)

    blocked_by_tick: dict[int, int] = {}
    for row in telemetry.robot_rows:
        if row["state"] == "blocked":
            blocked_by_tick[row["tick"]] = blocked_by_tick.get(row["tick"], 0) + 1

    rows = []
    for tick_row in telemetry.tick_rows:
        t = tick_row["tick"]
        zone_rows = zones_by_tick.get(t, [])
        rows.append(
            tick_feature_vector(
                active_orders=tick_row["active_orders"],
                near_miss_count=tick_row["near_miss_count"],
                zone_queue_depths=[z["queue_depth"] for z in zone_rows],
                zone_robot_densities=[z["robot_density"] for z in zone_rows],
                num_blocked_robots=blocked_by_tick.get(t, 0),
            )
        )
    return np.stack(rows) if rows else np.zeros((0, FEATURE_DIM), dtype=np.float32)
