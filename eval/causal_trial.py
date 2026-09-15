"""Runs one trial of the congestion-spike scenario with the M8 closed loop
either enabled or disabled, returning the M9 causal-eval metrics. "Loop
OFF" still runs the forecaster and logs its output every tick - the only
difference from "loop ON" is whether a triggered tick's signal is ever
handed to agent.orchestrator.run_loop_step, i.e. whether the action
executor ever touches simulator state. Everything else (seed, warehouse,
scenario, classical planner/avoidance) is identical between the two, which
is what makes the paired comparison causal rather than observational
(Locked Design Decision #6).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

from agent.orchestrator import run_loop_step
from ml.anomaly_labels import is_congestion_event
from ml.forecast_features import tick_feature_vector
from ml.forecaster import Forecaster
from rag.hybrid_retrieval import HybridRetriever
from sim.entities import OrderStatus
from sim.world import World


@dataclass(frozen=True)
class TrialMetrics:
    seed: int
    loop_enabled: bool
    throughput: int  # orders completed during the trial
    mean_wait: float  # mean wait_ticks over completed orders (nan if none completed)
    near_miss_total: int  # sum of near_miss_count over all ticks
    congestion_ticks: int  # count of ticks where active_orders crossed the congestion threshold


def run_trial(
    seed: int,
    ticks: int,
    robots: int,
    order_rate: Callable[[int], float],
    forecaster: Forecaster,
    loop_enabled: bool,
    retriever: HybridRetriever | None = None,
    agent_model: str | None = None,
) -> TrialMetrics:
    if loop_enabled and retriever is None:
        raise ValueError("retriever is required when loop_enabled=True")

    world = World(seed=seed, num_robots=robots, order_rate=order_rate)
    forecaster.window.clear()  # trials must not leak rolling-window state across each other

    congestion_ticks = 0
    for _ in range(ticks):
        world.tick()
        tick = world.tick_count - 1

        tick_row = world.telemetry.tick_rows[-1]
        if is_congestion_event(tick_row["active_orders"]):
            congestion_ticks += 1

        zone_rows = [z for z in world.telemetry.zone_rows if z["tick"] == tick]
        blocked = sum(
            1 for r in world.telemetry.robot_rows if r["tick"] == tick and r["state"] == "blocked"
        )
        features = tick_feature_vector(
            active_orders=tick_row["active_orders"],
            near_miss_count=tick_row["near_miss_count"],
            zone_queue_depths=[z["queue_depth"] for z in zone_rows],
            zone_robot_densities=[z["robot_density"] for z in zone_rows],
            num_blocked_robots=blocked,
        )
        forecaster_result = forecaster.step(features)

        if loop_enabled and forecaster_result["triggered"]:
            run_loop_step(world, forecaster_result, retriever, model=agent_model or "llama3.1:8b")

    completed = [o for o in world.orders.values() if o.status == OrderStatus.COMPLETED]
    picked_up = [o for o in world.orders.values() if o.wait_ticks is not None]
    mean_wait = sum(o.wait_ticks for o in picked_up) / len(picked_up) if picked_up else float("nan")
    near_miss_total = sum(row["near_miss_count"] for row in world.telemetry.tick_rows)

    return TrialMetrics(
        seed=seed,
        loop_enabled=loop_enabled,
        throughput=len(completed),
        mean_wait=mean_wait,
        near_miss_total=near_miss_total,
        congestion_ticks=congestion_ticks,
    )
