"""Run the scripted congestion-spike scenario with the live forecaster
wired in, tick by tick, logging calibrated confidence + both trigger
signals - satisfies M4's done-when condition (classifier + backstop both
produce visible trigger signals with confidence on a live run) and the
Review 1 demo checklist's early-warning requirement.

    python -m ml.run_forecast_demo --seed 0 --ticks 500 \
        --model data/models/forecaster --out data/runs/congestion_spike_demo
"""

from __future__ import annotations

import argparse

from ml.anomaly_labels import is_congestion_event
from ml.forecast_features import tick_feature_vector
from ml.forecaster import Forecaster
from ml.scripted_scenarios import SPIKE_END, SPIKE_START, congestion_spike_rate
from sim.world import World

# Threshold on the *raw* (uncalibrated) congestion probability used only for
# reporting an early-warning trend in this demo's summary - not a trigger
# decision (that's the calibrated threshold, which is what actually gates
# any future action). See CLAUDE.md's M4 note: raw output rises gradually
# ahead of a real event, calibrated output is closer to a step function
# once genuinely warranted, and both are worth showing for different
# reasons.
RAW_EARLY_WARNING_THRESHOLD = 0.05


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--ticks", type=int, default=500)
    parser.add_argument("--robots", type=int, default=8)
    parser.add_argument("--model", type=str, default="data/models/forecaster")
    parser.add_argument("--out", type=str, default="data/runs/congestion_spike_demo")
    args = parser.parse_args()

    world = World(seed=args.seed, num_robots=args.robots, order_rate=congestion_spike_rate)
    forecaster = Forecaster(args.model)
    thresholds = forecaster.thresholds

    first_actual_congestion_tick = None
    first_congestion_classifier_trigger_tick = None
    first_collision_classifier_trigger_tick = None
    first_any_trigger_tick = None
    first_raw_congestion_rise_tick = None

    for _ in range(args.ticks):
        world.tick()
        tick = world.tick_count - 1

        tick_row = world.telemetry.tick_rows[-1]
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

        result = forecaster.step(features)
        world.telemetry.log_forecast(
            tick,
            result["congestion_prob"],
            result["collision_prob"],
            result["classifier_triggered"],
            result["backstop_triggered"],
            result["congestion_prob_raw"],
            result["collision_prob_raw"],
        )

        if first_actual_congestion_tick is None and is_congestion_event(tick_row["active_orders"]):
            first_actual_congestion_tick = tick
        if (
            first_congestion_classifier_trigger_tick is None
            and result["congestion_prob"] is not None
            and result["congestion_prob"] >= thresholds["congestion"]
        ):
            first_congestion_classifier_trigger_tick = tick
        if (
            first_collision_classifier_trigger_tick is None
            and result["collision_prob"] is not None
            and result["collision_prob"] >= thresholds["collision"]
        ):
            first_collision_classifier_trigger_tick = tick
        if first_any_trigger_tick is None and result["triggered"]:
            first_any_trigger_tick = tick
        if (
            first_raw_congestion_rise_tick is None
            and result["congestion_prob_raw"] is not None
            and result["congestion_prob_raw"] >= RAW_EARLY_WARNING_THRESHOLD
        ):
            first_raw_congestion_rise_tick = tick

    world.telemetry.save(args.out, list(world.orders.values()), world.manifest())

    print(f"Saved {args.ticks}-tick congestion-spike run to {args.out}")
    print(f"Scripted spike window: ticks {SPIKE_START}-{SPIKE_END} (order arrival rate raised)")
    print(f"First actual congestion event (active_orders >= 40): tick {first_actual_congestion_tick}")
    print(f"First congestion-head classifier trigger (calibrated, gates any future action): tick {first_congestion_classifier_trigger_tick}")
    print(f"First collision-head classifier trigger (near-miss risk, independent signal): tick {first_collision_classifier_trigger_tick}")
    print(f"First trigger of any kind (either head OR backstop): tick {first_any_trigger_tick}")
    print(f"First raw congestion probability >= {RAW_EARLY_WARNING_THRESHOLD} (trend signal, not a trigger): tick {first_raw_congestion_rise_tick}")
    if first_congestion_classifier_trigger_tick is not None and first_actual_congestion_tick is not None:
        lead = first_actual_congestion_tick - first_congestion_classifier_trigger_tick
        print(f"Calibrated congestion-trigger lead time: {lead} ticks {'(early warning)' if lead > 0 else '(reactive, not predictive, on this run)'}")
    if first_raw_congestion_rise_tick is not None and first_actual_congestion_tick is not None:
        raw_lead = first_actual_congestion_tick - first_raw_congestion_rise_tick
        print(f"Raw-signal lead time: {raw_lead} ticks (the genuine early trend, ahead of the calibrated trigger)")


if __name__ == "__main__":
    main()
