"""Runs the M4 scripted congestion-spike scenario with the full M8 closed
loop wired in: forecaster/backstop trigger -> RAG retrieval -> LLM agent ->
action executor -> simulator state, every triggered tick. Satisfies M8's
done-when condition (a live run demonstrating at least one full
trigger->retrieve->decide->act cycle, logged) the same way
ml.run_forecast_demo satisfied M4's.

    python -m agent.run_agent_demo --seed 0 --ticks 500 \
        --forecaster-model data/models/forecaster \
        --out data/runs/agent_closed_loop_demo

Requires a running local Ollama server (embeddings + llama3.1:8b).
"""

from __future__ import annotations

import argparse
from collections import Counter

from agent.orchestrator import run_loop_step
from ml.forecast_features import tick_feature_vector
from ml.forecaster import Forecaster
from ml.scripted_scenarios import congestion_spike_rate
from rag.hybrid_retrieval import HybridRetriever
from sim.world import World


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--ticks", type=int, default=500)
    parser.add_argument("--robots", type=int, default=8)
    parser.add_argument("--forecaster-model", type=str, default="data/models/forecaster")
    parser.add_argument("--agent-model", type=str, default="llama3.1:8b")
    parser.add_argument("--out", type=str, default="data/runs/agent_closed_loop_demo")
    args = parser.parse_args()

    world = World(seed=args.seed, num_robots=args.robots, order_rate=congestion_spike_rate)
    forecaster = Forecaster(args.forecaster_model)
    retriever = HybridRetriever()

    num_triggers = 0
    num_loop_calls = 0
    action_counts: Counter = Counter()
    executed_counts: Counter = Counter()
    first_cycle_tick = None

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
        forecaster_result = forecaster.step(features)
        world.telemetry.log_forecast(
            tick,
            forecaster_result["congestion_prob"],
            forecaster_result["collision_prob"],
            forecaster_result["classifier_triggered"],
            forecaster_result["backstop_triggered"],
            forecaster_result["congestion_prob_raw"],
            forecaster_result["collision_prob_raw"],
        )

        if forecaster_result["triggered"]:
            num_triggers += 1
            result = run_loop_step(world, forecaster_result, retriever, model=args.agent_model)
            if result is not None:
                num_loop_calls += 1
                if first_cycle_tick is None:
                    first_cycle_tick = tick
                action_counts[result.decision.action] += 1
                executed_counts[result.decision.executed] += 1
                world.telemetry.log_agent_decision(
                    tick=tick,
                    anomaly_type=result.anomaly_type,
                    severity=result.severity,
                    confidence=result.confidence,
                    sop_doc_id=result.sop_doc_id,
                    retrieval_score=result.retrieval_score,
                    action=result.decision.action,
                    target_type=result.decision.target_type,
                    target_id=str(result.decision.target_id) if result.decision.target_id is not None else None,
                    reason=result.decision.reason,
                    executed=result.decision.executed,
                    raw_llm_output=result.decision.raw_llm_output,
                )

    world.telemetry.save(args.out, list(world.orders.values()), world.manifest())

    print(f"Saved {args.ticks}-tick closed-loop demo run to {args.out}")
    print(f"Ticks with a forecaster/backstop trigger: {num_triggers}")
    print(f"Closed-loop cycles run (trigger -> retrieve -> decide -> act): {num_loop_calls}")
    print(f"First full cycle at tick: {first_cycle_tick}")
    print(f"Action distribution: {dict(action_counts)}")
    print(f"Executed distribution (True=action applied, False=fell back safely): {dict(executed_counts)}")


if __name__ == "__main__":
    main()
