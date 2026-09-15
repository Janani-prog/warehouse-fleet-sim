"""Agent accuracy eval: given a fixed, correct (anomaly, retrieved SOP,
action candidates) input - i.e. retrieval is held constant at ground truth
so this isolates the LLM's action-selection accuracy from M7's retrieval
accuracy, matching the backlog's "(anomaly, SOP, correct action) test set"
wording - does the prompt-only agent choose the SOP's recommended action
(or correctly fall back to NO_ACTION when no valid candidate is offered)?

Run: python -m agent.benchmark_agent
Requires a running local Ollama server with llama3.1:8b pulled.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from agent.context import ActionCandidates
from agent.executor import ActionExecutor
from agent.ollama_client import DEFAULT_MODEL, chat_json
from agent.prompt import SYSTEM_PROMPT, build_user_prompt
from rag.corpus import get_doc
from sim.entities import Order, OrderStatus
from sim.world import World

EVAL_SET_PATH = Path(__file__).parent / "data" / "agent_eval_set.json"
RESULTS_DIR = Path(__file__).parent.parent / "data" / "results"

TARGET_ACCURACY = 0.75


def load_eval_set() -> list[dict]:
    with open(EVAL_SET_PATH, encoding="utf-8") as f:
        return json.load(f)


def _world_for_case(case: dict) -> World:
    world = World(seed=0, num_robots=8, order_rate=0.0)
    cand = case["candidates"]
    if cand.get("order_id") is not None:
        world.orders[cand["order_id"]] = Order(
            id=cand["order_id"], origin=(0, 1), destination=(3, 3), arrival_tick=0,
            status=OrderStatus.ASSIGNED, assigned_robot=cand["order_robot_id"], assign_tick=0,
        )
        robot = next(r for r in world.robots if r.id == cand["order_robot_id"])
        robot.order_id = cand["order_id"]
    return world


def run_case(case: dict, model: str = DEFAULT_MODEL) -> dict:
    sop = get_doc(case["sop_doc_id"])
    candidates = ActionCandidates(**case["candidates"])
    world = _world_for_case(case)

    user_prompt = build_user_prompt(case["query_text"], sop.title, sop.text, sop.recommended_action, candidates)
    raw = chat_json(SYSTEM_PROMPT, user_prompt, model=model)

    executor = ActionExecutor(world)
    decision = executor.execute(raw, candidates)

    return {
        "case_id": case["case_id"],
        "expected_action": case["expected_action"],
        "predicted_action": decision.action,
        "correct": decision.action == case["expected_action"],
        "executed": decision.executed,
        "error": decision.error,
        "raw_llm_output": raw,
    }


def main() -> None:
    eval_set = load_eval_set()
    rows = [run_case(case) for case in eval_set]

    accuracy = sum(r["correct"] for r in rows) / len(rows)

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(rows)
    df.to_csv(RESULTS_DIR / "agent_benchmark.csv", index=False)

    report = {
        "n_cases": len(rows),
        "accuracy": accuracy,
        "target_accuracy": TARGET_ACCURACY,
        "meets_target": accuracy >= TARGET_ACCURACY,
        "model": DEFAULT_MODEL,
        "misses": [r["case_id"] for r in rows if not r["correct"]],
    }
    with open(RESULTS_DIR / "agent_benchmark_report.json", "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    print(df[["case_id", "expected_action", "predicted_action", "correct", "executed"]])
    print()
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
