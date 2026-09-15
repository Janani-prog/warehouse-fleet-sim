"""Full closed-loop tests: forecaster signal -> RAG -> LLM agent -> executor.
Needs a running local Ollama server (both nomic-embed-text for retrieval and
llama3.1:8b for the agent) - skips cleanly if unreachable rather than
failing the rest of the suite, same pattern as rag/tests/test_dense_and_hybrid.py.
"""

from __future__ import annotations

import pytest

from agent.action_schema import WHITELISTED_ACTIONS
from agent.orchestrator import run_loop_step
from agent.ollama_client import OllamaUnavailableError, chat_json
from rag.hybrid_retrieval import HybridRetriever
from sim.world import World

BASE_FORECASTER_RESULT = {
    "classifier_ready": True,
    "congestion_prob": 0.1,
    "collision_prob": 0.1,
    "congestion_prob_raw": 0.1,
    "collision_prob_raw": 0.1,
    "congestion_classifier_triggered": False,
    "collision_classifier_triggered": False,
    "classifier_triggered": False,
    "backstop_triggered": False,
    "triggered": False,
}


@pytest.fixture(scope="module")
def ollama_available() -> bool:
    try:
        chat_json("You are a test.", 'Respond with {"ok": true}', timeout=15)
        return True
    except OllamaUnavailableError:
        return False


@pytest.fixture(scope="module")
def retriever(ollama_available):
    if not ollama_available:
        pytest.skip("Ollama not reachable - skipping closed-loop orchestrator tests")
    return HybridRetriever(use_cache=True)


def _world_with_tick(active_orders: int, near_miss_count: int) -> World:
    world = World(seed=0, num_robots=3, order_rate=0.0)
    world.tick()  # produce one real tick's worth of telemetry/state
    # overwrite the logged tick row to script a specific anomaly scenario,
    # independent of what the tiny 3-robot/0-order-rate run actually produced
    world.telemetry.tick_rows[-1]["active_orders"] = active_orders
    world.telemetry.tick_rows[-1]["near_miss_count"] = near_miss_count
    return world


def test_no_trigger_returns_none(retriever):
    world = _world_with_tick(active_orders=1, near_miss_count=0)
    result = run_loop_step(world, BASE_FORECASTER_RESULT, retriever)
    assert result is None


def test_congestion_backstop_trigger_resolves_congestion_sop(retriever):
    # realistic joint scenario: active_orders this high would also drive the
    # classifier's own congestion_prob high (M4's report notes congestion is
    # close to a deterministic function of the active-orders trajectory) -
    # an artificially low classifier confidence alongside a firing backstop
    # is itself an edge case the backstop-vs-classifier-precedence SOP
    # exists for, not what this test is checking.
    world = _world_with_tick(active_orders=999, near_miss_count=0)
    forecaster_result = {
        **BASE_FORECASTER_RESULT, "congestion_prob": 0.95,
        "congestion_classifier_triggered": True, "classifier_triggered": True,
        "backstop_triggered": True, "triggered": True,
    }
    result = run_loop_step(world, forecaster_result, retriever)
    assert result is not None
    assert result.anomaly_type == "congestion"
    assert result.severity == "critical"
    assert result.sop_doc_id is not None and result.sop_doc_id.startswith("congestion-")
    assert result.decision.action in WHITELISTED_ACTIONS
    assert isinstance(result.decision.executed, bool)


def test_collision_backstop_trigger_resolves_collision_sop(retriever):
    world = _world_with_tick(active_orders=1, near_miss_count=999)
    forecaster_result = {
        **BASE_FORECASTER_RESULT, "collision_prob": 0.95,
        "collision_classifier_triggered": True, "classifier_triggered": True,
        "backstop_triggered": True, "triggered": True,
    }
    result = run_loop_step(world, forecaster_result, retriever)
    assert result is not None
    assert result.anomaly_type == "collision_risk"
    assert result.sop_doc_id is not None and result.sop_doc_id.startswith("collision-")
    assert result.decision.action in WHITELISTED_ACTIONS


def test_concurrent_trigger_when_both_heads_critical(retriever):
    world = _world_with_tick(active_orders=999, near_miss_count=999)
    forecaster_result = {
        **BASE_FORECASTER_RESULT, "congestion_prob": 0.95, "collision_prob": 0.95,
        "congestion_classifier_triggered": True, "collision_classifier_triggered": True,
        "classifier_triggered": True, "backstop_triggered": True, "triggered": True,
    }
    result = run_loop_step(world, forecaster_result, retriever)
    assert result is not None
    assert result.anomaly_type == "concurrent"
    assert result.decision.action in WHITELISTED_ACTIONS


def test_ollama_unreachable_during_chat_falls_back_to_no_action(monkeypatch, retriever):
    from agent import orchestrator as orch_module

    def _raise(*args, **kwargs):
        raise OllamaUnavailableError("simulated outage")

    monkeypatch.setattr(orch_module, "chat_json", _raise)
    world = _world_with_tick(active_orders=999, near_miss_count=0)
    forecaster_result = {**BASE_FORECASTER_RESULT, "backstop_triggered": True, "triggered": True}
    result = run_loop_step(world, forecaster_result, retriever)
    assert result is not None
    assert result.decision.action == "NO_ACTION"
    assert result.decision.executed is False
