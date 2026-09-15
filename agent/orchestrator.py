"""Wires the full closed loop for one tick: forecaster/backstop trigger ->
RAG retrieval -> LLM agent -> action executor -> simulator state. This is
the concrete implementation of the data-flow diagram in
docs/TECHNICAL_ARCHITECTURE.md section 1 / 4.

Only called when ml.forecaster.Forecaster.step()'s combined `triggered` flag
is True for the current tick - the confidence gate (Locked Design Decision
#2) happens upstream of this module, not inside it.
"""

from __future__ import annotations

from dataclasses import dataclass

from agent.context import build_candidates
from agent.executor import ActionExecutor, AgentDecision
from agent.ollama_client import DEFAULT_MODEL, OllamaUnavailableError, chat_json
from agent.prompt import SYSTEM_PROMPT, build_user_prompt
from ml.anomaly_labels import is_collision_risk_event, is_congestion_event
from rag.hybrid_retrieval import HybridRetriever
from rag.query_builder import AnomalySignal, build_query, classify_severity
from sim.world import World


@dataclass(frozen=True)
class LoopResult:
    anomaly_type: str  # "congestion", "collision_risk", or "concurrent"
    severity: str
    confidence: float | None
    sop_doc_id: str | None
    retrieval_score: float | None
    decision: AgentDecision


def _signal_for_head(head: str, forecaster_result: dict, active_orders: int, near_miss_count: int) -> AnomalySignal:
    backstop_for_head = (
        is_congestion_event(active_orders) if head == "congestion" else is_collision_risk_event(near_miss_count)
    )
    return AnomalySignal(
        head=head,
        calibrated_prob=forecaster_result[f"{head}_prob"],
        classifier_triggered=forecaster_result[f"{head}_classifier_triggered"],
        backstop_triggered_for_head=backstop_for_head,
    )


def run_loop_step(
    world: World,
    forecaster_result: dict,
    retriever: HybridRetriever,
    model: str = DEFAULT_MODEL,
) -> LoopResult | None:
    """Returns None if neither head is genuinely triggered (nothing to do).
    Never raises: an unreachable Ollama server is treated the same as
    malformed LLM output - falls back to NO_ACTION via the executor path,
    logged with the connection error as the reason."""
    if not forecaster_result["triggered"]:
        return None

    tick_row = world.telemetry.tick_rows[-1]
    active_orders, near_miss_count = tick_row["active_orders"], tick_row["near_miss_count"]

    congestion_signal = _signal_for_head("congestion", forecaster_result, active_orders, near_miss_count)
    collision_signal = _signal_for_head("collision", forecaster_result, active_orders, near_miss_count)
    congestion_severity = classify_severity(congestion_signal)
    collision_severity = classify_severity(collision_signal)

    congestion_active = congestion_severity in ("high", "critical")
    collision_active = collision_severity in ("high", "critical")

    if congestion_active and collision_active:
        anomaly_type = "concurrent"
        severity = "critical" if "critical" in (congestion_severity, collision_severity) else "high"
        confidence = collision_signal.calibrated_prob  # collision-risk takes priority per the concurrent-multi-anomaly SOP
        query_text = (
            "Both congestion and collision-risk anomalies triggered this tick. "
            f"Congestion: {congestion_severity} severity, confidence "
            f"{congestion_signal.calibrated_prob if congestion_signal.calibrated_prob is not None else 'unavailable'}. "
            f"Collision risk: {collision_severity} severity, confidence "
            f"{collision_signal.calibrated_prob if collision_signal.calibrated_prob is not None else 'unavailable'}."
        )
    elif collision_active:
        anomaly_type = "collision_risk"
        severity = collision_severity
        confidence = collision_signal.calibrated_prob
        query_text = build_query(collision_signal)
    elif congestion_active:
        anomaly_type = "congestion"
        severity = congestion_severity
        confidence = congestion_signal.calibrated_prob
        query_text = build_query(congestion_signal)
    else:
        # backstop-only trigger with neither head reaching high/critical via
        # this proxy (rare edge case) - still worth a NO_ACTION-eligible pass
        anomaly_type = "congestion" if is_congestion_event(active_orders) else "collision_risk"
        severity = "critical"
        confidence = None
        signal = congestion_signal if anomaly_type == "congestion" else collision_signal
        query_text = build_query(signal)

    top = retriever.retrieve(query_text, k=1)
    sop_doc_id = top[0].doc.id if top else None
    retrieval_score = top[0].score if top else None
    candidates = build_candidates(world)

    executor = ActionExecutor(world)
    if not top:
        decision = executor.execute(None, candidates)
    else:
        sop = top[0].doc
        user_prompt = build_user_prompt(query_text, sop.title, sop.text, sop.recommended_action, candidates)
        try:
            raw = chat_json(SYSTEM_PROMPT, user_prompt, model=model)
        except OllamaUnavailableError:
            raw = None
        decision = executor.execute(raw, candidates)

    return LoopResult(
        anomaly_type=anomaly_type,
        severity=severity,
        confidence=confidence,
        sop_doc_id=sop_doc_id,
        retrieval_score=retrieval_score,
        decision=decision,
    )
