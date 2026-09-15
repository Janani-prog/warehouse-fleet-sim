"""Turns a forecaster signal for one anomaly head into a natural-language
retrieval query, plus the severity band used both to build that query and
(by the M8 agent orchestrator) to reason about urgency.

Severity bands mirror the "confidence-threshold-reference" SOP doc:
- critical: rule-based backstop fired for this anomaly type, this tick
- high:     classifier confidence >= the calibrated operating threshold
- moderate: classifier confidence >= 0.5 but below threshold
- low:      classifier confidence < 0.5 (or classifier not ready yet)

congestion and collision_risk each have their own backstop check
(ml.anomaly_labels.is_congestion_event / is_collision_risk_event) since
ml.forecaster.Forecaster.step()'s single `backstop_triggered` bool is an OR
of both and isn't specific enough to build a per-head query from.
"""

from __future__ import annotations

from dataclasses import dataclass

MODERATE_FLOOR = 0.5

HEAD_TO_ANOMALY_TYPE = {
    "congestion": "congestion",
    "collision": "collision_risk",
}


@dataclass(frozen=True)
class AnomalySignal:
    head: str  # "congestion" or "collision"
    calibrated_prob: float | None
    classifier_triggered: bool
    backstop_triggered_for_head: bool
    zone_context: str | None = None  # e.g. "zone Z_0_2, dock-adjacent"

    @property
    def anomaly_type(self) -> str:
        return HEAD_TO_ANOMALY_TYPE[self.head]


def classify_severity(signal: AnomalySignal) -> str:
    if signal.backstop_triggered_for_head:
        return "critical"
    if signal.classifier_triggered:
        return "high"
    if signal.calibrated_prob is not None and signal.calibrated_prob >= MODERATE_FLOOR:
        return "moderate"
    return "low"


def build_query(signal: AnomalySignal) -> str:
    severity = classify_severity(signal)
    prob_str = f"{signal.calibrated_prob:.2f}" if signal.calibrated_prob is not None else "unavailable"
    parts = [
        f"{signal.anomaly_type.replace('_', ' ')} risk, {severity} severity.",
        f"Calibrated confidence {prob_str}.",
    ]
    if signal.backstop_triggered_for_head:
        parts.append("Rule-based backstop is firing this tick.")
    if signal.zone_context:
        parts.append(f"Location: {signal.zone_context}.")
    return " ".join(parts)
