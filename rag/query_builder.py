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

# not derived from anomaly_type.replace('_', ' ') + " risk" - that produced
# "collision risk risk" for the collision_risk head (a real bug: caught by
# eyeballing generated query text directly, not by the tests, since they
# only substring-checked "collision risk" which is still true inside
# "collision risk risk").
ANOMALY_TYPE_LABEL = {
    "congestion": "Congestion risk",
    "collision_risk": "Collision risk",
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
    """Deliberately does NOT print the raw calibrated_prob magnitude in the
    query text (bug found integrating M8 end-to-end, documented in CLAUDE.md
    M8 notes): M4's temperature-scaled congestion head has an operating
    threshold of ~0.0007, so a genuinely triggering, "high severity"
    congestion confidence can be a tiny number like 0.0015 - printed as
    "confidence 0.00" at 2 decimal places, that read as contradictory right
    next to "high severity" and measurably confused the embedding retriever
    (it started preferring the confidence-band glossary doc over the correct
    congestion-high SOP). Severity is already computed relative to the
    model's own threshold in classify_severity() - the query text says so in
    words instead of repeating a magnitude on a scale that isn't comparable
    across the two forecaster heads (collision's threshold is a normal-
    looking 0.805, congestion's is not)."""
    severity = classify_severity(signal)
    parts = [f"{ANOMALY_TYPE_LABEL[signal.anomaly_type]}, {severity} severity."]
    if signal.backstop_triggered_for_head:
        parts.append("Rule-based backstop is firing this tick - this is happening now, not a forecast.")
    elif signal.classifier_triggered:
        parts.append("Calibrated classifier confidence has crossed the operating threshold - an early warning ahead of the rule-based backstop.")
    elif signal.calibrated_prob is not None and signal.calibrated_prob >= MODERATE_FLOOR:
        parts.append("Calibrated classifier confidence is elevated but has not yet crossed the operating threshold.")
    else:
        parts.append("Calibrated classifier confidence is low, or not yet available.")
    if signal.zone_context:
        parts.append(f"Location: {signal.zone_context}.")
    return " ".join(parts)
