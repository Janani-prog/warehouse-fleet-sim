"""Rule-based backstop: a simple threshold check on the *current* tick's raw
telemetry, computed entirely independently of the LSTM (same thresholds as
the ground-truth anomaly labels in ml/anomaly_labels.py, since the backstop's
job is literally "notice the anomaly is already happening", not predict it
ahead of time). This is the redundant, always-available trigger the
architecture doc calls for: if the classifier is unreachable, undertrained,
or just wrong, the fleet still gets *a* signal, degrading gracefully rather
than silently losing detection."""

from __future__ import annotations

from ml.anomaly_labels import is_collision_risk_event, is_congestion_event


def backstop_triggered(active_orders: int, near_miss_count: int) -> bool:
    return is_congestion_event(active_orders) or is_collision_risk_event(near_miss_count)
