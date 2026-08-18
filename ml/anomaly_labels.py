"""Ground-truth anomaly event rules, applied to raw per-tick telemetry.

Thresholds are data-driven, not guessed: chosen from the observed
active_orders/near_miss_count distributions across a mix of healthy and
deliberately overloaded (robots, order_rate) configs (see CLAUDE.md Current
Status for the exact sweep). Congestion at 40+ backlogged orders sits
~around the 70th percentile of that pooled distribution; collision-risk at
6+ near-misses in a tick sits between the 75th and 90th percentile - both
common enough to give the classifier real positive examples to learn from,
rare enough to still mean something.

These are independent of the LSTM classifier - they define what "actually
happened," which is what the classifier is trained to predict *ahead of
time*, and what the rule-based backstop (ml/rule_backstop.py) checks for
*right now* as a redundant, non-learned trigger.
"""

from __future__ import annotations

CONGESTION_ACTIVE_ORDERS_THRESHOLD = 40
COLLISION_RISK_NEAR_MISS_THRESHOLD = 6


def is_congestion_event(active_orders: int) -> bool:
    return active_orders >= CONGESTION_ACTIVE_ORDERS_THRESHOLD


def is_collision_risk_event(near_miss_count: int) -> bool:
    return near_miss_count >= COLLISION_RISK_NEAR_MISS_THRESHOLD
