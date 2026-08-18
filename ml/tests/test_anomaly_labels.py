from ml.anomaly_labels import (
    COLLISION_RISK_NEAR_MISS_THRESHOLD,
    CONGESTION_ACTIVE_ORDERS_THRESHOLD,
    is_collision_risk_event,
    is_congestion_event,
)


def test_congestion_threshold_boundary():
    assert not is_congestion_event(CONGESTION_ACTIVE_ORDERS_THRESHOLD - 1)
    assert is_congestion_event(CONGESTION_ACTIVE_ORDERS_THRESHOLD)
    assert is_congestion_event(CONGESTION_ACTIVE_ORDERS_THRESHOLD + 50)


def test_collision_risk_threshold_boundary():
    assert not is_collision_risk_event(COLLISION_RISK_NEAR_MISS_THRESHOLD - 1)
    assert is_collision_risk_event(COLLISION_RISK_NEAR_MISS_THRESHOLD)


def test_zero_is_never_anomalous():
    assert not is_congestion_event(0)
    assert not is_collision_risk_event(0)
