from rag.query_builder import AnomalySignal, build_query, classify_severity


def test_severity_critical_when_backstop_fires_regardless_of_confidence():
    signal = AnomalySignal(
        head="congestion", calibrated_prob=0.1, classifier_triggered=False,
        backstop_triggered_for_head=True,
    )
    assert classify_severity(signal) == "critical"


def test_severity_high_when_classifier_triggered():
    signal = AnomalySignal(
        head="collision", calibrated_prob=0.9, classifier_triggered=True,
        backstop_triggered_for_head=False,
    )
    assert classify_severity(signal) == "high"


def test_severity_moderate_band():
    signal = AnomalySignal(
        head="congestion", calibrated_prob=0.6, classifier_triggered=False,
        backstop_triggered_for_head=False,
    )
    assert classify_severity(signal) == "moderate"


def test_severity_low_band():
    signal = AnomalySignal(
        head="congestion", calibrated_prob=0.2, classifier_triggered=False,
        backstop_triggered_for_head=False,
    )
    assert classify_severity(signal) == "low"


def test_severity_low_when_prob_unavailable():
    signal = AnomalySignal(
        head="collision", calibrated_prob=None, classifier_triggered=False,
        backstop_triggered_for_head=False,
    )
    assert classify_severity(signal) == "low"


def test_build_query_includes_anomaly_type_and_severity():
    signal = AnomalySignal(
        head="collision", calibrated_prob=0.95, classifier_triggered=True,
        backstop_triggered_for_head=False, zone_context="Z_1_2",
    )
    q = build_query(signal)
    assert "collision risk" in q
    assert "high severity" in q
    assert "0.95" in q
    assert "Z_1_2" in q
