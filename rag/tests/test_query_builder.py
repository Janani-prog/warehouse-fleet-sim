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
    assert "Collision risk" in q
    assert "collision risk risk" not in q.lower()  # regression: double "risk" bug
    assert "high severity" in q
    assert "Z_1_2" in q


def test_build_query_does_not_print_raw_confidence_magnitude():
    """Regression test for a real bug found integrating M8: a tiny but
    genuinely-triggering congestion confidence (M4's threshold is ~0.0007)
    printed as "confidence 0.00" at 2dp, which reads as contradictory next
    to "high severity" and misled the embedding retriever in practice."""
    signal = AnomalySignal(
        head="congestion", calibrated_prob=0.0015, classifier_triggered=True,
        backstop_triggered_for_head=False,
    )
    q = build_query(signal)
    assert "high severity" in q
    assert "0.00" not in q
    assert "0.0015" not in q
