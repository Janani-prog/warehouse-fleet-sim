from ml.rule_backstop import backstop_triggered


def test_backstop_fires_on_congestion_alone():
    assert backstop_triggered(active_orders=100, near_miss_count=0)


def test_backstop_fires_on_collision_risk_alone():
    assert backstop_triggered(active_orders=0, near_miss_count=20)


def test_backstop_silent_when_both_normal():
    assert not backstop_triggered(active_orders=5, near_miss_count=1)


def test_backstop_independent_of_classifier_module():
    # the backstop must not import anything from the LSTM/calibration stack -
    # that's the whole point of it being a redundant, non-learned trigger.
    import ml.rule_backstop as mod

    assert "forecaster_model" not in mod.__dict__
    assert "torch" not in dir(mod)
