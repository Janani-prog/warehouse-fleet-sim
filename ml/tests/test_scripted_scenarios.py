from ml.scripted_scenarios import BASELINE_RATE, SPIKE_END, SPIKE_RATE, SPIKE_START, congestion_spike_rate, make_spike_rate


def test_make_spike_rate_returns_baseline_outside_window():
    rate = make_spike_rate(baseline=0.1, spike=0.5, start=10, end=20)
    assert rate(0) == 0.1
    assert rate(9) == 0.1
    assert rate(20) == 0.1
    assert rate(100) == 0.1


def test_make_spike_rate_returns_spike_inside_window():
    rate = make_spike_rate(baseline=0.1, spike=0.5, start=10, end=20)
    assert rate(10) == 0.5
    assert rate(15) == 0.5
    assert rate(19) == 0.5


def test_congestion_spike_rate_matches_m4_constants():
    assert congestion_spike_rate(SPIKE_START) == SPIKE_RATE
    assert congestion_spike_rate(SPIKE_END) == BASELINE_RATE
    assert congestion_spike_rate(0) == BASELINE_RATE
