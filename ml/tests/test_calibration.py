import numpy as np

from ml.calibration import apply_temperature, choose_threshold_for_precision, fit_temperature


def test_apply_temperature_matches_manual_sigmoid():
    logits = np.array([-2.0, 0.0, 2.0])
    out = apply_temperature(logits, temperature=1.0)
    expected = 1.0 / (1.0 + np.exp(-logits))
    assert np.allclose(out, expected)


def test_apply_temperature_higher_temp_pulls_toward_half():
    logits = np.array([4.0])
    sharp = apply_temperature(logits, temperature=1.0)
    softened = apply_temperature(logits, temperature=10.0)
    assert softened < sharp
    assert softened > 0.5  # still positive, just less extreme


def test_fit_temperature_recovers_reasonable_scale_on_well_calibrated_data():
    rng = np.random.default_rng(0)
    n = 2000
    true_logits = rng.normal(0, 2, size=n)
    probs = 1.0 / (1.0 + np.exp(-true_logits))
    labels = (rng.random(n) < probs).astype(np.float32)
    # logits are already well-calibrated for these labels -> T should land near 1
    T = fit_temperature(true_logits, labels)
    assert 0.5 < T < 2.0


def test_choose_threshold_for_precision_meets_target_when_achievable():
    labels = np.array([1, 1, 1, 0, 0, 0, 0, 0, 0, 0])
    probs = np.array([0.9, 0.85, 0.8, 0.75, 0.3, 0.2, 0.1, 0.05, 0.02, 0.01])
    threshold = choose_threshold_for_precision(probs, labels, target_precision=1.0)
    pred = (probs >= threshold).astype(int)
    tp = ((pred == 1) & (labels == 1)).sum()
    fp = ((pred == 1) & (labels == 0)).sum()
    assert tp / (tp + fp) >= 1.0


def test_choose_threshold_for_precision_falls_back_when_unachievable():
    labels = np.array([0, 0, 0, 0, 1])
    probs = np.array([0.9, 0.8, 0.7, 0.6, 0.5])  # best case: one positive is never highest-ranked alone
    threshold = choose_threshold_for_precision(probs, labels, target_precision=0.99)
    assert 0.0 <= threshold <= 1.0
