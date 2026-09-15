import numpy as np

from eval.stats import interpret, paired_test


def test_all_zero_deltas_not_significant():
    result = paired_test(np.zeros(10))
    assert result["p_value"] == 1.0
    assert result["significant_at_0.05"] is False
    assert result["mean_delta"] == 0.0


def test_consistent_negative_deltas_significant():
    deltas = np.array([-5, -6, -4, -7, -5, -6, -4, -5, -6, -5], dtype=float)
    result = paired_test(deltas)
    assert result["significant_at_0.05"] is True
    assert result["mean_delta"] < 0
    assert result["cohens_d"] < 0
    assert result["ci_95_high"] < 0  # whole CI on the negative side


def test_noisy_deltas_around_zero_not_significant():
    rng = np.random.default_rng(42)
    deltas = rng.normal(loc=0.0, scale=10.0, size=12)
    result = paired_test(deltas)
    assert result["significant_at_0.05"] is False


def test_single_trial_does_not_crash():
    result = paired_test(np.array([3.0]))
    assert result["n"] == 1
    assert result["std_delta"] == 0.0
    assert result["cohens_d"] == 0.0  # undefined with one point, defined as 0 not inf/nan


def test_paired_test_requires_at_least_one_trial():
    import pytest

    with pytest.raises(ValueError):
        paired_test(np.array([]))


def test_interpret_lower_is_better_significant_improvement():
    result = {"p_value": 0.001, "significant_at_0.05": True, "mean_delta": -3.0,
              "ci_95_low": -5.0, "ci_95_high": -1.0, "cohens_d": -0.9, "n": 20}
    text = interpret("near_miss_total", result, lower_is_better=True)
    assert "improved" in text


def test_interpret_lower_is_better_significant_worsening():
    result = {"p_value": 0.001, "significant_at_0.05": True, "mean_delta": 3.0,
              "ci_95_low": 1.0, "ci_95_high": 5.0, "cohens_d": 0.9, "n": 20}
    text = interpret("near_miss_total", result, lower_is_better=True)
    assert "worsened" in text


def test_interpret_not_significant():
    result = {"p_value": 0.6, "significant_at_0.05": False, "mean_delta": 0.4,
              "ci_95_low": -2.0, "ci_95_high": 2.8, "cohens_d": 0.1, "n": 20}
    text = interpret("throughput", result, lower_is_better=False)
    assert "no statistically significant difference" in text
