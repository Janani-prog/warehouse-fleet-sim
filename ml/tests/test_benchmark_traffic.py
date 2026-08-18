import pandas as pd

from ml.benchmark_traffic import paired_test_summary, run_scenario
from ml.traffic import resolve as classical_avoidance


def test_paired_test_summary_shape_and_columns():
    df = pd.DataFrame(
        {
            "policy": ["classical", "learned"] * 5,
            "seed": [i // 2 for i in range(10)],
            "completion_rate": [0.8, 0.7] * 5,
            "near_miss_count": [100, 80] * 5,
            "mean_detour_ticks": [10, 12] * 5,
        }
    )
    summary = paired_test_summary(df)
    assert set(summary["metric"]) == {"completion_rate", "near_miss_count", "mean_detour_ticks"}
    assert {"mean_classical", "mean_learned", "p_value", "wilcoxon_statistic"} <= set(summary.columns)


def test_run_scenario_returns_expected_keys_and_sane_values():
    result = run_scenario(seed=0, avoidance=classical_avoidance, num_robots=4, order_rate=0.15, ticks=100)
    expected_keys = {
        "seed",
        "total_orders",
        "completed",
        "completion_rate",
        "near_miss_count",
        "replans_around_blockage",
        "mean_detour_ticks",
    }
    assert expected_keys <= set(result.keys())
    assert 0 <= result["completed"] <= result["total_orders"]
    if result["total_orders"] > 0:
        assert 0.0 <= result["completion_rate"] <= 1.0
