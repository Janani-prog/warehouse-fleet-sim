import pandas as pd

from eval.causal_harness import METRICS_LOWER_IS_BETTER, build_report


def _synthetic_df(n=15):
    rows = []
    for seed in range(n):
        rows.append({
            "seed": seed,
            "throughput_on": 20 + (seed % 3), "throughput_off": 15 + (seed % 3),
            "mean_wait_on": 10.0, "mean_wait_off": 14.0,
            "near_miss_total_on": 5, "near_miss_total_off": 9,
            "congestion_ticks_on": 2, "congestion_ticks_off": 6,
        })
    return pd.DataFrame(rows)


def test_build_report_covers_all_metrics():
    df = _synthetic_df()
    report = build_report(df)
    assert set(report["metrics"]) == set(METRICS_LOWER_IS_BETTER)
    assert report["n_pairs"] == 15


def test_build_report_detects_consistent_improvement():
    df = _synthetic_df()
    report = build_report(df)
    # throughput_on is consistently higher than off by a fixed amount -> significant, "improved" (higher is better)
    assert report["metrics"]["throughput"]["significant_at_0.05"] is True
    assert "improved" in report["metrics"]["throughput"]["interpretation"]
    # congestion_ticks_on consistently lower -> significant, "improved" (lower is better)
    assert report["metrics"]["congestion_ticks"]["significant_at_0.05"] is True
    assert "improved" in report["metrics"]["congestion_ticks"]["interpretation"]


def test_build_report_handles_nan_deltas_gracefully():
    df = _synthetic_df(n=3)
    df.loc[0, "mean_wait_on"] = float("nan")
    report = build_report(df)
    # one NaN pair dropped, the other two still produce a valid result
    assert report["metrics"]["mean_wait"]["n"] == 2
