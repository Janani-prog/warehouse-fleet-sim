import json

import numpy as np
import pandas as pd
import pytest
import torch

from eval.causal_harness import METRICS_LOWER_IS_BETTER, build_report, run_paired_trials
from ml.forecast_features import FEATURE_DIM
from ml.forecaster_model import AnomalyLSTM
from ml.scripted_scenarios import make_spike_rate
from rag.embeddings import OllamaUnavailableError, embed_text


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


def _write_fake_forecaster(tmp_path):
    torch.manual_seed(0)
    model = AnomalyLSTM()
    torch.save(model.state_dict(), tmp_path / "model.pt")
    np.savez(tmp_path / "normalization.npz", mean=np.zeros(FEATURE_DIM), std=np.ones(FEATURE_DIM))
    with open(tmp_path / "thresholds.json", "w") as f:
        json.dump({"congestion": 0.5, "collision": 0.5}, f)
    with open(tmp_path / "temperatures.json", "w") as f:
        json.dump({"congestion": 1.0, "collision": 1.0}, f)
    return tmp_path


@pytest.fixture(scope="module")
def ollama_available() -> bool:
    try:
        embed_text("connectivity check")
        return True
    except OllamaUnavailableError:
        return False


def test_run_paired_trials_resumes_from_checkpoint(tmp_path, ollama_available):
    """A kill mid-run must lose at most one in-flight pair, not the whole
    thing (see CLAUDE.md's M9 note on real OOM kills hit while developing
    this) - so a rerun with the same checkpoint_path must skip seeds already
    recorded there rather than redoing them."""
    if not ollama_available:
        pytest.skip("Ollama not reachable - skipping checkpoint/resume test")

    forecaster_dir = _write_fake_forecaster(tmp_path)
    rate = make_spike_rate(0.3, 0.6, 3, 6)
    checkpoint = tmp_path / "checkpoint.csv"

    # simulate a prior partial run: one pair already recorded
    pd.DataFrame([{
        "seed": 100, "throughput_on": 1, "throughput_off": 1,
        "mean_wait_on": 1.0, "mean_wait_off": 1.0,
        "near_miss_total_on": 0, "near_miss_total_off": 0,
        "congestion_ticks_on": 0, "congestion_ticks_off": 0,
    }]).to_csv(checkpoint, index=False)

    df = run_paired_trials(
        n_pairs=2, seed_start=100, ticks=8, robots=2, order_rate=rate,
        forecaster_model_dir=str(forecaster_dir), checkpoint_path=checkpoint,
    )
    assert len(df) == 2  # seed 100 reused from checkpoint, seed 101 actually run
    assert set(df["seed"]) == {100, 101}
