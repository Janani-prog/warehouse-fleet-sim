"""loop_enabled=False needs no Ollama - it never calls run_loop_step. This
also implicitly tests that the same seed + same order_rate produces
identical simulator dynamics up to the point the loop would first diverge
them, which is the whole premise the causal comparison rests on."""

from __future__ import annotations

from ml.forecaster import Forecaster
from ml.scripted_scenarios import make_spike_rate
import json
import numpy as np
import torch

from eval.causal_trial import run_trial
from ml.forecast_features import FEATURE_DIM
from ml.forecaster_model import AnomalyLSTM
from ml.generate_forecast_dataset import WINDOW


def _write_fake_forecaster(tmp_path):
    torch.manual_seed(0)
    model = AnomalyLSTM()
    torch.save(model.state_dict(), tmp_path / "model.pt")
    np.savez(tmp_path / "normalization.npz", mean=np.zeros(FEATURE_DIM), std=np.ones(FEATURE_DIM))
    with open(tmp_path / "thresholds.json", "w") as f:
        json.dump({"congestion": 0.5, "collision": 0.5}, f)
    with open(tmp_path / "temperatures.json", "w") as f:
        json.dump({"congestion": 1.0, "collision": 1.0}, f)
    return Forecaster(str(tmp_path))


def test_loop_off_trial_runs_without_ollama_and_returns_metrics(tmp_path):
    forecaster = _write_fake_forecaster(tmp_path)
    rate = make_spike_rate(0.3, 0.6, 5, 15)
    metrics = run_trial(seed=0, ticks=30, robots=4, order_rate=rate, forecaster=forecaster, loop_enabled=False)
    assert metrics.seed == 0
    assert metrics.loop_enabled is False
    assert metrics.throughput >= 0
    assert metrics.near_miss_total >= 0
    assert metrics.congestion_ticks >= 0


def test_loop_enabled_without_retriever_raises(tmp_path):
    import pytest

    forecaster = _write_fake_forecaster(tmp_path)
    rate = make_spike_rate(0.3, 0.6, 5, 15)
    with pytest.raises(ValueError):
        run_trial(seed=0, ticks=10, robots=2, order_rate=rate, forecaster=forecaster, loop_enabled=True)


def test_same_seed_same_rate_reproducible_off_trials(tmp_path):
    forecaster = _write_fake_forecaster(tmp_path)
    rate = make_spike_rate(0.3, 0.6, 5, 15)
    m1 = run_trial(seed=7, ticks=25, robots=4, order_rate=rate, forecaster=forecaster, loop_enabled=False)
    m2 = run_trial(seed=7, ticks=25, robots=4, order_rate=rate, forecaster=forecaster, loop_enabled=False)
    assert m1.throughput == m2.throughput
    assert m1.near_miss_total == m2.near_miss_total
    assert m1.congestion_ticks == m2.congestion_ticks


def test_forecaster_window_is_cleared_between_trials(tmp_path):
    forecaster = _write_fake_forecaster(tmp_path)
    rate = make_spike_rate(0.3, 0.6, 5, 15)
    run_trial(seed=1, ticks=WINDOW + 5, robots=3, order_rate=rate, forecaster=forecaster, loop_enabled=False)
    assert len(forecaster.window) == WINDOW  # left full after the first trial
    run_trial(seed=2, ticks=1, robots=3, order_rate=rate, forecaster=forecaster, loop_enabled=False)
    assert len(forecaster.window) == 1  # cleared and restarted for the second trial, not accumulated to WINDOW+1
