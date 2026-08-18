import json

import numpy as np
import torch

from ml.forecast_features import FEATURE_DIM, tick_feature_vector
from ml.forecaster import Forecaster
from ml.forecaster_model import AnomalyLSTM
from ml.generate_forecast_dataset import WINDOW


def _write_fake_model_dir(tmp_path, threshold=0.5):
    torch.manual_seed(0)
    model = AnomalyLSTM()
    torch.save(model.state_dict(), tmp_path / "model.pt")
    np.savez(tmp_path / "normalization.npz", mean=np.zeros(FEATURE_DIM), std=np.ones(FEATURE_DIM))
    with open(tmp_path / "thresholds.json", "w") as f:
        json.dump({"congestion": threshold, "collision": threshold}, f)
    with open(tmp_path / "temperatures.json", "w") as f:
        json.dump({"congestion": 1.0, "collision": 1.0}, f)
    return tmp_path


def test_forecaster_not_ready_before_full_window(tmp_path):
    model_dir = _write_fake_model_dir(tmp_path)
    forecaster = Forecaster(str(model_dir))
    feat = tick_feature_vector(0, 0, [], [], 0)
    for _ in range(WINDOW - 1):
        result = forecaster.step(feat)
        assert result["classifier_ready"] is False
        assert result["congestion_prob"] is None
        assert result["classifier_triggered"] is False


def test_forecaster_ready_and_produces_bounded_probs(tmp_path):
    model_dir = _write_fake_model_dir(tmp_path)
    forecaster = Forecaster(str(model_dir))
    feat = tick_feature_vector(active_orders=10, near_miss_count=2, zone_queue_depths=[1], zone_robot_densities=[2], num_blocked_robots=0)
    result = None
    for _ in range(WINDOW):
        result = forecaster.step(feat)
    assert result["classifier_ready"] is True
    assert 0.0 <= result["congestion_prob"] <= 1.0
    assert 0.0 <= result["collision_prob"] <= 1.0
    assert 0.0 <= result["congestion_prob_raw"] <= 1.0
    assert isinstance(result["classifier_triggered"], bool)


def test_forecaster_backstop_fires_regardless_of_classifier_readiness(tmp_path):
    model_dir = _write_fake_model_dir(tmp_path)
    forecaster = Forecaster(str(model_dir))
    congested_feat = tick_feature_vector(active_orders=999, near_miss_count=0, zone_queue_depths=[], zone_robot_densities=[], num_blocked_robots=0)
    result = forecaster.step(congested_feat)  # first call, window not full yet
    assert result["classifier_ready"] is False
    assert result["backstop_triggered"] is True
    assert result["triggered"] is True


def test_forecaster_combined_trigger_is_or_of_both_signals(tmp_path):
    model_dir = _write_fake_model_dir(tmp_path, threshold=2.0)  # unreachable threshold -> classifier never triggers
    forecaster = Forecaster(str(model_dir))
    feat = tick_feature_vector(active_orders=999, near_miss_count=0, zone_queue_depths=[], zone_robot_densities=[], num_blocked_robots=0)
    result = None
    for _ in range(WINDOW):
        result = forecaster.step(feat)
    assert result["classifier_triggered"] is False
    assert result["backstop_triggered"] is True
    assert result["triggered"] is True  # OR still fires via backstop
