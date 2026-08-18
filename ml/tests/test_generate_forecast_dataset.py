import numpy as np

from ml.anomaly_labels import CONGESTION_ACTIVE_ORDERS_THRESHOLD
from ml.forecast_features import FEATURE_DIM
from ml.generate_forecast_dataset import HORIZON, WINDOW, build_windows


def test_build_windows_shapes_and_count():
    T = WINDOW + HORIZON + 5
    matrix = np.zeros((T, FEATURE_DIM), dtype=np.float32)
    rows = build_windows(matrix)
    assert len(rows) == T - HORIZON - (WINDOW - 1)
    assert len(rows[0]["window"]) == WINDOW * FEATURE_DIM


def test_build_windows_labels_future_congestion_correctly():
    T = WINDOW + HORIZON + 5
    matrix = np.zeros((T, FEATURE_DIM), dtype=np.float32)
    # spike active_orders (column 0) just after the first eligible window ends
    spike_tick = WINDOW + 2
    matrix[spike_tick, 0] = CONGESTION_ACTIVE_ORDERS_THRESHOLD

    rows = build_windows(matrix)
    row_by_tick = {r["tick"]: r for r in rows}

    # a window ending right before the spike, with the spike inside its horizon
    assert row_by_tick[WINDOW - 1]["label_congestion"] == 1.0
    # a window ending well after the spike has left the horizon
    late_tick = spike_tick + HORIZON
    if late_tick in row_by_tick:
        assert row_by_tick[late_tick]["label_congestion"] == 0.0


def test_build_windows_no_false_positive_on_flat_series():
    T = WINDOW + HORIZON + 5
    matrix = np.zeros((T, FEATURE_DIM), dtype=np.float32)
    rows = build_windows(matrix)
    assert all(r["label_congestion"] == 0.0 for r in rows)
    assert all(r["label_collision"] == 0.0 for r in rows)
