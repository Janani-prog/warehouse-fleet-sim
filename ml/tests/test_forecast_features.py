import numpy as np

from ml.forecast_features import FEATURE_DIM, episode_feature_matrix, tick_feature_vector


def test_tick_feature_vector_shape_and_values():
    v = tick_feature_vector(
        active_orders=12,
        near_miss_count=3,
        zone_queue_depths=[0, 2, 4],
        zone_robot_densities=[1, 3, 0],
        num_blocked_robots=2,
    )
    assert v.shape == (FEATURE_DIM,)
    assert v.tolist() == [12, 3, 4, 2.0, 3, 2]


def test_tick_feature_vector_handles_empty_zones():
    v = tick_feature_vector(0, 0, [], [], 0)
    assert v.tolist() == [0, 0, 0, 0.0, 0, 0]


class _FakeTelemetry:
    def __init__(self, tick_rows, zone_rows, robot_rows):
        self.tick_rows = tick_rows
        self.zone_rows = zone_rows
        self.robot_rows = robot_rows


def test_episode_feature_matrix_aligns_ticks():
    telemetry = _FakeTelemetry(
        tick_rows=[
            {"tick": 0, "active_orders": 5, "near_miss_count": 1},
            {"tick": 1, "active_orders": 6, "near_miss_count": 2},
        ],
        zone_rows=[
            {"tick": 0, "zone_id": "Z_0_0", "queue_depth": 1, "robot_density": 2},
            {"tick": 1, "zone_id": "Z_0_0", "queue_depth": 3, "robot_density": 1},
        ],
        robot_rows=[
            {"tick": 0, "robot_id": 0, "state": "blocked"},
            {"tick": 1, "robot_id": 0, "state": "moving"},
        ],
    )
    matrix = episode_feature_matrix(telemetry)
    assert matrix.shape == (2, FEATURE_DIM)
    assert matrix[0].tolist() == [5, 1, 1, 1.0, 2, 1]  # tick 0 has 1 blocked robot
    assert matrix[1].tolist() == [6, 2, 3, 3.0, 1, 0]  # tick 1 has 0 blocked robots


def test_episode_feature_matrix_empty_telemetry():
    telemetry = _FakeTelemetry([], [], [])
    matrix = episode_feature_matrix(telemetry)
    assert matrix.shape == (0, FEATURE_DIM)
