import numpy as np

from ml.traffic_features import BLOCKED, FEATURE_DIM, FREE, ROBOT, WINDOW_RADIUS, desired_direction_onehot, encode, local_traffic_window
from sim.grid import generate_default_layout


def test_feature_dim():
    assert FEATURE_DIM == (2 * WINDOW_RADIUS + 1) ** 2 + 4


def test_window_marks_other_robot_distinct_from_rack_and_free():
    wh = generate_default_layout()
    pos = (10, 5)
    other = (11, 5)
    window = local_traffic_window(wh, pos, {other}).reshape(5, 5)
    # (11,5) is dx=+1,dy=0 from (10,5) -> window index (radius+0, radius+1)
    assert window[WINDOW_RADIUS, WINDOW_RADIUS + 1] == ROBOT
    assert window[WINDOW_RADIUS, WINDOW_RADIUS] == FREE  # own cell, no robot recorded there


def test_window_out_of_bounds_is_blocked_not_free_or_robot():
    wh = generate_default_layout()
    window = local_traffic_window(wh, (0, 0), set()).reshape(5, 5)
    assert window[0, 0] == BLOCKED


def test_desired_direction_onehot_matches_delta():
    onehot = desired_direction_onehot((5, 5), (6, 5))  # +x
    assert onehot.tolist() == [1.0, 0.0, 0.0, 0.0]


def test_desired_direction_onehot_zero_when_not_moving():
    onehot = desired_direction_onehot((5, 5), (5, 5))
    assert onehot.sum() == 0.0


def test_encode_shape_and_dtype():
    wh = generate_default_layout()
    feat = encode(wh, (10, 5), (11, 5), {(12, 5)})
    assert feat.shape == (FEATURE_DIM,)
    assert feat.dtype == np.float32
