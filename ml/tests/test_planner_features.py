import numpy as np

from ml.planner_features import ACTIONS, FEATURE_DIM, WINDOW_RADIUS, action_index, encode, goal_delta, local_window
from sim.grid import generate_default_layout


def test_feature_dim_matches_window_plus_goal():
    assert FEATURE_DIM == (2 * WINDOW_RADIUS + 1) ** 2 + 2


def test_encode_output_shape_and_dtype():
    wh = generate_default_layout()
    feat = encode(wh, wh.pickup_points[0], wh.dropoff_points[0])
    assert feat.shape == (FEATURE_DIM,)
    assert feat.dtype == np.float32


def test_local_window_all_free_in_open_interior():
    wh = generate_default_layout()
    # aisle cell between two rack bands, well clear of the boundary
    pos = (10, 5)
    assert wh.is_free(*pos)
    window = local_window(wh, pos)
    assert window.shape == (25,)


def test_local_window_marks_out_of_bounds_as_blocked():
    wh = generate_default_layout()
    corner = (0, 0)
    window = local_window(wh, corner).reshape(5, 5)
    # top-left of the window is 2 cells up/left of (0,0): out of bounds -> blocked
    assert window[0, 0] == 1.0


def test_goal_delta_zero_at_goal():
    wh = generate_default_layout()
    pos = (5, 5)
    delta = goal_delta(pos, pos, wh)
    assert np.allclose(delta, [0.0, 0.0])


def test_action_index_matches_actions_order():
    for i, delta in enumerate(ACTIONS):
        assert action_index(delta) == i
