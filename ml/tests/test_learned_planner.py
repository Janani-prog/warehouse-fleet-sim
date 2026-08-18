import torch

from ml.learned_planner import rollout
from ml.planner_model import PlannerMLP
from sim.grid import generate_default_layout


def test_rollout_start_equals_goal_short_circuits():
    wh = generate_default_layout()
    model = PlannerMLP()
    start = wh.pickup_points[0]
    path, success = rollout(model, wh, start, start)
    assert path == [start]
    assert success is True


def test_rollout_path_stays_in_free_cells_regardless_of_model_quality():
    torch.manual_seed(0)
    wh = generate_default_layout()
    model = PlannerMLP()  # untrained: exercises the fallback/loop-avoidance logic, not just the happy path
    start = wh.pickup_points[0]
    goal = wh.dropoff_points[0]
    path, _ = rollout(model, wh, start, goal, max_steps=50)

    assert path[0] == start
    for cell in path:
        assert wh.is_free(*cell)
    for a, b in zip(path, path[1:]):
        assert abs(a[0] - b[0]) + abs(a[1] - b[1]) == 1


def test_rollout_returns_bool_success_flag():
    wh = generate_default_layout()
    model = PlannerMLP()
    start = wh.pickup_points[0]
    goal = wh.dropoff_points[0]
    _, success = rollout(model, wh, start, goal, max_steps=10)
    assert isinstance(success, bool)
