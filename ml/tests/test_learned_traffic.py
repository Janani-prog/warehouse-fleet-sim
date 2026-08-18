import torch

from ml.learned_traffic import make_learned_avoidance
from ml.traffic_model import TrafficMLP
from sim.grid import Warehouse


def _empty_warehouse(width=7, height=7):
    return Warehouse(
        width=width,
        height=height,
        racks=frozenset(),
        zone_size=(width, height),
        spawn_points=(),
        dropoff_points=(),
        pickup_points=(),
    )


class _AlwaysProceedModel(torch.nn.Module):
    """Adversarial stand-in: predicts 'proceed' for everyone, regardless of
    input. Used to prove the safety net catches bad model output rather
    than trusting it."""

    def forward(self, x):
        return torch.full((x.shape[0],), 10.0)  # large positive logit -> sigmoid ~1


def test_learned_avoidance_never_produces_a_collision_even_with_adversarial_model():
    wh = _empty_warehouse()
    avoidance = make_learned_avoidance(_AlwaysProceedModel())
    positions = {0: (2, 2), 1: (3, 2)}
    desired = {0: (3, 2), 1: (2, 2)}  # head-on swap: must never both proceed
    final = avoidance(wh, positions, desired)

    cells = list(final.values())
    assert len(cells) == len(set(cells)), "vertex collision despite adversarial model"
    assert not (final[0] == desired[0] and final[1] == desired[1]), "swap collision slipped through"


def test_learned_avoidance_no_conflict_case_with_untrained_model_stays_safe():
    torch.manual_seed(0)
    wh = _empty_warehouse()
    model = TrafficMLP()
    avoidance = make_learned_avoidance(model)
    positions = {0: (1, 1), 1: (5, 5)}
    desired = {0: (2, 1), 1: (5, 6)}
    final = avoidance(wh, positions, desired)
    for rid in positions:
        dx = abs(final[rid][0] - positions[rid][0])
        dy = abs(final[rid][1] - positions[rid][1])
        assert (dx, dy) in {(0, 0), (1, 0), (0, 1)}


def test_learned_avoidance_returns_all_robots():
    wh = _empty_warehouse()
    model = TrafficMLP()
    avoidance = make_learned_avoidance(model)
    positions = {0: (1, 1), 1: (2, 2), 2: (3, 3)}
    desired = dict(positions)  # everyone idle
    final = avoidance(wh, positions, desired)
    assert set(final.keys()) == {0, 1, 2}
    assert final == positions
