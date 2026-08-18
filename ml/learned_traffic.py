"""Learned avoidance policy: each robot's proceed/yield decision comes from
the trained MLP instead of "always propose the A* next cell", but every
prediction still passes through `ml.traffic.arbitrate` before touching
simulator state - the same "propose, then validate deterministically"
pattern used everywhere else raw model output could otherwise cause harm.
A predicted "proceed" is not a guarantee of moving; it's an attempt that can
still be arbitrated away by a same-tick conflict."""

from __future__ import annotations

import numpy as np
import torch

from ml.traffic import Cell, arbitrate
from ml.traffic_features import encode
from ml.traffic_model import TrafficMLP
from sim.grid import Warehouse

PROCEED_THRESHOLD = 0.5


def load_model(checkpoint_path: str) -> TrafficMLP:
    model = TrafficMLP()
    model.load_state_dict(torch.load(checkpoint_path, map_location="cpu", weights_only=True))
    model.eval()
    return model


def make_learned_avoidance(model: TrafficMLP):
    """Returns an AvoidanceFn-compatible callable closing over the model,
    so it can be passed as `World(avoidance=make_learned_avoidance(model))`."""

    def learned_avoidance(
        warehouse: Warehouse,
        positions: dict[int, Cell],
        desired: dict[int, Cell],
    ) -> dict[int, Cell]:
        attempted = dict(desired)
        movers = {rid: want for rid, want in desired.items() if want != positions[rid]}
        if movers:
            feats = []
            for rid, want in movers.items():
                others = {p for other_rid, p in positions.items() if other_rid != rid}
                feats.append(encode(warehouse, positions[rid], want, others))
            batch = torch.tensor(np.stack(feats), dtype=torch.float32)
            with torch.no_grad():
                proceed = torch.sigmoid(model(batch)) > PROCEED_THRESHOLD
            for (rid, want), should_proceed in zip(movers.items(), proceed.tolist()):
                if not should_proceed:
                    attempted[rid] = positions[rid]  # pre-emptive yield

        return arbitrate(positions, attempted)

    return learned_avoidance
