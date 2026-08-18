"""Autoregressive rollout of the learned planner: repeatedly query the MLP
for the next move and step, since (unlike A*) it has no global search -
it only ever sees a local window + goal delta at the current cell."""

from __future__ import annotations

import torch

from ml.planner_features import ACTIONS, encode
from ml.planner_model import PlannerMLP
from sim.grid import Warehouse

Cell = tuple[int, int]

# Each cell may be revisited a bounded number of times before the rollout
# treats it as stuck (prevents infinite 2-cycles between adjacent cells).
MAX_REVISITS = 3


def load_model(checkpoint_path: str) -> PlannerMLP:
    model = PlannerMLP()
    model.load_state_dict(torch.load(checkpoint_path, map_location="cpu", weights_only=True))
    model.eval()
    return model


def rollout(
    model: PlannerMLP,
    warehouse: Warehouse,
    start: Cell,
    goal: Cell,
    max_steps: int = 150,
) -> tuple[list[Cell], bool]:
    """Returns (path, success). path always starts at `start`; success is
    True iff the goal was reached within max_steps."""
    if start == goal:
        return [start], True

    pos = start
    path = [pos]
    visit_counts = {pos: 1}

    for _ in range(max_steps):
        feat = torch.tensor(encode(warehouse, pos, goal), dtype=torch.float32).unsqueeze(0)
        with torch.no_grad():
            logits = model(feat)[0]
        ranked = torch.argsort(logits, descending=True).tolist()

        next_pos = None
        for allow_revisit in (False, True):
            for a_idx in ranked:
                dx, dy = ACTIONS[a_idx]
                candidate = (pos[0] + dx, pos[1] + dy)
                if not warehouse.is_free(*candidate):
                    continue
                if not allow_revisit and visit_counts.get(candidate, 0) >= MAX_REVISITS:
                    continue
                next_pos = candidate
                break
            if next_pos is not None:
                break

        if next_pos is None:
            return path, False  # no free neighbor at all (shouldn't happen)

        pos = next_pos
        path.append(pos)
        visit_counts[pos] = visit_counts.get(pos, 0) + 1
        if pos == goal:
            return path, True

    return path, False
