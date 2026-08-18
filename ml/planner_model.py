"""Small MLP policy network: local window + goal delta -> 4-way move logits."""

from __future__ import annotations

import torch
import torch.nn as nn

from ml.planner_features import ACTIONS, FEATURE_DIM


class PlannerMLP(nn.Module):
    def __init__(self, input_dim: int = FEATURE_DIM, hidden: int = 64, n_actions: int = len(ACTIONS)):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, n_actions),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)
