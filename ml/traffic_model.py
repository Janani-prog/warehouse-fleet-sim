"""Small MLP binary classifier: local robot/rack occupancy window + desired
direction -> proceed (1) or yield (0) this tick."""

from __future__ import annotations

import torch
import torch.nn as nn

from ml.traffic_features import FEATURE_DIM


class TrafficMLP(nn.Module):
    def __init__(self, input_dim: int = FEATURE_DIM, hidden: int = 32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)  # logit
