"""Small LSTM classifier: a rolling window of fleet telemetry -> two
independent binary logits (congestion, collision-risk), each predicting
whether that anomaly type occurs within the next HORIZON ticks. Kept small
(1 layer, 32 hidden units) per the architecture doc's CPU-tractability
guidance - this trains in well under a minute on CPU at this dataset size."""

from __future__ import annotations

import torch
import torch.nn as nn

from ml.forecast_features import FEATURE_DIM

HIDDEN_SIZE = 32


class AnomalyLSTM(nn.Module):
    def __init__(self, input_dim: int = FEATURE_DIM, hidden_size: int = HIDDEN_SIZE):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden_size, num_layers=1, batch_first=True)
        self.head = nn.Linear(hidden_size, 2)  # [congestion_logit, collision_logit]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: [batch, window, FEATURE_DIM] -> [batch, 2] raw logits."""
        _, (h_n, _) = self.lstm(x)
        return self.head(h_n[-1])
