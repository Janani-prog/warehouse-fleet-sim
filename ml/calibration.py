"""Calibration + operating-threshold selection, shared by training and live
inference.

Temperature scaling (Guo et al. 2017), not isotonic regression: isotonic
regression was tried first and technically calibrates well in aggregate, but
its piecewise-constant fit collapses a wide range of raw probabilities to a
single output value (typically 0), which turns a genuinely gradual rising
signal within one trajectory into a near step function - fine for aggregate
calibration accuracy, bad for an early-warning demo where the whole point is
to show confidence climbing *before* the threshold is crossed. Temperature
scaling divides the logit by a single learned scalar T before the sigmoid,
so it calibrates while staying smooth and strictly monotonic in the raw
score. See CLAUDE.md's M4 note for the concrete before/after comparison
that motivated the switch (both are explicitly permitted as alternatives in
the architecture doc, so this isn't a locked-decision change)."""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import precision_recall_curve


def fit_temperature(logits: np.ndarray, labels: np.ndarray, max_iter: int = 200) -> float:
    logits_t = torch.tensor(logits, dtype=torch.float32)
    labels_t = torch.tensor(labels, dtype=torch.float32)
    log_temp = nn.Parameter(torch.zeros(1))  # optimize in log-space: T = exp(log_temp) > 0
    optimizer = torch.optim.LBFGS([log_temp], lr=0.05, max_iter=max_iter)
    loss_fn = nn.BCEWithLogitsLoss()

    def closure():
        optimizer.zero_grad()
        temperature = torch.exp(log_temp)
        loss = loss_fn(logits_t / temperature, labels_t)
        loss.backward()
        return loss

    optimizer.step(closure)
    return float(torch.exp(log_temp).item())


def apply_temperature(logits: np.ndarray, temperature: float) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-logits / temperature))


def choose_threshold_for_precision(
    calibrated_probs: np.ndarray, labels: np.ndarray, target_precision: float
) -> float:
    """Lowest threshold that still meets target_precision on this data (more
    recall at the same precision floor); falls back to the
    highest-precision achievable threshold if the target is unreachable."""
    precision, recall, thresholds = precision_recall_curve(labels, calibrated_probs)
    # precision_recall_curve returns one more precision/recall point than
    # thresholds (for threshold=+inf); align by dropping the last point.
    precision, recall = precision[:-1], recall[:-1]
    meets_target = precision >= target_precision
    if meets_target.any():
        candidates = thresholds[meets_target]
        return float(candidates.min())
    return float(thresholds[int(np.argmax(precision))])
