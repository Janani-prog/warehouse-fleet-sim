"""Live inference: rolling-window LSTM classifier (calibrated, thresholded)
+ independent rule-based backstop, combined into a single per-tick trigger
signal with confidence values attached - this is the "does anything wire
into the (future, Part 2) agent loop" gate. Confidence-gating and the
backstop are built now, in Part 1, precisely so Part 2's agent has a
trustworthy trigger to consume rather than raw model output (Locked Design
Decision #2 in CLAUDE.md)."""

from __future__ import annotations

import json
from collections import deque
from pathlib import Path

import numpy as np
import torch

from ml.calibration import apply_temperature
from ml.forecast_features import FEATURE_DIM
from ml.forecaster_model import AnomalyLSTM
from ml.generate_forecast_dataset import WINDOW
from ml.rule_backstop import backstop_triggered

HEADS = ["congestion", "collision"]


class Forecaster:
    def __init__(self, model_dir: str):
        model_dir = Path(model_dir)
        self.model = AnomalyLSTM()
        self.model.load_state_dict(torch.load(model_dir / "model.pt", map_location="cpu", weights_only=True))
        self.model.eval()

        with open(model_dir / "temperatures.json") as f:
            self.temperatures = json.load(f)
        with open(model_dir / "thresholds.json") as f:
            self.thresholds = json.load(f)
        norm = np.load(model_dir / "normalization.npz")
        self.mean, self.std = norm["mean"], norm["std"]

        self.window: deque = deque(maxlen=WINDOW)

    def step(self, feature_vector: np.ndarray) -> dict:
        """feature_vector: the current tick's raw (unnormalized) 6-dim
        vector from ml.forecast_features.tick_feature_vector. Returns a dict
        with calibrated confidence per head, both trigger signals, and the
        combined decision - always safe to call, even before WINDOW ticks
        of history exist (classifier just isn't ready yet)."""
        self.window.append(feature_vector)

        active_orders, near_miss_count = int(feature_vector[0]), int(feature_vector[1])
        backstop = backstop_triggered(active_orders, near_miss_count)

        classifier_ready = len(self.window) == WINDOW
        probs = {head: None for head in HEADS}
        raw_probs = {head: None for head in HEADS}
        classifier_triggered = False
        if classifier_ready:
            window_arr = np.stack(self.window)
            normalized = (window_arr - self.mean) / self.std
            x = torch.tensor(normalized, dtype=torch.float32).unsqueeze(0)
            with torch.no_grad():
                logits = self.model(x).numpy()[0]
            for i, head in enumerate(HEADS):
                raw_probs[head] = float(1.0 / (1.0 + np.exp(-logits[i])))
                calibrated = float(apply_temperature(np.array([logits[i]]), self.temperatures[head])[0])
                probs[head] = calibrated
                if calibrated >= self.thresholds[head]:
                    classifier_triggered = True

        return {
            "classifier_ready": classifier_ready,
            "congestion_prob": probs["congestion"],
            "collision_prob": probs["collision"],
            # raw (uncalibrated) sigmoid output - not used for the trigger
            # decision (only the calibrated probability is, since it's the
            # one that's honest about real-world positive rate at that
            # confidence level), but it rises gradually where the
            # calibrated signal can jump sharply once genuinely warranted
            # (see CLAUDE.md's M4 note on why both are kept) - useful for
            # showing the underlying trend, e.g. on the dashboard.
            "congestion_prob_raw": raw_probs["congestion"],
            "collision_prob_raw": raw_probs["collision"],
            "classifier_triggered": classifier_triggered,
            "backstop_triggered": backstop,
            "triggered": classifier_triggered or backstop,
        }
