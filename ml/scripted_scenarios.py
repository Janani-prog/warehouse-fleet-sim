"""Scripted scenarios for demoing the forecaster against a known, engineered
event rather than waiting for one to occur organically."""

from __future__ import annotations

from typing import Callable

BASELINE_RATE = 0.15
SPIKE_RATE = 0.6
SPIKE_START = 200
SPIKE_END = 260


def make_spike_rate(baseline: float, spike: float, start: int, end: int) -> Callable[[int], float]:
    """Factory for a step-function order-arrival-rate scenario: baseline,
    then an engineered burst from `start` to `end`, then back to baseline.
    Factored out so M9's causal eval harness can use a shorter, compressed
    version of the same shape without duplicating the spike logic."""

    def rate(tick: int) -> float:
        return spike if start <= tick < end else baseline

    return rate


congestion_spike_rate = make_spike_rate(BASELINE_RATE, SPIKE_RATE, SPIKE_START, SPIKE_END)
