"""Scripted scenarios for demoing the forecaster against a known, engineered
event rather than waiting for one to occur organically."""

from __future__ import annotations

BASELINE_RATE = 0.15
SPIKE_RATE = 0.6
SPIKE_START = 200
SPIKE_END = 260


def congestion_spike_rate(tick: int) -> float:
    """Healthy baseline order arrival rate, with an engineered burst of
    demand from tick 200-260 - enough to push an 8-robot fleet (tuned for
    the 0.15 baseline, see CLAUDE.md's throughput sweep) well past its
    capacity and into a genuine, labeled congestion event shortly after."""
    if SPIKE_START <= tick < SPIKE_END:
        return SPIKE_RATE
    return BASELINE_RATE
