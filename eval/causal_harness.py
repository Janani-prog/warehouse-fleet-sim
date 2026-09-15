"""M9 causal evaluation harness: for N seeds, run the congestion-spike
scenario twice per seed (loop ON, loop OFF) - a paired-trial design so the
comparison is causal (same seed -> identical order arrivals and classical
planning/avoidance decisions in both trials) rather than observational
(Locked Design Decision #6). Metrics and the statistical test are in
eval/causal_trial.py and eval/stats.py respectively; this module just
orchestrates N pairs and assembles the report.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd

from eval.causal_trial import TrialMetrics, run_trial
from eval.stats import interpret, paired_test
from ml.forecaster import Forecaster
from ml.scripted_scenarios import make_spike_rate
from rag.hybrid_retrieval import HybridRetriever

# Metric name -> whether a lower value is the better outcome, for
# plain-language interpretation.
METRICS_LOWER_IS_BETTER = {
    "throughput": False,
    "mean_wait": True,
    "near_miss_total": True,
    "congestion_ticks": True,
}


def run_paired_trials(
    n_pairs: int,
    seed_start: int,
    ticks: int,
    robots: int,
    order_rate: Callable[[int], float],
    forecaster_model_dir: str,
    agent_model: str = "llama3.1:8b",
) -> pd.DataFrame:
    forecaster = Forecaster(forecaster_model_dir)
    retriever = HybridRetriever()

    rows: list[dict] = []
    for i in range(n_pairs):
        seed = seed_start + i
        off = run_trial(seed, ticks, robots, order_rate, forecaster, loop_enabled=False)
        on = run_trial(
            seed, ticks, robots, order_rate, forecaster, loop_enabled=True,
            retriever=retriever, agent_model=agent_model,
        )
        rows.append(_trial_pair_row(seed, on, off))
    return pd.DataFrame(rows)


def _trial_pair_row(seed: int, on: TrialMetrics, off: TrialMetrics) -> dict:
    row = {"seed": seed}
    for metric in METRICS_LOWER_IS_BETTER:
        row[f"{metric}_on"] = getattr(on, metric)
        row[f"{metric}_off"] = getattr(off, metric)
    return row


def build_report(df: pd.DataFrame) -> dict:
    report: dict = {"n_pairs": len(df), "metrics": {}}
    for metric, lower_is_better in METRICS_LOWER_IS_BETTER.items():
        deltas = (df[f"{metric}_on"] - df[f"{metric}_off"]).to_numpy(dtype=float)
        deltas = deltas[~np.isnan(deltas)]
        if len(deltas) == 0:
            continue
        result = paired_test(deltas)
        result["lower_is_better"] = lower_is_better
        result["interpretation"] = interpret(metric, result, lower_is_better)
        report["metrics"][metric] = result
    return report


def run_and_report(
    n_pairs: int,
    seed_start: int,
    ticks: int,
    robots: int,
    baseline_rate: float,
    spike_rate: float,
    spike_start: int,
    spike_end: int,
    forecaster_model_dir: str,
    out_dir: str,
    agent_model: str = "llama3.1:8b",
) -> dict:
    order_rate = make_spike_rate(baseline_rate, spike_rate, spike_start, spike_end)
    df = run_paired_trials(n_pairs, seed_start, ticks, robots, order_rate, forecaster_model_dir, agent_model)
    report = build_report(df)
    report["scenario"] = {
        "ticks": ticks, "robots": robots, "baseline_rate": baseline_rate,
        "spike_rate": spike_rate, "spike_start": spike_start, "spike_end": spike_end,
        "seed_start": seed_start,
    }

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    df.to_csv(out / "causal_eval_trials.csv", index=False)
    with open(out / "causal_eval_report.json", "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    return report
