"""Paired-trial statistics: Wilcoxon signed-rank test (the architecture
doc's "safer default" nonparametric choice) plus effect size (paired
Cohen's d) and a bootstrap 95% CI on the mean paired delta - reported
together because a p-value alone doesn't say how big or how uncertain the
effect is. Pure functions over an array of paired deltas, independent of
the simulator/agent/Ollama, so this is fully unit-testable without a live
run.
"""

from __future__ import annotations

import numpy as np
from scipy import stats as scipy_stats

N_BOOTSTRAP = 5000
BOOTSTRAP_SEED = 0


def paired_test(deltas: np.ndarray) -> dict:
    """deltas[i] = metric(loop ON, seed i) - metric(loop OFF, seed i)."""
    deltas = np.asarray(deltas, dtype=float)
    n = len(deltas)
    if n < 1:
        raise ValueError("need at least one paired trial")

    nonzero = deltas[deltas != 0]
    if len(nonzero) == 0:
        wilcoxon_stat, p_value = 0.0, 1.0
    else:
        wilcoxon_stat, p_value = scipy_stats.wilcoxon(deltas)

    mean_delta = float(np.mean(deltas))
    median_delta = float(np.median(deltas))
    std_delta = float(np.std(deltas, ddof=1)) if n > 1 else 0.0
    cohens_d = mean_delta / std_delta if std_delta > 0 else 0.0

    rng = np.random.default_rng(BOOTSTRAP_SEED)
    boot_means = np.array(
        [np.mean(rng.choice(deltas, size=n, replace=True)) for _ in range(N_BOOTSTRAP)]
    )
    ci_low, ci_high = np.percentile(boot_means, [2.5, 97.5])

    return {
        "n": n,
        "mean_delta": mean_delta,
        "median_delta": median_delta,
        "std_delta": std_delta,
        "wilcoxon_stat": float(wilcoxon_stat),
        "p_value": float(p_value),
        "cohens_d": cohens_d,
        "ci_95_low": float(ci_low),
        "ci_95_high": float(ci_high),
        "significant_at_0.05": bool(p_value < 0.05),
    }


def interpret(metric_name: str, result: dict, lower_is_better: bool) -> str:
    if not result["significant_at_0.05"]:
        return (
            f"{metric_name}: no statistically significant difference between loop ON and OFF "
            f"(p={result['p_value']:.3f}, n={result['n']} paired trials)."
        )
    improved = (result["mean_delta"] < 0) if lower_is_better else (result["mean_delta"] > 0)
    direction = "improved" if improved else "worsened"
    return (
        f"{metric_name}: loop ON {direction} this metric relative to OFF, mean paired delta "
        f"{result['mean_delta']:+.2f} (95% CI [{result['ci_95_low']:+.2f}, {result['ci_95_high']:+.2f}]), "
        f"p={result['p_value']:.4f}, Cohen's d={result['cohens_d']:.2f} (n={result['n']})."
    )
