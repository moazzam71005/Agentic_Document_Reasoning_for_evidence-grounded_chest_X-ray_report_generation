"""Statistical significance testing."""

from __future__ import annotations

import numpy as np
from scipy import stats


def paired_tests(
    adre_scores: list[float],
    baseline_scores: list[float],
    n_bootstrap: int = 10000,
    alpha: float = 0.05,
) -> dict:
    deltas = np.array(adre_scores) - np.array(baseline_scores)
    t_stat, t_p = stats.ttest_rel(adre_scores, baseline_scores)
    w_stat, w_p = stats.wilcoxon(adre_scores, baseline_scores)

    rng = np.random.default_rng(42)
    boot_means = []
    for _ in range(n_bootstrap):
        sample = rng.choice(deltas, size=len(deltas), replace=True)
        boot_means.append(sample.mean())
    ci_low, ci_high = np.percentile(boot_means, [2.5, 97.5])

    return {
        "mean_delta": float(deltas.mean()),
        "paired_t_p": float(t_p),
        "wilcoxon_p": float(w_p),
        "bootstrap_ci_95": [float(ci_low), float(ci_high)],
        "significant_at_005": float(w_p) < alpha,
    }
