"""Aggregate repeated trials into a mean with a bootstrap confidence interval."""

from __future__ import annotations

import random
from statistics import mean


def bootstrap_ci(
    per_query: list[float], n_resamples: int = 2000, alpha: float = 0.05, seed: int = 0
) -> tuple[float, float]:
    """Percentile bootstrap over queries. `per_query` holds each query's mean over trials.

    Queries are the resampling unit: trials of the same query are correlated, so
    resampling individual trials would understate the interval.
    """
    if not per_query:
        return (0.0, 0.0)
    rng = random.Random(seed)
    n = len(per_query)
    means = sorted(mean(rng.choices(per_query, k=n)) for _ in range(n_resamples))
    lo = means[int((alpha / 2) * n_resamples)]
    hi = means[min(n_resamples - 1, int((1 - alpha / 2) * n_resamples))]
    return (lo, hi)
