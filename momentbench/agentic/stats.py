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


def cluster_bootstrap_ci(
    per_query: list[float], clusters: list[str], n_resamples: int = 2000, alpha: float = 0.05,
    seed: int = 0,
) -> tuple[float, float]:
    """Percentile bootstrap that resamples clusters (source videos), not queries.

    Queries on the same video are correlated, so resampling them independently
    understates uncertainty about new footage. Each resample draws whole videos with
    replacement and takes the mean over all their queries.
    """
    if not per_query:
        return (0.0, 0.0)
    groups: dict[str, list[float]] = {}
    for v, c in zip(per_query, clusters):
        groups.setdefault(c, []).append(v)
    keys = list(groups)
    rng = random.Random(seed)
    means = []
    for _ in range(n_resamples):
        picked = [x for k in rng.choices(keys, k=len(keys)) for x in groups[k]]
        means.append(mean(picked))
    means.sort()
    lo = means[int((alpha / 2) * n_resamples)]
    hi = means[min(n_resamples - 1, int((1 - alpha / 2) * n_resamples))]
    return (lo, hi)


def paired_difference_ci(
    a: list[float], b: list[float], n_resamples: int = 2000, alpha: float = 0.05, seed: int = 0
) -> tuple[float, tuple[float, float]]:
    """Mean of a - b over the same queries, with a bootstrap interval on the difference.

    Pairing removes query difficulty from the comparison, so it is the right interval
    for "is model A better than model B", which separate intervals answer poorly.
    """
    diffs = [x - y for x, y in zip(a, b)]
    return (mean(diffs) if diffs else 0.0), bootstrap_ci(diffs, n_resamples, alpha, seed)
