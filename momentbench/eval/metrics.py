"""Aggregate metrics computation from benchmark runs."""

from __future__ import annotations

from dataclasses import dataclass

from momentbench.types import BenchmarkRun, QueryTier


@dataclass
class AggregateMetrics:
    recall_at_1: float
    recall_at_5: float
    mean_temporal_iou: float
    num_queries: int


def compute_aggregate(run: BenchmarkRun) -> AggregateMetrics:
    """Compute overall aggregate metrics for a run."""
    if not run.results:
        return AggregateMetrics(0.0, 0.0, 0.0, 0)
    n = len(run.results)
    return AggregateMetrics(
        recall_at_1=sum(r.hit_at_1 for r in run.results) / n,
        recall_at_5=sum(r.hit_at_5 for r in run.results) / n,
        mean_temporal_iou=sum(r.best_temporal_iou for r in run.results) / n,
        num_queries=n,
    )


def compute_by_tier(run: BenchmarkRun) -> dict[QueryTier, AggregateMetrics]:
    """Compute metrics broken down by query tier."""
    by_tier: dict[QueryTier, list] = {}
    for r in run.results:
        by_tier.setdefault(r.tier, []).append(r)

    return {
        tier: AggregateMetrics(
            recall_at_1=sum(r.hit_at_1 for r in results) / len(results),
            recall_at_5=sum(r.hit_at_5 for r in results) / len(results),
            mean_temporal_iou=sum(r.best_temporal_iou for r in results) / len(results),
            num_queries=len(results),
        )
        for tier, results in by_tier.items()
    }


def compute_by_vertical(run: BenchmarkRun) -> dict[str, AggregateMetrics]:
    """Compute metrics broken down by vertical."""
    by_vertical: dict[str, list] = {}
    for r in run.results:
        by_vertical.setdefault(r.vertical, []).append(r)

    return {
        vertical: AggregateMetrics(
            recall_at_1=sum(r.hit_at_1 for r in results) / len(results),
            recall_at_5=sum(r.hit_at_5 for r in results) / len(results),
            mean_temporal_iou=sum(r.best_temporal_iou for r in results) / len(results),
            num_queries=len(results),
        )
        for vertical, results in by_vertical.items()
    }
