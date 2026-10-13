"""Scoring logic: temporal IoU, hit computation, and aggregate metrics."""

from __future__ import annotations

from momentbench.types import (
    BenchmarkRun,
    EvalResult,
    Moment,
    Prediction,
    Query,
)


def compute_temporal_iou(pred: Moment, gold: Moment) -> float:
    """Compute temporal Intersection-over-Union between two moments on the same asset."""
    if pred.asset_id != gold.asset_id:
        return 0.0
    intersection_start = max(pred.t_in, gold.t_in)
    intersection_end = min(pred.t_out, gold.t_out)
    intersection = max(0.0, intersection_end - intersection_start)
    union = (pred.t_out - pred.t_in) + (gold.t_out - gold.t_in) - intersection
    if union <= 0:
        return 0.0
    return intersection / union


def is_hit(pred_moment: Moment, gold_moments: list[Moment], iou_threshold: float = 0.5) -> bool:
    """A prediction is a hit if it matches any gold moment with IoU >= threshold."""
    return any(
        compute_temporal_iou(pred_moment, gold) >= iou_threshold for gold in gold_moments
    )


def best_iou(pred_moments: list[Moment], gold_moments: list[Moment]) -> float:
    """Best temporal IoU across all prediction-gold pairs."""
    if not pred_moments or not gold_moments:
        return 0.0
    return max(
        compute_temporal_iou(p, g) for p in pred_moments for g in gold_moments
    )


def evaluate_prediction(
    query: Query, prediction: Prediction, iou_threshold: float = 0.5
) -> EvalResult:
    """Score a single prediction against a query's gold moments."""
    ranked = prediction.ranked_moments
    if query.is_no_answer:
        # Correct only by abstaining. Any returned moment is a false positive.
        abstained = not ranked
        return EvalResult(
            query_id=query.query_id,
            model_id=prediction.model_id,
            hit_at_1=abstained,
            hit_at_5=abstained,
            best_temporal_iou=1.0 if abstained else 0.0,
            tier=query.tier,
            vertical=query.vertical,
        )
    hit_1 = is_hit(ranked[0], query.gold_moments, iou_threshold) if ranked else False
    hit_5 = any(
        is_hit(m, query.gold_moments, iou_threshold) for m in ranked[:5]
    ) if ranked else False
    b_iou = best_iou(ranked[:5], query.gold_moments)

    return EvalResult(
        query_id=query.query_id,
        model_id=prediction.model_id,
        hit_at_1=hit_1,
        hit_at_5=hit_5,
        best_temporal_iou=b_iou,
        tier=query.tier,
        vertical=query.vertical,
    )


def evaluate_run(
    queries: list[Query],
    predictions: list[Prediction],
    iou_threshold: float = 0.5,
    run_id: str = "default",
    model_id: str | None = None,
) -> BenchmarkRun:
    """Score a run against every expected query.

    Exactly one prediction per query is allowed. A query with no prediction scores
    zero and stays in the denominator; a duplicate or unknown query id is an error,
    because silently dropping or double-counting either one changes the score.
    """
    query_map = {q.query_id: q for q in queries}
    seen: dict[str, Prediction] = {}
    for pred in predictions:
        if pred.query_id not in query_map:
            raise ValueError(f"prediction for unknown query {pred.query_id!r}")
        if pred.query_id in seen:
            raise ValueError(f"more than one prediction for query {pred.query_id!r}")
        seen[pred.query_id] = pred
    model_id = model_id or (predictions[0].model_id if predictions else "unknown")
    results = []
    for query in queries:
        pred = seen.get(query.query_id)
        if pred is None:
            results.append(EvalResult(query_id=query.query_id, model_id=model_id,
                                      hit_at_1=False, hit_at_5=False, best_temporal_iou=0.0,
                                      tier=query.tier, vertical=query.vertical, failed=True))
        else:
            results.append(evaluate_prediction(query, pred, iou_threshold))
    return BenchmarkRun(run_id=run_id, model_id=model_id, results=results,
                        predictions=list(seen.values()))
