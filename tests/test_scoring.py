"""Tests for the evaluation scoring engine."""


from momentbench.eval.scoring import (
    compute_temporal_iou,
    evaluate_prediction,
    evaluate_run,
    is_hit,
)
from momentbench.types import Moment, Prediction, Query, QueryTier


def make_moment(asset_id="v1", t_in=10.0, t_out=15.0):
    return Moment(asset_id=asset_id, t_in=t_in, t_out=t_out)


class TestTemporalIoU:
    def test_perfect_overlap(self):
        m = make_moment(t_in=10, t_out=20)
        assert compute_temporal_iou(m, m) == 1.0

    def test_no_overlap(self):
        a = make_moment(t_in=0, t_out=5)
        b = make_moment(t_in=10, t_out=15)
        assert compute_temporal_iou(a, b) == 0.0

    def test_partial_overlap(self):
        a = make_moment(t_in=0, t_out=10)
        b = make_moment(t_in=5, t_out=15)
        # intersection: 5-10 = 5s, union: 10+10-5 = 15s
        assert abs(compute_temporal_iou(a, b) - 5 / 15) < 1e-9

    def test_different_assets(self):
        a = make_moment(asset_id="v1", t_in=0, t_out=10)
        b = make_moment(asset_id="v2", t_in=0, t_out=10)
        assert compute_temporal_iou(a, b) == 0.0

    def test_contained(self):
        outer = make_moment(t_in=0, t_out=20)
        inner = make_moment(t_in=5, t_out=10)
        # intersection: 5s, union: 20s
        assert abs(compute_temporal_iou(outer, inner) - 5 / 20) < 1e-9


class TestIsHit:
    def test_hit_above_threshold(self):
        pred = make_moment(t_in=10, t_out=20)
        golds = [make_moment(t_in=10, t_out=20)]
        assert is_hit(pred, golds, iou_threshold=0.5)

    def test_miss_below_threshold(self):
        pred = make_moment(t_in=0, t_out=5)
        golds = [make_moment(t_in=50, t_out=55)]
        assert not is_hit(pred, golds, iou_threshold=0.5)

    def test_hit_any_gold(self):
        pred = make_moment(t_in=10, t_out=20)
        golds = [make_moment(t_in=50, t_out=55), make_moment(t_in=10, t_out=20)]
        assert is_hit(pred, golds, iou_threshold=0.5)


class TestEvaluatePrediction:
    def test_perfect_prediction(self):
        query = Query(
            query_id="q1",
            text="test",
            tier=QueryTier.SINGLE_VISUAL,
            vertical="civic",
            gold_moments=[make_moment(t_in=10, t_out=20)],
        )
        pred = Prediction(
            query_id="q1",
            ranked_moments=[make_moment(t_in=10, t_out=20)],
            model_id="test",
        )
        result = evaluate_prediction(query, pred)
        assert result.hit_at_1
        assert result.hit_at_5
        assert result.best_temporal_iou == 1.0

    def test_empty_prediction(self):
        query = Query(
            query_id="q1",
            text="test",
            tier=QueryTier.CROSS_MODAL,
            vertical="aerospace",
            gold_moments=[make_moment()],
        )
        pred = Prediction(query_id="q1", ranked_moments=[], model_id="test")
        result = evaluate_prediction(query, pred)
        assert not result.hit_at_1
        assert not result.hit_at_5
        assert result.best_temporal_iou == 0.0


class TestEvaluateRun:
    def test_full_run(self):
        queries = [
            Query(
                query_id=f"q{i}",
                text=f"query {i}",
                tier=QueryTier.SINGLE_VISUAL,
                vertical="civic",
                gold_moments=[make_moment(t_in=i * 10, t_out=i * 10 + 5)],
            )
            for i in range(3)
        ]
        predictions = [
            Prediction(
                query_id=f"q{i}",
                ranked_moments=[make_moment(t_in=i * 10, t_out=i * 10 + 5)],
                model_id="test",
            )
            for i in range(3)
        ]
        run = evaluate_run(queries, predictions)
        assert len(run.results) == 3
        assert all(r.hit_at_1 for r in run.results)


class TestNoAnswerQueries:
    def _query(self):
        return Query(query_id="na", text="a moment that is not in the corpus",
                     tier=QueryTier.CORPUS_HARD_NEGATIVE, vertical="film", gold_moments=[])

    def test_abstaining_is_a_hit(self):
        pred = Prediction(query_id="na", ranked_moments=[], model_id="m")
        result = evaluate_prediction(self._query(), pred)
        assert result.hit_at_1 and result.hit_at_5 and result.best_temporal_iou == 1.0

    def test_returning_any_moment_is_a_miss(self):
        pred = Prediction(query_id="na", ranked_moments=[make_moment()], model_id="m")
        result = evaluate_prediction(self._query(), pred)
        assert not result.hit_at_1 and not result.hit_at_5 and result.best_temporal_iou == 0.0

    def test_abstaining_on_an_answerable_query_is_a_miss(self):
        q = Query(query_id="a", text="t", tier=QueryTier.SINGLE_VISUAL, vertical="film",
                  gold_moments=[make_moment()])
        result = evaluate_prediction(q, Prediction(query_id="a", ranked_moments=[], model_id="m"))
        assert not result.hit_at_1
