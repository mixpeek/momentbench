"""Adversarial checks for score integrity (audit 2026-10-13, findings 4, 5, 6, 7, 27).

Each test reproduces a probe from the audit that used to inflate a score.
"""

import json
from pathlib import Path

import pytest
from click.testing import CliRunner

from momentbench.agentic.budget import Budget, Usage
from momentbench.agentic.environment import InvalidSubmissionError, parse_submission
from momentbench.agentic.episode import Episode
from momentbench.agentic.model import AgentModel, AgentSession, ModelTurn, ToolCall
from momentbench.agentic.run import run_agentic_benchmark
from momentbench.eval.scoring import evaluate_run
from momentbench.types import Moment, Prediction, Query, QueryTier
from tests.test_agentic import SUBMIT_GOLD, ScriptedModel, make_corpus, make_query

PRICING = Path(__file__).resolve().parents[1] / "configs/pricing.json"
NO_ANSWER = Query(query_id="na", text="nothing like this exists", vertical="film",
                  tier=QueryTier.CORPUS_HARD_NEGATIVE, gold_moments=[])


def q(qid: str) -> Query:
    return Query(query_id=qid, text="t", tier=QueryTier.SINGLE_VISUAL, vertical="film",
                 gold_moments=[Moment(asset_id="v1", t_in=10, t_out=20)])


def pred(qid: str, t_in=10.0, t_out=20.0) -> Prediction:
    return Prediction(query_id=qid, model_id="m",
                      ranked_moments=[Moment(asset_id="v1", t_in=t_in, t_out=t_out)])


class TestLegacyCoverage:  # finding 4
    def test_a_missing_prediction_stays_in_the_denominator(self):
        run = evaluate_run([q("a"), q("b")], [pred("a")])
        assert len(run.results) == 2
        assert sum(r.hit_at_1 for r in run.results) / len(run.results) == 0.5
        assert [r.failed for r in run.results] == [False, True]

    def test_duplicate_predictions_are_rejected(self):
        with pytest.raises(ValueError, match="more than one"):
            evaluate_run([q("a"), q("b")], [pred("a")] * 4 + [pred("b", 60, 70)])

    def test_unknown_query_ids_are_rejected(self):
        with pytest.raises(ValueError, match="unknown query"):
            evaluate_run([q("a")], [pred("a"), pred("zz")])


class TestInvalidSubmission:  # finding 5
    @pytest.mark.parametrize("args", [
        {},
        {"moments": None},
        {"moments": [{"asset_id": "no-such-video", "t_in": 1, "t_out": 2}]},
        {"moments": [{"asset_id": "v1", "t_in": 5, "t_out": 5}]},
        {"moments": [{"asset_id": "v1", "t_in": "nan", "t_out": 9}]},
        {"moments": [{"asset_id": "v1", "t_in": 10, "t_out": 9999}]},
    ])
    def test_invalid_output_is_never_an_abstention(self, args):
        with pytest.raises(InvalidSubmissionError):
            parse_submission(args, {"v1": {"duration_s": 100}})

    def test_only_an_explicit_empty_list_abstains(self):
        assert parse_submission({"moments": []}, {"v1": {"duration_s": 100}}) == []

    async def test_malformed_answers_score_zero_on_no_answer_queries(self, tmp_path):
        for bad in ({}, {"moments": [{"asset_id": "nope", "t_in": 1, "t_out": 2}]}):
            run = await run_agentic_benchmark(
                [NO_ANSWER], ScriptedModel([[("submit_answer", bad)]]), make_corpus(tmp_path),
                trials=1)
            assert run.summary.recall_at_1 == 0.0, bad
            assert run.trials[0].outcome != "submitted"

    async def test_the_model_can_correct_an_invalid_submission(self, tmp_path):
        script = [[("submit_answer", {})], [("submit_answer", {"moments": []})]]
        run = await run_agentic_benchmark([NO_ANSWER], ScriptedModel(script),
                                          make_corpus(tmp_path), trials=1)
        assert run.trials[0].outcome == "submitted" and run.summary.recall_at_1 == 1.0
        assert run.trials[0].trace[0]["is_error"] is True


class TestEntrantIsolation:  # finding 7
    async def test_a_custom_loop_cannot_read_gold_or_report_its_own_budget(self, tmp_path):
        seen = {}

        class Cheater(AgentModel):
            model_id = "cheater"

            def start(self, system, task, tools):
                raise AssertionError

            async def run_episode(self, corpus, query, budget):
                seen["fields"] = set(type(query).model_fields)
                gold = getattr(query, "gold_moments", None)
                moments = [gold[0]] if gold else [Moment(asset_id="v2", t_in=0, t_out=5)]
                return Episode(query.query_id, moments,
                               Usage(steps=1000, frames=1000, tool_calls=1000), "submitted")

        run = await run_agentic_benchmark([make_query()], Cheater(), make_corpus(tmp_path),
                                          budget=Budget(max_steps=1, max_frames=1,
                                                        max_tool_calls=1), trials=1)
        assert seen["fields"] == {"query_id", "text"}
        assert run.summary.recall_at_1 == 0.0
        # Frames and tool calls are the harness's own count (none went through the
        # corpus); the claimed 1000 turns breaks the budget and scores zero.
        t = run.trials[0]
        assert t.usage.frames == 0 and t.usage.tool_calls == 0
        assert t.outcome == "budget_exceeded"


class TestUsageSurvivesErrors:  # finding 27
    async def test_tokens_spent_before_a_provider_error_are_kept(self, tmp_path):
        class Flaky(AgentSession):
            def __init__(self):
                self.n = 0

            async def step(self, results):
                self.n += 1
                if self.n == 1:
                    return ModelTurn([ToolCall("c", "list_assets", {})],
                                     Usage(input_tokens=1000, output_tokens=100))
                raise RuntimeError("upstream 500")

        class FlakyModel(AgentModel):
            model_id = "flaky"

            def start(self, system, task, tools):
                return Flaky()

        run = await run_agentic_benchmark([make_query()], FlakyModel(), make_corpus(tmp_path),
                                          trials=1)
        t = run.trials[0]
        assert t.outcome == "error" and t.error == "RuntimeError: upstream 500"
        assert t.usage.input_tokens == 1000 and t.usage.output_tokens == 100
        assert [x["tool"] for x in t.trace] == ["list_assets"]


class TestCompleteCorpus:  # finding 6
    def _files(self, tmp_path):
        manifest = tmp_path / "m.json"
        manifest.write_text(json.dumps([
            {"asset_id": "v1", "duration_s": 100, "vertical": "film", "title": "a",
             "file": {"sha256": "0" * 64}}]))
        queries = tmp_path / "q.json"
        queries.write_text(json.dumps([NO_ANSWER.model_dump(mode="json")]))
        corpus = tmp_path / "corpus"
        corpus.mkdir()
        return manifest, queries, corpus

    def test_an_incomplete_corpus_cannot_produce_a_reportable_run(self, tmp_path):
        from momentbench.cli import main

        manifest, queries, corpus = self._files(tmp_path)
        out = tmp_path / "out"
        r = CliRunner().invoke(main, ["agentic", "abstain", "--provider", "baseline",
                                      "-q", str(queries), "-c", str(manifest),
                                      "--corpus-dir", str(corpus), "-o", str(out)])
        assert r.exit_code != 0 and "failed verification" in r.output
        assert not out.exists()

    def test_partial_runs_are_marked_not_reportable(self, tmp_path, monkeypatch):
        from momentbench.cli import main

        manifest, queries, corpus = self._files(tmp_path)
        monkeypatch.chdir(tmp_path)
        r = CliRunner().invoke(main, ["agentic", "abstain", "--provider", "baseline",
                                      "--partial", "-q", str(queries), "-c", str(manifest),
                                      "--corpus-dir", str(corpus), "--trials", "1",
                                      "--pricing", str(PRICING)])
        assert r.exit_code == 0, r.output
        [f] = (tmp_path / "data/results/debug").glob("*.json")
        saved = json.loads(f.read_text())
        assert saved["reportable"] is False


class TestSplitMetrics:  # finding 17
    async def test_abstention_and_localization_are_reported_apart(self, tmp_path):
        from momentbench.agentic.run import summarize

        # One positive answered right, one no-answer wrongly answered with a moment.
        run = await run_agentic_benchmark(
            [make_query(), NO_ANSWER], ScriptedModel([[SUBMIT_GOLD]]), make_corpus(tmp_path),
            trials=1)
        s = summarize(run.trials, 1)
        assert s.positive_recall_at_1 == 1.0
        assert s.no_answer_accuracy == 0.0 and s.false_positive_rate == 1.0
        assert s.abstention_on_positives == 0.0
        assert s.mean_tiou_positive == 1.0  # the no-answer query does not count here


class TestIntervals:  # finding 22
    def test_clustered_interval_is_wider_when_queries_share_a_video(self):
        from momentbench.agentic.stats import bootstrap_ci, cluster_bootstrap_ci

        hits = [1.0] * 10 + [0.0] * 10
        videos = ["a"] * 10 + ["b"] * 10  # all hits on one video, all misses on the other
        lo_q, hi_q = bootstrap_ci(hits)
        lo_c, hi_c = cluster_bootstrap_ci(hits, videos)
        assert hi_c - lo_c > hi_q - lo_q

    def test_paired_difference_uses_the_same_queries(self):
        from momentbench.agentic.stats import paired_difference_ci

        a = [1.0, 1.0, 0.0, 1.0]
        b = [0.0, 1.0, 0.0, 0.0]
        diff, (lo, hi) = paired_difference_ci(a, b)
        assert diff == 0.5 and lo <= diff <= hi
