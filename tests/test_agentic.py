"""Tests for the agentic track, driven by a scripted model (no network, no ffmpeg)."""

from pathlib import Path

import pytest

from momentbench.agentic.anthropic_model import _result_block
from momentbench.agentic.budget import Budget, Pricing, Usage
from momentbench.agentic.environment import LocalCorpus, load_transcript, parse_submission
from momentbench.agentic.episode import run_episode
from momentbench.agentic.model import AgentModel, AgentSession, ModelTurn, ToolCall, ToolResult
from momentbench.agentic.run import run_agentic_benchmark
from momentbench.agentic.stats import bootstrap_ci
from momentbench.types import Moment, Query, QueryTier

ASSETS = [
    {"asset_id": "v1", "title": "Hearing", "duration_s": 100, "storage_path": None},
    {"asset_id": "v2", "title": "Launch", "duration_s": 50, "storage_path": None},
]
TRANSCRIPTS = {
    "v1": [
        {"t_in": 0.0, "t_out": 5.0, "text": "Good morning everyone"},
        {"t_in": 10.0, "t_out": 20.0, "text": "The hearing will come to order"},
    ]
}


def make_corpus(tmp_path: Path) -> LocalCorpus:
    for a in ASSETS:
        (tmp_path / f"{a['asset_id']}.mp4").write_bytes(b"")
    return LocalCorpus(
        ASSETS, tmp_path, frame_extractor=lambda video, t: b"jpeg", transcripts=TRANSCRIPTS
    )


def make_query(t_in=10.0, t_out=20.0) -> Query:
    return Query(
        query_id="q1", text="the hearing is called to order", tier=QueryTier.SINGLE_VISUAL,
        vertical="civic", gold_moments=[Moment(asset_id="v1", t_in=t_in, t_out=t_out)],
    )


class ScriptedSession(AgentSession):
    def __init__(self, script):
        self._script = list(script)
        self.seen: list[list[ToolResult] | None] = []

    async def step(self, tool_results):
        self.seen.append(tool_results)
        calls = self._script.pop(0) if self._script else []
        return ModelTurn(
            tool_calls=[ToolCall(f"c{i}", n, a) for i, (n, a) in enumerate(calls)],
            usage=Usage(input_tokens=100, output_tokens=10),
        )


class ScriptedModel(AgentModel):
    model_id = "scripted"

    def __init__(self, script):
        self._script = script
        self.sessions: list[ScriptedSession] = []

    def start(self, system, task, tools):
        self.sessions.append(ScriptedSession(self._script))
        return self.sessions[-1]


SUBMIT_GOLD = ("submit_answer", {"moments": [{"asset_id": "v1", "t_in": 10, "t_out": 20}]})


class TestEnvironment:
    def test_search_transcript_ranks_by_term_overlap(self, tmp_path):
        hits = make_corpus(tmp_path).search_transcript("hearing come to order")
        assert '"t_in": 10.0' in hits.split("}")[0]

    def test_sample_frames_charges_frame_budget(self, tmp_path):
        corpus, usage = make_corpus(tmp_path), Usage()
        call = ToolCall("c", "sample_frames",
                        {"asset_id": "v1", "t_start": 0, "t_end": 40, "num_frames": 4})
        result = corpus.execute(call, Budget(max_frames=6), usage)
        assert len(result.images_jpeg) == 4 and usage.frames == 4
        result = corpus.execute(call, Budget(max_frames=6), usage)
        assert len(result.images_jpeg) == 2 and usage.frames == 6
        result = corpus.execute(call, Budget(max_frames=6), usage)
        assert result.is_error and "frame budget" in result.text

    def test_unknown_asset_is_an_error_result_not_an_exception(self, tmp_path):
        call = ToolCall("c", "read_transcript", {"asset_id": "nope", "t_start": 0, "t_end": 5})
        assert make_corpus(tmp_path).execute(call, Budget(), Usage()).is_error

    def test_missing_videos_lists_assets_without_files(self, tmp_path):
        assert LocalCorpus(ASSETS, tmp_path).missing_videos() == ["v1", "v2"]

    def test_parse_submission_drops_unknown_assets_and_inverted_spans(self):
        moments = parse_submission(
            {"moments": [
                {"asset_id": "v1", "t_in": 1, "t_out": 2},
                {"asset_id": "ghost", "t_in": 1, "t_out": 2},
                {"asset_id": "v1", "t_in": 5, "t_out": 3},
                {"asset_id": "v1"},
            ]},
            {"v1"},
        )
        assert [(m.t_in, m.t_out) for m in moments] == [(1.0, 2.0)]

    def test_load_vtt(self, tmp_path):
        p = tmp_path / "a.vtt"
        p.write_text("WEBVTT\n\n00:00:01.000 --> 00:00:03.500\nhello\nworld\n")
        assert load_transcript(p) == [{"t_in": 1.0, "t_out": 3.5, "text": "hello world"}]

    def test_load_vtt_without_hours(self, tmp_path):
        p = tmp_path / "a.vtt"
        p.write_text("WEBVTT\n\n01:17.000 --> 01:30.250\nhello\n")
        assert load_transcript(p) == [{"t_in": 77.0, "t_out": 90.25, "text": "hello"}]


class TestEpisode:
    async def test_tool_results_reach_the_model_and_submission_ends_the_episode(self, tmp_path):
        model = ScriptedModel([
            [("search_transcript", {"query": "order"}),
             ("sample_frames", {"asset_id": "v1", "t_start": 10, "t_end": 20, "num_frames": 2})],
            [SUBMIT_GOLD],
        ])
        ep = await run_episode(model, make_corpus(tmp_path), make_query(), Budget())
        assert ep.outcome == "submitted"
        assert ep.moments == [Moment(asset_id="v1", t_in=10.0, t_out=20.0)]
        assert ep.usage.steps == 2 and ep.usage.frames == 2 and ep.usage.tool_calls == 2
        assert ep.usage.input_tokens == 200
        second = model.sessions[0].seen[1]
        assert [len(r.images_jpeg) for r in second] == [0, 2]

    async def test_ending_without_submit_is_no_submission(self, tmp_path):
        ep = await run_episode(ScriptedModel([[]]), make_corpus(tmp_path), make_query(), Budget())
        assert ep.outcome == "no_submission" and ep.moments == []

    async def test_step_limit_stops_a_model_that_never_submits(self, tmp_path):
        script = [[("list_assets", {})]] * 10
        ep = await run_episode(
            ScriptedModel(script), make_corpus(tmp_path), make_query(), Budget(max_steps=3)
        )
        assert ep.outcome == "step_limit" and ep.usage.steps == 3


class TestRun:
    async def test_trials_are_scored_costed_and_saved(self, tmp_path):
        run = await run_agentic_benchmark(
            [make_query()], ScriptedModel([[SUBMIT_GOLD]]), make_corpus(tmp_path), trials=3,
            pricing=Pricing(input_per_mtok=1.0, output_per_mtok=10.0), output_dir=tmp_path / "out",
        )
        assert len(run.trials) == 3 and run.summary.recall_at_1 == 1.0
        assert run.summary.mean_cost_usd_per_query == pytest.approx((100 + 100) / 1e6)
        assert (tmp_path / "out" / f"{run.run_id}.json").exists()

    async def test_a_wrong_span_misses(self, tmp_path):
        wrong = ("submit_answer", {"moments": [{"asset_id": "v1", "t_in": 60, "t_out": 70}]})
        run = await run_agentic_benchmark(
            [make_query()], ScriptedModel([[wrong]]), make_corpus(tmp_path), trials=2
        )
        assert run.summary.recall_at_1 == 0.0 and run.summary.mean_cost_usd_per_query is None

    async def test_a_crashing_model_is_counted_as_a_failed_trial(self, tmp_path):
        class Broken(AgentModel):
            model_id = "broken"

            def start(self, system, task, tools):
                raise RuntimeError("boom")

        run = await run_agentic_benchmark(
            [make_query()], Broken(), make_corpus(tmp_path), trials=2, output_dir=tmp_path / "out"
        )
        assert not (tmp_path / "out").exists()
        assert run.summary.failed_trials == 2 and run.summary.recall_at_1 == 0.0
        assert run.trials[0].error == "RuntimeError: boom"

    async def test_an_entrant_with_its_own_loop_is_scored_the_same_way(self, tmp_path):
        from momentbench.agentic.episode import Episode

        class OwnLoop(AgentModel):
            model_id = "own-loop"
            harness = "claude-code"

            def start(self, system, task, tools):
                raise AssertionError("the runner must call run_episode instead")

            async def run_episode(self, corpus, query, budget):
                return Episode(query.query_id, [Moment(asset_id="v1", t_in=10, t_out=20)],
                               Usage(steps=1), "submitted")

        run = await run_agentic_benchmark([make_query()], OwnLoop(), make_corpus(tmp_path))
        assert run.harness == "claude-code" and run.summary.recall_at_1 == 1.0

    async def test_no_answer_query_needs_an_explicit_empty_submission(self, tmp_path):
        query = Query(query_id="na", text="not in the corpus", vertical="civic",
                      tier=QueryTier.CORPUS_HARD_NEGATIVE, gold_moments=[])
        corpus = make_corpus(tmp_path)
        abstain = await run_agentic_benchmark(
            [query], ScriptedModel([[("submit_answer", {"moments": []})]]), corpus, trials=1
        )
        silent = await run_agentic_benchmark([query], ScriptedModel([[]]), corpus, trials=1)
        assert abstain.summary.recall_at_1 == 1.0
        assert silent.summary.recall_at_1 == 0.0 and silent.trials[0].best_temporal_iou == 0.0


class TestStatsAndWire:
    def test_bootstrap_ci_brackets_the_mean_and_is_deterministic(self):
        values = [1.0, 0.0, 1.0, 1.0, 0.0, 0.6, 0.2, 1.0]
        lo, hi = bootstrap_ci(values)
        assert lo <= sum(values) / len(values) <= hi
        assert (lo, hi) == bootstrap_ci(values)

    def test_bootstrap_ci_of_constant_is_a_point(self):
        assert bootstrap_ci([1.0, 1.0, 1.0]) == (1.0, 1.0)

    def test_anthropic_tool_result_carries_frames_as_image_blocks(self):
        block = _result_block(ToolResult("toolu_1", "{}", images_jpeg=[b"\xff\xd8"]))
        assert block["tool_use_id"] == "toolu_1"
        assert [c["type"] for c in block["content"]] == ["text", "image"]
        assert block["content"][1]["source"]["media_type"] == "image/jpeg"


class TestContaminationControls:
    def test_every_shipped_query_carries_the_canary(self):
        from momentbench.canary import CANARY
        from momentbench.runner import load_queries

        files = sorted(Path("data/queries").glob("*.json"))
        assert files
        for f in files:
            for q in load_queries(f):
                assert q.metadata.get("canary") == CANARY, f"{f}:{q.query_id}"

    async def test_run_files_do_not_contain_gold_moments(self, tmp_path):
        query = make_query(t_in=11.25, t_out=19.75)
        wrong = ("submit_answer", {"moments": [{"asset_id": "v1", "t_in": 60, "t_out": 70}]})
        run = await run_agentic_benchmark(
            [query], ScriptedModel([[wrong]]), make_corpus(tmp_path), trials=1
        )
        dumped = run.model_dump_json()
        assert "11.25" not in dumped and "19.75" not in dumped
