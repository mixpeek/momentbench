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


def verified_corpus(tmp_path):
    """make_corpus, pinned to its files as prepare_corpus would after hashing them."""
    corpus = make_corpus(tmp_path)
    files = {a: tmp_path / f"{a}.mp4" for a in corpus.assets}
    corpus.pin_files(files, {a: f"sha-{a}" for a in files})
    return corpus


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


class TestRunIdentity:  # findings 14, 24
    def test_cli_runs_record_what_they_ran(self, tmp_path, monkeypatch):
        from momentbench.cli import main

        manifest, queries, corpus = TestCompleteCorpus()._files(tmp_path)
        monkeypatch.chdir(tmp_path)
        r = CliRunner().invoke(main, ["agentic", "abstain", "--provider", "baseline",
                                      "--partial", "-q", str(queries), "-c", str(manifest),
                                      "--corpus-dir", str(corpus), "--trials", "1",
                                      "--pricing", str(PRICING)])
        assert r.exit_code == 0, r.output
        [f] = (tmp_path / "data/results/debug").glob("*.json")
        ident = json.loads(f.read_text())["identity"]
        for key in ("queries_sha", "queries_file_sha256", "manifest_sha256", "corpus_sha",
                    "system_prompt_sha", "tools_sha", "compatibility_key", "git", "packages",
                    "pricing_as_of"):
            assert ident.get(key) is not None, key
        assert ident["provider"] == "baseline" and ident["budget"]["max_steps"] > 0

    def test_compatibility_key_changes_with_the_task(self):
        from momentbench.agentic.identity import task_identity

        base = dict(queries=[NO_ANSWER, make_query()], receipt={"v1": "a"}, transcripts={},
                    system_prompt="p", tools=[], budget={"max_steps": 30},
                    modalities="frames")
        a = task_identity(**base)["compatibility_key"]
        assert task_identity(**base)["compatibility_key"] == a
        for change in ({"modalities": "frames+transcripts"}, {"receipt": {"v1": "b"}},
                       {"queries": [NO_ANSWER]},  # a subset is a different task (R9)
                       {"queries": [NO_ANSWER, make_query().model_copy(
                           update={"text": "reworded"})]}):
            assert task_identity(**{**base, **change})["compatibility_key"] != a, change

    async def test_resolved_model_is_recorded_per_trial(self, tmp_path):
        class Named(AgentSession):
            async def step(self, results):
                return ModelTurn([ToolCall("a", "submit_answer", {"moments": []})], Usage(),
                                 model="provider-model-2026-10-01")

        class NamedModel(AgentModel):
            model_id = "alias"

            def start(self, system, task, tools):
                return Named()

        run = await run_agentic_benchmark([NO_ANSWER], NamedModel(), make_corpus(tmp_path),
                                          trials=1)
        assert run.trials[0].resolved_model == "provider-model-2026-10-01"
        assert run.identity["resolved_models"] == ["provider-model-2026-10-01"]
        assert run.trials[0].query_text_sha


class TestRescoreDropsChangedText:  # finding 14
    async def test_a_reworded_query_is_not_scored_with_the_old_answer(self, tmp_path):
        from momentbench.agentic.run import rescore

        run = await run_agentic_benchmark([NO_ANSWER], ScriptedModel([[("submit_answer",
                                          {"moments": []})]]), make_corpus(tmp_path), trials=1)
        same = rescore(run, {"na": NO_ANSWER})
        assert len(same.trials) == 1 and same.rescore_excluded == 0
        reworded = NO_ANSWER.model_copy(update={"text": "a different question"})
        changed = rescore(run, {"na": reworded})
        assert changed.trials == [] and changed.rescore_excluded == 1

    async def test_old_runs_fall_back_to_the_refinement_date(self, tmp_path):
        from momentbench.agentic.run import rescore

        run = await run_agentic_benchmark([NO_ANSWER], ScriptedModel([[("submit_answer",
                                          {"moments": []})]]), make_corpus(tmp_path), trials=1)
        legacy = run.model_copy(update={
            "timestamp": "2026-10-01T00:00:00Z",
            "trials": [t.model_copy(update={"query_text_sha": None}) for t in run.trials]})
        def refined_on(day):
            return NO_ANSWER.model_copy(update={"metadata": {"text_refined": {"date": day}}})

        refined = refined_on("2026-10-05")
        assert rescore(legacy, {"na": refined}).rescore_excluded == 1
        earlier = refined_on("2026-09-20")
        assert rescore(legacy, {"na": earlier}).rescore_excluded == 0
        # Same day: the exact instant decides which wording the run saw.
        def applied(at):
            return NO_ANSWER.model_copy(update={"metadata": {"text_refined": {
                "date": at[:10], "applied_at": at}}})

        assert rescore(legacy, {"na": applied("2026-10-01T09:00:00Z")}).rescore_excluded == 1
        assert rescore(legacy, {"na": applied("2026-09-30T23:00:00Z")}).rescore_excluded == 0


class TestLegacyLabels:  # findings 2, 3
    EXPECTED = {
        "gemini": "metadata-only", "openai": "metadata-only", "claude": "metadata-only",
        "qwen-vl": "metadata-only", "hf-vlm": "metadata-only",
        "clip": "whole-video-control", "jina-clip": "whole-video-control",
        "mixedbread": "whole-video-control", "e5-omni": "whole-video-control",
        "twelvelabs": "indexed-platform", "mixpeek": "indexed-platform",
    }

    @pytest.mark.parametrize("name", sorted(EXPECTED))
    def test_every_builtin_adapter_declares_what_it_sees(self, name):
        from momentbench.models.base import INPUT_KINDS
        from momentbench.models.registry import get_adapter_class

        try:
            cls = get_adapter_class(name)
        except (ImportError, ValueError) as e:  # optional dependency not installed
            pytest.skip(str(e))
        assert cls.input_kind == self.EXPECTED[name] and cls.input_kind in INPUT_KINDS

    def test_run_warns_and_records_the_input_kind(self, tmp_path):
        from momentbench.cli import main
        from momentbench.models.base import ModelAdapter
        from momentbench.models.registry import register_adapter

        @register_adapter("title-guesser-test")
        class TitleGuesser(ModelAdapter):
            model_id = "title-guesser-test"

            async def predict(self, query, corpus_assets, max_results=5):
                return Prediction(query_id=query.query_id, model_id=self.model_id,
                                  ranked_moments=[])

        manifest, queries, _ = TestCompleteCorpus()._files(tmp_path)
        out = tmp_path / "out"
        r = CliRunner().invoke(main, ["run", "title-guesser-test", "-q", str(queries),
                                      "-c", str(manifest), "-o", str(out)])
        assert r.exit_code == 0, r.output
        text = " ".join(r.output.split())  # rich wraps to the terminal width
        assert "metadata-only" in text and "momentbench agentic" in text
        [f] = out.glob("*.json")
        assert json.loads(f.read_text())["config"]["input_kind"] == "metadata-only"


class TestFrameSampling:  # finding 26
    @pytest.mark.skipif(not __import__("shutil").which("ffmpeg"), reason="needs ffmpeg")
    def test_the_last_second_of_a_video_returns_a_frame(self, tmp_path):
        import subprocess

        from momentbench.agentic.environment import LocalCorpus

        # Picture stops at 2 s, audio runs to 3 s: the shape of Elephants Dream.
        video = tmp_path / "v.mp4"
        subprocess.run(["ffmpeg", "-v", "error", "-f", "lavfi", "-i",
                        "testsrc=duration=2:size=128x72:rate=10", "-f", "lavfi", "-i",
                        "sine=duration=3", "-c:v", "libx264", "-pix_fmt", "yuv420p",
                        "-c:a", "aac", str(video)], check=True)
        corpus = LocalCorpus([{"asset_id": "v", "title": "t", "duration_s": 3.0}], tmp_path,
                             transcripts={})
        info, frames = corpus.sample_frames("v", 3.0, 3.0, 1)
        assert frames and frames[0][:2] == b"\xff\xd8"
        assert json.loads(info)["frame_times_s"][0] <= 2.0  # reports what was decoded
        assert len(corpus.sample_frames("v", 0, 3.0, 4)[1]) == 4

    async def test_frame_extraction_does_not_block_other_episodes(self, tmp_path):
        import time

        from momentbench.agentic.environment import LocalCorpus
        from tests.test_agentic import ASSETS, TRANSCRIPTS

        for a in ASSETS:
            (tmp_path / f"{a['asset_id']}.mp4").write_bytes(b"")

        def slow(video, t):
            time.sleep(0.4)  # a blocking ffmpeg call
            return b"jpeg"

        corpus = LocalCorpus(ASSETS, tmp_path, frame_extractor=slow, transcripts=TRANSCRIPTS)
        script = [[("sample_frames", {"asset_id": "v1", "t_start": 0, "t_end": 1,
                                      "num_frames": 1})], [SUBMIT_GOLD]]
        queries = [make_query().model_copy(update={"query_id": f"q{i}"}) for i in range(4)]
        start = time.perf_counter()
        await run_agentic_benchmark(queries, ScriptedModel(script), corpus, trials=1,
                                    max_concurrency=4)
        assert time.perf_counter() - start < 1.2  # serial would take 1.6 s


class TestSubmission:  # finding 28
    async def _run(self, tmp_path, queries, reportable=True):
        script = [[("submit_answer", {"moments": []})]]
        return await run_agentic_benchmark(queries, ScriptedModel(script),
                                           verified_corpus(tmp_path), trials=1,
                                           reportable=reportable)

    def _release(self, tmp_path):
        """A manifest and transcripts dir matching verified_corpus, as a release."""
        from tests.test_agentic import ASSETS, TRANSCRIPTS

        manifest = tmp_path / "manifest.json"
        manifest.write_text(json.dumps([{**a, "file": {"sha256": f"sha-{a['asset_id']}"}}
                                        for a in ASSETS]))
        tdir = tmp_path / "transcripts"
        tdir.mkdir(exist_ok=True)
        for aid, segs in TRANSCRIPTS.items():
            (tdir / f"{aid}.transcript.json").write_text(json.dumps(segs))
        return manifest, tdir

    def _file(self, tmp_path, queries):
        path = tmp_path / "q_v0.2.json"
        path.write_text(json.dumps([q.model_dump(mode="json") for q in queries]))
        return path

    async def test_a_complete_reportable_run_is_packaged(self, tmp_path):
        from momentbench.submission import agentic_entry
        from tests.test_agentic import TRANSCRIPTS

        qs = [NO_ANSWER, make_query()]
        manifest, _ = self._release(tmp_path)
        entry = agentic_entry(await self._run(tmp_path, qs), self._file(tmp_path, qs),
                              manifest=manifest, transcripts=TRANSCRIPTS)
        assert entry["track"] == "agentic" and entry["num_queries"] == 2
        assert entry["recall_at_1"] == 0.5

    async def test_incomplete_or_partial_runs_are_refused(self, tmp_path):
        from momentbench.submission import SubmissionError, agentic_entry
        from tests.test_agentic import TRANSCRIPTS

        qs = [NO_ANSWER, make_query()]
        path = self._file(tmp_path, qs)
        manifest, _ = self._release(tmp_path)
        release = {"manifest": manifest, "transcripts": TRANSCRIPTS}
        with pytest.raises(SubmissionError, match="does not match the current release"):
            agentic_entry(await self._run(tmp_path, [NO_ANSWER]), path, **release)
        with pytest.raises(SubmissionError, match="not reportable"):
            agentic_entry(await self._run(tmp_path, qs, reportable=False), path, **release)
        # Listing older runs records coverage instead of refusing.
        entry = agentic_entry(await self._run(tmp_path, [NO_ANSWER]), path,
                              require_complete=False)
        assert (entry["num_queries"], entry["queries_in_file"]) == (1, 2)

    async def test_cli_submit_and_leaderboard_take_agentic_runs(self, tmp_path):
        from momentbench.cli import main

        qs = [NO_ANSWER, make_query()]
        qfile = self._file(tmp_path, qs)
        runs = tmp_path / "runs"
        run = await self._run(tmp_path, qs)
        runs.mkdir()
        run_file = runs / "run.json"
        run_file.write_text(run.model_dump_json())
        out = tmp_path / "subs"
        manifest, tdir = self._release(tmp_path)
        r = CliRunner().invoke(main, ["submit", "-r", str(run_file), "-q", str(qfile),
                                      "-c", str(manifest), "--transcripts-dir", str(tdir),
                                      "--model-name", "Floor", "-o", str(out)])
        assert r.exit_code == 0, r.output
        meta = json.loads((out / "floor" / "submission.json").read_text())
        assert meta["recall_at_1"] == 0.5 and meta["compatibility_key"]
        assert (out / "floor" / "run.json").exists()

        legacy = evaluate_run([q("a")], [pred("a")])
        legacy_file = tmp_path / "legacy.json"
        legacy_file.write_text(legacy.model_dump_json())
        r = CliRunner().invoke(main, ["submit", "-r", str(legacy_file), "--model-name", "x",
                                      "-o", str(out)])
        assert r.exit_code != 0 and "legacy-track run" in r.output


class TestAnonymizedTitles:  # finding 16
    def _corpus(self, tmp_path):
        from momentbench.agentic.environment import LocalCorpus
        from tests.test_agentic import ASSETS, TRANSCRIPTS

        for a in ASSETS:
            (tmp_path / f"{a['asset_id']}.mp4").write_bytes(b"")
        return LocalCorpus(ASSETS, tmp_path, frame_extractor=lambda v, t: b"jpeg",
                           transcripts=TRANSCRIPTS, anonymize_titles=True)

    def test_the_model_sees_no_title_and_no_real_id(self, tmp_path):
        corpus = self._corpus(tmp_path)
        listed = json.loads(corpus.list_assets())
        assert all(set(row) == {"asset_id", "duration_s", "has_transcript"} for row in listed)
        assert all(row["asset_id"].startswith("video-") for row in listed)
        for real in corpus.assets:
            assert real not in corpus.list_assets()
            assert real not in corpus.search_transcript("hearing order")

    def test_real_ids_are_refused_and_aliases_work(self, tmp_path):
        corpus = self._corpus(tmp_path)
        alias = corpus.shown_id("v1")
        with pytest.raises(ValueError, match="unknown asset_id"):
            corpus.read_transcript("v1", 0, 30)
        assert json.loads(corpus.sample_frames(alias, 0, 10, 2)[0])["asset_id"] == alias
        with pytest.raises(InvalidSubmissionError):
            corpus.parse_answer({"moments": [{"asset_id": "v1", "t_in": 10, "t_out": 20}]})

    async def test_an_answer_by_alias_is_scored_against_the_real_gold(self, tmp_path):
        corpus = self._corpus(tmp_path)
        answer = ("submit_answer", {"moments": [{"asset_id": corpus.shown_id("v1"),
                                                 "t_in": 10, "t_out": 20}]})
        run = await run_agentic_benchmark([make_query()], ScriptedModel([[answer]]), corpus,
                                          trials=1)
        assert run.summary.recall_at_1 == 1.0 and run.anonymize_titles is True
        assert run.trials[0].moments[0].asset_id == "v1"

    def test_the_condition_changes_the_compatibility_key(self):
        from momentbench.agentic.identity import task_identity

        base = dict(queries=[NO_ANSWER], receipt={}, transcripts={}, system_prompt="p",
                    tools=[], budget={}, modalities="frames+transcripts")
        assert (task_identity(**base)["compatibility_key"]
                != task_identity(**base, anonymize_titles=True)["compatibility_key"])



class TestAdmission:  # audit 2: R3, R5, R6, R7
    async def test_an_api_run_on_an_unverified_corpus_is_not_reportable(self, tmp_path):
        run = await run_agentic_benchmark([NO_ANSWER], ScriptedModel([[("submit_answer",
                                          {"moments": []})]]), make_corpus(tmp_path), trials=1)
        assert run.reportable is False and run.corpus_receipt == {}
        assert run.identity["compatibility_key"]

    async def test_a_wrong_receipt_or_key_cannot_be_submitted(self, tmp_path):
        from momentbench.submission import SubmissionError, agentic_entry
        from tests.test_agentic import TRANSCRIPTS

        sub = TestSubmission()
        qs = [NO_ANSWER, make_query()]
        path, (manifest, _) = sub._file(tmp_path, qs), sub._release(tmp_path)
        run = await sub._run(tmp_path, qs)
        release = {"manifest": manifest, "transcripts": TRANSCRIPTS}
        agentic_entry(run, path, **release)  # the honest run passes
        wrong_bytes = run.model_copy(update={"corpus_receipt": {**run.corpus_receipt,
                                                                 "v1": "other"}})
        with pytest.raises(SubmissionError, match="corpus receipt"):
            agentic_entry(wrong_bytes, path, **release)
        for ident in ({}, {**run.identity, "compatibility_key": "forged"}):
            with pytest.raises(SubmissionError, match="compatibility key"):
                agentic_entry(run.model_copy(update={"identity": ident}), path, **release)

    async def test_duplicated_or_missing_trials_are_rejected(self, tmp_path):
        from momentbench.submission import SubmissionError, agentic_entry

        script = [[SUBMIT_GOLD]]
        run = await run_agentic_benchmark([make_query()], ScriptedModel(script),
                                          verified_corpus(tmp_path), trials=2)
        hit = run.trials[0]
        stuffed = run.model_copy(update={"trials": [hit] * 4 + run.trials[1:]})
        with pytest.raises(SubmissionError, match="more than once"):
            agentic_entry(stuffed, tmp_path / "unused.json", require_complete=False)
        dropped = run.model_copy(update={"trials": run.trials[:1]})
        with pytest.raises(SubmissionError, match="expected 0 to 1"):
            agentic_entry(dropped, tmp_path / "unused.json", require_complete=False)

    def test_frames_come_from_the_verified_file(self, tmp_path):
        from momentbench.agentic.environment import LocalCorpus

        real, impostor = tmp_path / "viper.mp4", tmp_path / "ingenuity.mp4"
        real.write_bytes(b"viper")
        impostor.write_bytes(b"ingenuity")
        seen = []
        corpus = LocalCorpus([{"asset_id": "viper", "title": "t", "duration_s": 10,
                               "storage_path": str(impostor)}], tmp_path,
                             frame_extractor=lambda v, t: seen.append(v) or b"jpeg",
                             transcripts={})
        corpus.pin_files({"viper": real}, {"viper": "sha"})
        corpus.sample_frames("viper", 0, 5, 1)
        assert seen == [real]
        real.write_bytes(b"swapped after verification")
        with pytest.raises(RuntimeError, match="changed after verification"):
            corpus.sample_frames("viper", 0, 5, 1)


class TestMeteredCustomLoops:  # audit 2: R4
    async def test_direct_and_parallel_calls_cannot_draw_unmetered_frames(self, tmp_path):
        import asyncio

        got = {}

        class Greedy(AgentModel):
            model_id = "greedy"

            def start(self, system, task, tools):
                raise AssertionError

            async def run_episode(self, corpus, query, budget):
                frames, refused = 0, 0
                for _ in range(3):  # the audit's probe: 3 direct calls of 8 frames
                    try:
                        frames += len(corpus.sample_frames("v1", 0, 60, 8)[1])
                    except ValueError:
                        refused += 1

                async def grab():
                    try:
                        return len(await asyncio.to_thread(
                            corpus.sample_frames, "v1", 0, 60, 8))
                    except ValueError:
                        return 0

                frames += sum(await asyncio.gather(*(grab() for _ in range(8))))
                for name in ("_corpus", "_MeteredCorpus__corpus", "assets", "_extract"):
                    try:
                        getattr(corpus, name)
                        got.setdefault("reached", []).append(name)
                    except AttributeError:
                        pass
                got.update(frames=frames, refused=refused)
                return Episode(query.query_id, [Moment(asset_id="v1", t_in=10, t_out=20)],
                               Usage(steps=1), "submitted")

        run = await run_agentic_benchmark([make_query()], Greedy(), make_corpus(tmp_path),
                                          budget=Budget(max_frames=1), trials=1)
        t = run.trials[0]
        assert got["frames"] <= 1 and got["refused"] == 2
        assert t.usage.frames <= 1 and t.outcome == "submitted"
        # Name mangling only deters; it is the documented limit of in-process isolation.
        assert set(got.get("reached", [])) <= {"_MeteredCorpus__corpus"}


class TestRankedAnswers:  # audit 2: R8
    def test_an_invalid_top_candidate_rejects_the_ranking(self):
        good = {"asset_id": "v1", "t_in": 10, "t_out": 20}
        for bad in ({"asset_id": "nope", "t_in": 1, "t_out": 2},
                    {"asset_id": "v1", "t_in": 5, "t_out": 5}):
            with pytest.raises(InvalidSubmissionError, match="invalid submission"):
                parse_submission({"moments": [bad, good]}, {"v1": {"duration_s": 100}})
        assert parse_submission({"moments": [good]}, {"v1": {"duration_s": 100}})

    async def test_it_scores_zero_unless_the_model_resubmits(self, tmp_path):
        bad_first = ("submit_answer", {"moments": [
            {"asset_id": "nope", "t_in": 1, "t_out": 2},
            {"asset_id": "v1", "t_in": 10, "t_out": 20}]})
        run = await run_agentic_benchmark([make_query()], ScriptedModel([[bad_first]]),
                                          make_corpus(tmp_path), trials=1)
        assert run.summary.recall_at_1 == 0.0


class TestComparableOnly:  # audit 2: R9
    async def test_runs_with_different_tasks_are_never_ranked_together(self, tmp_path,
                                                                      monkeypatch):
        from momentbench import submission
        from momentbench.cli import main

        qs = [NO_ANSWER, make_query()]
        qdir = tmp_path / "queries"
        qdir.mkdir()
        (qdir / "q_v0.2.json").write_text(json.dumps([x.model_dump(mode="json") for x in qs]))
        monkeypatch.setattr(submission, "QUERIES_DIR", qdir)
        runs = tmp_path / "runs"
        runs.mkdir()
        script = [[("submit_answer", {"moments": []})]]
        for i, frames in enumerate((64, 8)):
            (tmp_path / f"c{i}").mkdir()
            run = await run_agentic_benchmark(qs, ScriptedModel(script),
                                              verified_corpus(tmp_path / f"c{i}"), trials=1,
                                              budget=Budget(max_frames=frames))
            (runs / f"r{i}.json").write_text(run.model_dump_json())
        r = CliRunner().invoke(main, ["leaderboard", str(runs)], env={"COLUMNS": "200"})
        titles = [line for line in r.output.splitlines() if "agentic track" in line]
        assert r.exit_code == 0 and len(titles) == 2, r.output


class TestTranscriptControl:  # audit 2: R12
    async def test_it_answers_from_speech_and_never_samples_a_frame(self, tmp_path):
        from momentbench.agentic.baselines import BaselineAgent

        run = await run_agentic_benchmark([make_query(), NO_ANSWER],
                                          BaselineAgent("transcript"),
                                          make_corpus(tmp_path), trials=1)
        by_id = {t.query_id: t for t in run.trials}
        assert by_id["q1"].moments and by_id["q1"].usage.frames == 0
        assert [x["tool"] for x in by_id["q1"].trace][0] == "search_transcript"
        assert run.model_id == "baseline-transcript"


class TestMergeRuns:  # MO-4530
    async def test_a_rerun_of_reworded_queries_fills_the_hole(self, tmp_path):
        import importlib.util

        from momentbench.submission import agentic_entry
        from tests.test_agentic import TRANSCRIPTS

        spec = importlib.util.spec_from_file_location(
            "merge_runs", Path(__file__).resolve().parents[1] / "scripts/merge_runs.py")
        m = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(m)

        sub = TestSubmission()
        manifest, _ = sub._release(tmp_path)
        abstain = [[("submit_answer", {"moments": []})]]
        base = await run_agentic_benchmark([NO_ANSWER, make_query()], ScriptedModel(abstain),
                                           verified_corpus(tmp_path), trials=1)
        reworded = make_query().model_copy(update={"text": "the hearing begins"})
        qfile = sub._file(tmp_path, [NO_ANSWER, reworded])
        patch = await run_agentic_benchmark([reworded], ScriptedModel([[SUBMIT_GOLD]]),
                                            verified_corpus(tmp_path), trials=1)
        merged = m.merge(base, patch, qfile, manifest, TRANSCRIPTS)
        assert {t.query_id for t in merged.trials} == {"na", "q1"}
        assert merged.summary.recall_at_1 == 1.0
        assert merged.identity["merged_from"] == [base.run_id, patch.run_id]
        entry = agentic_entry(merged, qfile, manifest=manifest, transcripts=TRANSCRIPTS)
        assert entry["compatibility_key"] == merged.identity["compatibility_key"]

        other = await run_agentic_benchmark([reworded], ScriptedModel([[SUBMIT_GOLD]]),
                                            verified_corpus(tmp_path), trials=1,
                                            budget=Budget(max_frames=8))
        with pytest.raises(SystemExit, match="budget"):
            m.merge(base, other, qfile, manifest, TRANSCRIPTS)


class SnapshotSession(AgentSession):
    """Replies from a fixed list of (tool call, reported model) turns."""

    def __init__(self, turns):
        self._turns = list(turns)

    async def step(self, tool_results):
        (name, args), model = self._turns.pop(0)
        return ModelTurn([ToolCall("c", name, args)], Usage(input_tokens=1), model=model)


class SnapshotModel(AgentModel):
    model_id = "Qwen/Qwen3-VL-32B-Instruct"

    def __init__(self, turns):
        self._turns = turns

    def start(self, system, task, tools):
        return SnapshotSession(self._turns)


class TestSafeRunIds:  # audit 3, finding 1
    async def test_a_slash_in_the_model_id_still_saves(self, tmp_path):
        model = SnapshotModel([(SUBMIT_GOLD, "Qwen/Qwen3-VL-32B-Instruct")])
        out = tmp_path / "runs"
        run = await run_agentic_benchmark([make_query()], model, verified_corpus(tmp_path),
                                          trials=1, output_dir=out)
        assert "/" not in run.run_id and run.model_id == "Qwen/Qwen3-VL-32B-Instruct"
        assert [p.name for p in out.iterdir()] == [f"{run.run_id}.json"]

    async def test_a_provider_failure_saves_the_partial_file(self, tmp_path):
        from momentbench.agentic.run import ProviderUnavailableError

        class UnauthorizedError(Exception):
            status_code = 401

        class FailsSecond(SnapshotModel):
            started = 0

            def start(self, system, task, tools):
                self.started += 1
                if self.started == 2:
                    raise UnauthorizedError("key revoked")
                return SnapshotSession([(SUBMIT_GOLD, None)])

        out = tmp_path / "runs"
        with pytest.raises(ProviderUnavailableError):
            await run_agentic_benchmark([make_query(), NO_ANSWER], FailsSecond([]),
                                        verified_corpus(tmp_path), trials=1,
                                        max_concurrency=1, output_dir=out)
        [partial] = list(out.iterdir())
        assert partial.name.startswith("agentic-Qwen-Qwen3-VL-32B-Instruct-")
        assert partial.name.endswith(".partial.json")
        assert json.loads(partial.read_text())["model_id"] == "Qwen/Qwen3-VL-32B-Instruct"


class TestSnapshotIdentity:  # audit 3, finding 3
    async def test_a_model_change_mid_trial_ends_it(self, tmp_path):
        list_assets = ("list_assets", {})
        model = SnapshotModel([(list_assets, "snapshot-A"), (SUBMIT_GOLD, "snapshot-B")])
        run = await run_agentic_benchmark([make_query()], model, verified_corpus(tmp_path),
                                          trials=1)
        t = run.trials[0]
        assert t.outcome == "model_changed" and not t.hit_at_1
        assert t.turn_models == ["snapshot-A", "snapshot-B"]
        assert run.identity["resolved_models"] == ["snapshot-A", "snapshot-B"]

    async def test_a_run_with_two_snapshots_is_not_admitted(self, tmp_path):
        from momentbench.submission import SubmissionError, check_model_identity

        run = await run_agentic_benchmark([make_query()],
                                          SnapshotModel([(SUBMIT_GOLD, "snapshot-A")]),
                                          verified_corpus(tmp_path), trials=1)
        check_model_identity(run)
        t = run.trials[0].model_copy(update={"trial": 1, "resolved_model": "snapshot-B",
                                             "turn_models": ["snapshot-B"]})
        mixed = run.model_copy(update={"trials": [*run.trials, t]})
        with pytest.raises(SubmissionError, match="2 model identities"):
            check_model_identity(mixed)

    async def test_merge_refuses_two_snapshots_and_keeps_provenance(self, tmp_path):
        import importlib.util

        from tests.test_agentic import TRANSCRIPTS

        spec = importlib.util.spec_from_file_location(
            "merge_runs", Path(__file__).resolve().parents[1] / "scripts/merge_runs.py")
        m = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(m)
        sub = TestSubmission()
        manifest, _ = sub._release(tmp_path)
        abstain = ("submit_answer", {"moments": []})
        base = await run_agentic_benchmark(
            [NO_ANSWER, make_query()], SnapshotModel([(abstain, "snapshot-A")] * 2),
            verified_corpus(tmp_path), trials=1)
        reworded = make_query().model_copy(update={"text": "the hearing begins"})
        qfile = sub._file(tmp_path, [NO_ANSWER, reworded])
        drifted = await run_agentic_benchmark(
            [reworded], SnapshotModel([(SUBMIT_GOLD, "snapshot-B")]),
            verified_corpus(tmp_path), trials=1)
        with pytest.raises(SystemExit, match="different model snapshots"):
            m.merge(base, drifted, qfile, manifest, TRANSCRIPTS)
        moved = drifted.model_copy(update={"identity": {**drifted.identity,
                                                        "endpoint": "http://other/v1"}})
        with pytest.raises(SystemExit, match="identity.endpoint differs"):
            m.merge(base, moved, qfile, manifest, TRANSCRIPTS)
        same = await run_agentic_benchmark(
            [reworded], SnapshotModel([(SUBMIT_GOLD, "snapshot-A")]),
            verified_corpus(tmp_path), trials=1)
        merged = m.merge(base, same, qfile, manifest, TRANSCRIPTS)
        assert "/" not in merged.run_id
        assert [i["run_id"] for i in merged.identity["merged_inputs"]] == [base.run_id,
                                                                            same.run_id]
        assert merged.identity["merged_inputs"][1]["resolved_models"] == ["snapshot-A"]


class TestTextControls:  # audit 3, finding 4
    def test_bm25_finds_a_cue_the_keyword_search_ranks_lower(self):
        from momentbench.agentic.text_controls import BM25Index, bm25_control

        transcripts = {"a": [{"t_in": 0.0, "t_out": 4.0, "text": "the the the launch the"},
                             {"t_in": 10.0, "t_out": 14.0, "text": "booster separation"}],
                       "b": [{"t_in": 0.0, "t_out": 4.0, "text": "the launch and the pad"}]}
        index = BM25Index(transcripts)
        assert index.search("booster separation after the launch")[0].t_in == 10.0
        query = make_query().model_copy(update={
            "text": "booster separation", "gold_moments": [Moment(asset_id="a", t_in=10.0,
                                                                  t_out=14.0)]})
        assert bm25_control([query], transcripts) == {"q1": True}
        assert BM25Index(transcripts).search("unrelated words") == []
