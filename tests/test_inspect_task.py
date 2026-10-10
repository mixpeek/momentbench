"""The Inspect AI wrapper, run end to end with Inspect's mock model."""

import json
from pathlib import Path

import pytest

pytest.importorskip("inspect_ai")

from inspect_ai import eval as inspect_eval  # noqa: E402
from inspect_ai.model import ModelOutput, get_model  # noqa: E402

from momentbench.agentic.budget import Budget  # noqa: E402
from momentbench.agentic.environment import LocalCorpus  # noqa: E402
from momentbench.agentic.inspect_task import (  # noqa: E402
    REPO_ROOT,
    build_task_from,
    momentbench_agentic,
    parse_answer,
)
from momentbench.types import Moment, Query, QueryTier  # noqa: E402

ASSETS = [{"asset_id": "v1", "title": "Hearing", "duration_s": 100, "storage_path": None}]
QUERY = Query(
    query_id="q1", text="the hearing is called to order", tier=QueryTier.SINGLE_VISUAL,
    vertical="civic", gold_moments=[Moment(asset_id="v1", t_in=10.0, t_out=20.0)],
)


def run(tmp_path: Path, outputs, budget: Budget | None = None, query: Query = QUERY):
    tmp_path.mkdir(parents=True, exist_ok=True)
    (tmp_path / "v1.mp4").write_bytes(b"")
    corpus = LocalCorpus(ASSETS, tmp_path, frame_extractor=lambda video, t: b"\xff\xd8jpeg")
    task = build_task_from([query], corpus, budget or Budget())
    model = get_model("mockllm/model", custom_outputs=outputs)
    [log] = inspect_eval(task, model=model, log_dir=str(tmp_path / "logs"), display="none")
    assert log.status == "success", log.error
    return log.samples[0].scores["recall_at_1"]


def call(name, **args):
    return ModelOutput.for_tool_call("mockllm/model", name, args)


def test_frames_then_correct_submission_scores_a_hit(tmp_path):
    score = run(tmp_path, [
        call("sample_frames", asset_id="v1", t_start=10, t_end=20, num_frames=3),
        call("submit", answer='[{"asset_id": "v1", "t_in": 10, "t_out": 20}]'),
    ])
    assert score.value == 1.0
    assert score.metadata["frames"] == 3 and score.metadata["tool_calls"] == 1


def test_wrong_span_scores_a_miss(tmp_path):
    score = run(tmp_path, [call("submit", answer='[{"asset_id": "v1", "t_in": 60, "t_out": 70}]')])
    assert score.value == 0.0 and score.metadata["best_temporal_iou"] == 0.0


def test_frame_budget_is_enforced_across_calls(tmp_path):
    frames = call("sample_frames", asset_id="v1", t_start=0, t_end=50, num_frames=4)
    score = run(
        tmp_path,
        [frames, frames, frames, call("submit", answer="[]")],
        Budget(max_frames=6),
    )
    assert score.metadata["frames"] == 6 and score.value == 0.0


def test_parse_answer_ignores_prose_and_garbage():
    assert parse_answer("no idea", {"v1"}) is None
    assert parse_answer("[not json]", {"v1"}) is None
    assert parse_answer("[]", {"v1"}) == []
    got = parse_answer('Here: [{"asset_id": "v1", "t_in": 1, "t_out": 2}]', {"v1"})
    assert [(m.t_in, m.t_out) for m in got] == [(1.0, 2.0)]


def test_seed_task_loads_without_a_corpus_dir():
    seed = json.loads((REPO_ROOT / "data/queries/sample.json").read_text())
    assert len(momentbench_agentic().dataset) == len(seed)


NO_ANSWER = Query(query_id="na", text="a moment that is not in the corpus",
                  tier=QueryTier.CORPUS_HARD_NEGATIVE, vertical="civic", gold_moments=[])


def test_no_answer_query_is_a_hit_only_on_an_explicit_empty_submission(tmp_path):
    assert run(tmp_path / "a", [call("submit", answer="[]")], query=NO_ANSWER).value == 1.0
    assert run(tmp_path / "b", [call("submit", answer="I give up")], query=NO_ANSWER).value == 0.0
