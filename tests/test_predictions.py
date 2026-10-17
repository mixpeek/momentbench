"""The prediction-file track: same scoring rules, its own key, accepted by submit."""

import json

import pytest

from momentbench.predictions import score_predictions
from momentbench.submission import SubmissionError, agentic_entry
from tests.test_agentic import make_query
from tests.test_score_integrity import NO_ANSWER, TestSubmission


@pytest.fixture
def release(tmp_path):
    sub = TestSubmission()
    second = make_query().model_copy(update={"query_id": "q2", "text": "another moment"})
    qs = [NO_ANSWER, make_query(), second]
    manifest, _ = sub._release(tmp_path)
    return sub._file(tmp_path, qs), manifest


def write(tmp_path, preds):
    path = tmp_path / "preds.json"
    path.write_text(json.dumps({"system": "Org/retriever v2", "predictions": preds}))
    return path


def test_scores_abstains_and_rejects(tmp_path, release):
    qfile, manifest = release
    preds = write(tmp_path, [
        {"query_id": "q1", "moments": [{"asset_id": "v1", "t_in": 10, "t_out": 20}]},
        {"query_id": "na", "moments": []},
        {"query_id": "q2", "moments": [{"asset_id": "nope", "t_in": 1, "t_out": 2}]}])
    run = score_predictions(preds, qfile, manifest, tmp_path / "out")
    by = {t.query_id: t for t in run.trials}
    assert by["q1"].hit_at_1 and by["na"].hit_at_1
    assert by["q2"].outcome == "invalid_submission" and not by["q2"].hit_at_1
    assert run.harness == "predictions" and "/" not in run.run_id
    entry = agentic_entry(run, qfile, manifest=manifest)
    assert entry["recall_at_1"] == pytest.approx(2 / 3, abs=1e-3)


def test_a_missing_query_counts_and_the_key_is_its_own(tmp_path, release):
    qfile, manifest = release
    run = score_predictions(write(tmp_path, [{"query_id": "na", "moments": []}]),
                            qfile, manifest)
    assert sum(t.outcome == "no_submission" for t in run.trials) == 2
    assert run.summary.recall_at_1 == pytest.approx(1 / 3, abs=1e-3)
    forged = run.model_copy(update={"identity": {**run.identity,
                                                 "compatibility_key": "x"}})
    with pytest.raises(SubmissionError, match="compatibility key"):
        agentic_entry(forged, qfile, manifest=manifest)


def test_unknown_or_duplicate_queries_are_errors(tmp_path, release):
    qfile, manifest = release
    with pytest.raises(ValueError, match="not in"):
        score_predictions(write(tmp_path, [{"query_id": "zzz", "moments": []}]),
                          qfile, manifest)
    with pytest.raises(ValueError, match="more than once"):
        score_predictions(write(tmp_path, [{"query_id": "na", "moments": []}] * 2),
                          qfile, manifest)


def test_a_bundle_carries_the_validated_numbers(tmp_path, release):
    from momentbench.bundle import make_bundle

    qfile, manifest = release
    preds = write(tmp_path, [{"query_id": "na", "moments": []}])
    run = score_predictions(preds, qfile, manifest, tmp_path / "out")
    out = make_bundle(next((tmp_path / "out").glob("*.json")), qfile, tmp_path / "b",
                      manifest)
    result = json.loads((out / "result.json").read_text())
    assert result["recall_at_1"] == pytest.approx(run.summary.recall_at_1, abs=1e-4)
    assert result["compatibility_key"] == run.identity["compatibility_key"]
    assert "33.3%" in (out / "model_card.md").read_text()
    assert "\\begin{tabular}" in (out / "table.tex").read_text()
    assert (out / "citation.bib").read_text().startswith("@misc{momentbench")
