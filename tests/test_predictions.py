"""The prediction-file track: same scoring rules, its own key, accepted by submit."""

import json
from pathlib import Path

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


def test_the_submission_validator_passes_and_fails(tmp_path, release, monkeypatch):
    import importlib.util
    from pathlib import Path

    from momentbench import submission as sub_mod

    qfile, manifest = release
    run = score_predictions(write(tmp_path, [{"query_id": "na", "moments": []}]),
                            qfile, manifest)
    spec = importlib.util.spec_from_file_location(
        "vs", Path(__file__).resolve().parents[1] / "scripts/validate_submissions.py")
    vs = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(vs)
    monkeypatch.setattr(vs, "ROOT", tmp_path)
    (tmp_path / "data/queries").mkdir(parents=True)
    (tmp_path / "data/queries" / qfile.name).write_text(qfile.read_text())
    monkeypatch.setattr(sub_mod, "MANIFEST", manifest)
    monkeypatch.setattr(vs, "agentic_entry",
                        lambda r, q: sub_mod.agentic_entry(r, q, manifest=manifest))
    d = tmp_path / "submissions/team"
    d.mkdir(parents=True)
    (d / "run.json").write_text(run.model_dump_json())
    (d / "submission.json").write_text(json.dumps({"queries_file": qfile.name}))
    assert vs.main([str(d)]) == 0
    bad = run.model_copy(update={"trials": run.trials[:-1]})
    (d / "run.json").write_text(bad.model_dump_json())
    assert vs.main([str(d)]) == 1


@pytest.mark.parametrize("custom_manifest", [False, True])
def test_the_printed_submit_command_runs_as_printed(tmp_path, monkeypatch, custom_manifest):
    import shlex
    import shutil

    from click.testing import CliRunner

    from momentbench.cli import main

    repo = Path(__file__).resolve().parents[1]
    monkeypatch.chdir(repo)
    queries = "data/queries/scored/robotics_v0.2_scored.json"
    ids = [q["query_id"] for q in json.loads(Path(queries).read_text())]
    preds = tmp_path / "preds.json"
    preds.write_text(json.dumps({"system": "my retriever",
                                 "predictions": [{"query_id": i, "moments": []} for i in ids]}))
    extra = []
    if custom_manifest:
        manifest = tmp_path / "manifest.json"
        shutil.copy("data/corpus_manifests/sample_manifest.json", manifest)
        extra = ["-c", str(manifest)]
    runner = CliRunner()
    scored = runner.invoke(main, ["score-predictions", "-p", str(preds), "-q", queries,
                                  "-o", str(tmp_path / "runs"), *extra], terminal_width=80)
    assert scored.exit_code == 0, scored.output
    saved = next(x for x in scored.output.splitlines() if "Saved:" in x)
    assert Path(saved.split("Saved:", 1)[1].strip()).is_file()
    line = next(x for x in scored.output.splitlines() if "Submit with:" in x)
    argv = shlex.split(line.split("Submit with:", 1)[1])
    assert argv[:2] == ["momentbench", "submit"] and "--contact" not in argv
    assert ("-c" in argv) == custom_manifest
    if custom_manifest:
        assert argv[argv.index("-c") + 1] == str(tmp_path / "manifest.json")
    submitted = runner.invoke(main, argv[1:] + ["-o", str(tmp_path / "subs")])
    assert submitted.exit_code == 0, submitted.output
