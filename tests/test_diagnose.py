"""The diagnostic profile is consistent, and diagnose fails clearly without media."""

import json
from pathlib import Path

from momentbench import diagnose

ROOT = Path(__file__).resolve().parents[1]
D = ROOT / "data/diagnostic"


def test_the_profile_is_a_subset_of_the_release():
    manifest = {a["asset_id"]: a for a in json.loads(
        (ROOT / "data/corpus_manifests/sample_manifest.json").read_text())}
    for a in json.loads((D / "manifest.json").read_text()):
        assert manifest[a["asset_id"]] == a
    scored = {q["query_id"]: q for d in ("film", "aerospace", "robotics")
              for q in json.loads((ROOT / f"data/queries/scored/{d}_v0.2_scored.json")
                                  .read_text())}
    queries = json.loads((D / "queries.json").read_text())
    assert queries and all(scored[q["query_id"]] == q for q in queries)
    expected = json.loads((D / "expected.json").read_text())
    assert set(expected["floors"]["transcript"]) == {q["query_id"] for q in queries}


def test_the_prediction_reference_reproduces():
    assert diagnose.prediction_recall() == json.loads(
        (D / "expected.json").read_text())["predictions_recall_at_1"]


def test_missing_media_fails_with_the_download_command(tmp_path, capsys):
    assert diagnose.main(tmp_path / "nothing-here") == 1
    out = capsys.readouterr().out
    assert "FAIL  profile videos verify" in out
    assert "momentbench corpus-download -m data/diagnostic/manifest.json" in out
