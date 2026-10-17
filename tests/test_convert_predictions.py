"""scripts/convert_predictions.py: retriever output to a prediction file the scorer accepts."""

from __future__ import annotations

import importlib
import json
import sys
from pathlib import Path

from momentbench.predictions import score_predictions

ROOT = Path(__file__).resolve().parents[1]
QUERIES = ROOT / "data/queries/scored/robotics_v0.2_scored.json"
MANIFEST = ROOT / "data/corpus_manifests/sample_manifest.json"


def converter():
    sys.path.insert(0, str(ROOT / "scripts"))
    return importlib.import_module("convert_predictions")


def first_two():
    qs = json.loads(QUERIES.read_text())[:2]
    assets = json.loads(MANIFEST.read_text())
    asset = next(a for a in assets if a["asset_id"] == "nasa-perseverance-twin-rover")
    return qs, asset


def test_qvh_ranks_windows_across_videos_by_score(tmp_path):
    cv = converter()
    (q1, q2), asset = first_two()
    src = tmp_path / "p.jsonl"
    src.write_text("\n".join(json.dumps(r) for r in [
        {"qid": q1["query_id"], "vid": asset["asset_id"], "pred_relevant_windows": [[1, 3, 0.2]]},
        {"qid": q1["query_id"], "vid": "elephants-dream", "pred_relevant_windows": [[5, 9, 0.9]]},
    ]))
    data, stats = cv.convert(cv.read_windows(src, "qvh"), QUERIES, MANIFEST, None)
    first = data["predictions"][0]["moments"]
    assert [m["asset_id"] for m in first] == ["elephants-dream", asset["asset_id"]]
    assert data["predictions"][1]["moments"] == []
    assert stats["missing"] == len(data["predictions"]) - 1


def test_clipping_low_scores_and_unknown_assets(tmp_path):
    cv = converter()
    (q1, q2), asset = first_two()
    end = float(asset["duration_s"])
    src = tmp_path / "p.csv"
    src.write_text("query_id,asset_id,t_in,t_out,score\n"
                   f"{q1['query_id']},{asset['asset_id']},{end - 2},{end + 5},0.8\n"
                   f"{q1['query_id']},no-such-video,1,2,0.7\n"
                   f"{q1['query_id']},{asset['asset_id']},{end + 1},{end + 3},0.6\n"
                   f"{q2['query_id']},{asset['asset_id']},1,2,0.1\n")
    data, stats = cv.convert(cv.read_windows(src, "csv"), QUERIES, MANIFEST, 0.5)
    first, second = data["predictions"][0]["moments"], data["predictions"][1]["moments"]
    assert first == [{"asset_id": asset["asset_id"], "t_in": round(end - 2, 3),
                      "t_out": round(end, 3)}]
    assert second == []  # best score 0.1 is below --min-score 0.5: abstain
    assert stats["unknown_asset"] == 1 and stats["dropped"] == 1
    assert stats["clipped"] == 2 and stats["abstained_low_score"] == 1
    # The scorer accepts every ranking the converter writes.
    out = tmp_path / "preds.json"
    out.write_text(json.dumps({"system": "t", **data}))
    run = score_predictions(out, QUERIES, MANIFEST)
    assert not any(t.outcome == "invalid_submission" for t in run.trials)
