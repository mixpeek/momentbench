import json

import numpy as np

from momentbench.predictions import score_predictions
from scripts.dense_predictions import predictions


def test_dense_ranks_cues_across_videos_and_writes_valid_moments(tmp_path):
    class Encoder:
        def encode(self, texts, *, normalize_embeddings, convert_to_numpy):
            assert normalize_embeddings and convert_to_numpy
            vectors = {"launch": [1, 0], "rocket": [1, 0], "ocean": [0, 1]}
            return np.array([vectors[text] for text in texts])

    queries = tmp_path / "queries.json"
    queries.write_text(json.dumps([{"query_id": "q", "text": "launch", "tier": 1,
                                    "vertical": "film"}]))
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps([
        {"asset_id": aid, "duration_s": 6, "file": {"sha256": "a" * 64}} for aid in ("b", "a")
    ]))
    a = tmp_path / "a.transcript.json"
    a.write_text(json.dumps([{"t_in": i, "t_out": i + 1, "text": "ocean"} for i in range(6)]))
    b = tmp_path / "b.transcript.json"
    b.write_text(json.dumps([{"t_in": 5, "t_out": 8, "text": "rocket"},
                             {"t_in": 8, "t_out": 9, "text": "outside the video"}]))
    out = predictions(queries, manifest, tmp_path, Encoder())
    moments = out["predictions"][0]["moments"]
    assert moments[0] == {"asset_id": "b", "t_in": 5, "t_out": 6}
    assert [(m["asset_id"], m["t_in"]) for m in moments[1:]] == [("a", i) for i in range(4)]
    path = tmp_path / "predictions.json"
    path.write_text(json.dumps(out))
    assert score_predictions(path, queries, manifest).trials[0].outcome == "submitted"

    a.unlink()
    b.unlink()
    assert predictions(queries, manifest, tmp_path, Encoder())["predictions"] == [
        {"query_id": "q", "moments": []}
    ]
