import json
import math
import subprocess
import sys
from pathlib import Path


def test_abstain_below_applies_to_the_top_score(tmp_path):
    script = Path(__file__).resolve().parents[1] / "scripts/bm25_predictions.py"
    manifest, queries, output = (tmp_path / name for name in ("manifest", "queries", "output"))
    manifest.write_text(json.dumps([{"asset_id": "clip"}]))
    (tmp_path / "clip.transcript.json").write_text(json.dumps([
        {"t_in": 0, "t_out": 1, "text": "launch orbit"},
        {"t_in": 1, "t_out": 2, "text": "launch tower"},
    ]))
    queries.write_text(json.dumps([
        {"query_id": text, "text": text, "tier": 1, "vertical": "film"}
        for text in ("launch", "launch orbit", "unmatched")
    ]))

    def predict(*args):
        subprocess.run([sys.executable, str(script), "-q", str(queries), "-c", str(manifest),
                        "--transcripts-dir", str(tmp_path), "-o", str(output), *args], check=True)
        return [row["moments"] for row in json.loads(output.read_text())["predictions"]]

    weak, strong, missing = predict()
    assert len(weak) == len(strong) == 2
    assert not missing
    # Both cues contain "launch" once and have average length, so its score equals its IDF.
    launch_score = math.log(1 + 0.5 / 2.5)
    assert predict("--abstain-below", str(launch_score)) == [weak, strong, []]
    assert predict("--abstain-below", "0.5") == [[], strong, []]
