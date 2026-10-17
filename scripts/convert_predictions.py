"""Convert a retrieval system's output into a MomentBench prediction file.

    python scripts/convert_predictions.py --format qvh -i preds.jsonl \
        -q data/queries/scored/film_v0.2_scored.json -o preds.json [--min-score 0.3]
    python scripts/convert_predictions.py --format csv -i preds.csv \
        -q data/queries/scored/film_v0.2_scored.json -o preds.json

Formats:

- `qvh`: JSON lines in the QVHighlights / Moment-DETR style, one line per (query,
  video): `{"qid": "<query_id>", "vid": "<asset_id>", "pred_relevant_windows":
  [[t_in, t_out, score], ...]}`. Windows from every video of a query are ranked together
  by score.
- `csv`: a header row with `query_id,asset_id,t_in,t_out` and an optional `score`
  column; rows without a score keep their file order.

The top five moments per query are kept. Windows are clipped to the video's duration
from the manifest, and a window that is empty after clipping is dropped, because the
scorer rejects a query's whole ranking if any entry is invalid. `--min-score` abstains
(an empty list) on a query whose best window scores below it, which is how a retriever
says "nothing in the corpus matches". A query the input never mentions gets an empty
list too, and the converter reports how many.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from momentbench.runner import load_corpus_assets, load_queries  # noqa: E402


def read_windows(path: Path, fmt: str) -> dict[str, list[tuple[float, str, float, float]]]:
    """query_id -> [(score, asset_id, t_in, t_out)] in input order."""
    out: dict[str, list[tuple[float, str, float, float]]] = defaultdict(list)
    if fmt == "qvh":
        for line in path.read_text().splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            for w in row.get("pred_relevant_windows") or []:
                score = float(w[2]) if len(w) > 2 else 0.0
                out[str(row["qid"])].append((score, str(row["vid"]), float(w[0]), float(w[1])))
    else:
        with path.open(newline="") as f:
            for i, row in enumerate(csv.DictReader(f)):
                score = float(row["score"]) if row.get("score") not in (None, "") else -i
                out[row["query_id"]].append((score, row["asset_id"], float(row["t_in"]),
                                             float(row["t_out"])))
    return out


def convert(windows: dict, queries_path: Path, manifest: Path,
            min_score: float | None) -> tuple[dict, dict[str, int]]:
    durations = {a["asset_id"]: float(a.get("duration_s") or "inf")
                 for a in load_corpus_assets(manifest)}
    stats = {"queries": 0, "abstained_low_score": 0, "missing": 0, "clipped": 0,
             "dropped": 0, "unknown_asset": 0}
    preds = []
    for q in load_queries(queries_path):
        stats["queries"] += 1
        ranked = sorted(windows.get(q.query_id, []), key=lambda w: -w[0])
        if not ranked:
            stats["missing"] += 1
        elif min_score is not None and ranked[0][0] < min_score:
            stats["abstained_low_score"] += 1
            ranked = []
        moments = []
        for _, asset, t_in, t_out in ranked:
            if asset not in durations:
                stats["unknown_asset"] += 1
                continue
            a, b = max(0.0, t_in), min(t_out, durations[asset])
            if (a, b) != (t_in, t_out):
                stats["clipped"] += 1
            if b <= a:
                stats["dropped"] += 1
                continue
            moments.append({"asset_id": asset, "t_in": round(a, 3), "t_out": round(b, 3)})
            if len(moments) == 5:
                break
        preds.append({"query_id": q.query_id, "moments": moments})
    return {"predictions": preds}, stats


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--format", choices=["qvh", "csv"], required=True)
    ap.add_argument("-i", "--input", type=Path, required=True)
    ap.add_argument("-q", "--queries", type=Path, required=True)
    ap.add_argument("-c", "--corpus", type=Path,
                    default=ROOT / "data/corpus_manifests/sample_manifest.json")
    ap.add_argument("--system", default=None, help="system name (default: input file stem)")
    ap.add_argument("--min-score", type=float, default=None,
                    help="abstain on a query whose best window scores below this")
    ap.add_argument("-o", "--output", type=Path, required=True)
    args = ap.parse_args()
    data, stats = convert(read_windows(args.input, args.format), args.queries, args.corpus,
                          args.min_score)
    data = {"system": args.system or args.input.stem, **data}
    args.output.write_text(json.dumps(data, indent=1) + "\n")
    print(f"wrote {args.output}: " + ", ".join(f"{k} {v}" for k, v in stats.items()))


if __name__ == "__main__":
    main()
