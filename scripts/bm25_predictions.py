"""Write the BM25 transcript baseline as a prediction file (the prediction-file track).

    python scripts/bm25_predictions.py -q data/queries/scored/film_v0.2_scored.json \
        -o bm25_film.json
    momentbench score-predictions -p bm25_film.json \
        -q data/queries/scored/film_v0.2_scored.json

BM25 (k1 1.2, b 0.75) over the shipped transcript cues of every corpus video, top five
cues as ranked moments; it never looks at a frame. This is the generator behind the
`submissions/bm25-transcript-*` rows, so those rows reproduce from these two commands.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from momentbench.agentic.text_controls import BM25Index, load_transcripts  # noqa: E402
from momentbench.runner import load_corpus_assets, load_queries  # noqa: E402


def predictions(queries_path: Path, manifest: Path, transcripts_dir: Path) -> dict:
    assets = [a["asset_id"] for a in load_corpus_assets(manifest)]
    index = BM25Index(load_transcripts(assets, transcripts_dir))
    return {"system": "bm25-transcript",
            "predictions": [{"query_id": q.query_id,
                             "moments": [m.model_dump(exclude={"metadata"})
                                         for m in index.search(q.text)]}
                            for q in load_queries(queries_path)]}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("-q", "--queries", type=Path, required=True)
    ap.add_argument("-c", "--corpus", type=Path,
                    default=ROOT / "data/corpus_manifests/sample_manifest.json")
    ap.add_argument("--transcripts-dir", type=Path, default=ROOT / "data/transcripts")
    ap.add_argument("-o", "--output", type=Path, required=True)
    args = ap.parse_args()
    out = predictions(args.queries, args.corpus, args.transcripts_dir)
    args.output.write_text(json.dumps(out, indent=1) + "\n")
    print(f"wrote {args.output}: {len(out['predictions'])} queries")


if __name__ == "__main__":
    main()
