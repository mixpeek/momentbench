"""Rank transcript cues by cosine similarity to the query using MiniLM on CPU.

    pip install -e ".[dense]"
    python scripts/dense_predictions.py -q data/queries/scored/film_v0.2_scored.json \
        -o dense_film.json
    momentbench score-predictions -p dense_film.json -q data/queries/scored/film_v0.2_scored.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from momentbench.agentic.text_controls import TOP_K, load_transcripts  # noqa: E402
from momentbench.runner import load_corpus_assets, load_queries  # noqa: E402

MODEL = "sentence-transformers/all-MiniLM-L6-v2"
REVISION = "1110a243fdf4706b3f48f1d95db1a4f5529b4d41"


def predictions(queries_path: Path, manifest: Path, transcripts_dir: Path, model) -> dict:
    durations = {a["asset_id"]: a["duration_s"] for a in load_corpus_assets(manifest)}
    transcripts = load_transcripts(sorted(durations), transcripts_dir)
    texts, moments = [], []
    for asset_id, cues in transcripts.items():
        for cue in cues:
            start = max(0, cue["t_in"])
            end = min(durations[asset_id], cue["t_out"])
            if start < end and cue["text"].strip():
                texts.append(cue["text"])
                moments.append({"asset_id": asset_id, "t_in": start, "t_out": end})

    queries = load_queries(queries_path)
    rows = [{"query_id": q.query_id, "moments": []} for q in queries]
    if texts and queries:
        cues = model.encode(texts, normalize_embeddings=True, convert_to_numpy=True)
        query_vectors = model.encode([q.text for q in queries], normalize_embeddings=True,
                                     convert_to_numpy=True)
        for row, vector in zip(rows, query_vectors):
            # Stable ties follow asset ID and then cue order within the transcript.
            top = np.argsort(-(cues @ vector), kind="stable")[:TOP_K]
            row["moments"] = [moments[i] for i in top]
    return {"system": "dense-transcript-all-MiniLM-L6-v2", "predictions": rows}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("-q", "--queries", type=Path, required=True)
    ap.add_argument("-c", "--corpus", type=Path,
                    default=ROOT / "data/corpus_manifests/sample_manifest.json")
    ap.add_argument("--transcripts-dir", type=Path, default=ROOT / "data/transcripts")
    ap.add_argument("-o", "--output", type=Path, required=True)
    args = ap.parse_args()
    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer(MODEL, revision=REVISION, device="cpu")
    out = predictions(args.queries, args.corpus, args.transcripts_dir, model)
    args.output.write_text(json.dumps(out, indent=1) + "\n")
    print(f"wrote {args.output}: {len(out['predictions'])} queries")


if __name__ == "__main__":
    main()
