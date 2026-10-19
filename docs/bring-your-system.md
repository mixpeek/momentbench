# Bring your system

You have a video retrieval or temporal-grounding system and want to know how it does
on MomentBench: finding the right video and the right seconds across a collection, or
saying that nothing matches. This page gets you from zero to a validated, citable
result.

## 1. Install and check your setup (about 10 minutes)

Clone `main`: it carries the v0.2.5 queries, manifests, transcripts and scorer unchanged,
plus the prediction-file tools added after the tag (`scripts/bm25_predictions.py`,
`scripts/convert_predictions.py`). Results still cite v0.2.5.

```bash
git clone https://github.com/mixpeek/momentbench.git && cd momentbench
pip install -e .                      # Python 3.10+, plus ffmpeg on your PATH
momentbench corpus-download -m data/diagnostic/manifest.json --corpus-dir diag/
momentbench diagnose --corpus-dir diag/
```

`diagnose` downloads nothing else and calls no model. It checks ffmpeg, the three
profile videos, frame extraction, the two deterministic floors against stored results,
and the prediction scorer. Every line should read PASS.

## 2. Pick a track

| Your system | Track | You provide |
|---|---|---|
| Ranks moments itself (an index, a retriever, a grounding model) | **Prediction file** | One JSON file: up to five ranked `(asset_id, t_in, t_out)` per query, or `[]` when nothing matches |
| Is a model that should search with tools (list videos, read transcripts, look at frames) | **Agentic** | A model endpoint; the harness runs the loop under a fixed budget |

The prediction-file format, and a converter for QVHighlights-style or CSV output, are in
[`prediction-files.md`](prediction-files.md). The agentic
track is in `docs/protocol.md`, and `docs/input-spec.md` lists exactly what a model
sees.

## 3. Run on the scored set

The scored set is the 333 queries two model families other than Claude accepted
(`docs/labels.md`), split by domain in `data/queries/scored/`.

What you need depends on the step:

- **Producing predictions** needs the media your system indexes: the full corpus
  (`momentbench corpus-download -m data/corpus_manifests/sample_manifest.json
  --corpus-dir corpus/`, 2.6 GB), and the transcripts in `data/transcripts/` if your
  system reads text.
- **Scoring predictions you already have** needs no media and no credentials: only the
  query files and the corpus manifest in this repository. `scripts/bm25_predictions.py`
  writes a complete prediction file from the shipped transcripts, so you can try the
  whole path before plugging in your own system.

Then:

```bash
# prediction-file track
momentbench score-predictions -p my_predictions.json -q data/queries/scored/film_v0.2_scored.json

# agentic track (an OpenAI-compatible endpoint, for example vLLM)
momentbench agentic my-model --provider openai-compatible --base-url http://localhost:8000/v1 \
  -q data/queries/scored/film_v0.2_scored.json -c data/corpus_manifests/sample_manifest.json \
  --corpus-dir corpus/ --anonymize-titles --trials 1
```

## 4. Get a result you can use

```bash
momentbench bundle -r data/results/<run>.json
```

This writes a `result.json`, a Markdown block for a model card or README, a LaTeX
table and a citation, every number taken from the validated run.

## 5. Submit (optional)

```bash
momentbench submit -r data/results/<run>.json --model-name "Your system" -o submissions/
```

Open a pull request with the new `submissions/<name>/` directory. A check runs on the
pull request and reports whether the run validates; a valid run joins the leaderboard
on the next rebuild.

## What the score means

Recall@1 counts a hit when the first answer is in the right video and overlaps the
true moment by tIoU 0.5 or more; on a no-answer query only an empty answer counts.
Net score adds +1 per correct answer, -1 per wrong moment, 0 per abstention. Labels
are model-annotated and model-adjudicated, with no human review yet; see
`docs/labels.md` for the method and its limits, and open an issue if a label is wrong.
