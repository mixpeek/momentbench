# Evaluating a video retrieval system with a prediction file

If your system already ranks moments (a text-to-video index, a moment-retrieval model
such as Moment-DETR, a hybrid of transcript and visual search), you can score it on
MomentBench without an agent, an API key or a GPU on our side. You write one JSON file
of ranked moments per query; `momentbench score-predictions` scores it with the same
rules as every other row on the leaderboard.

## What a query asks

A query describes one moment somewhere in a collection of 27 videos, for example a
specific action in a NASA rover test. Your system must return the right video and the
right few seconds. 49 of the 333 scored queries have no match in the collection, and
the correct answer to them is an empty list. A hit at rank 1 needs the right video and a
temporal IoU of at least 0.5 with the labelled span.

## The file

```json
{"system": "my-retriever-v2",
 "predictions": [
   {"query_id": "q-robotics-0101", "moments": [
     {"asset_id": "nasa-perseverance-twin-rover", "t_in": 34.1, "t_out": 39.0},
     {"asset_id": "nasa-minute-2026-05-08", "t_in": 34.0, "t_out": 39.0}]},
   {"query_id": "q-robotics-0102", "moments": []}]}
```

- Up to five moments per query, best first; times in seconds from the start of the video.
- `asset_id` values are the ids in `data/corpus_manifests/sample_manifest.json`.
- An empty list abstains ("nothing in the collection matches").
- An invalid entry (an unknown video, `t_in >= t_out`, a time past the end) rejects that
  query's whole ranking, and a missing query scores zero.

## Converting output you already have

`scripts/convert_predictions.py` reads two common shapes and writes a valid file:

```bash
# QVHighlights / Moment-DETR style JSON lines: one line per (query, video)
# {"qid": "q-robotics-0101", "vid": "nasa-perseverance-twin-rover",
#  "pred_relevant_windows": [[34.1, 39.0, 0.82], [12.0, 15.5, 0.41]]}
python scripts/convert_predictions.py --format qvh -i preds.jsonl \
  -q data/queries/scored/robotics_v0.2_scored.json -o preds.json --min-score 0.3

# CSV with a header: query_id,asset_id,t_in,t_out[,score]
python scripts/convert_predictions.py --format csv -i preds.csv \
  -q data/queries/scored/robotics_v0.2_scored.json -o preds.json
```

Windows from all videos of a query are ranked together by score, clipped to the video's
length, and cut to five. `--min-score` is your abstention threshold: a query whose best
window scores below it gets an empty list. The converter prints how many windows it
clipped or dropped and how many queries had no output, so a mapping mistake shows up
before you score.

## Scoring, and what you need for it

Scoring a file needs no video: only the query files and the manifest in this repository.

```bash
momentbench score-predictions -p preds.json -q data/queries/scored/robotics_v0.2_scored.json
momentbench bundle -r <the run file it wrote>     # JSON, model-card block, LaTeX, citation
```

Producing predictions does need the media your system indexes:
`momentbench corpus-download -m data/corpus_manifests/sample_manifest.json --corpus-dir corpus/`
(2.6 GB), plus the transcripts in `data/transcripts/` if your system reads text.

To try the whole path before wiring in your own system, generate the BM25 transcript
baseline, which is a complete prediction file built from the shipped transcripts:

```bash
python scripts/bm25_predictions.py -q data/queries/scored/robotics_v0.2_scored.json -o bm25.json
momentbench score-predictions -p bm25.json -q data/queries/scored/robotics_v0.2_scored.json
```

It scores 10.2% Recall@1 on robotics, the row on the leaderboard, which never looks at a
frame. The never-answer floor (an empty list everywhere) scores the share of no-answer
queries.

## Reading the result

- **Positive-query Recall@1** is the localization number the leaderboard leads with.
- **No-answer accuracy** and **net score** (+1 correct, -1 wrong moment, 0 abstain) show
  whether your threshold abstains where it should. They are exploratory: the no-answer
  labels were screened across the collection by model, not by people
  ([`labels.md`](labels.md)).
- Rows are compared only within a compatibility group. Prediction files form their own
  group per domain, so your row sits beside the BM25 baseline and the never-answer
  floor, never beside an agentic model's.

## Putting it on the leaderboard

```bash
momentbench submit -r <run file> --model-name "Your system" --contact you@your-lab.org -o submissions/
```

Open a pull request with the new `submissions/<name>/` folder. A check validates the
run, merges it and publishes it to [momentbench.com](https://momentbench.com) without a
maintainer. The contact address lets the result be attributed to your team.
