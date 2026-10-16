# Changelog

## 0.2.4 (2026-10-16)

Fixes from the fourth lab audit (docs/scope.md lists each). Same queries, corpus,
prompt, tools and budget as 0.2.3; the 0.2.3 record runs remain admissible.

- The Claude Code driver records the model of every assistant message and ends a
  trial on a change; the harness applies the same rule to entrants that run their own
  loop.
- The compatibility key includes `asset_view_sha`, the exact asset list the model
  sees. Runs without it are admitted only on the manifest they recorded.
- Review frames are cached by video bytes and timestamp, so a recheck after a span
  edit shows the new span. Pre-review and refinement verdict caches are bound to the
  annotation, and cross-checks record the media and reviewer configuration they used.
- No-answer queries store their unaltered description (`altered_from.original_text`);
  a cross-check without it is incomplete rather than agreed.

## 0.2.3 (2026-10-16)

Fixes from the third lab audit (docs/scope.md lists each). Same queries, corpus,
prompt, tools and budget as 0.2.2, so compatibility keys are unchanged and the 0.2.2
record runs remain valid.

- Run files are named from a filesystem-safe run ID, so model IDs with a slash
  (`Qwen/Qwen3-VL-32B-Instruct`) save, including partial and merged files.
- `examples/custom_agent.py` prepares and verifies the corpus like `momentbench
  agentic`, so its full runs are reportable; the default short run says it is a
  diagnostic.
- Each trial keeps the model identity reported on every turn. A change mid-trial ends
  it as `model_changed`; `momentbench submit` refuses a run with more than one
  identity; `scripts/merge_runs.py` refuses different snapshots, providers or
  endpoints and records each input's provenance.
- A BM25 transcript control joins the keyword control. The tier-2 split now reads
  "solved by a text-only control" against "not solved by either", and says that the
  second is no evidence a query needs the picture.
- Every query review records the hash of the annotation it saw (`annotation_sha`).
  126 queries edited after their Gemini cross-check moved it to
  `cross_check_history` and await a new independent check.
- The Claude record runs were rerun on the reworded queries and merged, so the 18
  record runs cover every public query (since 0.2.2).
- The dataset is on Hugging Face as `Mixpeek/momentbench`, with the 25 videos whose
  licences are confirmed.

## 0.2.2 (2026-10-14)

Query review. Same harness as 0.2.1. The maintainer delegated the review of the 457
published queries to Claude: each was checked against its frames, and the 69 that the
frame pre-review had flagged without a later fix were re-reviewed with extra frames, the
video and the transcript. 397 were accepted, 59 corrected (wording, span, or both) and 1
removed (q-film-1021, a no-answer query that nearly matches a real shot). Every query
records `metadata.model_review`; corrections carry `text_refined` or `boundary_refined`.
Human review is still pending, and the queries say so.

- 456 published queries (film 184, aerospace 202, robotics 70); 619 in total.
- Runs are rescored: corrected spans apply to saved answers, and reworded queries are
  left out of runs made before the rewording.
- `scripts/review_queries.py apply --kind model` records a model review without
  marking queries human-reviewed, and applies concrete text and span fixes.

## 0.2.1 (2026-10-13)

Same corpus and queries as 0.2.0; the harness, scoring and submission rules are
hardened after two lab-readiness audits. Cite 0.2.1 for any new result; 0.2.0 scores
some invalid answers and partial corpora that 0.2.1 refuses. docs/scope.md lists what
is fixed and what is still open.

Scoring

- Every expected query is scored once; missing predictions count as misses and
  duplicate or unknown ids are rejected.
- Only `{"moments": []}` abstains. Malformed output, unknown videos, out-of-range
  times, or an invalid entry anywhere in a ranked answer is rejected.
- Summaries report positive Recall@1, no-answer accuracy, false-positive rate and
  abstention on positives apart, with per-video clustered and paired intervals.
- A run is scored only on queries whose wording it saw (per-trial text hash).

Corpus and runs

- One preparation step for the CLI, the Python API and the Inspect task: every asset
  is hashed at the path its frames are read from and pinned there. A run is reportable
  only on a fully verified corpus; `--partial` debug runs are not.
- Runs record their identity: commit, selected queries, verified bytes, transcripts,
  prompt, tools, budget, conditions, provider, resolved model and a compatibility key.
- Custom agents get the request text only, and a closed corpus interface metered
  against the harness budget.
- Frames at a video's final second, ffmpeg off the event loop, download timeout and
  retries.

Controls and submission

- `baseline-transcript`, a transcript-only modality control, reported per domain with
  each model's tier-2 score split by whether speech alone solves the query.
- `--anonymize-titles`, and Inspect parity for both controls.
- `momentbench submit` takes agentic runs and requires a complete trial grid, a corpus
  receipt equal to the manifest and a matching compatibility key. `leaderboard` ranks
  only runs that share a key.
- Legacy adapters declare what they see (`input_kind`).

## 0.2.0 (2026-10-10)

Corpus and queries

- 27 hash-verified assets: 7 Blender open movies, 13 NASA aerospace videos and 7 NASA robotics videos (2019 to 2026). The two v0.1 assets with dead source links were removed.
- 620 queries across four tiers and three domains, drafted and verified by Claude and cross-checked by Gemini. 457 are published (`data/queries/film_v0.2.json`, `data/queries/aerospace_v0.2.json`, `data/queries/robotics_v0.2.json`); 163 are held out.
- Robotics domain: 7 NASA rover, free-flyer and helicopter videos, 70 public queries.
- 64 physical-event queries (`metadata.physical_event`), runnable alone with `--physical-only`.
- `rolling` split for footage published after 2026-06-30.
- Whisper transcripts for every asset with speech, in `data/transcripts`.

Harness

- Agentic track providers: Anthropic API, local Claude Code, OpenAI Responses, Gemini, any OpenAI-compatible Chat Completions server (vLLM, SGLang, Ollama), and `baseline` floors (random, abstain).
- Inspect AI task, defaulting to the three v0.2 query files.
- Visual-only ablation (`--no-transcripts`); runs record their `modalities`.
- Recall@1 at tIoU 0.3, 0.5 and 0.7, per-tier rows with bootstrap intervals, cost per query.
- A run stops when the provider refuses the account (billing, quota, auth) and keeps finished trials in a `.partial.json` file instead of scoring the failures.

Tooling and docs

- `docs/protocol.md` (what a result means), `docs/integrating.md` (how to evaluate your model), `docs/results.md` (generated from run files by `scripts/results_doc.py`).
- `scripts/draft_queries.py`, `scripts/cross_verify.py`, `scripts/review_queries.py` for building, cross-checking and human-reviewing queries.
- `examples/custom_agent.py`, a runnable template for your own agent.

## 0.1.0

Seed release: 9 manifest assets and 8 queries.
