# Evaluation protocol (v0.2)

This page specifies what a MomentBench number means. A result that departs from it should say how.
For what a v0.2 number does not cover (audio, archive scale, physical prediction, unseen footage) and the plan for each gap, see [`scope.md`](scope.md).

## Task

Given a text request and a corpus of videos, return the video and the start and end time, in seconds, of the one moment that matches. Some requests match nothing in the corpus. For those, the correct response is to return no moment.

The system sees the whole corpus for every query. It is never told which video to search.

## Corpus

Every asset is listed in [`data/corpus_manifests/sample_manifest.json`](../data/corpus_manifests/sample_manifest.json) with its licence, source URL, duration and, for downloadable assets, a `file` block giving the exact file and its sha256. `momentbench corpus-download` fetches those files and `momentbench corpus-verify` checks presence, hash and duration. Run `corpus-verify` before any reported run. A run against different bytes is a different benchmark.

## Queries

| Tier | What it tests | How gold is defined |
|---|---|---|
| 1 | One shot, described visually | One scene-cut-bounded shot |
| 2 | Dialogue or narration together with the picture | One shot whose transcript lines and visuals both match |
| 3 | No-answer near miss: a real shot with one concrete detail changed | No gold. Only an explicit empty answer is correct |
| 4 | An ordered event across two or three consecutive shots | The span from the first shot's start to the last shot's end |

Each query file is checked in CI (`tests/test_query_validation.py`): unique ids, the canary string, a `metadata.source` saying who annotated it and how, known assets, and spans inside the video.

**Provenance.** The v0.2 queries were drafted and verified by a model and are marked `pending human review` in `metadata.source`. See [Query construction](#query-construction) for the method and its measured error rate. Report results on them as model-annotated.

**Splits.** `public` queries ship with their gold. `held_out` queries are kept by the maintainers and scored on request, so a score on them cannot come from training on the answers. `rolling` queries use footage published after the training cutoff of the current frontier models (after 2026-06-30 for v0.2). A large gap between a system's public and rolling scores suggests contamination.

## Agentic track

The model cannot watch video. It gets five tools, the same for every entrant:

| Tool | Returns |
|---|---|
| `list_assets` | Every asset: id, title, duration, whether it has a transcript |
| `search_transcript` | Keyword-overlap search over transcripts. No embeddings, so no entrant depends on a vendor's index |
| `read_transcript` | Transcript lines between two times, at most 600 s per call |
| `sample_frames` | Up to 8 evenly spaced JPEG frames, 512 px wide, between two times |
| `submit_answer` | Ends the episode with a ranked list of moments, or an empty list |

Transcripts are the files in [`data/transcripts`](../data/transcripts), machine transcribed once, so speech recognition is held constant across entrants.

**Visual-only ablation:** `momentbench agentic ... --no-transcripts` removes every transcript, so the model can only look. Runs record `modalities` (`frames+transcripts` or `frames`) and the results table marks frames-only rows. Use it to separate visual grounding from finding the moment through speech, which is what world-model and physical-AI evaluations usually need.

**Title control:** `--anonymize-titles` hides every title and replaces asset ids with opaque aliases (`video-01`, ...), ordered by hash so the list order says nothing either. Answers are mapped back to the real ids before scoring. Without it, a model can choose the right video from a name like `nasa-viper-slope-lab-testing` before looking at a frame. Runs record `anonymize_titles`, it is part of the compatibility key, and the results table marks those rows. The Inspect task takes the same two controls as `-T no_transcripts=true` and `-T anonymize_titles=true`.

**Budget per query:** 30 turns, 64 frames, 60 tool calls. The harness enforces it and tells the model the limits in the task text. A tool call past the limit returns an error, and the model must then submit.

**System prompt:** fixed, in `momentbench/agentic/episode.py` (`SYSTEM_PROMPT`). It states the scoring rule and that some requests have no match.

**Failures count.** A refusal, a crash, hitting the turn limit or ending without `submit_answer` scores zero and stays in the denominator. A no-answer query is a hit only on an explicit empty submission, so a run that fails cannot score as a correct abstention.

**No silent fallback.** If a provider refuses or routes to another model, the trial ends. A turn answered by a different model is never scored under the entrant's name. Each trial records the model identity the provider reported on every turn (`turn_models`). The model ID you pass is an alias; within a run it must resolve to one snapshot. A second identity on any turn ends that trial with outcome `model_changed` (scored zero), and `momentbench submit` refuses a run that reports more than one identity. To compare two snapshots, run each as its own entrant. `scripts/merge_runs.py` refuses to merge runs with different snapshots, providers or endpoints, and records each input's provenance in `identity.merged_inputs`.

## Metrics

- **Hit at 1:** the first submitted moment is in a gold video and its temporal IoU with a gold span is at least 0.5. For a no-answer query, an empty submission.
- **Recall@1:** the mean over queries of the per-query hit rate across trials. Also reported at tIoU 0.3 and 0.7.
- **Recall@5, mean tIoU:** secondary.
- **Per tier:** Recall@1 for each tier with its own interval. Report it. An average over tiers hides that tier 3 measures abstention and tier 1 measures localisation.
- **Interval:** 95% percentile bootstrap, resampling queries (2,000 resamples, seed 0), so the interval reflects how many queries there are as well as trial noise.
- **Cost:** tokens per trial priced from [`configs/pricing.json`](../configs/pricing.json), which carries an `as_of` date and its sources. Report Recall@1 together with cost per query.

**Trials.** Run each query at least once. Five trials per query is the default for small sets. With 300 queries the query-resampled interval dominates, and one to three trials is enough. State the number.

**Measured run-to-run stability.** Two independent single-trial Claude Opus 5.5 runs on the 185 public film queries scored 54.1% and 56.2% Recall@1, inside each other's interval. Per query they agreed 78% of the time (Cohen's kappa 0.56): 82 queries were hits in both runs and 40 in only one. A single trial is enough for an aggregate over this many queries. Per-query or per-tier comparisons on small slices need several trials.

**What to report:** model id and snapshot, provider, harness (`api` or `claude-code`), effort setting, budget, trials, query file and its commit, Recall@1 with interval overall and per tier, cost per query and the pricing `as_of` date. The run JSON written by `momentbench agentic` holds all of it plus every trial's tool trace.

## Query construction

`scripts/draft_queries.py` builds model-annotated queries from footage:

1. Shots come from ffmpeg scene-cut detection (threshold 0.3), so gold boundaries are cuts rather than guesses.
2. A drafting call sees a labelled sheet of about 16 shots, plus the transcript for that window, and writes queries of each tier pointing at shot ids.
3. A separate verification call sees the gold candidate shuffled among the three corpus shots that look most like it (by thumbnail distance) and must pick the match, or "none". A positive survives only if the verifier picks its gold and reports no wrong detail. A no-answer query survives only if its altered text matches nothing while its unaltered original picks out the source shot cleanly.
4. The three look-alikes are stored as `hard_negatives`.

**Measured quality:** a random 20-query spot check against frames found 2 wrong queries (removed) and 1 gold span that starts with about 1.8 s of a missed soft cut. Expect roughly 10% of unreviewed queries to be wrong.

**Cross-check:** every kept query was then shown to a second model family (`gemini-3.5-flash`, `scripts/cross_verify.py`) against the same candidates, reshuffled. Only queries both models agree on are kept; 20 of 287 film queries were removed this way. The result is in each query's `metadata.cross_check`, with `annotation_sha`: a hash of the text, tier, gold spans and (for a no-answer query) its source shot, as the checker saw them (`momentbench/annotation.py`). A later edit to any of those moves the check to `metadata.cross_check_history`, and the query waits for a new independent check; `cross_verify.py` re-checks exactly those. Model and human review records carry `annotation_sha` (what was reviewed) and `result_sha` (what the review left). In 0.2.2, 330 of 456 published queries have a cross-check of their current annotation and 126 are pending, because their span or wording was refined after the check.

**Human review:** `scripts/review_queries.py build` writes a static page that shows each query with five frames (one second before the span, its start, middle and end, one second after). A reviewer marks accept, fix or reject, and `review_queries.py apply` records the decision in `metadata.human_review`, removing rejected queries. Accepted queries say `human-reviewed by <reviewer>` in `metadata.source`.

**Miss audit:** after the first Claude run on the film set, 8 randomly chosen tier 1 misses were compared against frames. 6 were genuine model errors. 1 query was not unique (another shot in the same film fits it) and 1 gold span contained a missed cut. Both are flagged in `metadata.review_flag` for human review rather than edited, because correcting only the queries a model missed would bias the set toward that model. 6 random tier 3 misses were also checked: in all 6 the model submitted the altered source shot, and in all 6 the frames confirm the changed detail is false (a green coat described as red, a pink sign as blue). Tier 3 misses are model errors of detail, which is what the tier is meant to measure.

**Known bias:** the drafter and verifier are Claude. Queries may suit Claude's way of describing images. Human review, or cross-model verification, is the remedy. Until then, compare Claude to other entrants with that caveat.

**Captions:** the NASA Minute episodes burn their narration into the picture as captions, so some aerospace tier 2 queries can be answered by reading the frames alone. They are still correct, but on those assets tier 2 measures reading as much as listening.

**Aerospace spot check:** 10 random aerospace queries were checked against frames before the cross-check. All 10 matched; one described a red-vested crew member as wearing a green flight suit.

**Robotics domain:** drafted and verified by Claude, then cross-checked by Gemini like the other domains (2 of 72 published robotics queries and 1 aerospace physical-event query were removed at that step). A spot check of 8 random robotics queries found all 8 accurate, but only 2 described a physical event: NASA's robotics videos are mostly public-outreach footage of people and control rooms. A dedicated pass (`draft_queries.py --physical-only`) therefore drafts only queries in which a robot, vehicle or mechanism visibly moves or makes contact, and tags them `metadata.physical_event`. Report that subset separately when the question is physical understanding. That pass added 64 queries (22 robotics, 42 aerospace: launches, parachute descents, splashdowns, rovers on test beds). A spot check of 6 found all 6 about visible machine motion and 5 accurate; the sixth gold span runs across undetected cuts.

**Pre-review and refinement:** a model pre-review of all published queries (`scripts/prereview_queries.py`) flagged 136 of 460 for a human to look at first, mostly gold spans that include part of a neighbouring shot. For queries it marked fix, `scripts/refine_boundaries.py` had Claude pick the moment's start and end from frames sampled across the span (or, when only a detail was wrong, make the smallest text edit), and applied a change only when an independent pre-review call accepted it. 67 queries were changed this way (28 film, 27 aerospace, 12 robotics), each recording its original span and text in `metadata.boundary_refined` and `metadata.text_refined`; 10 of 10 sampled changes looked right against frames, 3 of them with loose ends. Saved runs are rescored against the corrected gold, so no rerun was needed. Refined spans are model-chosen, not detected cuts.

**Known limitation:** scene detection misses some dissolves and dark cuts, mostly in Elephants Dream, so a few gold spans include a second or two of the neighbouring shot.
