# Scope and roadmap (v0.2)

What a MomentBench v0.2 number measures, what it does not, and the work each open gap
needs. It answers two lab readiness audits of 2026-10-13: the first (30 findings,
against 16e16bc) and a second onboarding pass (R1 to R20, against 8ac60e9). The last
two sections give the status of every finding.

## What v0.2 measures

Given a text request and a corpus of 27 videos (about 117 minutes), a system returns
the video and the start and end time of the matching moment. In the agentic track a
model does this through tools under a fixed budget (30 turns, 64 frames, 60 tool
calls): it lists the videos, searches and reads machine transcripts, and samples still
frames. A hit is the right video with temporal IoU of at least 0.5.

That makes a v0.2 score a measure of:

- **Localization** of a described moment in a small, fixed corpus, from sampled frames
  and a speech transcript (positive Recall@1, reported on its own).
- **Abstention**: saying nothing matches when nothing does (no-answer accuracy and
  false-positive rate, reported on their own). Tier 3 is entirely this.
- **Search under a budget**: how a model spends turns and frames, with cost per query
  reported next to every score.

## What it does not measure

- **Joint visual-language reasoning on every tier-2 query.** `baseline-transcript`
  answers from keyword transcript search alone and localizes 23 of 34 aerospace tier-2
  queries (4 of 32 film, 3 of 10 robotics). Those queries can be solved from speech.
  docs/results.md splits each model's tier-2 score by whether the control solved the
  query; read tier 2 through that split.

- **Archive-scale retrieval.** 27 assets, mostly short, from two source families
  (Blender open movies and NASA outreach). Query density is uneven: per video, positive
  queries range from 6 to 47 in film and 1 to 28 in robotics. There is no distractor
  scaling, long-form recording or source-disjoint evaluation yet.
- **Audio or native video understanding.** Models get still frames and a fixed
  Whisper transcript. There are no audio windows, sound events, speaker identity from
  audio, or video clips. Fast motion and contact can fall between sampled frames.
- **Physical intelligence or world modelling.** 53 queries (18 robotics, 35 aerospace)
  carry the `physical_event` tag. They test whether a model can find a moment where a
  machine moves or makes contact. Nothing in v0.2 asks a model to predict dynamics,
  forces, trajectories, state changes or counterfactual outcomes, and there is no
  manipulation, driving or egocentric footage.
- **Generalization to unseen footage.** Held-out queries are assigned by hashing the
  query id, so they concern the same videos as public ones. The held-out set therefore
  tests memorization of published answers on footage a model has already been scored on.
- **Freedom from contamination.** The 38 rolling queries come from three NASA Minute
  episodes published after 2026-06-30 and are not matched in difficulty to the rest.
  Runs do not certify a model's training cutoff, and canaries only support cooperative
  filtering.
- **Model differences free of harness effects.** Claude runs use the Claude Code SDK
  loop and other models the API loop; turn semantics and image transport differ. Each
  row names its harness.
- **Human-verified labels.** All 457 published queries were drafted and verified by
  Claude and cross-checked by Gemini. None has a human review record yet. Verification
  compared each candidate with three look-alike shots, which does not prove a positive
  is unique or a no-answer query has no match anywhere in the corpus.
- **Event boundaries.** Gold spans mostly follow shot cuts, while an event can fill
  part of a shot. The median positive span is 4 seconds, so small timing differences
  matter; 67 spans were refined by a model and 34 queries reworded.

## Roadmap

Each item names the work and what it needs. "Maintainers" means it can be done in
this repository without outside input.

| Gap | Work | Needs |
|---|---|---|
| Human labels (11, 12) | Independent reviewers watch every published clip, adjudicate alternative matches and no-answer queries across the whole corpus, record agreement, and estimate the error rate on a stratified random sample. Freeze reviewed labels as a release before comparing models. | Human reviewers and their time |
| Event boundaries (13) | State per query whether the target is a shot or an event; annotate boundary uncertainty and acceptable alternatives; report boundary error next to IoU. | Human reviewers, then maintainers |
| Corpus scale (18) | Fixed distractor-scaling runs, longer recordings, more source families, balanced slices; report score against corpus hours, candidate count and event duration. | Rights-cleared footage, then maintainers |
| Audio and native video (15) | Separate, separately budgeted tracks with audio windows and video clips; report pixels, video seconds and audio seconds consumed. | Maintainers |
| Physical intelligence (19) | Event and state annotations, unseen environments, continuous interactions, predictive tasks with task-specific scoring; manipulation and driving footage. | Openly licensed robot and driving footage, licence review, annotators |
| Source-disjoint held-out (20) | Group the split by video and event; publish a held-out evaluation route with rules, turnaround and auditable aggregate records. | Maintainers, plus an evaluation service for entrant code |
| Contamination evidence (21) | Model-specific cutoff evidence, publisher release dates, temporarily sealed queries, rolling queries matched in source and difficulty. | Maintainers and new footage |
| Harness effects and coverage (23) | Same-model crossover between the Claude Code and API loops; GPT-5.5 on v0.2; Gemini on aerospace; open-weight runs on all domains; transcript, visual-segment and hybrid baselines; several budgets. | API budget for the frontier runs; maintainers for the rest |
| Modality controls (16, R12, R15) | Transcript-only control is built in (`baseline-transcript`). Still to add: metadata-only, masked captions, shuffled or reversed frames, mismatched speech and picture; new tier-2 queries kept only when the transcript control fails them. | Maintainers and API budget |
| Reruns under the new contract (R10) | Rerun every record row on the full 0.2.1 release with receipts, identity and resolved model; keep the current rows as historical. The Claude reruns use the local Claude Code subscription; the Gemini rerun needs API budget, and GPT-5.5 needs an OpenAI account. | Owner decision on API spend |
| Indexed retrieval reference (R11) | One end-to-end indexed baseline with deterministic segments, an id map, an index receipt and measured build and query cost, wrapped as an agent. | Maintainers |
| Untrusted entrant isolation (R4) | Custom agents now get a metered, closed corpus interface in-process. Code from outside entrants, especially on held-out queries, needs a separate process or service. | Maintainers |
| Mixpeek workflow (8) | Expose server build, feature URI and retriever id on the CLI; persist ingestion state; map robotics; check collection readiness. | Maintainers |
| Vendor index identity (9) | Ingestion tools that write benchmark ids into vendor metadata, id maps, index receipts and a round-trip check for TwelveLabs and Mixpeek. | Maintainers, vendor accounts |
| Segment-level embedding baselines (3) | Timestamped frame or clip embeddings, segment ranking and a calibrated abstention threshold. Today these adapters are labelled whole-video controls. | Maintainers, GPU time |
| Pinned environments (25) | Lock file, pinned Docker base, tested vLLM serving recipes (model revision, tool parser, dtype, hardware). | Maintainers |
| Long-run controls (26) | Checkpoint and resume, per-query deadline, token and dollar caps, frame cache, resumable downloads. | Maintainers |
| Replay artifacts (27) | Keep returned transcript text, frame hashes and provider error bodies in traces; report indexing cost and latency percentiles for retrieval systems. | Maintainers |
| Licence clearance (29) | Confirm Spring and Agent 327 with Blender Studio; record clearance per component (Tears of Steel's soundtrack is CC BY-ND). | Blender Studio's answer |

## Status of the audit findings

Fixed means the change is on main with a test that reproduces the audit's probe.

| # | Finding | Status | Where |
|---|---|---|---|
| 1 | Site shows placeholder rankings | Fixed: the site is generated from `docs/results.json` | cc0d864 |
| 2 | Legacy VLM adapters see no media | Fixed: each adapter declares `input_kind`, the CLI warns, README routes to the agentic track | f70bbd7 |
| 3 | Embedding baselines cannot localize | Labelled whole-video controls; segment baselines open | f70bbd7, roadmap |
| 4 | Legacy evaluator drops and duplicates | Fixed | d624330 |
| 5 | Invalid answers earn abstention credit | Fixed | d624330 |
| 6 | Incomplete corpus still reportable | Fixed: full verification with a receipt; `--partial` runs are not reportable | d624330 |
| 7 | Custom agents see gold and self-report usage | Fixed for the harness (stripped request, metered corpus); an isolated service for held-out entrant code is open | d624330, roadmap |
| 8 | Mixpeek workflow incomplete | Open | roadmap |
| 9 | Vendor index identity | Open | roadmap |
| 10 | Build gate compares shas as strings | Fixed: git ancestry or commit time | 94ddb76 |
| 11 | No human annotation | Open | roadmap |
| 12 | Uniqueness and absence unproven | Open | roadmap |
| 13 | Shot versus event boundaries | Open | roadmap |
| 14 | Rescoring reuses answers to old wording | Fixed: per-trial text hash, reworded queries excluded | 19fa5f1 |
| 15 | No audio or native video | Open; scope stated above | roadmap |
| 16 | Ablations do not isolate modalities | Partly: anonymized titles and Inspect parity added; other controls open | 4023db6, roadmap |
| 17 | Tiers mix abstention and localization | Fixed: split metrics in every summary and results table | fc906af |
| 18 | Corpus too small for archive claims | Open; scope stated above | roadmap |
| 19 | Physical events are thin evidence | Open; scope stated above | roadmap |
| 20 | Held-out is not source-disjoint | Open; scope stated above | roadmap |
| 21 | Rolling does not prove no contamination | Open; scope stated above | roadmap |
| 22 | Query bootstrap alone | Fixed: per-video clustered and paired intervals | fc906af |
| 23 | Model and harness effects mixed | Open | roadmap |
| 24 | Runs lack identity | Fixed: identity block and compatibility key | 19fa5f1 |
| 25 | Environment not pinned; `--effort` ignored | Partly: `--effort` forwarded or rejected; pinning open | 94ddb76, roadmap |
| 26 | Weak execution controls | Partly: end-of-video frames, ffmpeg off the event loop, download timeout and retries; checkpoints and caps open | 97eb562, roadmap |
| 27 | Usage and traces lost on error | Partly: usage and trace kept through errors; richer replay artifacts open | d624330, roadmap |
| 28 | Submit and leaderboard ignore the agentic track | Fixed | 8bfc896 |
| 29 | Licence exceptions | Partly: `license_status` per asset, pending films not sellable; confirmation open | 94ddb76, roadmap |
| 30 | Release drift | Fixed: consistency tests over card, README, SUBMISSION and report; lossless manifest | 94ddb76 |

## Status of the second audit (R1 to R20)

| # | Finding | Status | Where |
|---|---|---|---|
| R1 | Install path does not deliver a runnable benchmark | Fixed: README, SUBMISSION and the integration guide clone a release tag, name ffmpeg, and say the package is not on PyPI | 0.2.1 |
| R2 | Tagged release is the old code | Fixed: v0.2.1 tags the hardened harness; v0.2.0 is left as it was | 0.2.1 |
| R3 | Verification and frames can read different files | Fixed: one resolver; frames are pinned to the verified file | 359b333 |
| R4 | Custom agents can bypass frame metering | Fixed in process: closed, metered, serialised corpus interface; process isolation is on the roadmap | 359b333 |
| R5 | Inspect runs without verification | Fixed: Inspect uses the same preparation and refuses an unverified corpus | 359b333 |
| R6 | API runs reportable by default; submit accepts no receipt | Fixed: reportable only on a pinned, fully verified corpus; submit checks the receipt and recomputes the key | 359b333 |
| R7 | Duplicate or dropped trials change scores | Fixed: complete, unique trial grid required | 359b333 |
| R8 | Invalid first candidate promotes the second | Fixed: any invalid entry rejects the ranking | 359b333 |
| R9 | Keys not enforced; subset shares key | Fixed: key covers the selected queries; leaderboard ranks only within a key | 359b333 |
| R10 | Record runs predate the identity contract | Open: rows are labelled historical; reruns are on the roadmap | roadmap |
| R11 | Retrieval-system integration incomplete | Open | roadmap |
| R12 | Aerospace tier 2 solvable from transcripts | Measured and reported: `baseline-transcript` control and the tier-2 split in docs/results.md; new queries open | 0.2.1, roadmap |
| R13 | No human label review | Open | roadmap |
| R14 | Narrow generalization | Open; stated above | roadmap |
| R15 | No audio, continuous video or temporal controls | Open; stated above | roadmap |
| R16 | Physical location is not dynamics | Open; stated above | roadmap |
| R17 | Model comparisons incomplete and harness-confounded | Open | roadmap |
| R18 | Site overstated contamination evidence | Fixed: site and report describe the held-out comparison as limited evidence | 0.2.1 |
| R19 | Long-run operations and serving recipes | Open | roadmap |
| R20 | Reuse claims broader than the manifest | Fixed in README and ATTRIBUTION (exceptions named, per-asset fields); confirmation from Blender Studio still open | 0.2.1 |
