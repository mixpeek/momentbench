# Input specification

Everything a model sees and how it is scored, at the level of detail needed to
reimplement the agentic track in another harness and get the same numbers. When an
internal reimplementation and this page disagree, the code is the reference: each
value below names the constant or function that defines it.

## What the model receives

**System prompt** (`episode.SYSTEM_PROMPT`, hashed into `identity.system_prompt_sha`):

> You are locating a specific moment in a corpus of videos. You cannot watch video directly. Use the tools to list the videos, search and read transcripts, and look at sampled frames, then call submit_answer with the video and the start and end time in seconds of the moment that matches the request. A submitted moment is correct when it is in the right video and overlaps the true moment with temporal IoU of at least 0.5, so make the span tight. Some requests have no matching moment anywhere in the corpus. For those, the correct answer is to submit an empty list. Frames and tool calls are limited by a budget. If the budget runs out, submit your best answer.

**First user turn** (`episode.build_task`):

```
Find this moment: <query text>

Budget for this task: <max_frames> frames, <max_tool_calls> tool calls, <max_steps> turns.
```

The model never sees the query's tier, gold spans, hard negatives or metadata
(`Query.request()`).

## Tools (`environment.TOOLS`, hashed into `identity.tools_sha`)

| Tool | Arguments | Returns |
|---|---|---|
| `list_assets` | none | JSON list of `{asset_id, title, duration_s, has_transcript}` in manifest order. With `--anonymize-titles`: no title, opaque `video-NN` ids ordered by hash |
| `search_transcript` | `query`, optional `asset_id`, `top_k` 1–20 (default 5) | Cues ranked by the number of distinct query words they share (lowercase `[a-z0-9']+` tokens); ties by asset id, then start time |
| `read_transcript` | `asset_id`, `t_start`, `t_end` | Cues overlapping the window; the window is capped at 600 s (`MAX_TRANSCRIPT_WINDOW_S`) |
| `sample_frames` | `asset_id`, `t_start`, `t_end`, `num_frames` 1–8 (`MAX_FRAMES_PER_CALL`) | JSON `{asset_id, frame_times_s}` plus that many JPEG images |
| `submit_answer` | `moments`: ranked list of `{asset_id, t_in, t_out}`, or `[]` to abstain | Ends the trial |

**Frames.** `num_frames` evenly spaced times from `t_start` to `t_end` inclusive (one
frame takes the midpoint). `t_end` is clamped to 0.25 s before the end of the video
stream (`END_MARGIN_S`). Each frame is extracted by ffmpeg as
`-ss <t> -frames:v 1 -vf scale=512:-2 -vcodec mjpeg` (`ffmpeg_frame`), so it is a
**512 px-wide JPEG at ffmpeg's default MJPEG quality** (no `-q:v` is set in the standard condition; the resolution condition below sets both), with height following the
aspect ratio (288 px for 16:9). On the Anthropic API, frames go back as `image` blocks
inside the `tool_result`, after the JSON text.

Frame width is a declared condition. Current Claude models read images up to 2,576 px
on the long edge, so 512 px is well inside every model's native resolution. A 512×288
frame costs 19 × 11 = 209 visual tokens on Claude (28 px patches). A reimplementation must use the same width,
or report it as a different condition.

**Errors** go back to the model as tool results with `is_error`, and the trial
continues: an unknown asset id, an over-budget call, or an invalid submission (any
invalid entry rejects the whole ranking, and the model may submit again).

## Conditions (each has its own compatibility key)

| Condition | CLI | What changes |
|---|---|---|
| Standard | (default) | Everything above |
| Abstain prompt | `--prompt-condition abstain` | The last sentence of the system prompt becomes: "A wrong moment counts against you and an empty answer does not, so submit an empty list whenever you are not confident that a moment matches." Net score is read as a measure of false claims only under this condition |
| Frame resolution | `--frame-width 1024 --jpeg-quality 2` | Frames at the given width and ffmpeg `-q:v` |
| Anonymized titles | `--anonymize-titles` | No titles; opaque `video-NN` ids |
| Frames only | `--no-transcripts` | No transcript tools' content |

Record runs from release 0.2.5 on use anonymized titles, because the film corpus is
well known. A run records its condition in `identity` (`prompt_condition`, `frames`,
`anonymize_titles`, `modalities`), and the results tables label every non-standard row.

## Budget (`budget.Budget`, part of the compatibility key)

Defaults: 30 turns, 64 frames, 60 tool calls per query. A request that would exceed
the frame budget returns only the frames that fit; once a budget is spent, the tool
returns an error that tells the model to submit.

## Scoring

- **Hit at 1:** the first submitted moment is in a gold video with temporal IoU of at
  least 0.5 against a gold span. On a no-answer query, only `[]` is a hit.
- **Recall@1:** mean over queries of the per-query hit rate across trials; also
  reported at tIoU 0.3 and 0.7.
- **Net score** (`run.trial_net`): +1 per correct trial, −1 per trial that submitted a
  wrong moment (including a span below tIoU 0.5), 0 per abstention or no answer;
  averaged over trials. It is the rule closed-book factuality benchmarks use. Most of
  its penalty comes from search and timing errors on positive queries, and the prompt
  asks for a best answer when the budget runs out, so the no-answer accuracy and
  false-positive rate are the measures of abstention.
- **Intervals:** 95% percentile bootstrap over queries, 2,000 resamples, seed 0
  (`stats.bootstrap_ci`); a video-clustered bootstrap for the positive-query table.
- **Cost:** tokens priced from `configs/pricing.json`, which carries its `as_of` date.

A failed, refused or over-budget trial scores zero and stays in the denominator.

## Identity

A run records a `compatibility_key` over the selected queries and their text, the
verified video bytes, the transcripts, the prompt, the tools, the budget, the
modalities, the title condition and the asset list exactly as `list_assets` shows it.
Two runs with the same key ran the same task. `momentbench submit` recomputes it.

## Reproducing in another harness

1. Download the corpus and verify it: `momentbench corpus-download` then the hashes in
   `data/corpus_manifests/sample_manifest.json`.
2. Use the prompt, task text and tool schemas above verbatim (or import `SYSTEM_PROMPT`,
   `build_task` and `TOOLS`).
3. Serve frames at 512 px wide as above, and charge the budget the same way.
4. Score with `momentbench submit`, or check your scorer against it on one run file.
   The Inspect task (`momentbench/agentic/inspect_task.py`) is a working second harness
   to compare against.
