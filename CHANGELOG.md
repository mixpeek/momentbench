# Changelog

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
