# Evaluating your model

Four ways in, from least to most code. All of them score the same way (see [`protocol.md`](protocol.md)) and write a run file that [`scripts/results_table.py`](../scripts/results_table.py) and [`scripts/results_doc.py`](../scripts/results_doc.py) can read.

First get the corpus, once:

```bash
git clone --branch v0.2.1 https://github.com/mixpeek/momentbench.git
cd momentbench
pip install -e ".[openai]"   # Python 3.10+; or .[gemini], .[claude-code], .[inspect]
momentbench corpus-download -m data/corpus_manifests/sample_manifest.json --corpus-dir corpus/
momentbench corpus-verify   -m data/corpus_manifests/sample_manifest.json --corpus-dir corpus/
```

About 2.6 GB for the 27 assets. `corpus-verify` must pass before a reported run.

## 1. A model behind an OpenAI-compatible endpoint (most open-weight VLMs)

vLLM, SGLang, Ollama, LM Studio, TGI and most hosted routers serve `/v1/chat/completions` with tool calling. Point the harness at it:

```bash
# start your server with tool calling enabled (see its docs for the flags), then:
momentbench agentic Qwen/Qwen3-VL-32B-Instruct --provider openai-compatible \
  --base-url http://localhost:8000/v1 \
  -q data/queries/film_v0.2.json -c data/corpus_manifests/sample_manifest.json \
  --corpus-dir corpus/ --trials 1
```

The server must support tool calls and image inputs. Frames reach the model as a user message right after the tool results, because Chat Completions has no images inside tool results. Set `OPENAI_API_KEY` if the server checks one.

## 2. A hosted frontier model

```bash
momentbench agentic gpt-5.5 --provider openai ...                  # OPENAI_API_KEY
momentbench agentic gemini-3.1-pro-preview --provider gemini ...   # GEMINI_API_KEY
momentbench agentic claude-opus-5-5 --provider anthropic ...       # ANTHROPIC_API_KEY
momentbench agentic claude-opus-5-5 --provider claude-code ...     # a local Claude Code login
```

Or under [Inspect AI](https://inspect.aisi.org.uk/), with any model Inspect supports:

```bash
inspect eval momentbench/agentic/inspect_task.py --model openai/gpt-5.5 -T corpus_dir=corpus/
```

## 3. Your own agent or harness

Subclass `AgentModel` and implement one method that takes the last tool results and returns the next tool calls. [`examples/custom_agent.py`](../examples/custom_agent.py) is a complete, runnable starting point:

```bash
python examples/custom_agent.py --corpus-dir corpus/ --limit 5
```

The harness owns the tools, the budget and scoring, so a custom agent cannot accidentally read the gold or exceed the frame budget. If your system runs its own loop (an agent SDK, a planner), implement `run_episode(corpus, query, budget)` instead, as `momentbench/agentic/claude_code_model.py` does, and call the corpus tools through `corpus.execute`, which charges the budget.

## 4. A retrieval system that ranks moments directly

Embedding and video-search systems that do not use tools can produce ranked moments per query in the `Prediction` schema (`momentbench/types.py`) and score them with `momentbench.eval.scoring.evaluate_run`, which counts a query with no prediction as a miss. The legacy adapters in `momentbench/models/` (`momentbench run <adapter>`) are examples. This track is not ranked: to put such a system on the leaderboard, wrap it as an agent whose tools call your index (section 3).

## Reading your result

```bash
python scripts/results_table.py data/results/agentic/*.json
```

Compare against the floors in [`results.md`](results.md): `baseline-random` is chance and `baseline-abstain` scores only the no-answer queries. Report per-tier Recall@1, not only the overall number: tier 3 measures abstention and tier 1 measures localisation, and an average hides which one a system is good at.

## Submitting

Package the run with `momentbench submit --results <run file> --model-name "..."` and open a pull request adding the `submissions/<model-name>/` directory it writes. [`SUBMISSION.md`](../SUBMISSION.md) has the steps, what is checked, and how held-out queries are scored.
