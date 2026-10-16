# Submitting Results to MomentBench

We welcome external submissions. MomentBench is only credible if teams beyond Mixpeek run models against it, find weaknesses, and push the state of the art.

There is one route onto the leaderboard: an agentic-track run on a whole domain's query file, on the full hash-verified corpus.

## 1. Get the corpus

```bash
git clone --branch v0.2.3 https://github.com/mixpeek/momentbench.git
cd momentbench
pip install -e ".[openai]"   # Python 3.10+; or .[gemini], .[claude-code], .[inspect]
momentbench corpus-download -m data/corpus_manifests/sample_manifest.json --corpus-dir corpus/
```

MomentBench is not on PyPI; the commands read the queries, manifest and transcripts in the checkout, so run them inside it. `ffmpeg` and `ffprobe` must be on the PATH. `corpus-download` checks every file against the sha256 in the manifest.

## 2. Run the agentic track

```bash
momentbench agentic <model> --provider <provider> \
  -q data/queries/robotics_v0.2.json \
  -c data/corpus_manifests/sample_manifest.json \
  --corpus-dir corpus/ -o data/results/agentic/
```

Run every query in the file (the default). [`docs/integrating.md`](docs/integrating.md) covers each provider (`anthropic`, `openai`, `gemini`, `openai-compatible` for vLLM, SGLang or any OpenAI-compatible server) and how to plug in your own agent or loop.

The run file records what it ran: repo commit, hashes of the queries, manifest, transcripts, prompt and tools, the budget, the provider and the model version it reported, and a compatibility key. A run on an incomplete corpus is refused unless you pass `--partial`, and a `--partial` run is marked not reportable and cannot be submitted.

## 3. Package it

```bash
momentbench submit \
  --results data/results/agentic/agentic-<model>-<id>.json \
  --model-name "Your Model Name" \
  --model-description "Brief description of approach" \
  --contact your@email.com
```

`submit` rescores the run against the query file as it stands and refuses a run that is not reportable or skipped any query. It writes `submissions/<model-name>/` with the run file and a `submission.json` holding the scores, the compatibility key and the run identity.

## 4. Open a pull request

Add the `submissions/<model-name>/` directory and say in the PR whether the entry should be public now or anonymous until the next leaderboard update. Maintainers check the compatibility key against the current release, move the run into `data/results/agentic-v0.2/record/`, and regenerate `docs/results.md` and momentbench.com.

To see how your run ranks before submitting:

```bash
momentbench leaderboard data/results/agentic/
```

## What we verify

1. **Reportable and complete:** the run covers every query in the file and its corpus receipt lists every asset.
2. **Same benchmark:** the compatibility key matches runs on the current release (same queries, corpus, transcripts, prompt, tools, budget and modalities).
3. **Reproducibility:** we may rerun the model, so include how to call it.
4. **No query-set overfitting:** see the held-out set below.

## Evaluation on the private set

The 163-query private held-out set is never distributed. To evaluate against it:

1. Tell us how to call your model (an endpoint, or an agent class like `examples/custom_agent.py`).
2. We run it on the held-out queries with the same harness and budget.
3. Held-out scores are reported next to the public-set scores.

This prevents contamination while letting every submission be scored on both sets. [`GOVERNANCE.md`](GOVERNANCE.md) states how Mixpeek, which maintains the benchmark and holds the held-out labels, handles its own entries.

## The legacy track

`momentbench run <adapter>` is the older single-call track. Its adapters see titles and durations, or return whole videos (see each adapter's `input_kind`), so its scores do not measure multimodal moment retrieval and are not ranked. `momentbench submit` refuses them.

## What makes a good submission

- A model or system evaluated on all three domains, or a clear reason for one
- Reproducible from what you send us
- A short write-up of what the approach does differently

We especially welcome submissions from teams working on video-language models, temporal grounding, world models and multimodal retrieval.

## Questions?

Open an issue on GitHub or email team@mixpeek.com.
