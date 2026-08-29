<p align="center">
  <img src="site/assets/banner.svg" alt="MomentBench — Corpus-scale video retrieval benchmark" width="100%">
</p>

# MomentBench

**[momentbench.com](https://momentbench.com)** — Corpus-scale video moment retrieval benchmark.

**Design target:** 75 hours of rights-cleared video, 500 queries across 4 difficulty tiers, evaluated against frontier models and retrieval pipelines.

**Current seed release (v0.1):** 9 corpus assets, 6 queries, spanning all 3 verticals and 4 tiers — see [`data/`](data/). The corpus and query set are actively growing toward the full target; PRs adding rights-cleared footage or queries are welcome (see [`SUBMISSION.md`](SUBMISSION.md)).

Built on [Mixpeek](https://mixpeek.com).

## Quick Start

```bash
pip install -e ".[models]"

# Run a single model
momentbench run gemini -q data/queries/sample.json -c data/corpus_manifests/sample_manifest.json

# Run all baselines
./scripts/run_all_baselines.sh

# View leaderboard
momentbench leaderboard data/results/
```

## Query Tiers

Difficulty targets were pre-registered before any baseline evaluations were run.

| Tier | Name | Description | Pre-registered difficulty target |
|------|------|-------------|----------------------------------|
| 1 | Single-condition visual | Identify a single visual event | >80% recall for frontier models |
| 2 | Cross-modal joint | Joint reasoning across modalities | 50-70% recall |
| 3 | Corpus-scale hard negatives | Discriminate near-duplicates at corpus scale | <35% recall |
| 4 | Temporal reasoning | Temporal ordering and sequence reasoning | <25% recall |

## Verticals

Target scope per vertical, with current seed-release asset count:

- **Civic** (target ~30h, seed: 1 asset): U.S. congressional hearings, federal agency briefings
- **Aerospace** (target ~25h, seed: 1 asset): NASA missions, NOAA field operations
- **Film** (target ~20h, seed: 7 assets): Blender open movies, Prelinger Archives, National Archives

## Scoring

- **Hit**: correct asset AND temporal IoU ≥ 0.5 vs gold moment
- **Metrics**: Recall@1, Recall@5, mean temporal IoU, per-tier breakdowns

## Corpus Licensing

All footage is public domain (17 U.S.C. § 105) or CC BY, permitting commercial derivatives. See [`ATTRIBUTION.md`](ATTRIBUTION.md) for full provenance.

## Project Structure

```
momentbench/
├── momentbench/           # Core library
│   ├── eval/              # Scoring engine
│   ├── models/            # Model adapters (Gemini, GPT, Claude, Mixpeek)
│   ├── corpus/            # Corpus management & sourcing
│   ├── taxonomy/          # Query tier definitions & generation
│   └── mixpeek_integration/  # Mixpeek pipeline integration
├── site/                  # Leaderboard (momentbench.com)
├── scripts/               # Automation scripts
├── configs/               # Benchmark configuration
├── data/                  # Queries, manifests, results
└── tests/                 # Test suite
```

## Model Adapters

MomentBench ships adapters for both closed and open-source models.

### Closed-source (API-based)

| Adapter | Model | Notes |
|---------|-------|-------|
| `gemini` | Gemini 2.5 Pro | Native multimodal, long-context |
| `openai` | GPT-4.1 | Frames + transcript in context |
| `claude` | Claude Sonnet 4 | Frames + transcript in context |
| `mixpeek` | Mixpeek pipeline | Decompose→embed→search |
| `twelvelabs` | TwelveLabs Marengo | Video search API with temporal segments |
| `mixedbread` | mxbai-embed-large | Text embedding retrieval via MixedBread API |
| `jina-clip` | Jina CLIP v2 | Multilingual multimodal embeddings |

### Open-source (self-hosted)

| Adapter | Model | Modalities | Notes |
|---------|-------|------------|-------|
| `qwen-vl` | Qwen3-VL / Qwen2.5-VL | Text, image, video | Closest OSS analog to Gemini for video understanding. Serve via vLLM or SGLang. |
| `e5-omni` | E5-Omni (Microsoft) | Text, image, audio, video | Closest to Gemini Embedding 2 — shared 3584-dim space, MIT license. |
| `clip` | OpenAI CLIP (ViT-L/14) | Text, image | Zero-shot image-text retrieval via OpenCLIP. Ranks by cosine similarity. |
| `hf-vlm` | Any OpenAI-compatible VLM | Varies | Generic adapter for InternVL3, MiniCPM-V, Llama 4, Phi-4, etc. |

```bash
# Run API-based models
TWELVELABS_API_KEY=... momentbench run twelvelabs -q data/queries/sample.json -c data/corpus_manifests/sample_manifest.json
MIXEDBREAD_API_KEY=... momentbench run mixedbread -q data/queries/sample.json -c data/corpus_manifests/sample_manifest.json

# Run OSS models (requires a local inference server or GPU)
QWEN_API_BASE=http://localhost:8000/v1 momentbench run qwen-vl -q data/queries/sample.json -c data/corpus_manifests/sample_manifest.json
momentbench run clip -q data/queries/sample.json -c data/corpus_manifests/sample_manifest.json

# Run only OSS baselines
OSS_ONLY=1 ./scripts/run_all_baselines.sh

# Run only closed-source baselines
CLOSED_ONLY=1 ./scripts/run_all_baselines.sh
```

Install OSS dependencies: `pip install -e ".[oss]"`

## Submit Results

We welcome external submissions. See [`SUBMISSION.md`](SUBMISSION.md) for details.

```bash
momentbench submit \
  --results data/results/your-run.json \
  --model-name "Your Model" \
  --contact you@example.com
```

Or open a PR adding your results to [`submissions/`](submissions/).

## Run Your Own Footage

```bash
# Ingest your video corpus into Mixpeek
momentbench ingest -m your_manifest.json

# Run decomposition
momentbench decompose -m your_manifest.json

# Evaluate
momentbench run mixpeek -q your_queries.json -c your_manifest.json
```

## Citation

```bibtex
@software{momentbench2025,
  title     = {MomentBench: Corpus-Scale Video Moment Retrieval Benchmark},
  author    = {Mixpeek},
  year      = {2025},
  url       = {https://github.com/mixpeek/momentbench},
  license   = {Apache-2.0}
}
```

## License

Apache-2.0
