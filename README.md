# MomentBench

Corpus-scale video moment retrieval benchmark. 75 hours of rights-cleared video, 500 queries across 4 difficulty tiers, evaluated against frontier models and retrieval pipelines.

**Built on [Mixpeek](https://mixpeek.com)** — the pipeline that built the benchmark scores 81% on it.

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

| Tier | Name | Description | Expected frontier perf |
|------|------|-------------|----------------------|
| 1 | Single-condition visual | Identify a single visual event | >80% |
| 2 | Cross-modal joint | Joint reasoning across modalities | 50-70% |
| 3 | Corpus-scale hard negatives | Discriminate near-duplicates at corpus scale | <35% |
| 4 | Temporal reasoning | Temporal ordering and sequence reasoning | <25% |

## Verticals

- **Civic** (~30h): U.S. congressional hearings, federal agency briefings
- **Aerospace** (~25h): NASA missions, NOAA field operations
- **Film** (~20h): Blender open movies, Prelinger Archives, National Archives

## Scoring

- **Hit**: correct asset AND temporal IoU ≥ 0.5 vs gold moment
- **Metrics**: Recall@1, Recall@5, mean temporal IoU, per-tier breakdowns

## Corpus Licensing

All footage is public domain (17 U.S.C. § 105) or CC BY, permitting commercial derivatives. See `ATTRIBUTION.md` for full provenance.

## Project Structure

```
momentbench/
├── momentbench/           # Core library
│   ├── eval/              # Scoring engine
│   ├── models/            # Model adapters (Gemini, GPT, Claude, Mixpeek)
│   ├── corpus/            # Corpus management & sourcing
│   ├── taxonomy/          # Query tier definitions & generation
│   └── mixpeek_integration/  # Mixpeek pipeline integration
├── site/                  # Leaderboard web app
├── scripts/               # Automation scripts
├── configs/               # Benchmark configuration
├── data/                  # Queries, manifests, results
└── tests/                 # Test suite
```

## Run Your Own Footage

```bash
# Ingest your video corpus into Mixpeek
momentbench ingest -m your_manifest.json

# Run decomposition
momentbench decompose -m your_manifest.json

# Evaluate
momentbench run mixpeek -q your_queries.json -c your_manifest.json
```

## License

Apache-2.0
