# Submitting Results to MomentBench

We welcome external submissions. MomentBench is only credible if teams beyond Mixpeek run models against it, find weaknesses, and push the state of the art.

## How to submit

### 1. Run the benchmark

```bash
pip install momentbench

# Using a built-in adapter
momentbench run <model-name> \
  -q data/queries/sample.json \
  -c data/corpus_manifests/sample_manifest.json \
  -o data/results/

# Using a custom adapter (see below)
momentbench run my-model \
  -q data/queries/sample.json \
  -c data/corpus_manifests/sample_manifest.json \
  -o data/results/
```

### 2. Submit via CLI

```bash
momentbench submit \
  --results data/results/your-run.json \
  --model-name "Your Model Name" \
  --model-description "Brief description of approach" \
  --contact your@email.com
```

### 3. Or submit via pull request

1. Fork the repo
2. Add your results file to `submissions/`
3. Open a PR with:
   - Results JSON file
   - Brief description of your model/approach
   - Whether you want your submission public or anonymous until leaderboard update

## Results format

Your results JSON must follow the `BenchmarkRun` schema:

```json
{
  "run_id": "your-model-20250712",
  "model_id": "your-model-name",
  "results": [
    {
      "query_id": "q-civic-001",
      "model_id": "your-model-name",
      "hit_at_1": true,
      "hit_at_5": true,
      "best_temporal_iou": 0.82,
      "tier": 1,
      "vertical": "civic"
    }
  ],
  "config": {},
  "timestamp": "2025-07-12T00:00:00Z"
}
```

## Writing a custom adapter

```python
from momentbench.models.base import ModelAdapter
from momentbench.models.registry import register_adapter

@register_adapter("my-model")
class MyModelAdapter(ModelAdapter):
    model_id = "my-model-v1"

    async def predict(self, query, corpus_assets, max_results=5):
        # Your retrieval logic here
        # Return a Prediction with ranked_moments
        ...
```

Save this in `momentbench/models/my_model.py` and it will be auto-discovered.

## What we verify

Before results appear on the leaderboard:

1. **Format check** — results JSON parses and covers all public queries
2. **Reproducibility** — we may re-run your adapter if the code is included
3. **No query-set overfitting** — the private held-out set guards against this; we may run your model against it via our scoring endpoint

## Evaluation on the private set

The 150-query private held-out set is never distributed. To evaluate against it:

1. Submit your model adapter code
2. We run it against the private set on our infrastructure
3. Private-set scores are reported alongside public-set scores on the leaderboard

This prevents contamination while letting every submission be scored on both sets.

## What makes a good submission

- Novel approach or architecture (not just a parameter sweep)
- Reproducible from the adapter code
- Brief write-up of what the approach does differently

We especially welcome submissions from teams working on video-language models, temporal grounding, or multimodal retrieval — including approaches that use decomposition, indexing, or retrieval-augmented methods.

## Questions?

Open an issue on GitHub or email team@mixpeek.com.
