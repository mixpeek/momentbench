# Contributing to MomentBench

We welcome contributions! Here's how to get started.

## Development Setup

```bash
git clone https://github.com/mixpeek/momentbench.git
cd momentbench
pip install -e ".[dev,models]"
```

## Running Tests

```bash
pytest tests/ -v
```

## Linting

```bash
ruff check momentbench/ tests/ site/
```

## Adding a Model Adapter

1. Create a new file in `momentbench/models/`
2. Subclass `ModelAdapter` and implement the `predict` method
3. Register it with `@register_adapter("your-model-name")`
4. Add it to the `_load_builtin` map in `registry.py`

## Adding Queries

Queries go in `data/queries/` as JSON arrays. Each query needs:
- `query_id`: unique identifier
- `text`: natural language query
- `tier`: 1-4 (see taxonomy)
- `vertical`: "civic", "aerospace", or "film"
- `gold_moments`: array of `{asset_id, t_in, t_out}`

## Pull Requests

- Keep PRs focused on a single change
- Include tests for new scoring logic
- Run `ruff check` before submitting
- Describe what tier/vertical your query additions target

## Corpus Contributions

If adding new video sources, ensure:
- License is public domain or CC BY (not CC BY-NC, not CC BY-ND)
- Source URL is stable and verifiable
- Attribution string is complete
- Duration is accurate
