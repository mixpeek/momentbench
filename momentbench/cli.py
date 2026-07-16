"""MomentBench CLI — run evaluations, manage corpus, view results."""

from __future__ import annotations

import asyncio
from pathlib import Path

import click
from rich.console import Console

console = Console()


@click.group()
@click.version_option()
def main():
    """MomentBench: Corpus-scale video moment retrieval benchmark."""
    pass


@main.command()
@click.argument("model", type=str)
@click.option("--queries", "-q", type=click.Path(exists=True), required=True)
@click.option("--corpus", "-c", type=click.Path(exists=True), required=True)
@click.option("--output", "-o", type=click.Path(), default="data/results")
@click.option("--concurrency", type=int, default=5)
def run(model: str, queries: str, corpus: str, output: str, concurrency: int):
    """Run benchmark evaluation for a model."""
    from momentbench.runner import load_corpus_assets, load_queries, print_results, run_benchmark

    query_list = load_queries(Path(queries))
    corpus_assets = load_corpus_assets(Path(corpus))

    console.print(f"[bold]Running {model}[/bold] on {len(query_list)} queries...")
    result = asyncio.run(
        run_benchmark(
            queries=query_list,
            model_name=model,
            corpus_assets=corpus_assets,
            max_concurrency=concurrency,
            output_dir=Path(output),
        )
    )
    print_results(result)


@main.command()
@click.argument("results_dir", type=click.Path(exists=True))
def leaderboard(results_dir: str):
    """Display leaderboard from saved results."""
    from rich.table import Table

    from momentbench.eval.metrics import compute_aggregate
    from momentbench.runner import load_run

    results_path = Path(results_dir)
    runs = [load_run(f) for f in results_path.glob("*.json")]

    if not runs:
        console.print("[yellow]No results found.[/yellow]")
        return

    table = Table(title="MomentBench Leaderboard")
    table.add_column("Rank", justify="right")
    table.add_column("Model")
    table.add_column("R@1", justify="right")
    table.add_column("R@5", justify="right")
    table.add_column("mIoU", justify="right")
    table.add_column("Queries", justify="right")

    scored = []
    for r in runs:
        m = compute_aggregate(r)
        scored.append((r.model_id, m))

    scored.sort(key=lambda x: x[1].recall_at_1, reverse=True)
    for i, (model_id, m) in enumerate(scored, 1):
        table.add_row(
            str(i), model_id, f"{m.recall_at_1:.1%}", f"{m.recall_at_5:.1%}",
            f"{m.mean_temporal_iou:.3f}", str(m.num_queries),
        )

    console.print(table)


@main.command()
@click.option("--manifest", "-m", type=click.Path(exists=True), required=True)
@click.option("--namespace", default="momentbench")
def ingest(manifest: str, namespace: str):
    """Ingest corpus into Mixpeek from a manifest file."""
    from momentbench.corpus.manifest import load_manifest
    from momentbench.mixpeek_integration.ingest import ensure_collections, ingest_corpus

    m = load_manifest(Path(manifest))
    console.print(f"Manifest: {len(m.entries)} assets, {m.total_duration_hours:.1f} hours")

    asyncio.run(ensure_collections(namespace))
    results = asyncio.run(ingest_corpus(m, namespace))
    console.print(f"Ingested: {results['ingested']}, Errors: {len(results['errors'])}")
    for err in results["errors"][:5]:
        console.print(f"  [red]{err}[/red]")


@main.command()
@click.option("--manifest", "-m", type=click.Path(exists=True), required=True)
def decompose(manifest: str):
    """Run Mixpeek decomposition pipelines on ingested assets."""
    from momentbench.corpus.manifest import load_manifest
    from momentbench.mixpeek_integration.decompose import run_decomposition

    m = load_manifest(Path(manifest))
    results = asyncio.run(run_decomposition(m))
    console.print(f"Processed: {results['processed']}, Errors: {len(results['errors'])}")


@main.command()
@click.option("--manifest", "-m", type=click.Path(exists=True), required=True)
@click.option("--output", "-o", type=click.Path(), default="ATTRIBUTION.md")
def attribution(manifest: str, output: str):
    """Generate ATTRIBUTION.md from corpus manifest."""
    from momentbench.corpus.manifest import load_manifest

    m = load_manifest(Path(manifest))
    content = m.generate_attribution_md()
    Path(output).write_text(content)
    console.print(f"Written to {output}")


@main.command()
@click.option("--results", "-r", type=click.Path(exists=True), required=True)
@click.option("--model-name", required=True, help="Display name for the leaderboard")
@click.option("--model-description", default="", help="Brief description of approach")
@click.option("--contact", default="", help="Contact email for verification")
@click.option("--output-dir", "-o", type=click.Path(), default="submissions")
def submit(results: str, model_name: str, model_description: str, contact: str, output_dir: str):
    """Package a benchmark run for leaderboard submission."""
    import json
    import shutil

    from momentbench.eval.metrics import compute_aggregate, compute_by_tier
    from momentbench.runner import load_run

    run = load_run(Path(results))
    overall = compute_aggregate(run)
    by_tier = compute_by_tier(run)

    if overall.num_queries == 0:
        console.print("[red]No results found in the run file.[/red]")
        return

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    slug = model_name.lower().replace(" ", "-").replace("/", "-")
    submission_dir = out / slug
    submission_dir.mkdir(parents=True, exist_ok=True)

    shutil.copy(results, submission_dir / "results.json")

    meta = {
        "model_name": model_name,
        "model_description": model_description,
        "contact": contact,
        "run_id": run.run_id,
        "model_id": run.model_id,
        "num_queries": overall.num_queries,
        "recall_at_1": round(overall.recall_at_1, 4),
        "recall_at_5": round(overall.recall_at_5, 4),
        "mean_temporal_iou": round(overall.mean_temporal_iou, 4),
        "by_tier": {
            f"tier_{k.value}": {
                "recall_at_1": round(v.recall_at_1, 4),
                "recall_at_5": round(v.recall_at_5, 4),
                "mean_temporal_iou": round(v.mean_temporal_iou, 4),
                "num_queries": v.num_queries,
            }
            for k, v in by_tier.items()
        },
        "timestamp": run.timestamp,
    }
    (submission_dir / "submission.json").write_text(json.dumps(meta, indent=2))

    console.print(f"\n[bold]Submission packaged:[/bold] {submission_dir}/")
    console.print(f"  Model: {model_name}")
    console.print(f"  Recall@1: {overall.recall_at_1:.1%}")
    console.print(f"  Recall@5: {overall.recall_at_5:.1%}")
    console.print(f"  Mean tIoU: {overall.mean_temporal_iou:.3f}")
    console.print(f"  Queries: {overall.num_queries}")
    console.print(
        "\nTo submit: open a PR adding this directory to the repo, "
        "or email the submission.json to team@mixpeek.com"
    )


if __name__ == "__main__":
    main()
