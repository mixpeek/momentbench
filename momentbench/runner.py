"""Benchmark runner — orchestrates model evaluation across the query set."""

from __future__ import annotations

import asyncio
import json
import time
import uuid
from pathlib import Path
from typing import Any

from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.table import Table

from momentbench.eval.metrics import compute_aggregate, compute_by_tier, compute_by_vertical
from momentbench.eval.scoring import evaluate_run
from momentbench.models import get_adapter
from momentbench.types import BenchmarkRun, Prediction, Query

console = Console()


async def run_benchmark(
    queries: list[Query],
    model_name: str,
    corpus_assets: list[dict[str, Any]],
    max_concurrency: int = 5,
    output_dir: Path | None = None,
    **model_kwargs,
) -> BenchmarkRun:
    """Run a full benchmark evaluation for a single model."""
    adapter = get_adapter(model_name, **model_kwargs)
    run_id = f"{model_name}-{uuid.uuid4().hex[:8]}"
    predictions: list[Prediction] = []
    semaphore = asyncio.Semaphore(max_concurrency)

    async def predict_one(query: Query) -> Prediction | None:
        async with semaphore:
            try:
                # Adapters see only the request; gold and tier stay with the scorer.
                return await adapter.predict(query.request(), corpus_assets)
            except Exception as e:
                console.print(f"[red]Error on {query.query_id}: {e}[/red]")
                return None

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
    ) as progress:
        task = progress.add_task(
            f"Running {model_name} on {len(queries)} queries...", total=len(queries)
        )
        tasks = [predict_one(q) for q in queries]
        for coro in asyncio.as_completed(tasks):
            result = await coro
            if result:
                predictions.append(result)
            progress.advance(task)

    # Every query is scored: one that raised has no prediction and counts as a miss.
    bench_run = evaluate_run(queries, predictions, run_id=run_id, model_id=model_name)
    bench_run.config = {"model_kwargs": model_kwargs}
    bench_run.timestamp = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())

    if output_dir:
        save_run(bench_run, output_dir)

    return bench_run


def save_run(run: BenchmarkRun, output_dir: Path) -> Path:
    """Save a benchmark run to JSON."""
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / f"{run.run_id}.json"
    path.write_text(run.model_dump_json(indent=2))
    return path


def load_run(path: Path) -> BenchmarkRun:
    """Load a benchmark run from JSON."""
    return BenchmarkRun.model_validate_json(path.read_text())


def print_results(run: BenchmarkRun) -> None:
    """Print a rich summary table of benchmark results."""
    overall = compute_aggregate(run)
    by_tier = compute_by_tier(run)
    by_vertical = compute_by_vertical(run)

    console.print(f"\n[bold]Results: {run.model_id}[/bold]")
    console.print(f"Run ID: {run.run_id} | Queries: {overall.num_queries}\n")

    # Overall
    table = Table(title="Overall")
    table.add_column("Metric")
    table.add_column("Value", justify="right")
    table.add_row("Recall@1", f"{overall.recall_at_1:.1%}")
    table.add_row("Recall@5", f"{overall.recall_at_5:.1%}")
    table.add_row("Mean tIoU", f"{overall.mean_temporal_iou:.3f}")
    console.print(table)

    # By tier
    tier_table = Table(title="By Tier")
    tier_table.add_column("Tier")
    tier_table.add_column("R@1", justify="right")
    tier_table.add_column("R@5", justify="right")
    tier_table.add_column("mIoU", justify="right")
    tier_table.add_column("N", justify="right")
    for tier in sorted(by_tier.keys()):
        m = by_tier[tier]
        tier_table.add_row(
            f"T{tier.value}", f"{m.recall_at_1:.1%}", f"{m.recall_at_5:.1%}",
            f"{m.mean_temporal_iou:.3f}", str(m.num_queries),
        )
    console.print(tier_table)

    # By vertical
    vert_table = Table(title="By Vertical")
    vert_table.add_column("Vertical")
    vert_table.add_column("R@1", justify="right")
    vert_table.add_column("R@5", justify="right")
    vert_table.add_column("mIoU", justify="right")
    vert_table.add_column("N", justify="right")
    for vert in sorted(by_vertical.keys()):
        m = by_vertical[vert]
        vert_table.add_row(
            vert, f"{m.recall_at_1:.1%}", f"{m.recall_at_5:.1%}",
            f"{m.mean_temporal_iou:.3f}", str(m.num_queries),
        )
    console.print(vert_table)


def load_queries(path: Path) -> list[Query]:
    """Load queries from a JSON file."""
    data = json.loads(path.read_text())
    return [Query.model_validate(q) for q in data]


def load_corpus_assets(path: Path) -> list[dict[str, Any]]:
    """Load corpus asset metadata from a JSON file."""
    return json.loads(path.read_text())
