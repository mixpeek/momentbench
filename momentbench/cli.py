"""MomentBench CLI — run evaluations, manage corpus, view results."""

from __future__ import annotations

import asyncio
import json
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

    from momentbench.models.base import INPUT_KINDS
    from momentbench.models.registry import get_adapter_class

    kind = get_adapter_class(model).input_kind
    console.print(f"[bold]Running {model}[/bold] ({kind}) on {len(query_list)} queries...")
    if kind != "indexed-platform":
        console.print(f"[yellow]{model} {INPUT_KINDS[kind]}. This legacy-track score is not a "
                      "multimodal result; run `momentbench agentic` for that.[/yellow]")
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
@click.argument("model", type=str)
@click.option("--queries", "-q", type=click.Path(exists=True), required=True)
@click.option("--corpus", "-c", type=click.Path(exists=True), required=True)
@click.option("--corpus-dir", type=click.Path(exists=True, file_okay=False), required=True,
              help="Directory holding <asset_id>.mp4 and optional transcripts")
@click.option("--transcripts-dir", type=click.Path(file_okay=False), default="data/transcripts",
              help="Directory of <asset_id>.vtt transcripts, used when --corpus-dir has none")
@click.option("--partial", is_flag=True,
              help="Debug only: allow a corpus with missing or unverified assets. The run is "
              "marked not reportable and saved under data/results/debug")
@click.option("--physical-only", is_flag=True,
              help="Run only queries tagged metadata.physical_event (machine motion or contact)")
@click.option("--no-transcripts", is_flag=True,
              help="Visual-only ablation: no transcript is available to any tool")
@click.option("--anonymize-titles", is_flag=True,
              help="Control: hide titles and show opaque asset ids (video-01, ...), so the "
              "model cannot pick a video by its name")
@click.option("--provider", default="anthropic",
              type=click.Choice(["anthropic", "claude-code", "openai", "gemini",
                                 "openai-compatible", "baseline"]),
              help="anthropic, openai, gemini: provider API key and our loop. "
              "claude-code: the local claude CLI login. openai-compatible: any "
              "/v1/chat/completions server (vLLM, SGLang, Ollama), with --base-url. "
              "baseline: reference floors, MODEL is random or abstain")
@click.option("--base-url", default=None,
              help="openai-compatible only: server URL, e.g. http://localhost:8000/v1")
@click.option("--effort", type=str, default=None, help="Provider effort level, e.g. low, high")
@click.option("--trials", type=int, default=5)
@click.option("--max-steps", type=int, default=30)
@click.option("--max-frames", type=int, default=64)
@click.option("--max-tool-calls", type=int, default=60)
@click.option("--pricing", type=click.Path(exists=True), default="configs/pricing.json")
@click.option("--output", "-o", type=click.Path(), default="data/results/agentic")
@click.option("--concurrency", type=int, default=4)
def agentic(model: str, queries: str, corpus: str, corpus_dir: str, transcripts_dir: str,
            partial: bool, physical_only: bool, no_transcripts: bool,
            anonymize_titles: bool, provider: str,
            base_url: str | None,
            effort: str | None, trials: int, max_steps: int, max_frames: int,
            max_tool_calls: int, pricing: str, output: str, concurrency: int):
    """Run the agentic track: a tool-using model searches the corpus for each moment."""
    from momentbench.agentic.anthropic_model import AnthropicAgent
    from momentbench.agentic.baselines import BaselineAgent
    from momentbench.agentic.budget import Budget, load_pricing
    from momentbench.agentic.claude_code_model import ClaudeCodeAgent
    from momentbench.agentic.environment import LocalCorpus
    from momentbench.agentic.gemini_model import GeminiAgent
    from momentbench.agentic.openai_compat_model import OpenAICompatAgent
    from momentbench.agentic.openai_model import OpenAIAgent
    from momentbench.agentic.run import run_agentic_benchmark
    from momentbench.runner import load_corpus_assets, load_queries

    query_list = load_queries(Path(queries))
    if physical_only:
        query_list = [q for q in query_list if q.metadata.get("physical_event")]
        if not query_list:
            raise click.ClickException(f"no query in {queries} is tagged physical_event")
    local = LocalCorpus(load_corpus_assets(Path(corpus)), Path(corpus_dir),
                        transcripts={} if no_transcripts else None,
                        transcripts_dir=Path(transcripts_dir),
                        anonymize_titles=anonymize_titles)
    from momentbench.corpus.verify import verify_corpus

    # A reportable run needs every asset present and hash-verified: with assets missing,
    # positives drop out while every no-answer query stays, which inflates the score.
    receipt, failures = verify_corpus(list(local.assets.values()), Path(corpus_dir),
                                      Path(corpus_dir) / ".momentbench-verify-cache.json")
    runnable = query_list
    if failures:
        detail = "; ".join(f"{c.asset_id}: {', '.join(c.problems)}" for c in failures[:5])
        if not partial:
            raise click.ClickException(
                f"{len(failures)} of {len(local.assets)} assets failed verification ({detail}). "
                "Run corpus-download, or pass --partial for a non-reportable debug run.")
        bad = {c.asset_id for c in failures}
        runnable = [q for q in query_list if not any(g.asset_id in bad for g in q.gold_moments)]
        console.print(f"[yellow]--partial: {len(failures)} assets unverified; running "
                      f"{len(runnable)} of {len(query_list)} queries. This run is NOT "
                      "reportable and is saved under data/results/debug.[/yellow]")
        output = str(Path("data/results/debug"))
    if not runnable:
        raise click.ClickException("no query is runnable on this corpus")

    prices = load_pricing(Path(pricing), model)
    if prices is None:
        console.print(f"[yellow]No price for {model} in {pricing}; cost will be omitted.[/yellow]")

    from momentbench.agentic.environment import TOOLS
    from momentbench.agentic.episode import SYSTEM_PROMPT
    from momentbench.agentic.identity import build_identity
    from momentbench.agentic.run import ProviderUnavailableError

    budget = Budget(max_steps=max_steps, max_frames=max_frames, max_tool_calls=max_tool_calls)
    try:
        pricing_as_of = json.loads(Path(pricing).read_text()).get("as_of")
    except (OSError, ValueError):
        pricing_as_of = None
    identity = build_identity(
        queries_path=Path(queries), manifest_path=Path(corpus),
        transcripts=local.transcripts, system_prompt=SYSTEM_PROMPT, tools=TOOLS,
        budget=budget.model_dump(),
        modalities="frames+transcripts" if local.has_transcripts() else "frames",
        anonymize_titles=anonymize_titles,
        provider=provider, endpoint=base_url, pricing_as_of=pricing_as_of)
    identity["query_filter"] = "physical_only" if physical_only else None

    try:
        agent = (
            OpenAICompatAgent(model=model, effort=effort, base_url=base_url)
            if provider == "openai-compatible" else {
                "anthropic": AnthropicAgent,
                "claude-code": ClaudeCodeAgent,
                "openai": OpenAIAgent,
                "gemini": GeminiAgent,
                "baseline": BaselineAgent,
            }[provider](model=model, effort=effort)
        )
    except ValueError as e:
        raise click.ClickException(str(e)) from e

    try:
        result = asyncio.run(
            run_agentic_benchmark(
                queries=runnable,
                model=agent,
                corpus=local,
                trials=trials,
                budget=budget,
                pricing=prices,
                max_concurrency=concurrency,
                output_dir=Path(output),
                reportable=not failures,
                corpus_receipt=receipt,
                identity=identity,
            )
        )
    except ProviderUnavailableError as e:
        raise click.ClickException(
            f"the provider refused the account, run stopped and nothing saved: {e}"
        ) from e
    errors = [t.error for t in result.trials if t.error]
    if len(errors) == len(result.trials):
        raise click.ClickException(f"every trial errored, nothing saved. First error: {errors[0]}")
    if errors:
        console.print(f"[red]{len(errors)} trials errored. First: {errors[0]}[/red]")
    s = result.summary
    lo, hi = s.recall_at_1_ci95
    console.print(
        f"\n[bold]{result.model_id}[/bold] effort={result.effort} harness={result.harness} "
        f"run={result.run_id}"
    )
    console.print(f"  Queries: {s.num_queries} x {s.num_trials} trials")
    console.print(f"  Recall@1: {s.recall_at_1:.1%} (95% CI {lo:.1%} to {hi:.1%})")
    console.print(f"  Recall@5: {s.recall_at_5:.1%}   Mean tIoU: {s.mean_temporal_iou:.3f}")
    if s.mean_cost_usd_per_query is not None:
        console.print(f"  Cost per query: ${s.mean_cost_usd_per_query:.4f}")
    console.print(
        f"  Per query: {s.mean_frames_per_query:.1f} frames, {s.mean_steps_per_query:.1f} turns, "
        f"{s.mean_input_tokens_per_query:.0f} in / {s.mean_output_tokens_per_query:.0f} out tokens"
    )
    by_iou = "  ".join(f"@{th} {r:.1%}" for th, r in s.recall_at_1_by_iou.items())
    console.print(f"  Recall@1 by tIoU: {by_iou}")
    for tier, row in s.by_tier.items():
        t_lo, t_hi = row["recall_at_1_ci95"]
        console.print(f"  Tier {tier}: {row['recall_at_1']:.1%} ({t_lo:.1%} to {t_hi:.1%}), "
                      f"{row['num_queries']} queries")
    if s.failed_trials:
        console.print(f"  [yellow]{s.failed_trials} trials ended without a submission[/yellow]")


@main.command("corpus-download")
@click.option("--manifest", "-m", type=click.Path(exists=True), required=True)
@click.option("--corpus-dir", type=click.Path(file_okay=False), required=True)
def corpus_download(manifest: str, corpus_dir: str):
    """Download every manifest asset that has a source file, then check its sha256."""
    import json

    from momentbench.corpus.download import download_asset

    out = Path(corpus_dir)
    out.mkdir(parents=True, exist_ok=True)
    lines = [download_asset(a, out) for a in json.loads(Path(manifest).read_text())]
    for line in lines:
        console.print(line)
    if any(line.startswith("FAIL") for line in lines):
        raise SystemExit(1)


@main.command("corpus-verify")
@click.option("--manifest", "-m", type=click.Path(exists=True), required=True)
@click.option("--corpus-dir", type=click.Path(exists=True, file_okay=False), required=True)
def corpus_verify(manifest: str, corpus_dir: str):
    """Check corpus files against the manifest: presence, sha256 and duration."""
    import json

    from momentbench.corpus.verify import check_asset

    assets = json.loads(Path(manifest).read_text())
    checks = [check_asset(a, Path(corpus_dir)) for a in assets]
    for c in checks:
        if c.ok:
            console.print(f"[green]ok[/green]   {c.asset_id}  {c.duration_s:.2f}s  {c.sha256[:12]}")
        else:
            console.print(f"[red]FAIL[/red] {c.asset_id}: {'; '.join(c.problems)}")
    bad = sum(1 for c in checks if not c.ok)
    console.print(f"{len(checks) - bad} of {len(checks)} assets verified")
    if bad:
        raise SystemExit(1)


@main.command()
@click.argument("results_dir", type=click.Path(exists=True))
def leaderboard(results_dir: str):
    """Rank the runs in a directory: agentic runs per query file, legacy runs apart."""
    from rich.table import Table

    from momentbench.agentic.run import AgenticRun
    from momentbench.eval.metrics import compute_aggregate
    from momentbench.submission import (
        SubmissionError,
        agentic_entry,
        find_queries_file,
        load_any_run,
    )

    files = sorted(Path(results_dir).glob("*.json"))
    if not files:
        console.print("[yellow]No results found.[/yellow]")
        return

    def pct(x):
        return "n/a" if x is None else f"{100 * x:.1f}"

    boards: dict[str, list[dict]] = {}
    legacy, excluded = [], []
    for f in files:
        try:
            run = load_any_run(f)
        except ValueError as e:
            excluded.append((f.name, f"unreadable: {str(e).splitlines()[0]}"))
            continue
        if isinstance(run, AgenticRun):
            try:
                entry = agentic_entry(run, find_queries_file(run), require_complete=False)
            except SubmissionError as e:
                excluded.append((f.name, str(e)))
                continue
            boards.setdefault(entry["queries_file"], []).append({**entry, "file": f.name})
        else:
            legacy.append((run, compute_aggregate(run)))

    for name, entries in sorted(boards.items()):
        table = Table(title=f"MomentBench agentic track: {name}")
        for col in ("Rank", "Model", "R@1 (95% CI)", "Positive R@1", "No-answer acc",
                    "Cost/query", "Coverage x trials", "Run"):
            table.add_column(col, justify="right" if col not in ("Model", "Run") else "left")
        entries.sort(key=lambda e: e["recall_at_1"], reverse=True)
        for i, e in enumerate(entries, 1):
            lo, hi = e["recall_at_1_ci95"]
            cost = e["mean_cost_usd_per_query"]
            table.add_row(str(i), e["model_id"],
                          f"{pct(e['recall_at_1'])} ({pct(lo)} to {pct(hi)})",
                          pct(e["positive_recall_at_1"]), pct(e["no_answer_accuracy"]),
                          "n/a" if cost is None else f"${cost:.3f}",
                          f"{e['num_queries']}/{e['queries_in_file']} x {e['num_trials']}",
                          e["file"])
        console.print(table)
        if any(e["num_queries"] < e["queries_in_file"] for e in entries):
            console.print("Coverage below the file size: the run predates queries added or "
                          "reworded since, and is scored on the queries it answered. "
                          "docs/results.md compares runs on their common queries.")

    if legacy:
        table = Table(title="Legacy track (single call, not comparable with the agentic track)")
        for col in ("Model", "Input kind", "R@1", "Queries"):
            table.add_column(col)
        for run, m in sorted(legacy, key=lambda x: x[1].recall_at_1, reverse=True):
            table.add_row(run.model_id, run.config.get("input_kind", "unrecorded"),
                          f"{m.recall_at_1:.1%}", str(m.num_queries))
        console.print(table)

    for name, reason in excluded:
        console.print(f"[yellow]Not ranked: {name}: {reason}[/yellow]")


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
@click.option("--results", "-r", type=click.Path(exists=True), required=True,
              help="An agentic run file written by `momentbench agentic`")
@click.option("--queries", "-q", type=click.Path(exists=True), default=None,
              help="Query file the run answered (found in data/queries when omitted)")
@click.option("--model-name", required=True, help="Display name for the leaderboard")
@click.option("--model-description", default="", help="Brief description of approach")
@click.option("--contact", default="", help="Contact email for verification")
@click.option("--output-dir", "-o", type=click.Path(), default="submissions")
def submit(results: str, queries: str | None, model_name: str, model_description: str,
           contact: str, output_dir: str):
    """Package an agentic run for leaderboard submission."""
    import json
    import shutil

    from momentbench.agentic.run import AgenticRun
    from momentbench.submission import (
        SubmissionError,
        agentic_entry,
        find_queries_file,
        load_any_run,
    )

    run = load_any_run(Path(results))
    if not isinstance(run, AgenticRun):
        raise click.ClickException(
            "this is a legacy-track run. The leaderboard takes agentic runs: run "
            "`momentbench agentic` and submit its run file (see SUBMISSION.md).")
    try:
        queries_path = Path(queries) if queries else find_queries_file(run)
        entry = agentic_entry(run, queries_path)
    except SubmissionError as e:
        raise click.ClickException(str(e)) from e

    slug = model_name.lower().replace(" ", "-").replace("/", "-")
    submission_dir = Path(output_dir) / slug
    submission_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy(results, submission_dir / "run.json")
    meta = {"model_name": model_name, "model_description": model_description,
            "contact": contact, **entry}
    (submission_dir / "submission.json").write_text(json.dumps(meta, indent=2) + "\n")

    lo, hi = entry["recall_at_1_ci95"]
    console.print(f"\n[bold]Submission packaged:[/bold] {submission_dir}/")
    console.print(f"  Model: {model_name} ({entry['model_id']})")
    console.print(f"  Queries: {entry['queries_file']}, {entry['num_queries']} x "
                  f"{entry['num_trials']} trials")
    console.print(f"  Recall@1: {entry['recall_at_1']:.1%} ({lo:.1%} to {hi:.1%})")
    console.print(f"  Compatibility key: {entry['compatibility_key']}")
    console.print("\nTo submit: open a PR adding this directory under submissions/ "
                  "(see SUBMISSION.md).")


if __name__ == "__main__":
    main()
