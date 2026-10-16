"""Read run files of either track and package agentic runs for the leaderboard.

The agentic track is the reported one (audit finding 28): `momentbench submit` and
`momentbench leaderboard` take its run files, rescore them against the query file as
it stands, and refuse a run that cannot be compared (not reportable, or missing
queries).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from momentbench.agentic.run import AgenticRun, rescore, resolved_models
from momentbench.types import BenchmarkRun, Query

ROOT = Path(__file__).resolve().parents[1]
QUERIES_DIR = ROOT / "data" / "queries"
MANIFEST = ROOT / "data" / "corpus_manifests" / "sample_manifest.json"
TRANSCRIPTS_DIR = ROOT / "data" / "transcripts"


class SubmissionError(ValueError):
    """A run that cannot go on the leaderboard, with the reason."""


def load_any_run(path: Path) -> AgenticRun | BenchmarkRun:
    data = json.loads(path.read_text())
    if "trials" in data:
        return AgenticRun.model_validate(data)
    return BenchmarkRun.model_validate(data)


def load_queries_file(path: Path) -> dict[str, Query]:
    return {q["query_id"]: Query(**q) for q in json.loads(path.read_text())}


def find_queries_file(run: AgenticRun, search: Path | None = None) -> Path:
    """The published query file the run answered: the one holding most of its query ids.

    Older runs can include queries since removed by review, so containment is not
    required; at least half of the run's ids must be in the file.
    """
    search = search or QUERIES_DIR
    ids = {t.query_id for t in run.trials}
    best, overlap = None, 0
    for path in sorted(search.glob("*_v*.json")):
        n = len(ids & {q["query_id"] for q in json.loads(path.read_text())})
        if n > overlap:
            best, overlap = path, n
    if best is None or overlap * 2 < len(ids):
        raise SubmissionError(f"no query file in {search} matches the query ids of "
                              f"{run.run_id}; pass --queries")
    return best


def expected_key(run: AgenticRun, queries: dict[str, Query], manifest: Path,
                 transcripts: dict[str, list[dict[str, Any]]] | None = None) -> str:
    """The compatibility key a complete run on the current release must carry."""
    from momentbench.agentic.environment import TOOLS, LocalCorpus
    from momentbench.agentic.episode import SYSTEM_PROMPT
    from momentbench.agentic.identity import task_identity
    from momentbench.runner import load_corpus_assets

    assets = load_corpus_assets(manifest)
    if transcripts is None:
        transcripts = LocalCorpus(assets, None, transcripts_dir=TRANSCRIPTS_DIR).transcripts
    if run.modalities == "frames":
        transcripts = {}
    receipt = {a["asset_id"]: a["file"]["sha256"] for a in assets}
    return task_identity(queries=list(queries.values()), receipt=receipt,
                         transcripts=transcripts, system_prompt=SYSTEM_PROMPT, tools=TOOLS,
                         budget=run.budget.model_dump(), modalities=run.modalities,
                         anonymize_titles=run.anonymize_titles)["compatibility_key"]


def check_trial_grid(run: AgenticRun) -> None:
    """Exactly one trial per (query, trial number), num_trials of them per query."""
    n = run.summary.num_trials
    seen: dict[str, set[int]] = {}
    for t in run.trials:
        trials = seen.setdefault(t.query_id, set())
        if t.trial in trials:
            raise SubmissionError(f"{run.run_id}: query {t.query_id} trial {t.trial} "
                                  "appears more than once")
        trials.add(t.trial)
    for qid, trials in seen.items():
        if trials != set(range(n)):
            raise SubmissionError(f"{run.run_id}: query {qid} has trials {sorted(trials)}, "
                                  f"expected 0 to {n - 1}")


def check_model_identity(run: AgenticRun) -> None:
    """One leaderboard row is one model snapshot (audit 3, finding 3).

    Every model identity the provider reported, on any turn of any trial, must be the
    same. A run whose alias resolved to two snapshots is two systems; run each one
    separately under its own model ID.
    """
    models = set(resolved_models(run.trials)) | set(run.identity.get("resolved_models") or [])
    if len(models) > 1:
        raise SubmissionError(f"{run.run_id}: the provider reported {len(models)} model "
                              f"identities ({', '.join(sorted(models))}); one row is one "
                              "snapshot. Rerun so every turn is served by the same model.")
    changed = [t.query_id for t in run.trials if t.outcome == "model_changed"]
    if changed:
        raise SubmissionError(f"{run.run_id}: {len(changed)} trial(s) changed model "
                              f"mid-trial, e.g. {changed[0]}")


def agentic_entry(run: AgenticRun, queries_path: Path,
                  require_complete: bool = True, manifest: Path = MANIFEST,
                  transcripts: dict[str, list[dict[str, Any]]] | None = None
                  ) -> dict[str, Any]:
    """Rescore `run` against `queries_path` and return its leaderboard entry.

    Raises SubmissionError when the run is not reportable, or (with require_complete,
    as for a new submission) when it skipped any query in the file. Without it, as for
    listing older runs, the entry records its coverage instead.
    """
    if not run.reportable:
        raise SubmissionError(f"{run.run_id} is marked not reportable (a --partial run on an "
                              "incomplete corpus)")
    check_trial_grid(run)
    check_model_identity(run)
    queries = load_queries_file(queries_path)
    if require_complete:
        # A new submission must be the current release, end to end: its receipt names
        # the manifest's bytes and its key matches one recomputed here.
        assets = json.loads(manifest.read_text())
        want = {a["asset_id"]: a["file"]["sha256"] for a in assets}
        if run.corpus_receipt != want:
            bad = sorted(set(want) ^ set(run.corpus_receipt)
                         | {a for a in want if run.corpus_receipt.get(a) not in (None, want[a])})
            raise SubmissionError(
                f"{run.run_id}: corpus receipt does not match {manifest.name} "
                f"({len(run.corpus_receipt)} of {len(want)} assets verified; "
                f"differs on {', '.join(bad[:3]) or 'none'}). Rerun on the verified corpus.")
        key = run.identity.get("compatibility_key")
        if key != expected_key(run, queries, manifest, transcripts):
            raise SubmissionError(
                f"{run.run_id}: compatibility key {key} does not match the current "
                f"release of {queries_path.name}; rerun with the current harness and files.")
    scored = rescore(run, queries)
    covered = {t.query_id for t in scored.trials}
    missing = sorted(set(queries) - covered)
    if missing and require_complete:
        raise SubmissionError(
            f"{run.run_id} covers {len(covered)} of {len(queries)} queries in "
            f"{queries_path.name} (missing {', '.join(missing[:3])}"
            f"{', ...' if len(missing) > 3 else ''}). Rerun on the whole file.")
    s = scored.summary
    return {
        "track": "agentic",
        "run_id": run.run_id,
        "model_id": run.model_id,
        "effort": run.effort,
        "harness": run.harness,
        "resolved_models": resolved_models(run.trials),
        "modalities": run.modalities,
        "queries_file": queries_path.name,
        "num_queries": s.num_queries,
        "queries_in_file": len(queries),
        "rescore_excluded": scored.rescore_excluded,
        "num_trials": s.num_trials,
        "recall_at_1": round(s.recall_at_1, 4),
        "recall_at_1_ci95": [round(x, 4) for x in s.recall_at_1_ci95],
        "positive_recall_at_1": s.positive_recall_at_1,
        "no_answer_accuracy": s.no_answer_accuracy,
        "false_positive_rate": s.false_positive_rate,
        "mean_cost_usd_per_query": s.mean_cost_usd_per_query,
        "by_tier": s.by_tier,
        "budget": run.budget.model_dump(),
        "corpus_assets_verified": len(run.corpus_receipt),
        "compatibility_key": run.identity.get("compatibility_key"),
        "identity": run.identity,
        "timestamp": run.timestamp,
    }
