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

from momentbench.agentic.run import AgenticRun, rescore
from momentbench.types import BenchmarkRun, Query

QUERIES_DIR = Path(__file__).resolve().parents[1] / "data" / "queries"


class SubmissionError(ValueError):
    """A run that cannot go on the leaderboard, with the reason."""


def load_any_run(path: Path) -> AgenticRun | BenchmarkRun:
    data = json.loads(path.read_text())
    if "trials" in data:
        return AgenticRun.model_validate(data)
    return BenchmarkRun.model_validate(data)


def load_queries_file(path: Path) -> dict[str, Query]:
    return {q["query_id"]: Query(**q) for q in json.loads(path.read_text())}


def find_queries_file(run: AgenticRun, search: Path = QUERIES_DIR) -> Path:
    """The published query file the run answered: the one holding most of its query ids.

    Older runs can include queries since removed by review, so containment is not
    required; at least half of the run's ids must be in the file.
    """
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


def agentic_entry(run: AgenticRun, queries_path: Path,
                  require_complete: bool = True) -> dict[str, Any]:
    """Rescore `run` against `queries_path` and return its leaderboard entry.

    Raises SubmissionError when the run is not reportable, or (with require_complete,
    as for a new submission) when it skipped any query in the file. Without it, as for
    listing older runs, the entry records its coverage instead.
    """
    if not run.reportable:
        raise SubmissionError(f"{run.run_id} is marked not reportable (a --partial run on an "
                              "incomplete corpus)")
    queries = load_queries_file(queries_path)
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
