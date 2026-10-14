"""Merge a run with a rerun of the queries reworded since, into one full-coverage run.

    python scripts/merge_runs.py BASE.json PATCH.json -q data/queries/film_v0.2.json \\
        -o data/results/agentic-v0.2.1/record

A run is scored only on queries whose wording it saw, so a rewording leaves a hole in
older runs. Rerunning just the reworded queries (PATCH) and merging fills it without
rerunning the rest. The merge is refused unless both runs share everything that
defines the task apart from the query selection: model, harness, effort, budget,
modalities, conditions, verified corpus bytes, transcripts, prompt and tools. The
result keeps BASE trials whose text still matches, adds PATCH trials, must cover the
query file exactly once per trial, and gets the compatibility key of the full file.
It records `merged_from`, so it never reads as a single uninterrupted run.
"""

from __future__ import annotations

import argparse
import sys
import time
import uuid
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from momentbench.agentic.identity import queries_sha, text_sha  # noqa: E402
from momentbench.agentic.run import AgenticRun, rescore, summarize  # noqa: E402
from momentbench.submission import (  # noqa: E402
    MANIFEST,
    check_trial_grid,
    expected_key,
    load_queries_file,
)

SAME_TASK = ("corpus_sha", "transcripts_sha", "system_prompt_sha", "tools_sha", "budget",
             "modalities", "anonymize_titles")


def merge(base: AgenticRun, patch: AgenticRun, queries_path: Path,
          manifest: Path = MANIFEST, transcripts: dict | None = None) -> AgenticRun:
    for field in ("model_id", "harness", "effort", "modalities", "anonymize_titles"):
        if getattr(base, field) != getattr(patch, field):
            raise SystemExit(f"{field} differs: {getattr(base, field)!r} vs "
                             f"{getattr(patch, field)!r}")
    if base.budget != patch.budget or base.corpus_receipt != patch.corpus_receipt:
        raise SystemExit("budget or corpus receipt differs")
    for key in SAME_TASK:
        if base.identity.get(key) != patch.identity.get(key):
            raise SystemExit(f"identity.{key} differs; these runs are not the same task")
    if not (base.reportable and patch.reportable):
        raise SystemExit("both runs must be reportable")
    queries = load_queries_file(queries_path)
    current = {qid: text_sha(q.text) for qid, q in queries.items()}
    patched = {t.query_id for t in patch.trials}
    kept = [t for t in base.trials if t.query_id in current and t.query_id not in patched
            and t.query_text_sha == current[t.query_id]]
    added = [t for t in patch.trials if current.get(t.query_id) == t.query_text_sha]
    if len(added) != len(patch.trials):
        raise SystemExit("PATCH has trials whose text does not match the current file")
    n = base.summary.num_trials
    merged = base.model_copy(update={
        "run_id": f"agentic-{base.model_id}-{uuid.uuid4().hex[:8]}",
        "timestamp": max(base.timestamp, patch.timestamp),
        "trials": kept + added,
    })
    check_trial_grid(merged)
    missing = set(current) - {t.query_id for t in merged.trials}
    if missing:
        raise SystemExit(f"{len(missing)} queries still uncovered, e.g. {sorted(missing)[:3]}")
    merged = rescore(merged, queries)
    identity = {**base.identity,
                "compatibility_key": expected_key(merged, queries, manifest, transcripts),
                "queries_sha": queries_sha(list(queries.values())),
                "num_queries": len(queries),
                "resolved_models": sorted(set(base.identity.get("resolved_models", []))
                                          | set(patch.identity.get("resolved_models", []))),
                "merged_from": [base.run_id, patch.run_id],
                "merged_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
    return merged.model_copy(update={"identity": identity,
                                     "summary": summarize(merged.trials, n)})


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("base")
    p.add_argument("patch")
    p.add_argument("-q", "--queries", required=True)
    p.add_argument("-o", "--output", required=True)
    args = p.parse_args()
    base = AgenticRun.model_validate_json(Path(args.base).read_text())
    patch = AgenticRun.model_validate_json(Path(args.patch).read_text())
    run = merge(base, patch, Path(args.queries))
    out = Path(args.output) / f"{run.run_id}.json"
    out.write_text(run.model_dump_json(indent=2))
    s = run.summary
    print(f"{out}: {run.model_id} {s.num_queries} x {s.num_trials}, "
          f"R@1 {100 * s.recall_at_1:.1f}, key {run.identity['compatibility_key']}")


if __name__ == "__main__":
    main()
