"""The prediction-file track: score a system's ranked answers without the agent loop.

A retrieval or grounding system that already produces ranked moments does not need to
be wrapped in a tool-using agent. It writes one JSON file:

    {"system": "my-retriever-v2",
     "predictions": [
        {"query_id": "q-film-0100", "moments": [
            {"asset_id": "big-buck-bunny", "t_in": 63.8, "t_out": 68.0}, ...]},
        {"query_id": "q-film-0102", "moments": []}]}

An empty `moments` list abstains ("nothing in the corpus matches"). At most five moments
count, in rank order, scored with the same rules as the agentic track: Recall@1 at tIoU
0.5, Recall@5, net score, per-tier results and bootstrap intervals. An invalid entry
(an unknown video, an impossible span) rejects that query's whole ranking, and a query
with no prediction scores zero; neither can count as a correct abstention.

The result is an ordinary run file with `harness: "predictions"`, so `momentbench
submit`, the results tables and the leaderboard read it unchanged. Its compatibility
key covers the query set and text, the declared corpus, and the track, so it never
ranks against an agentic row. The corpus receipt is the manifest's hashes as
declared by the submitter: this track does not see which bytes the system indexed.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

from momentbench.agentic.budget import Budget, Usage
from momentbench.agentic.environment import InvalidSubmissionError, parse_submission
from momentbench.agentic.identity import new_run_id, task_identity, text_sha
from momentbench.agentic.run import AgenticRun, AgenticTrial, score_trial, summarize
from momentbench.runner import load_corpus_assets, load_queries

TRACK = "predictions"
TRACK_PROMPT = "prediction-file track: ranked moments per query, no agent loop"


def task(queries: list, receipt: dict[str, str]) -> dict[str, Any]:
    """Task identity for the track; submission recomputes it the same way."""
    return task_identity(queries=queries, receipt=receipt, transcripts={},
                         system_prompt=TRACK_PROMPT, tools=[], budget={},
                         modalities=TRACK)


def load_predictions(path: Path) -> tuple[str, dict[str, Any]]:
    data = json.loads(Path(path).read_text())
    system = str(data.get("system") or Path(path).stem)
    rows = data.get("predictions", [])
    seen: dict[str, Any] = {}
    for row in rows:
        qid = str(row.get("query_id"))
        if qid in seen:
            raise ValueError(f"query {qid} is predicted more than once")
        seen[qid] = row
    return system, seen


def score_predictions(predictions_path: Path, queries_path: Path, manifest: Path,
                      output_dir: Path | None = None) -> AgenticRun:
    system, preds = load_predictions(predictions_path)
    queries = load_queries(Path(queries_path))
    known = {q.query_id for q in queries}
    unknown = sorted(set(preds) - known)
    if unknown:
        raise ValueError(f"{len(unknown)} predictions for queries not in "
                         f"{Path(queries_path).name}, e.g. {unknown[:3]}")
    assets = {a["asset_id"]: a for a in load_corpus_assets(Path(manifest))}
    trials = []
    for q in queries:
        row = preds.get(q.query_id)
        error = None
        if row is None:
            moments, outcome, error = [], "no_submission", "no prediction for this query"
        else:
            try:
                moments, outcome = parse_submission({"moments": row.get("moments")}, assets), \
                    "submitted"
            except InvalidSubmissionError as e:
                moments, outcome, error = [], "invalid_submission", str(e)
        hit1, hit5, best, top1 = score_trial(q, moments, outcome)
        trials.append(AgenticTrial(
            query_id=q.query_id, trial=0, tier=int(q.tier), vertical=q.vertical,
            no_answer=q.is_no_answer, query_text_sha=text_sha(q.text), outcome=outcome,
            hit_at_1=hit1, hit_at_5=hit5, best_temporal_iou=best, top1_iou=top1,
            usage=Usage(), latency_s=0.0, moments=moments, error=error))
    receipt = {aid: a["file"]["sha256"] for aid, a in assets.items()}
    run = AgenticRun(
        run_id=new_run_id(system, prefix="predictions"), model_id=system, effort=None,
        harness=TRACK, modalities=TRACK, corpus_assets=len(assets), reportable=True,
        corpus_receipt=receipt,
        identity={**task(queries, receipt), "track": TRACK, "corpus_receipt": "declared",
                  "predictions_file_sha256": text_sha(Path(predictions_path).read_text())},
        budget=Budget(), pricing=None,
        timestamp=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        summary=summarize(trials, 1), trials=trials)
    if output_dir:
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        (Path(output_dir) / f"{run.run_id}.json").write_text(run.model_dump_json(indent=2))
    return run
