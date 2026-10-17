"""Build docs/leaderboard.json: validated rows, ranked only within a compatibility group.

    python scripts/leaderboard_data.py

Sources: the 0.2.5 record runs (`data/results/agentic-v0.2.5/record/<domain>/`) and
merged submissions (`submissions/*/run.json`). Every run is validated with the same
admission checks as `momentbench submit` against the query file it answered exactly
(a scored file or a full file). A run that fails is listed under `excluded` with the
reason and never ranked.

Rows are grouped by compatibility key, so a group's rows ran the same task: the same
queries and text, verified corpus, transcripts, prompt, tools, budget and conditions.
Groups are never merged or intersected, so adding a run cannot change another row's
evaluation set. Each group lists the rows the release plan expected and marks the ones
that did not run. Positive-query localization leads; no-answer accuracy and net score
are reported with the caveats in docs/labels.md.
"""

from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import momentbench  # noqa: E402
from momentbench.agentic.run import AgenticRun  # noqa: E402
from momentbench.submission import SubmissionError, agentic_entry, find_queries_file  # noqa: E402

RECORD = ROOT / "data/results/agentic-v0.2.5/record"
OUT = ROOT / "docs/leaderboard.json"
# Rows the 0.2.5 release planned per scored domain file (distribution plan, A6).
EXPECTED = {
    "standard": [("claude-opus-5-5", "claude-code"), ("claude-sonnet-5-5", "claude-code"),
                 ("claude-haiku-5-5", "claude-code"), ("claude-haiku-5-5", "api"),
                 ("gpt-5.4-mini", "api"), ("gemini-3.8-flash", "api")],
    "abstain": [("claude-opus-5-5", "claude-code"), ("claude-sonnet-5-5", "claude-code"),
                ("claude-haiku-5-5", "claude-code")],
}


def sha16(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()[:16]


def git_head() -> str:
    return subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=ROOT,
                          capture_output=True, text=True).stdout.strip()


def condition(run: AgenticRun) -> str:
    if run.harness == "predictions":
        return "prediction-file track"
    bits = ["anonymized titles" if run.anonymize_titles else "real titles"]
    if run.modalities == "frames":
        bits.append("frames only")
    bits.append(f"{run.identity.get('prompt_condition', 'standard')} prompt")
    if run.identity.get("frames"):
        bits.append(f"frames {run.identity['frames']['width']} px")
    return ", ".join(bits)


def sources() -> list[tuple[Path, str]]:
    found = [(p, "record") for p in sorted(RECORD.glob("*/*.json"))
             if not p.name.endswith(".partial.json")]
    found += [(p, "submission") for p in sorted((ROOT / "submissions").glob("*/run.json"))]
    return found


def build() -> dict:
    groups: dict[str, dict] = {}
    excluded = []
    for path, origin in sources():
        rel = str(path.relative_to(ROOT))
        try:
            run = AgenticRun.model_validate_json(path.read_text())
            qfile = find_queries_file(run, ROOT / "data/queries")
            entry = agentic_entry(run, qfile)
        except (SubmissionError, ValueError) as e:
            excluded.append({"file": rel, "reason": str(e)[:300]})
            continue
        key = entry["compatibility_key"]
        g = groups.setdefault(key, {
            "compatibility_key": key, "queries_file": qfile.name,
            "queries_file_sha256_16": sha16(qfile),
            "domain": qfile.name.split("_")[0], "track": "predictions"
            if run.harness == "predictions" else "agentic",
            "condition": condition(run), "rows": []})
        g["rows"].append({
            **{k: entry.get(k) for k in (
                "model_id", "harness", "resolved_models", "num_queries", "num_trials",
                "positive_recall_at_1", "recall_at_1", "recall_at_1_ci95", "net_score",
                "no_answer_accuracy", "false_positive_rate", "mean_cost_usd_per_query",
                "run_id")},
            "origin": origin, "file": rel,
            "code_commit": (run.identity.get("git") or {}).get("commit"),
            "timestamp": run.timestamp})
    for g in groups.values():
        g["rows"].sort(key=lambda r: -(r["positive_recall_at_1"] or 0))
        prompt = "abstain" if "abstain prompt" in g["condition"] else "standard"
        if g["track"] == "agentic" and g["queries_file"].endswith("_scored.json") \
                and "anonymized" in g["condition"]:
            have = {(r["model_id"], r["harness"]) for r in g["rows"]}
            g["not_run"] = [{"model_id": m, "harness": h} for m, h in EXPECTED[prompt]
                            if (m, h) not in have]
    return {"benchmark": "momentbench", "release": momentbench.__version__,
            "generated_at_commit": git_head(),
            "ranking_rule": "rows are compared only within a group (one compatibility key)",
            "groups": sorted(groups.values(), key=lambda g: (g["domain"], g["track"],
                                                            g["condition"])),
            "excluded": excluded}


def main() -> None:
    data = build()
    OUT.write_text(json.dumps(data, indent=1) + "\n")
    n = sum(len(g["rows"]) for g in data["groups"])
    print(f"{len(data['groups'])} groups, {n} rows, {len(data['excluded'])} excluded")


if __name__ == "__main__":
    main()
