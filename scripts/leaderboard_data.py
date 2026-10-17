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
from momentbench.agentic.stats import bootstrap_ci  # noqa: E402
from momentbench.submission import SubmissionError, agentic_entry, find_queries_file  # noqa: E402

RECORD = ROOT / "data/results/agentic-v0.2.5/record"
OUT = ROOT / "docs/leaderboard.json"
SPEC = json.loads((ROOT / f"configs/release_{momentbench.__version__}.json").read_text()) \
    if (ROOT / f"configs/release_{momentbench.__version__}.json").exists() else json.loads(
        sorted((ROOT / "configs").glob("release_*.json"))[-1].read_text())


def sha16(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()[:16]


INPUTS = ["data/results/agentic-v0.2.5", "submissions", "data/queries", "configs",
          "scripts/leaderboard_data.py", "momentbench"]


def git_head() -> str:
    """The last commit that changed an input of the leaderboard. Publishing commits
    only the outputs, so this stays fixed until the inputs change."""
    return subprocess.run(["git", "log", "-1", "--format=%h", "--", *INPUTS], cwd=ROOT,
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
        pos = {}
        for t in run.trials:
            if not t.no_answer:
                pos.setdefault(t.query_id, []).append(float(t.hit_at_1))
        per_q = [sum(v) / len(v) for v in pos.values()]
        g["rows"].append({
            # Interval over queries (query sampling); run-to-run variation is separate.
            "positive_recall_at_1_ci95": list(bootstrap_ci(per_q)) if per_q else None,
            **{k: entry.get(k) for k in (
                "model_id", "harness", "resolved_models", "num_queries", "num_trials",
                "positive_recall_at_1", "recall_at_1", "recall_at_1_ci95", "net_score",
                "no_answer_accuracy", "false_positive_rate", "mean_cost_usd_per_query",
                "run_id")},
            "origin": origin, "file": rel,
            "code_commit": (run.identity.get("git") or {}).get("commit"),
            "timestamp": run.timestamp})
    # Planned groups come from the frozen release spec, so a domain with no run of a
    # condition still lists its planned rows as not run.
    for domain, qname in SPEC["domains"].items():
        for prompt in SPEC["planned"]:
            cond = f"anonymized titles, {prompt} prompt"
            if not any(g["queries_file"] == qname and g["condition"] == cond
                       for g in groups.values()):
                groups[f"planned:{domain}:{prompt}"] = {
                    "compatibility_key": None, "queries_file": qname, "domain": domain,
                    "track": "agentic", "condition": cond, "rows": []}
    for g in groups.values():
        g["rows"].sort(key=lambda r: -(r["positive_recall_at_1"] or 0))
        prompt = "abstain" if "abstain prompt" in g["condition"] else "standard"
        if g["track"] == "agentic" and g["queries_file"] in SPEC["domains"].values() \
                and "anonymized" in g["condition"]:
            have = {(r["model_id"], r["harness"]) for r in g["rows"]}
            g["not_run"] = [
                {"model_id": m, "harness": h,
                 "reason": SPEC["not_run"].get(f"{g['domain']}/{m}/{h}", "not run yet")}
                for m, h in SPEC["planned"][prompt] if (m, h) not in have]
    # The primary group per domain is pinned to the key of the release's reference row,
    # so a submission under another condition can never replace it.
    ref_model, ref_harness = SPEC["primary"]["reference_row"]
    primary_keys = {}
    for g in groups.values():
        if g["condition"] == SPEC["primary"]["condition"] and any(
                r["model_id"] == ref_model and r["harness"] == ref_harness and
                r["origin"] == "record" for r in g["rows"]):
            primary_keys[g["domain"]] = g["compatibility_key"]
    return {"benchmark": "momentbench", "release": momentbench.__version__,
            "generated_at_commit": git_head(),
            "ranking_rule": "rows are compared only within a group (one compatibility key)",
            "primary_keys": primary_keys,
            "notes": {"no_answer": "No-answer accuracy and net score are exploratory: the "
                      "corpus-wide absence screen recovered 12 of 26 known positives "
                      "(docs/labels.md)."},
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
