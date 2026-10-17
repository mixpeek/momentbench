"""Write the machine-readable release manifest: everything needed to reproduce a number.

    python scripts/release_manifest.py            # data/release/<version>.json

Ties together, with full sha256 hashes: the code commits the runs were made with and
the packaging commit; Python, ffmpeg and dependency versions (data/release/
requirements-<version>.txt); the corpus manifest; the scored query files and the label
log; the leaderboard; every run file with the command that reproduces its row; and the
diagnostic profile's expected outputs. Generated last, before the tag; the publish
loop treats its existence as the release gate.
"""

from __future__ import annotations

import hashlib
import json
import platform
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import momentbench  # noqa: E402
from momentbench.agentic.run import AgenticRun  # noqa: E402


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def rel(path: Path) -> str:
    return str(path.relative_to(ROOT))


def command(run: AgenticRun, qfile: str) -> str:
    """The CLI call that reproduces a record row."""
    flags = [f"momentbench agentic {run.model_id} --provider "
             + ({"claude-code": "claude-code"}.get(run.harness)
                or ("anthropic" if run.model_id.startswith("claude")
                    else "openai" if run.model_id.startswith("gpt")
                    else "gemini")),
             f"-q {qfile}", "-c data/corpus_manifests/sample_manifest.json",
             "--corpus-dir corpus/", "--transcripts-dir data/transcripts",
             f"--trials {run.summary.num_trials}"]
    if run.anonymize_titles:
        flags.append("--anonymize-titles")
    if run.identity.get("prompt_condition", "standard") != "standard":
        flags.append(f"--prompt-condition {run.identity['prompt_condition']}")
    if run.effort:
        flags.append(f"--effort {run.effort}")
    return " ".join(flags)


def main() -> None:
    version = momentbench.__version__
    out_dir = ROOT / "data/release"
    out_dir.mkdir(parents=True, exist_ok=True)
    py = Path(sys.executable)
    freeze = subprocess.run([str(py), "-m", "pip", "freeze", "--exclude-editable"],
                            capture_output=True, text=True).stdout
    req = out_dir / f"requirements-{version}.txt"
    req.write_text(freeze)
    ffmpeg = subprocess.run(["ffmpeg", "-version"], capture_output=True,
                            text=True).stdout.splitlines()[0]
    board = json.loads((ROOT / "docs/leaderboard.json").read_text())
    runs, commits = [], set()
    for g in board["groups"]:
        for r in g["rows"]:
            path = ROOT / r["file"]
            run = AgenticRun.model_validate_json(path.read_text())
            commits.add(r.get("code_commit"))
            runs.append({"file": r["file"], "sha256": sha(path), "group": g["compatibility_key"],
                         "model_id": r["model_id"], "harness": r["harness"],
                         "code_commit": r.get("code_commit"),
                         "command": command(run, f"data/queries/scored/{g['queries_file']}")})
    manifest = {
        "benchmark": "momentbench", "version": version,
        "packaging_commit": subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT,
                                           capture_output=True, text=True).stdout.strip(),
        "run_code_commits": sorted(c for c in commits if c),
        "environment": {"python": platform.python_version(), "ffmpeg": ffmpeg,
                        "requirements": {"file": rel(req), "sha256": sha(req)}},
        "corpus_manifest": {"file": "data/corpus_manifests/sample_manifest.json",
                            "sha256": sha(ROOT / "data/corpus_manifests/sample_manifest.json")},
        "scored_queries": {rel(p): sha(p)
                           for p in sorted((ROOT / "data/queries/scored").glob("*.json"))},
        "label_log": {"file": "data/labels/adjudication.json",
                      "sha256": sha(ROOT / "data/labels/adjudication.json")},
        "leaderboard": {"file": "docs/leaderboard.json",
                        "sha256": sha(ROOT / "docs/leaderboard.json"),
                        "primary_keys": board["primary_keys"]},
        "diagnostic": {rel(p): sha(p) for p in sorted((ROOT / "data/diagnostic").glob("*.json"))},
        "heldout_commitments": {"file": "data/heldout_commitments.json",
                                "sha256": sha(ROOT / "data/heldout_commitments.json")},
        "runs": runs,
        "limitations": "docs/labels.md (model adjudication, absence-screen recall 12 of 26), "
                       "docs/scope.md",
    }
    out = out_dir / f"{version}.json"
    out.write_text(json.dumps(manifest, indent=1) + "\n")
    print(f"wrote {rel(out)}: {len(runs)} runs, {len(manifest['scored_queries'])} query files")


if __name__ == "__main__":
    main()
