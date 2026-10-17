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
import shlex
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import momentbench  # noqa: E402
from momentbench.agentic.budget import Budget  # noqa: E402
from momentbench.agentic.run import AgenticRun  # noqa: E402


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def rel(path: Path) -> str:
    return str(path.relative_to(ROOT))


def command(run: AgenticRun, qfile: str) -> tuple[str, str]:
    """(command, source). A run made after 0.2.5 records its own CLI arguments
    (identity.cli), which are used as is; older runs get a reconstruction from their
    recorded identity, labelled as such."""
    env = (f"MOMENTBENCH_SPEND_LEDGER=<ledger> MOMENTBENCH_MAX_TRIAL_INPUT_TOKENS="
           f"{run.identity['input_token_cap']} " if run.identity.get("input_token_cap") else "")
    if run.identity.get("cli"):
        return env + shlex.join(["momentbench", *run.identity["cli"]]), "recorded"
    if run.harness == "predictions":
        return (f"momentbench score-predictions -p <predictions.json> -q {qfile}",
                "reconstructed")
    provider = run.identity.get("provider") or (
        "claude-code" if run.harness == "claude-code" else None)
    if provider is None:
        return "", "unsupported: the run identity does not record its provider"
    flags = [f"momentbench agentic {shlex.quote(run.model_id)} --provider {provider}",
             f"-q {qfile}", "-c data/corpus_manifests/sample_manifest.json",
             "--corpus-dir corpus/", "--transcripts-dir data/transcripts",
             f"--trials {run.summary.num_trials}"]
    if run.anonymize_titles:
        flags.append("--anonymize-titles")
    if run.identity.get("prompt_condition", "standard") != "standard":
        flags.append(f"--prompt-condition {run.identity['prompt_condition']}")
    if run.effort:
        flags.append(f"--effort {run.effort}")
    default = Budget()
    for field, flag in (("max_steps", "--max-steps"), ("max_frames", "--max-frames"),
                        ("max_tool_calls", "--max-tool-calls")):
        if getattr(run.budget, field) != getattr(default, field):
            flags.append(f"{flag} {getattr(run.budget, field)}")
    if run.modalities == "frames":
        flags.append("--no-transcripts")
    if run.identity.get("frames"):
        f = run.identity["frames"]
        flags.append(f"--frame-width {f['width']}")
        if f.get("jpeg_quality"):
            flags.append(f"--jpeg-quality {f['jpeg_quality']}")
    if run.identity.get("endpoint"):
        flags.append(f"--base-url {run.identity['endpoint']}")
    return env + " ".join(flags), "reconstructed from the run identity"


def main() -> None:
    version = momentbench.__version__
    out_dir = ROOT / "data/release"
    out_dir.mkdir(parents=True, exist_ok=True)
    py = Path(sys.executable)
    frozen = subprocess.run([str(py), "-m", "pip", "freeze", "--exclude-editable"],
                            capture_output=True, text=True)
    if frozen.returncode != 0 or not frozen.stdout.strip():
        sys.exit(f"pip freeze failed: {frozen.stderr[:200]}")
    freeze = frozen.stdout
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
            qpath = next(p for p in (ROOT / "data/queries/scored" / g["queries_file"],
                                     ROOT / "data/queries" / g["queries_file"]) if p.exists())
            cmd, source = command(run, rel(qpath))
            runs.append({"file": r["file"], "sha256": sha(path), "group": g["compatibility_key"],
                         "model_id": r["model_id"], "harness": r["harness"],
                         "code_commit": r.get("code_commit"), "command": cmd,
                         "command_source": source, "recall_at_1": r["recall_at_1"],
                         "positive_recall_at_1": r["positive_recall_at_1"]})
    # The release is frozen as archived copies; the current leaderboard may change later.
    archive_dir = ROOT / "docs/releases" / version
    archive_dir.mkdir(parents=True, exist_ok=True)
    archive = {}
    for src in ("docs/leaderboard.json", "docs/report.md", "docs/labels.md"):
        dst = archive_dir / Path(src).name
        dst.write_bytes((ROOT / src).read_bytes())
        archive[Path(src).name] = {"file": rel(dst), "sha256": sha(dst)}
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
        "archive": archive,
        "runs": runs,
        "limitations": "docs/labels.md (model adjudication, absence-screen recall 12 of 26), "
                       "docs/scope.md",
    }
    out = out_dir / f"{version}.json"
    out.write_text(json.dumps(manifest, indent=1) + "\n")
    print(f"wrote {rel(out)}: {len(runs)} runs, {len(manifest['scored_queries'])} query files")


if __name__ == "__main__":
    main()
