"""Validate submission directories the way `momentbench submit` does.

    python scripts/validate_submissions.py submissions/team-a submissions/team-b

Each directory holds `run.json` and `submission.json` (written by `momentbench submit`).
The run is rescored against the query file named in submission.json, from
data/queries or data/queries/scored, with every admission check: reportable, trial
grid, model identity, corpus receipt and compatibility key. Prints a Markdown report
(the CI job summary) and exits 1 if any submission fails.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from momentbench.agentic.run import AgenticRun  # noqa: E402
from momentbench.submission import SubmissionError, agentic_entry  # noqa: E402


def check(directory: Path) -> tuple[bool, str]:
    try:
        meta = json.loads((directory / "submission.json").read_text())
        run = AgenticRun.model_validate_json((directory / "run.json").read_text())
    except (OSError, ValueError) as e:
        return False, f"cannot read run.json and submission.json: {e}"
    name = meta.get("queries_file", "")
    qfile = next((p for p in (ROOT / "data/queries" / name,
                              ROOT / "data/queries/scored" / name) if p.is_file()), None)
    if qfile is None:
        return False, f"query file {name!r} is not in data/queries or data/queries/scored"
    try:
        entry = agentic_entry(run, qfile)
    except SubmissionError as e:
        return False, str(e)
    return True, (f"{entry['model_id']} on {name}: Recall@1 {100 * entry['recall_at_1']:.1f}, "
                  f"net {entry.get('net_score') or 0:+.2f}, key {entry['compatibility_key']}")


def main(dirs: list[str]) -> int:
    lines, ok_all = ["## Submission check", ""], True
    for d in dirs:
        ok, msg = check(Path(d))
        ok_all &= ok
        lines.append(f"- {'PASS' if ok else 'FAIL'} `{d}`: {msg}")
    if not dirs:
        lines.append("No submission directories changed.")
    print("\n".join(lines))
    return 0 if ok_all else 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
