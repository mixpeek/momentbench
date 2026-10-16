"""Bind recorded reviews to the annotation version they saw, from git history.

    python scripts/bind_reviews.py data/queries/*_v0.2.json

Run once on the 0.2.2 query files (audit 3, finding 5). A cross-check is bound to the
annotation in the commit where its result was written; cross_verify.py skips queries
that already agree, so the result is not rewritten when the annotation changes later.
A model or human review is bound to the annotation before the commit that recorded it
(`annotation_sha`, what it reviewed) and in that commit (`result_sha`, what it left).
A cross-check whose annotation has changed since moves to `cross_check_history`.
New reviews record these hashes when they are written (review_queries.py,
cross_verify.py), so this script is not needed again.
"""

from __future__ import annotations

import argparse
import datetime
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from momentbench.annotation import annotation_sha, invalidate_cross_check  # noqa: E402


def versions(path: Path) -> list[tuple[str, dict[str, dict]]]:
    rel = path.resolve().relative_to(ROOT)
    commits = subprocess.run(["git", "log", "--reverse", "--format=%h", "--", str(rel)],
                             cwd=ROOT, capture_output=True, text=True, check=True).stdout.split()
    out = []
    for c in commits:
        show = subprocess.run(["git", "show", f"{c}:{rel}"], cwd=ROOT, capture_output=True,
                              text=True)
        if show.returncode == 0:
            out.append((c, {q["query_id"]: q for q in json.loads(show.stdout)}))
    return out


def bind(path: Path) -> dict[str, int]:
    history = versions(path)
    queries = json.loads(path.read_text())
    today = datetime.date.today().isoformat()
    counts = {"cross_check_current": 0, "cross_check_superseded": 0, "reviews_bound": 0}
    for q in queries:
        qid, meta = q["query_id"], q["metadata"]
        seen: dict[str, tuple[str, str]] = {}  # field -> (reviewed sha, result sha)
        prev_value: dict[str, str] = {}
        before: dict | None = None
        for _, snapshot in history:
            old = snapshot.get(qid)
            if old is None:
                before = None
                continue
            for field in ("cross_check", "model_review", "human_review"):
                value = json.dumps(old["metadata"].get(field), sort_keys=True)
                if old["metadata"].get(field) and value != prev_value.get(field):
                    if field == "cross_check":  # written onto the annotation it checked
                        seen[field] = (annotation_sha(old), annotation_sha(old))
                    else:  # reviewed the previous version, left this one
                        reviewed = annotation_sha(before) if before else annotation_sha(old)
                        seen[field] = (reviewed, annotation_sha(old))
                prev_value[field] = value
            before = old
        if "cross_check" in meta and "cross_check" in seen:
            meta["cross_check"]["annotation_sha"] = seen["cross_check"][0]
            if invalidate_cross_check(q, "text or gold span edited after this check", today):
                counts["cross_check_superseded"] += 1
            else:
                counts["cross_check_current"] += 1
        for field in ("model_review", "human_review"):
            if field in meta and field in seen:
                meta[field]["annotation_sha"], meta[field]["result_sha"] = seen[field]
                counts["reviews_bound"] += 1
    path.write_text(json.dumps(queries, indent=2, ensure_ascii=False) + "\n")
    return counts


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("queries", nargs="+")
    for f in p.parse_args().queries:
        print(f, json.dumps(bind(Path(f))))


if __name__ == "__main__":
    main()
