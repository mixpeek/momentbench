"""Commit to the held-out queries now, so they cannot be changed unseen later.

    python scripts/heldout_commit.py commit CORPUS/heldout   # data/heldout_commitments.json
    python scripts/heldout_commit.py verify CORPUS/heldout   # after the reveal

The held-out queries and their gold stay private; Mixpeek holds them. Publishing a
salted SHA-256 of each query (its full JSON: text, tier, gold spans, everything) means
any later change to a held-out query or its answer is detectable when the queries are
revealed. The salt is random per query and stays with the private files
(`heldout/commitment_salts.json`), because gold spans are short enough to guess from an
unsalted hash. At reveal, publish the queries and the salts; `verify` recomputes
every commitment.
"""

from __future__ import annotations

import hashlib
import json
import secrets
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data/heldout_commitments.json"
FILES = ("film_v0.2_held_out.json", "aerospace_v0.2_held_out.json",
         "robotics_v0.2_held_out.json")


def canonical(query: dict) -> bytes:
    return json.dumps(query, sort_keys=True, ensure_ascii=False,
                      separators=(",", ":")).encode()


def digest(salt: str, query: dict) -> str:
    return hashlib.sha256(bytes.fromhex(salt) + canonical(query)).hexdigest()


def load(heldout: Path) -> list[dict]:
    return [q for name in FILES for q in json.loads((heldout / name).read_text())]


def commit(heldout: Path) -> None:
    salts_path = heldout / "commitment_salts.json"
    salts = json.loads(salts_path.read_text()) if salts_path.exists() else {}
    queries = load(heldout)
    for q in queries:
        salts.setdefault(q["query_id"], secrets.token_hex(16))
    salts_path.write_text(json.dumps(salts, indent=1) + "\n")
    out = {
        "scheme": "sha256(salt_bytes || canonical_json(query)); canonical_json = "
                  "json.dumps(query, sort_keys=True, ensure_ascii=False, "
                  "separators=(',', ':')); one random 16-byte salt per query",
        "files": list(FILES),
        "count": len(queries),
        "commitments": {q["query_id"]: digest(salts[q["query_id"]], q)
                        for q in sorted(queries, key=lambda q: q["query_id"])},
    }
    OUT.write_text(json.dumps(out, indent=1) + "\n")
    print(f"committed {len(queries)} held-out queries -> {OUT}")


def verify(heldout: Path) -> int:
    salts = json.loads((heldout / "commitment_salts.json").read_text())
    published = json.loads(OUT.read_text())["commitments"]
    bad = [q["query_id"] for q in load(heldout)
           if published.get(q["query_id"]) != digest(salts[q["query_id"]], q)]
    print(f"{len(published) - len(bad)} of {len(published)} match" +
          (f"; changed: {', '.join(bad[:10])}" if bad else ""))
    return 1 if bad else 0


if __name__ == "__main__":
    if len(sys.argv) != 3 or sys.argv[1] not in ("commit", "verify"):
        sys.exit(__doc__)
    path = Path(sys.argv[2])
    sys.exit(commit(path) if sys.argv[1] == "commit" else verify(path))
