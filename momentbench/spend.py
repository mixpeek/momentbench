"""A shared cash ledger with a hard cap, for maintainers running paid evaluations.

Off unless `MOMENTBENCH_SPEND_LEDGER` names a ledger file, so nobody else using the
package is affected. When it is set, every paid path asks `reserve` before spending
and calls `record` after: a call that would push the ledger past the cap
(`MOMENTBENCH_SPEND_CAP_USD`, default 100) raises `SpendCapError` and spends
nothing. The ledger is append-only JSON lines, one entry per paid call or trial, and
a file lock makes concurrent runs share one total.

Only cash counts. Runs on a subscription (the Claude Code harness) and the reference
floors cost nothing here, whatever their tokens would cost at list price.
"""

from __future__ import annotations

import fcntl
import json
import os
import time
from contextlib import contextmanager
from pathlib import Path

DEFAULT_CAP_USD = 100.0
# Reserved for one agentic trial before any trial of the run has a measured cost.
TRIAL_ESTIMATE_USD = float(os.environ.get("MOMENTBENCH_TRIAL_ESTIMATE_USD", "0.5"))


class SpendCapError(RuntimeError):
    """The next paid call would take the ledger past its cap."""


def ledger_path() -> Path | None:
    p = os.environ.get("MOMENTBENCH_SPEND_LEDGER")
    return Path(p).expanduser() if p else None


def cap_usd() -> float:
    return float(os.environ.get("MOMENTBENCH_SPEND_CAP_USD", DEFAULT_CAP_USD))


def active() -> bool:
    return ledger_path() is not None


@contextmanager
def _locked():
    path = ledger_path()
    assert path is not None
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path.with_suffix(path.suffix + ".lock"), "w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        try:
            yield path
        finally:
            fcntl.flock(lock, fcntl.LOCK_UN)


def _entries(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def spent_usd() -> float:
    path = ledger_path()
    return round(sum(e["usd"] for e in _entries(path)), 6) if path else 0.0


def reserve(estimate_usd: float, what: str) -> None:
    """Refuse a paid call whose estimate would take the ledger past the cap."""
    if not active():
        return
    with _locked() as path:
        total = sum(e["usd"] for e in _entries(path))
        if total + max(0.0, estimate_usd) > cap_usd():
            raise SpendCapError(
                f"spend cap: {what} needs about ${estimate_usd:.2f}, ${total:.2f} of "
                f"${cap_usd():.2f} is already spent")


def record(usd: float, what: str, **detail) -> None:
    if not active() or usd <= 0:
        return
    with _locked() as path:
        with path.open("a") as f:
            f.write(json.dumps({"at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                                "usd": round(usd, 6), "what": what, **detail}) + "\n")


def summary() -> dict:
    path = ledger_path()
    entries = _entries(path) if path else []
    by: dict[str, float] = {}
    for e in entries:
        key = e["what"].split(":", 1)[0]
        by[key] = round(by.get(key, 0.0) + e["usd"], 6)
    total = round(sum(e["usd"] for e in entries), 6)
    return {"ledger": str(path) if path else None, "cap_usd": cap_usd(),
            "spent_usd": total, "remaining_usd": round(cap_usd() - total, 6),
            "entries": len(entries), "by_item": by}


# For models with no entry in configs/pricing.json: a ceiling well above any current
# vision model's list price, so the ledger can only over-count, never under-count.
CEILING_PER_MTOK = {"input": 5.0, "output": 30.0}


def token_cost_usd(model: str, input_tokens: int, output_tokens: int,
                   pricing_path: Path | None = None) -> float:
    """Cash cost of one call: the repository price when it has one, else the ceiling."""
    path = pricing_path or Path(__file__).resolve().parents[1] / "configs/pricing.json"
    price = None
    if path.exists():
        price = (json.loads(path.read_text()).get("models") or {}).get(model)
    inp = (price or {}).get("input_per_mtok", CEILING_PER_MTOK["input"])
    out = (price or {}).get("output_per_mtok", CEILING_PER_MTOK["output"])
    return (input_tokens * inp + output_tokens * out) / 1_000_000
