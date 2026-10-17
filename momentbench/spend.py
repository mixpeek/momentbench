"""A shared cash ledger with a hard cap, for maintainers running paid evaluations.

Off unless `MOMENTBENCH_SPEND_LEDGER` names a ledger file, so nobody else using the
package is affected. When it is set, every paid call goes through two steps:

1. `reserve(max_usd, what)` writes a reservation for the most the call could cost and
   returns its id. It refuses (`SpendCapError`) when settled spend plus every
   outstanding reservation plus this one would pass the cap. Reservations are
   persisted under a file lock, so concurrent processes share one total.
2. `settle(reservation, actual_usd)` replaces the reservation with the measured cost.

A reservation that is never settled (a crash, or a timeout where billing is unknown)
stays counted at its maximum, so uncertainty only ever lowers what can be spent.

The cap is `MOMENTBENCH_SPEND_CAP_USD`, default 100. It can be set lower and never
higher than `HARD_CAP_USD`, which is in this file and not in any configuration an
agent edits. Only cash counts: subscription runs (the Claude Code harness) and the
reference floors are never charged.
"""

from __future__ import annotations

import fcntl
import json
import os
import time
import uuid
from contextlib import contextmanager
from pathlib import Path

HARD_CAP_USD = 100.0
DEFAULT_CAP_USD = HARD_CAP_USD
# Most one agentic trial may use, in tokens, for its reservation: 30 turns with up to
# 64 frames of context is well under this.
MAX_TRIAL_INPUT_TOKENS = int(os.environ.get("MOMENTBENCH_MAX_TRIAL_INPUT_TOKENS", "600000"))
MAX_TRIAL_OUTPUT_TOKENS = int(os.environ.get("MOMENTBENCH_MAX_TRIAL_OUTPUT_TOKENS", "60000"))
# Kept for callers that predate per-trial bounds.
TRIAL_ESTIMATE_USD = float(os.environ.get("MOMENTBENCH_TRIAL_ESTIMATE_USD", "0.5"))
# For models with no entry in configs/pricing.json: a ceiling well above any current
# vision model's list price, so the ledger can only over-count, never under-count.
CEILING_PER_MTOK = {"input": 5.0, "output": 30.0}


class SpendCapError(RuntimeError):
    """The next paid call would take the ledger past its cap."""


def ledger_path() -> Path | None:
    p = os.environ.get("MOMENTBENCH_SPEND_LEDGER")
    return Path(p).expanduser() if p else None


def cap_usd() -> float:
    return min(HARD_CAP_USD, float(os.environ.get("MOMENTBENCH_SPEND_CAP_USD",
                                                  DEFAULT_CAP_USD)))


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


def _committed(entries: list[dict]) -> tuple[float, float]:
    """(settled spend, outstanding reservations). Entries without a kind are settled
    spend written before reservations existed."""
    settled_ids = {e["id"] for e in entries if e.get("kind") == "settle"}
    settled = sum(e["usd"] for e in entries if e.get("kind") in (None, "settle"))
    outstanding = sum(e["usd"] for e in entries
                      if e.get("kind") == "reserve" and e["id"] not in settled_ids)
    return settled, outstanding


def _append(path: Path, entry: dict) -> None:
    with path.open("a") as f:
        f.write(json.dumps({"at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                            **entry}) + "\n")
        f.flush()
        os.fsync(f.fileno())


def spent_usd() -> float:
    """Settled spend plus outstanding reservations: what the cap is checked against."""
    path = ledger_path()
    if not path:
        return 0.0
    settled, outstanding = _committed(_entries(path))
    return round(settled + outstanding, 6)


def reserve(max_usd: float, what: str) -> str | None:
    """Reserve the most a call can cost; returns the reservation id (None when off)."""
    if not active():
        return None
    with _locked() as path:
        settled, outstanding = _committed(_entries(path))
        if settled + outstanding + max(0.0, max_usd) > cap_usd():
            raise SpendCapError(
                f"spend cap: {what} may cost up to ${max_usd:.2f}; ${settled:.2f} spent "
                f"and ${outstanding:.2f} reserved of ${cap_usd():.2f}")
        rid = uuid.uuid4().hex[:12]
        _append(path, {"kind": "reserve", "id": rid, "usd": round(max(0.0, max_usd), 6),
                       "what": what})
        return rid


def settle(reservation: str | None, actual_usd: float, what: str, **detail) -> None:
    """Close a reservation at the measured cost."""
    if not active():
        return
    with _locked() as path:
        _append(path, {"kind": "settle", "id": reservation or uuid.uuid4().hex[:12],
                       "usd": round(max(0.0, actual_usd), 6), "what": what, **detail})


def record(usd: float, what: str, reservation: str | None = None, **detail) -> None:
    """Settle `reservation` (or record unreserved spend) at `usd`."""
    settle(reservation, usd, what, **detail)


def summary() -> dict:
    path = ledger_path()
    entries = _entries(path) if path else []
    settled, outstanding = _committed(entries)
    by: dict[str, float] = {}
    for e in entries:
        if e.get("kind") in (None, "settle"):
            key = e["what"].split(":", 1)[0]
            by[key] = round(by.get(key, 0.0) + e["usd"], 6)
    open_ids = {e["id"] for e in entries if e.get("kind") == "reserve"} - {
        e["id"] for e in entries if e.get("kind") == "settle"}
    return {"ledger": str(path) if path else None, "cap_usd": cap_usd(),
            "spent_usd": round(settled, 6), "reserved_usd": round(outstanding, 6),
            "remaining_usd": round(cap_usd() - settled - outstanding, 6),
            "open_reservations": len(open_ids), "entries": len(entries), "by_item": by}


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
