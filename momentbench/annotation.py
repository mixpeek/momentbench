"""Which version of a query annotation a review saw (audit 3, finding 5).

`annotation_sha` hashes what a reviewer judges: the request text, the tier, the gold
spans and, for a no-answer query, the source shot it was altered from. Every review
record (`cross_check`, `model_review`, `human_review`) stores the hash of the
annotation it reviewed. When the annotation changes, a review whose hash no longer
matches is about an earlier version: an edit moves the cross-check to
`cross_check_history`, and a new independent check is needed.
"""

from __future__ import annotations

import hashlib
import json
from typing import Any


def annotation_sha(query: dict[str, Any]) -> str:
    meta = query.get("metadata") or {}
    src = meta.get("altered_from")
    payload = {
        "text": query["text"],
        "tier": int(query["tier"]),
        "gold": [[g["asset_id"], float(g["t_in"]), float(g["t_out"])]
                 for g in query.get("gold_moments") or []],
        "altered_from": ([src.get("asset_id"), float(src["t_in"]), float(src["t_out"])]
                         if src else None),
    }
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()[:16]


def invalidate_cross_check(query: dict[str, Any], reason: str, date: str) -> bool:
    """Move a cross-check that no longer matches the annotation into history.

    Returns True when it moved one. The query keeps no `cross_check` until a new
    independent check runs on the current annotation.
    """
    meta = query["metadata"]
    check = meta.get("cross_check")
    if not check or check.get("annotation_sha") == annotation_sha(query):
        return False
    meta.setdefault("cross_check_history", []).append(
        {**check, "superseded": {"date": date, "reason": reason}})
    del meta["cross_check"]
    model = check.get("model", "")
    meta["source"] = meta.get("source", "").replace(
        f"cross-checked by {model}",
        f"cross-checked by {model} before a later edit; new cross-check pending")
    return True
