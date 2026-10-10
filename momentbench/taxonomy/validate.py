"""Static checks every query file must pass before it ships."""

from __future__ import annotations

from typing import Any

from momentbench.canary import CANARY
from momentbench.types import Query


def check_queries(queries: list[Query], assets: list[dict[str, Any]]) -> list[str]:
    """Return one message per problem. An empty list means the set is clean."""
    by_id = {a["asset_id"]: a for a in assets}
    problems: list[str] = []
    seen: set[str] = set()
    for q in queries:
        where = q.query_id
        if q.query_id in seen:
            problems.append(f"{where}: duplicate query_id")
        seen.add(q.query_id)
        if q.metadata.get("canary") != CANARY:
            problems.append(f"{where}: missing canary")
        if not q.metadata.get("source"):
            problems.append(f"{where}: metadata.source must say who annotated it and how")
        for m in [*q.gold_moments, *q.hard_negatives]:
            asset = by_id.get(m.asset_id)
            if asset is None:
                problems.append(f"{where}: unknown asset {m.asset_id}")
                continue
            if m.t_out <= m.t_in:
                problems.append(f"{where}: empty or inverted span {m.t_in}-{m.t_out}")
            if m.t_out > float(asset["duration_s"]):
                problems.append(
                    f"{where}: span ends at {m.t_out}s, past the end of {m.asset_id} "
                    f"({asset['duration_s']}s)"
                )
    return problems
