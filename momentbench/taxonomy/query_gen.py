"""Query generation: LLM-assisted candidate generation + hard-negative mining via clustering."""

from __future__ import annotations

import uuid
from typing import Any

from momentbench.types import Moment, Query, QueryTier

QUERY_GEN_SYSTEM_PROMPT = """\
You are a benchmark query generator for MomentBench, a video moment retrieval benchmark.
Given scene metadata (transcript, visual descriptions, temporal context), generate
candidate queries at the specified tier.

Tier definitions:
- Tier 1 (single visual): simple visual event identification
- Tier 2 (cross-modal): requires joint reasoning across modalities
- Tier 3 (corpus hard negative): target is similar to many other moments in corpus
- Tier 4 (temporal reasoning): requires understanding temporal ordering/sequences

Output JSON array of objects with: text, tier, reasoning, gold_t_in, gold_t_out
"""


def generate_candidate_queries(
    scene_metadata: list[dict[str, Any]],
    target_tier: QueryTier,
    count: int = 10,
    vertical: str = "",
    asset_id: str = "",
) -> list[Query]:
    """Generate candidate queries from scene metadata (placeholder for LLM call)."""
    queries = []
    for i, scene in enumerate(scene_metadata[:count]):
        query_id = f"q-{vertical}-{uuid.uuid4().hex[:8]}"
        queries.append(
            Query(
                query_id=query_id,
                text=scene.get("suggested_query", f"Find the moment at {scene.get('t_in', 0)}s"),
                tier=target_tier,
                vertical=vertical,
                gold_moments=[
                    Moment(
                        asset_id=asset_id,
                        t_in=scene.get("t_in", 0),
                        t_out=scene.get("t_out", 5),
                    )
                ],
                is_public=True,
            )
        )
    return queries


def mine_hard_negatives_from_clusters(
    query: Query,
    cluster_members: list[dict[str, Any]],
) -> list[Moment]:
    """Given a query and its cluster (near-duplicate moments), extract hard negatives."""
    hard_negatives = []
    gold_times = {(m.asset_id, m.t_in, m.t_out) for m in query.gold_moments}

    for member in cluster_members:
        key = (member["asset_id"], member["t_in"], member["t_out"])
        if key not in gold_times:
            hard_negatives.append(
                Moment(
                    asset_id=member["asset_id"],
                    t_in=member["t_in"],
                    t_out=member["t_out"],
                    metadata={"similarity_score": member.get("score", 0)},
                )
            )
    return hard_negatives


def split_public_private(
    queries: list[Query], private_ratio: float = 0.3
) -> tuple[list[Query], list[Query]]:
    """Split queries into public (free) and private (sellable) sets."""
    sellable = [q for q in queries if all(
        m.metadata.get("is_sellable", True) for m in q.gold_moments
    )]
    non_sellable = [q for q in queries if q not in sellable]

    n_private = int(len(sellable) * private_ratio)
    private_set = sellable[:n_private]
    public_set = sellable[n_private:] + non_sellable

    for q in private_set:
        q.is_public = False
    for q in public_set:
        q.is_public = True

    return public_set, private_set
