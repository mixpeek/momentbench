"""Search Mixpeek moment index for benchmark queries."""

from __future__ import annotations

from momentbench.types import Moment, Query


async def search_moments(
    query: Query,
    collection: str | None = None,
    limit: int = 10,
) -> list[Moment]:
    """Search the Mixpeek moment index for a given query."""
    from mixpeek import Mixpeek

    client = Mixpeek()
    target_collection = collection or f"momentbench-{query.vertical}"

    response = client.search.text(
        query=query.text,
        collection=target_collection,
        limit=limit,
    )

    moments = []
    for result in getattr(response, "results", []):
        moments.append(
            Moment(
                asset_id=result.asset_id,
                t_in=result.start_time,
                t_out=result.end_time,
                metadata={
                    "score": result.score,
                    "collection": target_collection,
                },
            )
        )
    return moments


async def search_with_hard_negative_filter(
    query: Query,
    hard_negative_cluster_id: str | None = None,
    limit: int = 10,
) -> list[Moment]:
    """Search with hard negative awareness — uses clustering to surface confusors."""
    from mixpeek import Mixpeek

    client = Mixpeek()
    target_collection = f"momentbench-{query.vertical}"

    response = client.search.text(
        query=query.text,
        collection=target_collection,
        limit=limit * 3,  # Over-fetch to allow filtering
    )

    moments = []
    for result in getattr(response, "results", []):
        moments.append(
            Moment(
                asset_id=result.asset_id,
                t_in=result.start_time,
                t_out=result.end_time,
                metadata={
                    "score": result.score,
                    "cluster_id": getattr(result, "cluster_id", None),
                },
            )
        )
    return moments[:limit]
