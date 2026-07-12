"""Mixpeek pipeline adapter — decompose → MVS → multi-stage retrieval."""

from __future__ import annotations

import time
from typing import Any

from momentbench.models.base import ModelAdapter
from momentbench.models.registry import register_adapter
from momentbench.types import Moment, Prediction, Query


@register_adapter("mixpeek")
class MixpeekAdapter(ModelAdapter):
    model_id: str = "mixpeek-pipeline"

    def __init__(self, namespace: str | None = None, collection: str | None = None, **kwargs):
        self._namespace = namespace
        self._collection = collection
        self._kwargs = kwargs

    async def predict(
        self,
        query: Query,
        corpus_assets: list[dict[str, Any]],
        max_results: int = 5,
    ) -> Prediction:
        from mixpeek import Mixpeek

        client = Mixpeek()

        start = time.perf_counter()
        response = client.search.text(
            query=query.text,
            collection=self._collection,
            filters={"vertical": query.vertical} if query.vertical else None,
            limit=max_results,
        )
        latency = (time.perf_counter() - start) * 1000

        moments = []
        for result in getattr(response, "results", []):
            moments.append(
                Moment(
                    asset_id=result.asset_id,
                    t_in=result.start_time,
                    t_out=result.end_time,
                )
            )

        return Prediction(
            query_id=query.query_id,
            ranked_moments=moments[:max_results],
            model_id=self.model_id,
            latency_ms=latency,
        )
