"""TwelveLabs adapter — API-based video understanding platform.

Uses TwelveLabs Marengo for video search with temporal segment retrieval.
Requires a TwelveLabs API key and a pre-created index containing the corpus.
"""

from __future__ import annotations

import os
import time
from typing import Any

from momentbench.models.base import ModelAdapter
from momentbench.models.registry import register_adapter
from momentbench.types import Moment, Prediction, Query


@register_adapter("twelvelabs")
class TwelveLabsAdapter(ModelAdapter):
    model_id: str = "twelvelabs-marengo"

    def __init__(
        self,
        model: str = "marengo-retrieval-2.7",
        index_id: str | None = None,
        api_key: str | None = None,
        **kwargs,
    ):
        self.model_id = f"twelvelabs-{model}"
        self._model = model
        self._index_id = index_id or os.environ.get("TWELVELABS_INDEX_ID", "")
        self._api_key = api_key or os.environ.get("TWELVELABS_API_KEY", "")
        self._kwargs = kwargs

    async def predict(
        self,
        query: Query,
        corpus_assets: list[dict[str, Any]],
        max_results: int = 5,
    ) -> Prediction:
        import httpx

        start = time.perf_counter()
        async with httpx.AsyncClient() as client:
            response = await client.post(
                "https://api.twelvelabs.io/v1.3/search",
                headers={"x-api-key": self._api_key},
                json={
                    "query_text": query.text,
                    "index_id": self._index_id,
                    "search_options": ["visual", "conversation", "text_in_video"],
                    "page_limit": max_results,
                },
                timeout=120,
            )
            response.raise_for_status()
            data = response.json()
        latency = (time.perf_counter() - start) * 1000

        moments = []
        for clip in data.get("data", []):
            moments.append(
                Moment(
                    asset_id=clip.get("video_id", ""),
                    t_in=float(clip.get("start", 0)),
                    t_out=float(clip.get("end", 0)),
                )
            )

        return Prediction(
            query_id=query.query_id,
            ranked_moments=moments[:max_results],
            model_id=self.model_id,
            latency_ms=latency,
        )
