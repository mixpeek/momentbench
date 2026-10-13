"""MixedBread (mxbai) adapter — embedding-based retrieval via MixedBread API.

Uses mxbai-embed-large or mxbai-embed-2d-large for text embedding, then
ranks corpus assets by cosine similarity. Supports both the hosted API
and local model loading via sentence-transformers.

WHOLE-VIDEO CONTROL: ranks assets by their title/description text and returns
each whole video as the span. It does not look at frames or localize a moment;
it is a retrieval floor, not a multimodal result.
"""

from __future__ import annotations

import os
import time
from typing import Any

import numpy as np

from momentbench.models.base import ModelAdapter
from momentbench.models.registry import register_adapter
from momentbench.types import Moment, Prediction, Query


def _cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return float(np.dot(a, b) / (norm_a * norm_b))


@register_adapter("mixedbread")
class MixedBreadAdapter(ModelAdapter):
    model_id: str = "mxbai-embed-large-v1"
    input_kind = "whole-video-control"

    def __init__(
        self,
        model: str = "mxbai-embed-large-v1",
        api_key: str | None = None,
        base_url: str | None = None,
        **kwargs,
    ):
        self.model_id = f"mxbai-{model}"
        self._model = model
        self._api_key = api_key or os.environ.get("MIXEDBREAD_API_KEY", "")
        self._base_url = base_url or os.environ.get(
            "MIXEDBREAD_API_BASE", "https://api.mixedbread.ai"
        )
        self._kwargs = kwargs

    async def _embed(self, texts: list[str]) -> np.ndarray:
        import httpx

        async with httpx.AsyncClient() as client:
            response = await client.post(
                f"{self._base_url}/v1/embeddings",
                headers={"Authorization": f"Bearer {self._api_key}"},
                json={"model": self._model, "input": texts},
                timeout=60,
            )
            response.raise_for_status()
            data = response.json()

        embeddings = np.array([d["embedding"] for d in data["data"]])
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        norms[norms == 0] = 1
        return embeddings / norms

    async def predict(
        self,
        query: Query,
        corpus_assets: list[dict[str, Any]],
        max_results: int = 5,
    ) -> Prediction:
        start = time.perf_counter()

        query_text = f"Represent this sentence for searching relevant passages: {query.text}"
        asset_texts = [
            f"{a.get('title', '')} {a.get('description', '')}"
            for a in corpus_assets
        ]

        all_texts = [query_text] + asset_texts
        all_embeddings = await self._embed(all_texts)
        query_emb = all_embeddings[0]
        asset_embs = all_embeddings[1:]

        scores = [
            (i, _cosine_sim(query_emb, asset_embs[i]))
            for i in range(len(corpus_assets))
        ]
        scores.sort(key=lambda x: x[1], reverse=True)

        latency = (time.perf_counter() - start) * 1000

        moments = []
        for idx, _score in scores[:max_results]:
            asset = corpus_assets[idx]
            moments.append(
                Moment(
                    asset_id=asset["asset_id"],
                    t_in=0.0,
                    t_out=float(asset.get("duration_s", 0)),
                )
            )

        return Prediction(
            query_id=query.query_id,
            ranked_moments=moments,
            model_id=self.model_id,
            latency_ms=latency,
        )
