"""Jina CLIP v2 adapter — multilingual multimodal embeddings via Jina API.

Projects text and images into a shared embedding space. For video retrieval,
embeds query text and corpus asset descriptions, ranks by cosine similarity.
Supports the hosted Jina API or local model via transformers.
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


@register_adapter("jina-clip")
class JinaCLIPAdapter(ModelAdapter):
    model_id: str = "jina-clip-v2"

    def __init__(
        self,
        model: str = "jina-clip-v2",
        api_key: str | None = None,
        **kwargs,
    ):
        self.model_id = model
        self._model = model
        self._api_key = api_key or os.environ.get("JINA_API_KEY", "")
        self._kwargs = kwargs

    async def _embed(self, texts: list[str]) -> np.ndarray:
        import httpx

        async with httpx.AsyncClient() as client:
            response = await client.post(
                "https://api.jina.ai/v1/embeddings",
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

        asset_texts = [
            f"{a.get('title', '')} {a.get('description', '')}"
            for a in corpus_assets
        ]

        all_texts = [query.text] + asset_texts
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
