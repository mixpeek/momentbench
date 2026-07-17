"""E5-Omni adapter — Microsoft's omnimodal embedding model (MIT license).

Closest OSS analog to Gemini Embedding 2 for cross-modal retrieval.
Projects text, image, audio, and video into a shared 3584-dim space
via Qwen2.5-Omni backbone. Commercially friendly (MIT).

This adapter uses E5-Omni as an embedding-based retriever: embed the query
and all corpus asset descriptions, then rank by cosine similarity. For full
video-level retrieval, provide video_urls in corpus asset metadata.
"""

from __future__ import annotations

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


@register_adapter("e5-omni")
class E5OmniAdapter(ModelAdapter):
    model_id: str = "intfloat/e5-omni"

    def __init__(self, model: str = "intfloat/e5-omni", device: str = "auto", **kwargs):
        self.model_id = model
        self._device = device
        self._model = None
        self._processor = None
        self._kwargs = kwargs

    def _load_model(self):
        if self._model is not None:
            return
        from transformers import AutoModel, AutoProcessor

        self._processor = AutoProcessor.from_pretrained(
            self.model_id, trust_remote_code=True
        )
        self._model = AutoModel.from_pretrained(
            self.model_id, trust_remote_code=True, device_map=self._device
        )

    def _embed_texts(self, texts: list[str]) -> np.ndarray:
        self._load_model()
        inputs = self._processor(
            text=texts, return_tensors="pt", padding=True, truncation=True
        ).to(self._model.device)
        outputs = self._model(**inputs)
        embeddings = outputs.last_hidden_state[:, 0, :].detach().cpu().numpy()
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

        query_text = f"query: {query.text}"
        asset_texts = [
            f"passage: {a.get('title', '')} {a.get('description', '')}"
            for a in corpus_assets
        ]

        all_texts = [query_text] + asset_texts
        all_embeddings = self._embed_texts(all_texts)
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
