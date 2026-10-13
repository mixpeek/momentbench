"""CLIP adapter — zero-shot image-text retrieval via OpenCLIP.

Encodes query text and corpus asset descriptions into a shared
embedding space, then ranks by cosine similarity. Works with any OpenCLIP
checkpoint (ViT-B/32, ViT-L/14, ViT-H/14, SigLIP, etc.).

WHOLE-VIDEO CONTROL: ranks assets by their title/description text and returns
each whole video as the span. It does not look at frames or localize a moment;
it is a retrieval floor, not a multimodal result.
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


@register_adapter("clip")
class CLIPAdapter(ModelAdapter):
    model_id: str = "openai/clip-vit-large-patch14"
    input_kind = "whole-video-control"

    def __init__(
        self,
        model: str = "ViT-L-14",
        pretrained: str = "openai",
        device: str = "auto",
        **kwargs,
    ):
        self.model_id = f"clip-{model}"
        self._model_name = model
        self._pretrained = pretrained
        self._device = device
        self._model = None
        self._tokenizer = None
        self._kwargs = kwargs

    def _load_model(self):
        if self._model is not None:
            return
        import open_clip
        import torch

        device = self._device
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self._model, _, self._preprocess = open_clip.create_model_and_transforms(
            self._model_name, pretrained=self._pretrained, device=device
        )
        self._tokenizer = open_clip.get_tokenizer(self._model_name)
        self._device_resolved = device

    def _embed_texts(self, texts: list[str]) -> np.ndarray:
        import torch

        self._load_model()
        tokens = self._tokenizer(texts).to(self._device_resolved)
        with torch.no_grad():
            embeddings = self._model.encode_text(tokens)
        embeddings = embeddings.cpu().numpy()
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

        query_emb = self._embed_texts([query.text])
        asset_texts = [
            f"{a.get('title', '')} {a.get('description', '')}"
            for a in corpus_assets
        ]
        asset_embs = self._embed_texts(asset_texts)

        scores = [
            (i, _cosine_sim(query_emb[0], asset_embs[i]))
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
