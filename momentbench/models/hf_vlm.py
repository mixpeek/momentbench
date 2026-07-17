"""Generic HuggingFace VLM adapter — run any OpenAI-compatible self-hosted model.

Covers vLLM, SGLang, TGI, Ollama, or any server exposing an OpenAI-compatible
chat completions endpoint. Use this for models like InternVL3, MiniCPM-V,
Llama 4 Maverick, Phi-4 Multimodal, or any other self-hosted VLM.
"""

from __future__ import annotations

import os
import time
from typing import Any

from momentbench.models.base import ModelAdapter
from momentbench.models.registry import register_adapter
from momentbench.types import Prediction, Query


@register_adapter("hf-vlm")
class HuggingFaceVLMAdapter(ModelAdapter):
    model_id: str = "self-hosted-vlm"

    def __init__(
        self,
        model: str = "self-hosted-vlm",
        base_url: str | None = None,
        api_key: str | None = None,
        **kwargs,
    ):
        self.model_id = model
        self.base_url = base_url or os.environ.get(
            "HF_VLM_API_BASE", "http://localhost:8000/v1"
        )
        self.api_key = api_key or os.environ.get("HF_VLM_API_KEY", "unused")
        self._kwargs = kwargs

    async def predict(
        self,
        query: Query,
        corpus_assets: list[dict[str, Any]],
        max_results: int = 5,
    ) -> Prediction:
        from openai import AsyncOpenAI

        client = AsyncOpenAI(base_url=self.base_url, api_key=self.api_key)
        prompt = self.build_prompt(query, corpus_assets)

        start = time.perf_counter()
        response = await client.chat.completions.create(
            model=self.model_id,
            messages=[{"role": "user", "content": prompt}],
            temperature=0,
        )
        latency = (time.perf_counter() - start) * 1000

        raw = response.choices[0].message.content or ""
        moments = self.parse_moments(raw)
        return Prediction(
            query_id=query.query_id,
            ranked_moments=moments[:max_results],
            model_id=self.model_id,
            latency_ms=latency,
        )
