"""Qwen3-VL / Qwen2.5-VL adapter — OSS multimodal vision-language model.

Closest OSS analog to Gemini for video understanding. Supports text, image,
and video inputs in a unified architecture. Runs via the OpenAI-compatible
API served by vLLM, SGLang, or any compatible inference server.
"""

from __future__ import annotations

import os
import time
from typing import Any

from momentbench.models.base import ModelAdapter
from momentbench.models.registry import register_adapter
from momentbench.types import Prediction, Query


@register_adapter("qwen-vl")
class QwenVLAdapter(ModelAdapter):
    model_id: str = "Qwen/Qwen3-VL-4B"

    def __init__(
        self,
        model: str = "Qwen/Qwen3-VL-4B",
        base_url: str | None = None,
        **kwargs,
    ):
        self.model_id = model
        self.base_url = base_url or os.environ.get(
            "QWEN_API_BASE", "http://localhost:8000/v1"
        )
        self._kwargs = kwargs

    async def predict(
        self,
        query: Query,
        corpus_assets: list[dict[str, Any]],
        max_results: int = 5,
    ) -> Prediction:
        from openai import AsyncOpenAI

        client = AsyncOpenAI(
            base_url=self.base_url,
            api_key=os.environ.get("QWEN_API_KEY", "unused"),
        )
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
