"""Google Gemini adapter (legacy track, metadata-only).

METADATA-ONLY: the prompt carries asset titles and durations, no frames,
audio or transcript. A score here measures guessing from titles; use
`momentbench agentic` to evaluate the model on the media.
"""

from __future__ import annotations

import time
from typing import Any

from momentbench.models.base import ModelAdapter
from momentbench.models.registry import register_adapter
from momentbench.types import Prediction, Query


@register_adapter("gemini")
class GeminiAdapter(ModelAdapter):
    model_id: str = "gemini-2.5-pro"

    def __init__(self, model: str = "gemini-2.5-pro", **kwargs):
        self.model_id = model
        self._kwargs = kwargs

    async def predict(
        self,
        query: Query,
        corpus_assets: list[dict[str, Any]],
        max_results: int = 5,
    ) -> Prediction:
        from google import genai

        client = genai.Client()
        prompt = self.build_prompt(query, corpus_assets)

        start = time.perf_counter()
        response = await client.aio.models.generate_content(
            model=self.model_id,
            contents=prompt,
        )
        latency = (time.perf_counter() - start) * 1000

        moments = self.parse_moments(response.text or "")
        return Prediction(
            query_id=query.query_id,
            ranked_moments=moments[:max_results],
            model_id=self.model_id,
            latency_ms=latency,
        )
