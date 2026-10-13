"""Anthropic Claude adapter (legacy track, metadata-only).

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


@register_adapter("claude")
class ClaudeAdapter(ModelAdapter):
    model_id: str = "claude-sonnet-4-20250514"

    def __init__(self, model: str = "claude-sonnet-4-20250514", **kwargs):
        self.model_id = model
        self._kwargs = kwargs

    async def predict(
        self,
        query: Query,
        corpus_assets: list[dict[str, Any]],
        max_results: int = 5,
    ) -> Prediction:
        import anthropic

        client = anthropic.AsyncAnthropic()
        prompt = self.build_prompt(query, corpus_assets)

        start = time.perf_counter()
        response = await client.messages.create(
            model=self.model_id,
            max_tokens=2048,
            messages=[{"role": "user", "content": prompt}],
        )
        latency = (time.perf_counter() - start) * 1000

        raw = response.content[0].text if response.content else ""
        moments = self.parse_moments(raw)
        return Prediction(
            query_id=query.query_id,
            ranked_moments=moments[:max_results],
            model_id=self.model_id,
            latency_ms=latency,
        )
