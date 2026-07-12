"""Base class for model adapters."""

from __future__ import annotations

import abc
from typing import Any

from momentbench.types import Moment, Prediction, Query


class ModelAdapter(abc.ABC):
    """Base adapter that all model implementations must subclass."""

    model_id: str

    @abc.abstractmethod
    async def predict(
        self,
        query: Query,
        corpus_assets: list[dict[str, Any]],
        max_results: int = 5,
    ) -> Prediction:
        """Given a query and corpus metadata, return ranked moment predictions.

        Args:
            query: The benchmark query to answer.
            corpus_assets: List of asset metadata dicts with at minimum
                           {asset_id, duration_s, title, vertical}.
            max_results: Maximum number of moments to return.

        Returns:
            Prediction with ranked_moments.
        """
        ...

    def build_prompt(self, query: Query, corpus_assets: list[dict[str, Any]]) -> str:
        """Build the text prompt sent to the model. Override for custom prompting."""
        asset_list = "\n".join(
            f"- [{a['asset_id']}] {a.get('title', 'untitled')} ({a.get('duration_s', '?')}s)"
            for a in corpus_assets
        )
        return (
            "You are a video moment retrieval system. Given a natural language query "
            "and a corpus of video assets, identify the specific moments (asset_id, "
            "start_time, end_time in seconds) that best match the query.\n\n"
            f"Query: {query.text}\n\n"
            f"Available assets:\n{asset_list}\n\n"
            "Return your answer as a JSON array of objects with keys: "
            "asset_id, t_in, t_out. Most relevant first. "
            f"Return at most {5} results."
        )

    def parse_moments(self, raw_output: str) -> list[Moment]:
        """Parse model output into Moment objects. Override for custom parsing."""
        import json
        import re

        json_match = re.search(r"\[.*\]", raw_output, re.DOTALL)
        if not json_match:
            return []
        try:
            items = json.loads(json_match.group())
        except json.JSONDecodeError:
            return []
        moments = []
        for item in items:
            if isinstance(item, dict) and all(k in item for k in ("asset_id", "t_in", "t_out")):
                moments.append(
                    Moment(
                        asset_id=str(item["asset_id"]),
                        t_in=float(item["t_in"]),
                        t_out=float(item["t_out"]),
                    )
                )
        return moments
