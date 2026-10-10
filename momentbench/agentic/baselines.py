"""Reference floors for the agentic track. No model, no network, no cost.

- `random`: lists the assets, then submits one uniformly random asset and a random span
  whose length is drawn from the corpus's typical shot length. This is chance.
- `abstain`: always submits an empty answer. It scores exactly the share of no-answer
  (tier 3) queries, which is the most a system can get by never committing.

A model that does not clear both floors on a tier has learned nothing there.
"""

from __future__ import annotations

import json
import random
from typing import Any

from momentbench.agentic.budget import Usage
from momentbench.agentic.model import AgentModel, AgentSession, ModelTurn, ToolCall, ToolResult

# Median gold span length over the 295 public v0.2 gold spans, in seconds.
TYPICAL_SPAN_S = 4.25


class _Session(AgentSession):
    def __init__(self, kind: str, rng: random.Random):
        self._kind = kind
        self._rng = rng
        self._turn = 0

    async def step(self, tool_results: list[ToolResult] | None) -> ModelTurn:
        self._turn += 1
        if self._kind == "abstain":
            return ModelTurn([ToolCall("a", "submit_answer", {"moments": []})], Usage())
        if self._turn == 1:
            return ModelTurn([ToolCall("l", "list_assets", {})], Usage())
        assets = json.loads(tool_results[0].text) if tool_results else []
        if not assets:
            return ModelTurn([ToolCall("a", "submit_answer", {"moments": []})], Usage())
        asset = self._rng.choice(assets)
        duration = float(asset.get("duration_s") or 60)
        length = min(duration, self._rng.uniform(0.5, 2.0) * TYPICAL_SPAN_S)
        start = self._rng.uniform(0, max(0.0, duration - length))
        moment = {"asset_id": asset["asset_id"], "t_in": round(start, 2),
                  "t_out": round(start + length, 2)}
        return ModelTurn([ToolCall("a", "submit_answer", {"moments": [moment]})], Usage())


class BaselineAgent(AgentModel):
    """`model` is "random" or "abstain"."""

    def __init__(self, model: str = "random", effort: str | None = None, seed: int = 0):
        if model not in ("random", "abstain"):
            raise ValueError('baseline model must be "random" or "abstain"')
        self.model_id = f"baseline-{model}"
        self.effort = None
        self._kind = model
        self._rng = random.Random(seed)

    def start(self, system: str, task: str, tools: list[dict[str, Any]]) -> AgentSession:
        return _Session(self._kind, random.Random(self._rng.random()))
