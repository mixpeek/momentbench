"""Reference floors for the agentic track. No model, no network, no cost.

- `random`: lists the assets, then submits one uniformly random asset and a random span
  whose length is drawn from the corpus's typical shot length. This is chance.
- `abstain`: always submits an empty answer. It scores exactly the share of no-answer
  (tier 3) queries, which is the most a system can get by never committing.
- `transcript`: a modality control. It sends the request text to the corpus's own
  keyword search_transcript and submits the top five cue spans, never looking at a
  frame. Where it scores, a query can be answered from speech alone; on tier 2
  (dialogue together with the picture) that marks a shortcut (audit 2, R12). It
  abstains only when the search finds nothing, so it scores near zero on no-answer
  queries.

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
        if self._kind == "transcript":
            return self._transcript(tool_results)
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


    def _transcript(self, tool_results: list[ToolResult] | None) -> ModelTurn:
        if self._turn == 1:
            return ModelTurn([ToolCall("s", "search_transcript",
                                       {"query": self._request, "top_k": 5})], Usage())
        hits = json.loads(tool_results[0].text) if tool_results and not tool_results[0].is_error \
            else []
        moments = [{"asset_id": h["asset_id"], "t_in": h["t_in"], "t_out": h["t_out"]}
                   for h in hits if h["t_out"] > h["t_in"]]
        return ModelTurn([ToolCall("a", "submit_answer", {"moments": moments})], Usage())


class BaselineAgent(AgentModel):
    """`model` is "random", "abstain" or "transcript"."""

    def __init__(self, model: str = "random", effort: str | None = None, seed: int = 0):
        if model not in ("random", "abstain", "transcript"):
            raise ValueError('baseline model must be "random", "abstain" or "transcript"')
        self.model_id = f"baseline-{model}"
        self.effort = None
        self._kind = model
        self._rng = random.Random(seed)

    def start(self, system: str, task: str, tools: list[dict[str, Any]]) -> AgentSession:
        session = _Session(self._kind, random.Random(self._rng.random()))
        # The request is the first line of the task text ("Find this moment: ...").
        session._request = task.split("\n", 1)[0].removeprefix("Find this moment: ")
        return session
