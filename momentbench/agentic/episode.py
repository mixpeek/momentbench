"""Drive one model through one query: tool loop, budget enforcement, final answer."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from momentbench.agentic.budget import Budget, Usage
from momentbench.agentic.environment import TOOLS, LocalCorpus, parse_submission
from momentbench.agentic.model import AgentModel, ToolResult
from momentbench.types import Moment, Query

SYSTEM_PROMPT = (
    "You are locating a specific moment in a corpus of videos. You cannot watch "
    "video directly. Use the tools to list the videos, search and read "
    "transcripts, and look at sampled frames, then call submit_answer with the "
    "video and the start and end time in seconds of the moment that matches the "
    "request. A submitted moment is correct when it is in the right video and "
    "overlaps the true moment with temporal IoU of at least 0.5, so make the "
    "span tight. Some requests have no matching moment anywhere in the corpus. "
    "For those, the correct answer is to submit an empty list. Frames and tool "
    "calls are limited by a budget. If the budget runs out, submit your best answer."
)


def build_task(query: Query, budget: Budget) -> str:
    return (
        f"Find this moment: {query.text}\n\n"
        f"Budget for this task: {budget.max_frames} frames, "
        f"{budget.max_tool_calls} tool calls, {budget.max_steps} turns."
    )


@dataclass
class Episode:
    query_id: str
    moments: list[Moment]
    usage: Usage
    # "submitted", "no_submission", "step_limit", or the provider stop ("refusal", ...).
    outcome: str
    trace: list[dict[str, Any]] = field(default_factory=list)


async def run_episode(
    model: AgentModel, corpus: LocalCorpus, query: Query, budget: Budget
) -> Episode:
    usage = Usage()
    trace: list[dict[str, Any]] = []
    session = model.start(SYSTEM_PROMPT, build_task(query, budget), TOOLS)
    results: list[ToolResult] | None = None

    while usage.steps < budget.max_steps:
        turn = await session.step(results)
        usage.steps += 1
        usage.add(turn.usage)
        if turn.stopped:
            return Episode(query.query_id, [], usage, turn.stopped, trace)
        if not turn.tool_calls:
            return Episode(query.query_id, [], usage, "no_submission", trace)

        results = []
        for call in turn.tool_calls:
            if call.name == "submit_answer":
                moments = parse_submission(call.arguments, set(corpus.assets))
                trace.append({"tool": call.name, "arguments": call.arguments})
                return Episode(query.query_id, moments, usage, "submitted", trace)
            result = corpus.execute(call, budget, usage)
            trace.append(
                {
                    "tool": call.name,
                    "arguments": call.arguments,
                    "is_error": result.is_error,
                    "frames": len(result.images_jpeg),
                }
            )
            results.append(result)

    return Episode(query.query_id, [], usage, "step_limit", trace)
