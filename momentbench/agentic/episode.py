"""Drive one model through one query: tool loop, budget enforcement, final answer."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from typing import Any

from momentbench.agentic.budget import Budget, Usage
from momentbench.agentic.environment import (
    TOOLS,
    InvalidSubmissionError,
    LocalCorpus,
)
from momentbench.agentic.model import AgentModel, ToolResult
from momentbench.types import Moment, Query, QueryRequest

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
    resolved_model: str | None = None


class EpisodeError(RuntimeError):
    """A failure partway through an episode; carries what was spent and done so far."""

    def __init__(self, cause: BaseException, usage: Usage, trace: list[dict[str, Any]]):
        super().__init__(f"{type(cause).__name__}: {cause}")
        self.cause = cause
        self.usage = usage
        self.trace = trace


async def run_episode(
    model: AgentModel, corpus: LocalCorpus, query: Query | QueryRequest, budget: Budget
) -> Episode:
    usage = Usage()
    trace: list[dict[str, Any]] = []
    try:
        return await _episode(model, corpus, query, budget, usage, trace)
    except Exception as e:
        # Keep tokens already spent and the tool trace: a failed trial is still costed
        # and inspectable (audit finding 27).
        raise EpisodeError(e, usage, trace) from e


async def _episode(model, corpus, query, budget, usage, trace) -> Episode:
    session = model.start(SYSTEM_PROMPT, build_task(query, budget), TOOLS)
    results: list[ToolResult] | None = None
    resolved: list[str | None] = [None]

    while usage.steps < budget.max_steps:
        turn = await session.step(results)
        usage.steps += 1
        usage.add(turn.usage)
        resolved[0] = turn.model or resolved[0]
        if turn.stopped:
            return Episode(query.query_id, [], usage, turn.stopped, trace, resolved[0])
        if not turn.tool_calls:
            return Episode(query.query_id, [], usage, "no_submission", trace, resolved[0])

        results = []
        for call in turn.tool_calls:
            if call.name == "submit_answer":
                try:
                    moments = corpus.parse_answer(call.arguments)
                except InvalidSubmissionError as e:
                    # Not an abstention: tell the model and let it submit again.
                    trace.append({"tool": call.name, "arguments": call.arguments,
                                  "is_error": True, "error": str(e)})
                    results.append(ToolResult(call.call_id, f"Invalid submission: {e}",
                                              is_error=True))
                    continue
                trace.append({"tool": call.name, "arguments": call.arguments})
                return Episode(query.query_id, moments, usage, "submitted", trace, resolved[0])
            # A worker thread, so ffmpeg in one episode does not stall the others.
            result = await asyncio.to_thread(corpus.execute, call, budget, usage)
            trace.append(
                {
                    "tool": call.name,
                    "arguments": call.arguments,
                    "is_error": result.is_error,
                    "frames": len(result.images_jpeg),
                }
            )
            results.append(result)

    return Episode(query.query_id, [], usage, "step_limit", trace, resolved[0])
