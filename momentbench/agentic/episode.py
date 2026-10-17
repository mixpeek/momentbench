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


# The standard prompt asks for a best answer when the budget runs out, so a wrong
# guess there is partly the prompt's doing. The "abstain" condition says abstaining
# costs nothing, so its net score can be read as a measure of when a model claims a
# match (distribution plan, audit 4 review). Each prompt has its own compatibility key.
ABSTAIN_PROMPT = SYSTEM_PROMPT.replace(
    "If the budget runs out, submit your best answer.",
    "A wrong moment counts against you and an empty answer does not, so submit an "
    "empty list whenever you are not confident that a moment matches.")
PROMPTS = {"standard": SYSTEM_PROMPT, "abstain": ABSTAIN_PROMPT}


def system_prompt(condition: str = "standard") -> str:
    if condition not in PROMPTS:
        raise ValueError(f"prompt condition must be one of {sorted(PROMPTS)}")
    return PROMPTS[condition]


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
    # The model the provider reported on each turn, in order (None where it reported none).
    turn_models: list[str | None] = field(default_factory=list)


class EpisodeError(RuntimeError):
    """A failure partway through an episode; carries what was spent and done so far."""

    def __init__(self, cause: BaseException, usage: Usage, trace: list[dict[str, Any]],
                 turn_models: list[str | None] | None = None):
        super().__init__(f"{type(cause).__name__}: {cause}")
        self.cause = cause
        self.usage = usage
        self.trace = trace
        self.turn_models = turn_models or []


async def run_episode(
    model: AgentModel, corpus: LocalCorpus, query: Query | QueryRequest, budget: Budget,
    system: str = SYSTEM_PROMPT,
) -> Episode:
    usage = Usage()
    trace: list[dict[str, Any]] = []
    turn_models: list[str | None] = []
    try:
        return await _episode(model, corpus, query, budget, usage, trace, turn_models, system)
    except Exception as e:
        # Keep tokens already spent and the tool trace: a failed trial is still costed
        # and inspectable (audit finding 27).
        raise EpisodeError(e, usage, trace, turn_models) from e


async def _episode(model, corpus, query, budget, usage, trace, turn_models,
                   system=SYSTEM_PROMPT) -> Episode:
    session = model.start(system, build_task(query, budget), TOOLS)
    results: list[ToolResult] | None = None

    def end(moments: list[Moment], outcome: str) -> Episode:
        resolved = next((m for m in turn_models if m), None)
        return Episode(query.query_id, moments, usage, outcome, trace, resolved, turn_models)

    while usage.steps < budget.max_steps:
        turn = await session.step(results)
        usage.steps += 1
        usage.add(turn.usage)
        turn_models.append(turn.model)
        if len({m for m in turn_models if m}) > 1:
            # The protocol's no-silent-fallback rule: a turn answered by another model
            # ends the trial, which scores zero (audit 3, finding 3).
            trace.append({"error": "model_changed",
                          "models": [m for m in turn_models if m]})
            return end([], "model_changed")
        if turn.stopped:
            return end([], turn.stopped)
        if not turn.tool_calls:
            return end([], "no_submission")

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
                return end(moments, "submitted")
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

    return end([], "step_limit")
