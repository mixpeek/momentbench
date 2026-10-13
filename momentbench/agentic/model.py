"""Provider-neutral interface between the episode driver and a tool-using model."""

from __future__ import annotations

import abc
from dataclasses import dataclass, field
from typing import Any

from momentbench.agentic.budget import Usage


@dataclass
class ToolCall:
    call_id: str
    name: str
    arguments: dict[str, Any]


@dataclass
class ToolResult:
    call_id: str
    text: str
    images_jpeg: list[bytes] = field(default_factory=list)
    is_error: bool = False


@dataclass
class ModelTurn:
    tool_calls: list[ToolCall]
    usage: Usage
    text: str = ""
    # Set when the provider declined or cut the turn short ("refusal", "max_tokens").
    stopped: str | None = None
    # The model identifier the provider reported for this response, when it gives one.
    model: str | None = None


class AgentSession(abc.ABC):
    """One conversation. The session owns provider-specific message history."""

    @abc.abstractmethod
    async def step(self, tool_results: list[ToolResult] | None) -> ModelTurn:
        """Advance one model turn. Pass None on the first call."""


class AgentModel(abc.ABC):
    model_id: str
    effort: str | None = None

    @abc.abstractmethod
    def start(self, system: str, task: str, tools: list[dict[str, Any]]) -> AgentSession:
        """Open a session. `tools` are {name, description, input_schema} dicts."""
