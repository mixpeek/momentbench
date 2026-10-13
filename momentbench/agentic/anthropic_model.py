"""Claude as an agentic-track entrant, via the Anthropic Messages API tool loop."""

from __future__ import annotations

import base64
from typing import Any

from momentbench.agentic.budget import Usage
from momentbench.agentic.model import AgentModel, AgentSession, ModelTurn, ToolCall, ToolResult

DEFAULT_MODEL = "claude-opus-5-5"
MAX_TOKENS = 16000


class AnthropicSession(AgentSession):
    def __init__(self, client: Any, model: str, effort: str | None, system: str, task: str,
                 tools: list[dict[str, Any]]):
        self._client = client
        self._model = model
        self._effort = effort
        self._system = system
        self._tools = tools
        self._messages: list[dict[str, Any]] = [{"role": "user", "content": task}]

    async def step(self, tool_results: list[ToolResult] | None) -> ModelTurn:
        if tool_results:
            self._messages.append(
                {"role": "user", "content": [_result_block(r) for r in tool_results]}
            )
        kwargs: dict[str, Any] = {}
        if self._effort:
            kwargs["output_config"] = {"effort": self._effort}
        # No refusal fallback: a turn answered by a different model would be
        # scored under this model's name. A refusal ends the trial instead.
        response = await self._client.messages.create(
            model=self._model,
            max_tokens=MAX_TOKENS,
            system=self._system,
            tools=self._tools,
            messages=self._messages,
            **kwargs,
        )
        # History is append-only and content goes back unchanged, thinking blocks included.
        self._messages.append({"role": "assistant", "content": response.content})

        u = response.usage
        usage = Usage(
            input_tokens=u.input_tokens or 0,
            output_tokens=u.output_tokens or 0,
            cache_read_tokens=getattr(u, "cache_read_input_tokens", 0) or 0,
            cache_write_tokens=getattr(u, "cache_creation_input_tokens", 0) or 0,
        )
        calls = [
            ToolCall(call_id=b.id, name=b.name, arguments=dict(b.input or {}))
            for b in response.content
            if b.type == "tool_use"
        ]
        text = "".join(b.text for b in response.content if b.type == "text")
        stop = response.stop_reason
        stopped = stop if stop in ("refusal", "max_tokens") else None
        return ModelTurn(tool_calls=[] if stopped else calls, usage=usage, text=text,
                         stopped=stopped, model=getattr(response, "model", None))


def _result_block(result: ToolResult) -> dict[str, Any]:
    content: list[dict[str, Any]] = [{"type": "text", "text": result.text}]
    for jpeg in result.images_jpeg:
        content.append(
            {
                "type": "image",
                "source": {
                    "type": "base64",
                    "media_type": "image/jpeg",
                    "data": base64.standard_b64encode(jpeg).decode("ascii"),
                },
            }
        )
    block: dict[str, Any] = {"type": "tool_result", "tool_use_id": result.call_id,
                             "content": content}
    if result.is_error:
        block["is_error"] = True
    return block


class AnthropicAgent(AgentModel):
    def __init__(self, model: str = DEFAULT_MODEL, effort: str | None = None, client: Any = None):
        self.model_id = model
        self.effort = effort
        self._client = client

    def start(self, system: str, task: str, tools: list[dict[str, Any]]) -> AgentSession:
        if self._client is None:
            import anthropic

            self._client = anthropic.AsyncAnthropic()
        return AnthropicSession(self._client, self.model_id, self.effort, system, task, tools)
