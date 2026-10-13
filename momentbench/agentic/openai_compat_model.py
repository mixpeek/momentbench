"""Any model behind an OpenAI-compatible Chat Completions endpoint, as an agentic entrant.

This is how most open-weight vision-language models are served: vLLM, SGLang, Ollama,
LM Studio, TGI and hosted routers all speak `/v1/chat/completions` with tool calling.

    momentbench agentic Qwen/Qwen3-VL-32B-Instruct --provider openai-compatible \\
        --base-url http://localhost:8000/v1 ...

Chat Completions cannot put images inside a tool result, so each tool result goes back
as a `tool` message with its text, and any frames follow in one `user` message that
says which call they came from. The model sees the same frames, at the same
resolution, as every other entrant.
"""

from __future__ import annotations

import base64
import json
import os
from typing import Any

from momentbench.agentic.budget import Usage
from momentbench.agentic.model import AgentModel, AgentSession, ModelTurn, ToolCall, ToolResult

MAX_TOKENS = 8000


def _tool_spec(tool: dict[str, Any]) -> dict[str, Any]:
    return {"type": "function", "function": {"name": tool["name"],
                                             "description": tool["description"],
                                             "parameters": tool["input_schema"]}}


class OpenAICompatSession(AgentSession):
    def __init__(self, client: Any, model: str, system: str, task: str,
                 tools: list[dict[str, Any]], extra_body: dict[str, Any] | None = None):
        self._client = client
        self._model = model
        self._tools = [_tool_spec(t) for t in tools]
        self._extra_body = extra_body or {}
        self._messages: list[dict[str, Any]] = [
            {"role": "system", "content": system},
            {"role": "user", "content": task},
        ]

    async def step(self, tool_results: list[ToolResult] | None) -> ModelTurn:
        if tool_results:
            frames: list[dict[str, Any]] = []
            for r in tool_results:
                text = ("ERROR: " if r.is_error else "") + r.text
                if r.images_jpeg:
                    text += f"\n[{len(r.images_jpeg)} frames follow in the next message]"
                self._messages.append({"role": "tool", "tool_call_id": r.call_id, "content": text})
                if r.images_jpeg:
                    frames.append({"type": "text", "text": f"Frames from tool call {r.call_id}, "
                                                           "in order:"})
                    frames += [{"type": "image_url", "image_url": {
                        "url": "data:image/jpeg;base64,"
                               + base64.standard_b64encode(j).decode("ascii")}}
                        for j in r.images_jpeg]
            if frames:
                self._messages.append({"role": "user", "content": frames})
        response = await self._client.chat.completions.create(
            model=self._model, messages=self._messages, tools=self._tools,
            max_tokens=MAX_TOKENS, extra_body=self._extra_body or None,
        )
        choice = response.choices[0]
        message = choice.message
        self._messages.append(message.model_dump(exclude_none=True))

        u = response.usage
        details = getattr(u, "prompt_tokens_details", None) if u else None
        cached = (getattr(details, "cached_tokens", 0) or 0) if details else 0
        usage = Usage(input_tokens=((u.prompt_tokens or 0) - cached) if u else 0,
                      output_tokens=(u.completion_tokens or 0) if u else 0,
                      cache_read_tokens=cached)
        calls = []
        for c in message.tool_calls or []:
            try:
                args = json.loads(c.function.arguments or "{}")
            except json.JSONDecodeError:
                args = {"_unparseable": c.function.arguments}
            calls.append(ToolCall(call_id=c.id, name=c.function.name, arguments=args))
        stopped = {"length": "max_tokens", "content_filter": "refusal"}.get(choice.finish_reason)
        if getattr(message, "refusal", None):
            stopped = "refusal"
        return ModelTurn(tool_calls=[] if stopped else calls, usage=usage,
                         text=message.content or "", stopped=stopped,
                         model=getattr(response, "model", None))


class OpenAICompatAgent(AgentModel):
    """`base_url` defaults to $OPENAI_BASE_URL; the key to $OPENAI_API_KEY, or "none"."""

    def __init__(self, model: str, effort: str | None = None, base_url: str | None = None,
                 api_key: str | None = None, client: Any = None,
                 extra_body: dict[str, Any] | None = None):
        self.model_id = model
        self.effort = effort
        self._base_url = base_url or os.environ.get("OPENAI_BASE_URL")
        self._api_key = api_key or os.environ.get("OPENAI_API_KEY") or "none"
        self._client = client
        self._extra_body = extra_body

    def start(self, system: str, task: str, tools: list[dict[str, Any]]) -> AgentSession:
        if self._client is None:
            import openai

            if not self._base_url:
                raise ValueError("openai-compatible needs --base-url or OPENAI_BASE_URL")
            self._client = openai.AsyncOpenAI(base_url=self._base_url, api_key=self._api_key)
        return OpenAICompatSession(self._client, self.model_id, system, task, tools,
                                   self._extra_body)
