"""GPT as an agentic-track entrant, via the OpenAI Responses API tool loop."""

from __future__ import annotations

import base64
import json
from typing import Any

from momentbench.agentic.budget import Usage
from momentbench.agentic.model import AgentModel, AgentSession, ModelTurn, ToolCall, ToolResult

DEFAULT_MODEL = "gpt-5.5"
MAX_OUTPUT_TOKENS = 16000


def _tool_spec(tool: dict[str, Any]) -> dict[str, Any]:
    return {
        "type": "function",
        "name": tool["name"],
        "description": tool["description"],
        "parameters": tool["input_schema"],
        "strict": False,
    }


def _result_item(result: ToolResult) -> dict[str, Any]:
    text = ("ERROR: " if result.is_error else "") + result.text
    output: list[dict[str, Any]] = [{"type": "input_text", "text": text}]
    for jpeg in result.images_jpeg:
        data = base64.standard_b64encode(jpeg).decode("ascii")
        # "high" so frames are not downscaled below what the other entrants receive.
        output.append({"type": "input_image", "detail": "high",
                       "image_url": f"data:image/jpeg;base64,{data}"})
    return {"type": "function_call_output", "call_id": result.call_id, "output": output}


class OpenAISession(AgentSession):
    def __init__(self, client: Any, model: str, effort: str | None, system: str, task: str,
                 tools: list[dict[str, Any]]):
        self._client = client
        self._model = model
        self._effort = effort
        self._system = system
        self._tools = [_tool_spec(t) for t in tools]
        self._input: list[Any] = [{"role": "user", "content": task}]

    async def step(self, tool_results: list[ToolResult] | None) -> ModelTurn:
        if tool_results:
            self._input.extend(_result_item(r) for r in tool_results)
        kwargs: dict[str, Any] = {}
        if self._effort:
            kwargs["reasoning"] = {"effort": self._effort}
        response = await self._client.responses.create(
            model=self._model,
            instructions=self._system,
            input=self._input,
            tools=self._tools,
            max_output_tokens=MAX_OUTPUT_TOKENS,
            store=False,
            include=["reasoning.encrypted_content"],
            **kwargs,
        )
        # Reasoning items go back unchanged so the model keeps its own chain across turns.
        self._input.extend(item.model_dump(exclude_none=True) for item in response.output)

        u = response.usage
        cached = (u.input_tokens_details.cached_tokens or 0) if u.input_tokens_details else 0
        usage = Usage(
            input_tokens=(u.input_tokens or 0) - cached,
            output_tokens=u.output_tokens or 0,
            cache_read_tokens=cached,
        )
        calls = [
            ToolCall(call_id=item.call_id, name=item.name,
                     arguments=json.loads(item.arguments or "{}"))
            for item in response.output
            if item.type == "function_call"
        ]
        text_parts, refused = [], False
        for item in response.output:
            if item.type == "message":
                for part in item.content:
                    if part.type == "refusal":
                        refused = True
                    elif part.type == "output_text":
                        text_parts.append(part.text)
        stopped = None
        if refused:
            stopped = "refusal"
        elif response.status == "incomplete":
            reason = getattr(response.incomplete_details, "reason", None)
            stopped = "max_tokens" if reason == "max_output_tokens" else (reason or "incomplete")
        return ModelTurn(tool_calls=[] if stopped else calls, usage=usage,
                         text="".join(text_parts), stopped=stopped,
                         model=getattr(response, "model", None))


class OpenAIAgent(AgentModel):
    def __init__(self, model: str = DEFAULT_MODEL, effort: str | None = None, client: Any = None):
        self.model_id = model
        self.effort = effort
        self._client = client

    def start(self, system: str, task: str, tools: list[dict[str, Any]]) -> AgentSession:
        if self._client is None:
            import openai

            self._client = openai.AsyncOpenAI()
        return OpenAISession(self._client, self.model_id, self.effort, system, task, tools)
