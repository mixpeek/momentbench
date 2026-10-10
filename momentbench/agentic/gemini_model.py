"""Gemini as an agentic-track entrant, via the google-genai function-calling loop."""

from __future__ import annotations

import asyncio
import os
import re
from typing import Any

from momentbench.agentic.budget import Usage
from momentbench.agentic.model import AgentModel, AgentSession, ModelTurn, ToolCall, ToolResult

DEFAULT_MODEL = "gemini-3.1-pro-preview"
MAX_OUTPUT_TOKENS = 16000
RATE_LIMIT_RETRIES = 8
STOPS = {"SAFETY": "refusal", "PROHIBITED_CONTENT": "refusal", "BLOCKLIST": "refusal",
         "RECITATION": "refusal", "MAX_TOKENS": "max_tokens"}


class GeminiSession(AgentSession):
    def __init__(self, client: Any, model: str, effort: str | None, system: str, task: str,
                 tools: list[dict[str, Any]]):
        from google.genai import types

        self._types = types
        self._client = client
        self._model = model
        thinking = types.ThinkingConfig(thinking_level=effort) if effort else None
        self._config = types.GenerateContentConfig(
            system_instruction=system,
            max_output_tokens=MAX_OUTPUT_TOKENS,
            thinking_config=thinking,
            tools=[types.Tool(function_declarations=[
                types.FunctionDeclaration(name=t["name"], description=t["description"],
                                          parameters_json_schema=t["input_schema"])
                for t in tools
            ])],
            automatic_function_calling=types.AutomaticFunctionCallingConfig(disable=True),
        )
        self._contents: list[Any] = [types.Content(role="user", parts=[types.Part(text=task)])]
        self._names: dict[str, str] = {}
        self._server_ids: set[str] = set()

    async def step(self, tool_results: list[ToolResult] | None) -> ModelTurn:
        t = self._types
        if tool_results:
            parts = [
                t.Part(function_response=t.FunctionResponse(
                    id=r.call_id if r.call_id in self._server_ids else None,
                    name=self._names.get(r.call_id, ""),
                    response={"error": r.text} if r.is_error else {"output": r.text},
                    parts=[
                        t.FunctionResponsePart(inline_data=t.FunctionResponseBlob(
                            mime_type="image/jpeg", data=jpeg))
                        for jpeg in r.images_jpeg
                    ] or None,
                ))
                for r in tool_results
            ]
            self._contents.append(t.Content(role="user", parts=parts))
        response = await self._generate()
        u = response.usage_metadata
        cached = (u.cached_content_token_count or 0) if u else 0
        usage = Usage(
            input_tokens=((u.prompt_token_count or 0) - cached) if u else 0,
            # Thinking tokens bill at the output rate.
            output_tokens=((u.candidates_token_count or 0) + (u.thoughts_token_count or 0))
            if u else 0,
            cache_read_tokens=cached,
        )
        if not response.candidates:
            return ModelTurn(tool_calls=[], usage=usage, stopped="refusal")
        candidate = response.candidates[0]
        finish = candidate.finish_reason.name if candidate.finish_reason else None
        stopped = STOPS.get(finish or "")
        content = candidate.content
        parts = (content.parts if content else None) or []
        # Content goes back unchanged, thought signatures included.
        if content:
            self._contents.append(content)
        calls = []
        for i, part in enumerate(parts):
            fc = part.function_call
            if fc:
                call_id = fc.id or f"call-{len(self._contents)}-{i}"
                if fc.id:
                    self._server_ids.add(fc.id)
                self._names[call_id] = fc.name
                calls.append(ToolCall(call_id=call_id, name=fc.name, arguments=dict(fc.args or {})))
        text = "".join(p.text for p in parts if p.text and not p.thought)
        return ModelTurn(tool_calls=[] if stopped else calls, usage=usage, text=text,
                         stopped=stopped)


    async def _generate(self) -> Any:
        """Call the model, waiting out per-minute quota errors (429) rather than failing."""
        for attempt in range(RATE_LIMIT_RETRIES + 1):
            try:
                return await self._client.aio.models.generate_content(
                    model=self._model, contents=self._contents, config=self._config
                )
            except Exception as e:
                if getattr(e, "code", None) != 429 or attempt == RATE_LIMIT_RETRIES:
                    raise
                hint = re.search(r"retry in ([\d.]+)s", str(e))
                await asyncio.sleep(float(hint.group(1)) + 1 if hint else 30 * (attempt + 1))
        raise AssertionError("unreachable")


class GeminiAgent(AgentModel):
    def __init__(self, model: str = DEFAULT_MODEL, effort: str | None = None, client: Any = None):
        self.model_id = model
        self.effort = effort
        self._client = client

    def start(self, system: str, task: str, tools: list[dict[str, Any]]) -> AgentSession:
        if self._client is None:
            from google import genai

            key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
            self._client = genai.Client(api_key=key)
        return GeminiSession(self._client, self.model_id, self.effort, system, task, tools)
