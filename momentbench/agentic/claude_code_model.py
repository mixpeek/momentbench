"""Run the agentic track through a local Claude Code install (Claude Agent SDK).

Claude Code runs the tool loop itself, so this entrant brings its own episode
driver. The corpus tools are served in-process over MCP and charge the same
budget as the API entrant. Built-in Claude Code tools, settings files, CLAUDE.md
and other MCP servers are all switched off, and the system prompt is replaced
with the benchmark's, so the model sees only the task and the corpus tools.

Two differences from the API entrant remain and are recorded on every run:
the tool names carry an `mcp__momentbench__` prefix, and the turn limit is
Claude Code's `max_turns`. Authentication is whatever the local `claude` CLI is
signed in with.
"""

from __future__ import annotations

import base64
import tempfile
from typing import Any

from momentbench.agentic.budget import Budget, Usage
from momentbench.agentic.environment import (
    TOOLS,
    InvalidSubmissionError,
    LocalCorpus,
    parse_submission,
)
from momentbench.agentic.episode import SYSTEM_PROMPT, Episode, build_task
from momentbench.agentic.model import AgentModel, AgentSession, ToolCall
from momentbench.types import Query, QueryRequest

SERVER = "momentbench"


class ClaudeCodeAgent(AgentModel):
    """Entrant that drives its own loop. `run_agentic_benchmark` calls `run_episode`."""

    harness = "claude-code"

    def __init__(self, model: str = "claude-opus-5-5", effort: str | None = None):
        self.model_id = model
        self.effort = effort

    def start(self, system: str, task: str, tools: list[dict[str, Any]]) -> AgentSession:
        raise NotImplementedError("ClaudeCodeAgent runs whole episodes; use run_episode")

    async def run_episode(self, corpus: LocalCorpus, query: Query | QueryRequest,
                          budget: Budget) -> Episode:
        from claude_agent_sdk import (
            ClaudeAgentOptions,
            ResultMessage,
            create_sdk_mcp_server,
            tool,
        )
        from claude_agent_sdk import (
            query as sdk_query,
        )

        usage = Usage()
        trace: list[dict[str, Any]] = []
        submitted: dict[str, Any] = {}

        def make(spec: dict[str, Any]):
            name = spec["name"]

            @tool(name, spec["description"], spec["input_schema"])
            async def handler(args: dict[str, Any]) -> dict[str, Any]:
                if name == "submit_answer":
                    if not submitted:
                        try:
                            submitted["moments"] = parse_submission(args, corpus.assets)
                        except InvalidSubmissionError as e:
                            trace.append({"tool": name, "arguments": args, "is_error": True,
                                          "error": str(e)})
                            return {"content": [{"type": "text",
                                                 "text": f"Invalid submission: {e}"}],
                                    "is_error": True}
                        submitted["arguments"] = args
                        trace.append({"tool": name, "arguments": args})
                    return {"content": [{"type": "text", "text": "Answer recorded. Stop now."}]}
                result = corpus.execute(ToolCall(name, name, args), budget, usage)
                trace.append(
                    {
                        "tool": name,
                        "arguments": args,
                        "is_error": result.is_error,
                        "frames": len(result.images_jpeg),
                    }
                )
                content: list[dict[str, Any]] = [{"type": "text", "text": result.text}]
                content += [
                    {
                        "type": "image",
                        "data": base64.b64encode(jpeg).decode(),
                        "mimeType": "image/jpeg",
                    }
                    for jpeg in result.images_jpeg
                ]
                return {"content": content, "is_error": result.is_error}

            return handler

        server = create_sdk_mcp_server(SERVER, tools=[make(spec) for spec in TOOLS])
        tool_names = [f"mcp__{SERVER}__{spec['name']}" for spec in TOOLS]
        system = SYSTEM_PROMPT + (
            f" The tools are named with the prefix mcp__{SERVER}__, for example "
            f"mcp__{SERVER}__submit_answer."
        )
        with tempfile.TemporaryDirectory(prefix="momentbench-cc-") as cwd:
            options = ClaudeAgentOptions(
                model=self.model_id,
                effort=self.effort,
                system_prompt=system,
                tools=[],
                mcp_servers={SERVER: server},
                strict_mcp_config=True,
                allowed_tools=tool_names,
                permission_mode="dontAsk",
                setting_sources=[],
                max_turns=budget.max_steps,
                cwd=cwd,
            )
            result: ResultMessage | None = None
            async for message in sdk_query(prompt=build_task(query, budget), options=options):
                if isinstance(message, ResultMessage):
                    result = message

        if result is not None:
            raw = result.usage or {}
            usage.input_tokens += raw.get("input_tokens", 0)
            usage.output_tokens += raw.get("output_tokens", 0)
            usage.cache_read_tokens += raw.get("cache_read_input_tokens", 0)
            usage.cache_write_tokens += raw.get("cache_creation_input_tokens", 0)
            usage.steps = result.num_turns

        if submitted:
            moments = submitted["moments"]
            return Episode(query.query_id, moments, usage, "submitted", trace)
        if result is not None and result.subtype == "error_max_turns":
            return Episode(query.query_id, [], usage, "step_limit", trace)
        if result is not None and result.is_error:
            raise RuntimeError(f"claude code ended with {result.subtype}: {result.errors}")
        return Episode(query.query_id, [], usage, "no_submission", trace)
