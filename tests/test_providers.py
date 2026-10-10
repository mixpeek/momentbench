"""OpenAI and Gemini entrants, driven by fake clients."""

import json
from types import SimpleNamespace as Ns

import pytest

from momentbench.agentic.model import ToolResult
from momentbench.agentic.openai_model import OpenAISession, _result_item

TOOLS = [{"name": "list_assets", "description": "d", "input_schema": {"type": "object"}}]


class FakeResponses:
    def __init__(self, response):
        self.response = response
        self.calls = []

    async def create(self, **kwargs):
        self.calls.append({**kwargs, "input": list(kwargs["input"])})
        return self.response


def openai_response(output, status="completed"):
    item = lambda d: Ns(**d, model_dump=lambda exclude_none=True: d)  # noqa: E731
    return Ns(
        output=[item(o) for o in output],
        status=status,
        incomplete_details=None,
        usage=Ns(input_tokens=100, output_tokens=20,
                 input_tokens_details=Ns(cached_tokens=40)),
    )


class TestOpenAI:
    def test_frames_go_back_inside_the_function_output(self):
        item = _result_item(ToolResult(call_id="c1", text="ok", images_jpeg=[b"x"]))
        assert item["call_id"] == "c1"
        assert item["output"][1]["type"] == "input_image"
        assert item["output"][1]["image_url"].startswith("data:image/jpeg;base64,")

    async def test_a_function_call_becomes_a_tool_call_and_history_grows(self):
        call = {"type": "function_call", "call_id": "c1", "name": "list_assets",
                "arguments": json.dumps({})}
        fake = FakeResponses(openai_response([call]))
        session = OpenAISession(Ns(responses=fake), "gpt", "high", "sys", "task", TOOLS)
        turn = await session.step(None)
        assert [c.name for c in turn.tool_calls] == ["list_assets"]
        assert turn.usage.input_tokens == 60 and turn.usage.cache_read_tokens == 40
        assert fake.calls[0]["reasoning"] == {"effort": "high"}
        await session.step([ToolResult(call_id="c1", text="assets")])
        kinds = [i.get("type") for i in fake.calls[1]["input"]]
        assert kinds[-2:] == ["function_call", "function_call_output"]

    async def test_a_refusal_ends_the_turn(self):
        refusal = {"type": "message", "content": [Ns(type="refusal", refusal="no")]}
        session = OpenAISession(Ns(responses=FakeResponses(openai_response([refusal]))),
                                "gpt", None, "sys", "task", TOOLS)
        turn = await session.step(None)
        assert turn.stopped == "refusal" and turn.tool_calls == []


class TestGemini:
    async def test_function_calls_and_multimodal_responses(self):
        pytest.importorskip("google.genai")
        from google.genai import types

        from momentbench.agentic.gemini_model import GeminiSession

        content = types.Content(role="model", parts=[
            types.Part(function_call=types.FunctionCall(name="list_assets", args={}))
        ])
        response = Ns(
            candidates=[Ns(content=content, finish_reason=types.FinishReason.STOP)],
            usage_metadata=Ns(prompt_token_count=100, cached_content_token_count=30,
                              candidates_token_count=10, thoughts_token_count=5),
        )
        sent = []

        async def generate_content(model, contents, config):
            sent.append(list(contents))
            return response

        client = Ns(aio=Ns(models=Ns(generate_content=generate_content)))
        session = GeminiSession(client, "gemini", None, "sys", "task", TOOLS)
        turn = await session.step(None)
        assert [c.name for c in turn.tool_calls] == ["list_assets"]
        assert turn.usage.input_tokens == 70 and turn.usage.output_tokens == 15
        await session.step([ToolResult(call_id=turn.tool_calls[0].call_id, text="a",
                                       images_jpeg=[b"x"])])
        fr = sent[1][-1].parts[0].function_response
        assert fr.name == "list_assets" and fr.id is None
        assert fr.parts[0].inline_data.mime_type == "image/jpeg"


async def test_gemini_waits_out_a_rate_limit(monkeypatch):
    pytest.importorskip("google.genai")
    from google.genai import types

    from momentbench.agentic import gemini_model

    class QuotaError(Exception):
        code = 429

    calls = []
    content = types.Content(role="model", parts=[types.Part(text="done")])
    ok = Ns(candidates=[Ns(content=content, finish_reason=types.FinishReason.STOP)],
            usage_metadata=None)

    async def generate_content(model, contents, config):
        calls.append(1)
        if len(calls) == 1:
            raise QuotaError("Please retry in 0.01s.")
        return ok

    async def no_sleep(_):
        return None

    monkeypatch.setattr(gemini_model.asyncio, "sleep", no_sleep)
    client = Ns(aio=Ns(models=Ns(generate_content=generate_content)))
    session = gemini_model.GeminiSession(client, "gemma", None, "sys", "task", TOOLS)
    turn = await session.step(None)
    assert len(calls) == 2 and turn.text == "done"


async def test_openai_compatible_round_trip_through_the_real_sdk(tmp_path):
    """A vLLM-style server, scripted: the real openai SDK serialises every request."""
    openai = pytest.importorskip("openai")
    httpx = pytest.importorskip("httpx")

    from momentbench.agentic.budget import Budget
    from momentbench.agentic.environment import LocalCorpus
    from momentbench.agentic.episode import run_episode
    from momentbench.agentic.openai_compat_model import OpenAICompatAgent
    from momentbench.types import Moment, Query, QueryTier

    requests = []

    def reply(tool_name, args, call_id):
        return {"id": "r", "object": "chat.completion", "created": 0, "model": "qwen-vl",
                "choices": [{"index": 0, "finish_reason": "tool_calls", "message": {
                    "role": "assistant", "content": None, "tool_calls": [{
                        "id": call_id, "type": "function",
                        "function": {"name": tool_name, "arguments": json.dumps(args)}}]}}],
                "usage": {"prompt_tokens": 50, "completion_tokens": 5, "total_tokens": 55}}

    def handler(request):
        body = json.loads(request.content)
        requests.append(body)
        if len(requests) == 1:
            out = reply("sample_frames", {"asset_id": "v1", "t_start": 0, "t_end": 10,
                                          "num_frames": 2}, "c1")
        else:
            out = reply("submit_answer", {"moments": [{"asset_id": "v1", "t_in": 10,
                                                       "t_out": 20}]}, "c2")
        return httpx.Response(200, json=out)

    client = openai.AsyncOpenAI(base_url="http://vllm.test/v1", api_key="none",
                                http_client=httpx.AsyncClient(transport=httpx.MockTransport(handler)))
    (tmp_path / "v1.mp4").write_bytes(b"")
    corpus = LocalCorpus([{"asset_id": "v1", "title": "t", "duration_s": 100}], tmp_path,
                         frame_extractor=lambda video, t: b"jpeg", transcripts={})
    query = Query(query_id="q", text="x", tier=QueryTier.SINGLE_VISUAL, vertical="film",
                  gold_moments=[Moment(asset_id="v1", t_in=10, t_out=20)])
    agent = OpenAICompatAgent(model="qwen-vl", client=client)
    episode = await run_episode(agent, corpus, query, Budget())

    assert episode.outcome == "submitted" and episode.usage.frames == 2
    assert requests[0]["tools"][0]["type"] == "function"
    second = requests[1]["messages"]
    assert second[-2]["role"] == "tool" and second[-2]["tool_call_id"] == "c1"
    frames = second[-1]["content"]
    assert second[-1]["role"] == "user"
    assert sum(p["type"] == "image_url" for p in frames) == 2
