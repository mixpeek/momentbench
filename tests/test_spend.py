"""The cash ledger: off by default, a hard cap on paid runs, free runs never counted."""

import json

import pytest

from momentbench import spend
from momentbench.agentic.budget import Pricing
from momentbench.agentic.run import ProviderUnavailableError, run_agentic_benchmark
from tests.test_agentic import SUBMIT_GOLD, ScriptedModel, make_query
from tests.test_score_integrity import NO_ANSWER, verified_corpus

PRICE = Pricing(input_per_mtok=1000.0, output_per_mtok=1000.0)  # $0.11 per scripted trial


@pytest.fixture
def ledger(tmp_path, monkeypatch):
    path = tmp_path / "spend.jsonl"
    monkeypatch.setenv("MOMENTBENCH_SPEND_LEDGER", str(path))
    monkeypatch.setenv("MOMENTBENCH_SPEND_CAP_USD", "1.0")
    monkeypatch.setattr(spend, "TRIAL_ESTIMATE_USD", 0.2)
    return path


def test_off_unless_configured(monkeypatch):
    monkeypatch.delenv("MOMENTBENCH_SPEND_LEDGER", raising=False)
    spend.reserve(1e9, "anything")  # no ledger, no cap
    spend.record(5.0, "anything")
    assert spend.spent_usd() == 0.0


def test_reserve_refuses_past_the_cap(ledger):
    spend.record(0.9, "x:a")
    spend.reserve(0.1, "fits")
    with pytest.raises(spend.SpendCapError, match="already spent"):
        spend.reserve(0.2, "too much")
    assert spend.summary()["by_item"] == {"x": 0.9}


async def test_an_api_run_stops_at_the_cap_and_keeps_its_trials(ledger, tmp_path):
    out = tmp_path / "runs"
    with pytest.raises(ProviderUnavailableError, match="spend cap"):
        await run_agentic_benchmark([make_query(), NO_ANSWER] * 1, ScriptedModel([[SUBMIT_GOLD]]),
                                    verified_corpus(tmp_path), trials=20, pricing=PRICE,
                                    max_concurrency=1, output_dir=out)
    assert spend.spent_usd() <= 1.0
    entries = [json.loads(line) for line in ledger.read_text().splitlines()]
    assert entries and all(e["what"] == "agentic:scripted" for e in entries)
    assert list(out.glob("*.partial.json"))


async def test_subscription_and_floor_runs_are_free(ledger, tmp_path):
    class Subscription(ScriptedModel):
        harness = "claude-code"

    await run_agentic_benchmark([make_query()], Subscription([[SUBMIT_GOLD]]),
                                verified_corpus(tmp_path), trials=3, pricing=PRICE)
    assert spend.spent_usd() == 0.0


async def test_a_paid_run_without_pricing_is_refused(ledger, tmp_path):
    with pytest.raises(spend.SpendCapError, match="no pricing"):
        await run_agentic_benchmark([make_query()], ScriptedModel([[SUBMIT_GOLD]]),
                                    verified_corpus(tmp_path), trials=1)


def test_unknown_models_are_priced_at_the_ceiling():
    assert spend.token_cost_usd("no-such-model", 1_000_000, 0) == 5.0
    assert spend.token_cost_usd("claude-haiku-5-5", 1_000_000, 0) == 0.1


def test_gemini_client_prefers_a_key_then_vertex(monkeypatch):
    pytest.importorskip("google.genai")
    from momentbench.agentic import gemini_client

    for var in ("GEMINI_API_KEY", "GOOGLE_API_KEY", "MOMENTBENCH_VERTEX_PROJECT",
                "MOMENTBENCH_GCLOUD_ACCOUNT"):
        monkeypatch.delenv(var, raising=False)
    with pytest.raises(RuntimeError, match="MOMENTBENCH_VERTEX_PROJECT"):
        gemini_client.make_client()
    monkeypatch.setenv("GEMINI_API_KEY", "test-key")
    assert not gemini_client.make_client()._api_client.vertexai
    monkeypatch.delenv("GEMINI_API_KEY")
    monkeypatch.setenv("MOMENTBENCH_VERTEX_PROJECT", "some-project")
    monkeypatch.setenv("MOMENTBENCH_GCLOUD_ACCOUNT", "sa@example.iam.gserviceaccount.com")
    monkeypatch.setattr(gemini_client.subprocess, "run", lambda *a, **k: type(
        "R", (), {"stdout": "token\n"})())
    assert gemini_client.make_client()._api_client.vertexai
