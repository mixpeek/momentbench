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
    # Bound each scripted trial at $0.22 (it really costs $0.11).
    monkeypatch.setattr(spend, "MAX_TRIAL_INPUT_TOKENS", 200)
    monkeypatch.setattr(spend, "MAX_OUTPUT_TOKENS_PER_TURN", 1)  # 30 turns -> 30 tokens
    return path


def test_off_unless_configured(monkeypatch):
    monkeypatch.delenv("MOMENTBENCH_SPEND_LEDGER", raising=False)
    spend.reserve(1e9, "anything")  # no ledger, no cap
    spend.record(5.0, "anything")
    assert spend.spent_usd() == 0.0


def test_reserve_refuses_past_the_cap(ledger):
    spend.record(0.9, "x:a")
    spend.reserve(0.1, "fits")
    with pytest.raises(spend.SpendCapError, match="spend cap"):
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


def test_unknown_models_are_refused():
    with pytest.raises(spend.SpendCapError, match="no price"):
        spend.token_cost_usd("no-such-model", 1_000_000, 0)
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


def test_reservations_are_persisted_across_callers(ledger, monkeypatch):
    """The reviewer's reproduction: two $60 reservations against a $100 cap."""
    monkeypatch.setenv("MOMENTBENCH_SPEND_CAP_USD", "100")
    first = spend.reserve(60.0, "a")
    with pytest.raises(spend.SpendCapError, match="reserved"):
        spend.reserve(60.0, "b")
    spend.settle(first, 12.0, "a")
    assert spend.summary()["spent_usd"] == 12.0
    spend.reserve(60.0, "b")  # fits once the first settled at its real cost


def test_an_unsettled_reservation_keeps_counting(ledger):
    spend.reserve(0.8, "crashed:call")  # never settled: billing unknown
    with pytest.raises(spend.SpendCapError):
        spend.reserve(0.3, "next")
    assert spend.summary()["open_reservations"] == 1


def test_the_cap_cannot_be_raised_past_the_hard_cap(ledger, monkeypatch):
    monkeypatch.setenv("MOMENTBENCH_SPEND_CAP_USD", "1000")
    assert spend.cap_usd() == spend.HARD_CAP_USD


async def test_an_errored_trial_keeps_its_reservation_open(ledger, tmp_path):
    from momentbench.agentic.model import AgentModel

    class Broken(AgentModel):
        model_id = "broken"

        def start(self, system, task, tools):
            raise TimeoutError("read timed out")

    run = await run_agentic_benchmark([make_query()], Broken(), verified_corpus(tmp_path),
                                      trials=1, pricing=PRICE)
    assert run.trials[0].outcome == "error"
    s = spend.summary()
    assert s["open_reservations"] == 1 and s["spent_usd"] == 0.0 and s["reserved_usd"] > 0


def test_frozen_paid_work_is_refused(ledger):
    ledger.with_suffix(ledger.suffix + ".frozen").write_text("0.2.5 released\n")
    with pytest.raises(spend.SpendCapError, match="frozen"):
        spend.reserve(0.01, "anything")


async def test_a_trial_stops_before_its_input_reservation(ledger, tmp_path, monkeypatch):
    monkeypatch.setattr(spend, "MAX_TRIAL_INPUT_TOKENS", 250)  # stop past 200 input tokens
    monkeypatch.setattr(spend, "MAX_OUTPUT_TOKENS_PER_TURN", 1)
    looping = [[("list_assets", {})]] * 10 + [[SUBMIT_GOLD]]
    run = await run_agentic_benchmark([make_query()], ScriptedModel(looping),
                                      verified_corpus(tmp_path), trials=1, pricing=PRICE)
    assert run.trials[0].outcome == "token_cap"
    assert run.trials[0].usage.input_tokens <= 300
