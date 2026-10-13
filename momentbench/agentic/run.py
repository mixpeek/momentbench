"""Run the agentic track: every query, several trials, scored and costed."""

from __future__ import annotations

import asyncio
import json
import time
import uuid
from pathlib import Path
from statistics import mean
from typing import Any

from pydantic import BaseModel, Field

from momentbench.agentic.budget import Budget, Pricing, Usage
from momentbench.agentic.environment import LocalCorpus
from momentbench.agentic.episode import EpisodeError, run_episode
from momentbench.agentic.model import AgentModel
from momentbench.agentic.stats import bootstrap_ci
from momentbench.eval.scoring import compute_temporal_iou, evaluate_prediction
from momentbench.types import Moment, Prediction, Query


class ProviderUnavailableError(RuntimeError):
    """The provider refused the account itself (billing, quota, auth). No trial can succeed."""


# Plain "billing" is not a marker: Gemini's per-minute quota 429 mentions billing too.
_FATAL_MARKERS = ("insufficient_quota", "account is not active", "spend cap", "session limit",
                  "invalid api key",
                  "incorrect api key", "permission denied", "api key not valid")


def is_fatal_provider_error(error: Exception) -> bool:
    status = getattr(error, "status_code", None) or getattr(error, "code", None)
    text = str(error).lower()
    return status in (401, 403) or any(m in text for m in _FATAL_MARKERS)


class AgenticTrial(BaseModel):
    query_id: str
    trial: int
    tier: int
    vertical: str
    outcome: str
    hit_at_1: bool
    hit_at_5: bool
    best_temporal_iou: float
    # tIoU of the first submitted moment against its best gold span; 1.0 for a correct
    # abstention on a no-answer query. Drives Recall@1 at other thresholds.
    top1_iou: float = 0.0
    usage: Usage
    cost_usd: float | None = None
    latency_s: float
    moments: list[Moment] = Field(default_factory=list)
    trace: list[dict[str, Any]] = Field(default_factory=list)
    error: str | None = None


class AgenticSummary(BaseModel):
    num_queries: int
    num_trials: int
    recall_at_1: float
    recall_at_1_ci95: tuple[float, float]
    recall_at_5: float
    mean_temporal_iou: float
    mean_cost_usd_per_query: float | None
    mean_input_tokens_per_query: float
    mean_output_tokens_per_query: float
    mean_frames_per_query: float
    mean_steps_per_query: float
    failed_trials: int
    # Trials that raised (provider or harness errors), a subset of failed_trials. A run of
    # record should have none; a nonzero count means the score understates the model.
    error_trials: int = 0
    # Recall@1 at tIoU 0.3, 0.5 and 0.7, keyed by the threshold as a string.
    recall_at_1_by_iou: dict[str, float] = Field(default_factory=dict)
    # Per tier: number of queries, Recall@1 and its 95% CI.
    by_tier: dict[str, dict[str, Any]] = Field(default_factory=dict)


class AgenticRun(BaseModel):
    run_id: str
    track: str = "agentic"
    model_id: str
    effort: str | None
    # How the model was driven: "api" (our loop) or "claude-code" (Agent SDK loop).
    harness: str = "api"
    # "frames+transcripts" (default) or "frames" for the visual-only ablation.
    modalities: str = "frames+transcripts"
    # Number of assets the model could search (list_assets). 0 in runs saved before
    # this field existed.
    corpus_assets: int = 0
    # False for debugging runs on an incomplete or unverified corpus; such runs never
    # enter a results table (no-answer queries survive a partial corpus and inflate it).
    reportable: bool = True
    # asset_id -> sha256 of every corpus file the run could search, from verification.
    corpus_receipt: dict[str, str] = Field(default_factory=dict)
    budget: Budget
    pricing: Pricing | None
    timestamp: str
    summary: AgenticSummary
    trials: list[AgenticTrial]


IOU_THRESHOLDS = (0.3, 0.5, 0.7)


class MeteredCorpus:
    """A corpus proxy that counts every tool call and frame an entrant draws."""

    def __init__(self, corpus: LocalCorpus):
        self._corpus = corpus
        self.usage = Usage()

    def __getattr__(self, name: str) -> Any:
        return getattr(self._corpus, name)

    def execute(self, call, budget: Budget, usage: Usage):
        before = (self.usage.tool_calls, self.usage.frames)
        result = self._corpus.execute(call, budget, self.usage)
        usage.tool_calls += self.usage.tool_calls - before[0]
        usage.frames += self.usage.frames - before[1]
        return result


def budget_violation(usage: Usage, budget: Budget) -> str | None:
    """Why an episode broke its budget, or None. Over budget scores zero."""
    for used, limit, what in ((usage.steps, budget.max_steps, "turns"),
                              (usage.frames, budget.max_frames, "frames"),
                              (usage.tool_calls, budget.max_tool_calls, "tool calls")):
        if used > limit:
            return f"used {used} {what}, budget {limit}"
    return None


def score_trial(query: Query, moments: list[Moment], outcome: str
                ) -> tuple[bool, bool, float, float]:
    """(hit@1, hit@5, best tIoU, top-1 tIoU) for one trial's submission."""
    scored = evaluate_prediction(
        query, Prediction(query_id=query.query_id, ranked_moments=moments, model_id="")
    )
    # Only an explicit submission can be right. Without this, a refusal or a
    # crash would score as a correct abstention on a no-answer query.
    submitted = outcome == "submitted"
    if query.is_no_answer:
        top1 = 1.0 if submitted and not moments else 0.0
    elif submitted and moments:
        top1 = max(compute_temporal_iou(moments[0], g) for g in query.gold_moments)
    else:
        top1 = 0.0
    return (submitted and scored.hit_at_1, submitted and scored.hit_at_5,
            scored.best_temporal_iou if submitted else 0.0, top1)


def rescore(run: AgenticRun, queries: dict[str, Query]) -> AgenticRun:
    """Score a saved run against the current gold, dropping queries no longer in the set.

    A run stores each trial's submitted moments, so a corrected label changes the score
    without rerunning the model.
    """
    trials = []
    for t in run.trials:
        q = queries.get(t.query_id)
        if q is None:
            continue
        hit1, hit5, best, top1 = score_trial(q, t.moments, t.outcome)
        trials.append(t.model_copy(update={"hit_at_1": hit1, "hit_at_5": hit5,
                                           "best_temporal_iou": best, "top1_iou": top1,
                                           "tier": int(q.tier)}))
    return run.model_copy(update={"trials": trials,
                                  "summary": summarize(trials, run.summary.num_trials)})


def summarize(trials: list[AgenticTrial], num_trials: int) -> AgenticSummary:
    by_query: dict[str, list[AgenticTrial]] = {}
    for t in trials:
        by_query.setdefault(t.query_id, []).append(t)
    per_query_r1 = [mean(t.hit_at_1 for t in ts) for ts in by_query.values()]
    costs = [t.cost_usd for t in trials if t.cost_usd is not None]

    def avg(values: list[float]) -> float:
        return mean(values) if values else 0.0

    return AgenticSummary(
        num_queries=len(by_query),
        num_trials=num_trials,
        recall_at_1=avg(per_query_r1),
        recall_at_1_ci95=bootstrap_ci(per_query_r1),
        recall_at_5=avg([mean(t.hit_at_5 for t in ts) for ts in by_query.values()]),
        mean_temporal_iou=avg([t.best_temporal_iou for t in trials]),
        # Only when every trial was priced: a mean over a priced subset would mislead.
        mean_cost_usd_per_query=mean(costs) if costs and len(costs) == len(trials) else None,
        mean_input_tokens_per_query=avg([t.usage.input_tokens for t in trials]),
        mean_output_tokens_per_query=avg([t.usage.output_tokens for t in trials]),
        mean_frames_per_query=avg([t.usage.frames for t in trials]),
        mean_steps_per_query=avg([t.usage.steps for t in trials]),
        failed_trials=sum(1 for t in trials if t.outcome != "submitted"),
        error_trials=sum(1 for t in trials if t.outcome == "error"),
        recall_at_1_by_iou={
            str(th): avg([mean(t.top1_iou >= th for t in ts) for ts in by_query.values()])
            for th in IOU_THRESHOLDS
        },
        by_tier=_by_tier(by_query),
    )


def _by_tier(by_query: dict[str, list[AgenticTrial]]) -> dict[str, dict[str, Any]]:
    tiers: dict[int, list[float]] = {}
    for ts in by_query.values():
        tiers.setdefault(ts[0].tier, []).append(mean(t.hit_at_1 for t in ts))
    return {
        str(tier): {"num_queries": len(r1), "recall_at_1": mean(r1),
                    "recall_at_1_ci95": bootstrap_ci(r1)}
        for tier, r1 in sorted(tiers.items())
    }


async def run_agentic_benchmark(
    queries: list[Query],
    model: AgentModel,
    corpus: LocalCorpus,
    trials: int = 5,
    budget: Budget | None = None,
    pricing: Pricing | None = None,
    max_concurrency: int = 4,
    output_dir: Path | None = None,
    reportable: bool = True,
    corpus_receipt: dict[str, str] | None = None,
) -> AgenticRun:
    budget = budget or Budget()
    semaphore = asyncio.Semaphore(max_concurrency)

    async def one(query: Query, trial: int) -> AgenticTrial:
        async with semaphore:
            start = time.perf_counter()
            error = None
            try:
                request = query.request()  # entrants never see gold, tier or negatives
                if hasattr(model, "run_episode"):  # entrants that run their own tool loop
                    metered = MeteredCorpus(corpus)
                    episode = await model.run_episode(metered, request, budget)
                    # The harness owns tool accounting: replace whatever the entrant
                    # reported with what actually went through the corpus tools.
                    episode.usage.tool_calls = metered.usage.tool_calls
                    episode.usage.frames = metered.usage.frames
                else:
                    episode = await run_episode(model, corpus, request, budget)
                moments, usage, outcome, trace = (
                    episode.moments, episode.usage, episode.outcome, episode.trace
                )
                over = budget_violation(usage, budget)
                if over:
                    moments, outcome, error = [], "budget_exceeded", over
            except Exception as e:  # a failed trial scores zero and is counted, never dropped
                cause = e.cause if isinstance(e, EpisodeError) else e
                if is_fatal_provider_error(cause):
                    raise ProviderUnavailableError(f"{type(cause).__name__}: {cause}") from e
                usage = e.usage if isinstance(e, EpisodeError) else Usage()
                trace = e.trace if isinstance(e, EpisodeError) else []
                moments, outcome = [], "error"
                error = f"{type(cause).__name__}: {cause}"
            latency = time.perf_counter() - start
        hit1, hit5, best, top1 = score_trial(query, moments, outcome)
        return AgenticTrial(
            query_id=query.query_id,
            trial=trial,
            tier=int(query.tier),
            vertical=query.vertical,
            outcome=outcome,
            hit_at_1=hit1,
            hit_at_5=hit5,
            best_temporal_iou=best,
            top1_iou=top1,
            usage=usage,
            cost_usd=pricing.cost_usd(usage) if pricing else None,
            latency_s=latency,
            moments=moments,
            trace=trace,
            error=error,
        )

    tasks = [asyncio.ensure_future(one(q, t)) for q in queries for t in range(trials)]
    try:
        results = await asyncio.gather(*tasks)
    except ProviderUnavailableError:
        # Nothing further can succeed, so stop spending. The run is not a result, but
        # trials that finished were paid for: keep them in a .partial.json file, which
        # no results table reads, so they can be inspected or resumed from.
        for task in tasks:
            task.cancel()
        outcomes = await asyncio.gather(*tasks, return_exceptions=True)
        done = [t for t in outcomes if isinstance(t, AgenticTrial)]
        if output_dir and done:
            output_dir.mkdir(parents=True, exist_ok=True)
            partial = output_dir / f"agentic-{model.model_id}-{uuid.uuid4().hex[:8]}.partial.json"
            partial.write_text(json.dumps(
                {"model_id": model.model_id, "complete": False,
                 "trials": [t.model_dump() for t in done]}, indent=2))
        raise
    run = AgenticRun(
        run_id=f"agentic-{model.model_id}-{uuid.uuid4().hex[:8]}",
        model_id=model.model_id,
        effort=model.effort,
        harness=getattr(model, "harness", "api"),
        modalities="frames+transcripts" if corpus.has_transcripts() else "frames",
        corpus_assets=len(corpus.assets),
        reportable=reportable,
        corpus_receipt=corpus_receipt or {},
        budget=budget,
        pricing=pricing,
        timestamp=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        summary=summarize(list(results), trials),
        trials=list(results),
    )
    # A run where nothing executed (bad credentials, missing dependency) is not a result.
    if output_dir and any(t.outcome != "error" for t in run.trials):
        output_dir.mkdir(parents=True, exist_ok=True)
        (output_dir / f"{run.run_id}.json").write_text(run.model_dump_json(indent=2))
    return run
