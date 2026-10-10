"""Run the agentic track: every query, several trials, scored and costed."""

from __future__ import annotations

import asyncio
import time
import uuid
from pathlib import Path
from statistics import mean
from typing import Any

from pydantic import BaseModel, Field

from momentbench.agentic.budget import Budget, Pricing, Usage
from momentbench.agentic.environment import LocalCorpus
from momentbench.agentic.episode import run_episode
from momentbench.agentic.model import AgentModel
from momentbench.agentic.stats import bootstrap_ci
from momentbench.eval.scoring import evaluate_prediction
from momentbench.types import Moment, Prediction, Query


class AgenticTrial(BaseModel):
    query_id: str
    trial: int
    tier: int
    vertical: str
    outcome: str
    hit_at_1: bool
    hit_at_5: bool
    best_temporal_iou: float
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


class AgenticRun(BaseModel):
    run_id: str
    track: str = "agentic"
    model_id: str
    effort: str | None
    budget: Budget
    pricing: Pricing | None
    timestamp: str
    summary: AgenticSummary
    trials: list[AgenticTrial]


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
    )


async def run_agentic_benchmark(
    queries: list[Query],
    model: AgentModel,
    corpus: LocalCorpus,
    trials: int = 5,
    budget: Budget | None = None,
    pricing: Pricing | None = None,
    max_concurrency: int = 4,
    output_dir: Path | None = None,
) -> AgenticRun:
    budget = budget or Budget()
    semaphore = asyncio.Semaphore(max_concurrency)

    async def one(query: Query, trial: int) -> AgenticTrial:
        async with semaphore:
            start = time.perf_counter()
            error = None
            try:
                episode = await run_episode(model, corpus, query, budget)
                moments, usage, outcome, trace = (
                    episode.moments, episode.usage, episode.outcome, episode.trace
                )
            except Exception as e:  # a failed trial scores zero and is counted, never dropped
                moments, usage, outcome, trace = [], Usage(), "error", []
                error = f"{type(e).__name__}: {e}"
            latency = time.perf_counter() - start
        scored = evaluate_prediction(
            query,
            Prediction(query_id=query.query_id, ranked_moments=moments, model_id=model.model_id),
        )
        # Only an explicit submission can be right. Without this, a refusal or a
        # crash would score as a correct abstention on a no-answer query.
        submitted = outcome == "submitted"
        return AgenticTrial(
            query_id=query.query_id,
            trial=trial,
            tier=int(query.tier),
            vertical=query.vertical,
            outcome=outcome,
            hit_at_1=submitted and scored.hit_at_1,
            hit_at_5=submitted and scored.hit_at_5,
            best_temporal_iou=scored.best_temporal_iou if submitted else 0.0,
            usage=usage,
            cost_usd=pricing.cost_usd(usage) if pricing else None,
            latency_s=latency,
            moments=moments,
            trace=trace,
            error=error,
        )

    results = await asyncio.gather(*(one(q, t) for q in queries for t in range(trials)))
    run = AgenticRun(
        run_id=f"agentic-{model.model_id}-{uuid.uuid4().hex[:8]}",
        model_id=model.model_id,
        effort=model.effort,
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
