"""Run the agentic track: every query, several trials, scored and costed."""

from __future__ import annotations

import asyncio
import json
import threading
import time
from pathlib import Path
from statistics import mean
from typing import Any

from pydantic import BaseModel, Field

from momentbench import spend
from momentbench.agentic.budget import Budget, Pricing, Usage
from momentbench.agentic.environment import TOOLS, LocalCorpus
from momentbench.agentic.episode import EpisodeError, run_episode, system_prompt
from momentbench.agentic.identity import new_run_id, task_identity, text_sha
from momentbench.agentic.model import AgentModel, ToolCall, ToolResult
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
    # True when the query has no gold moment (the right answer is to abstain).
    no_answer: bool = False
    # Hash of the request text the model saw; rescoring skips a trial whose query text
    # has changed since (a new wording can deserve a different answer).
    query_text_sha: str | None = None
    # Model identifier the provider reported for this trial's responses.
    resolved_model: str | None = None
    # The model reported on each turn. More than one distinct value ends the trial with
    # outcome "model_changed" and makes the run inadmissible.
    turn_models: list[str | None] = Field(default_factory=list)
    # tIoU of the first submitted moment against its best gold span; 1.0 for a correct
    # abstention on a no-answer query. Drives Recall@1 at other thresholds.
    top1_iou: float = 0.0
    usage: Usage
    cost_usd: float | None = None
    latency_s: float
    moments: list[Moment] = Field(default_factory=list)
    trace: list[dict[str, Any]] = Field(default_factory=list)
    error: str | None = None


def frame_condition(corpus: Any) -> dict[str, Any] | None:
    """Frame width and JPEG quality when they differ from the defaults, else None."""
    from momentbench.agentic.environment import FRAME_WIDTH_PX

    width = getattr(corpus, "frame_width", FRAME_WIDTH_PX)
    quality = getattr(corpus, "jpeg_quality", None)
    if (width, quality) == (FRAME_WIDTH_PX, None):
        return None
    return {"width": width, "jpeg_quality": quality}


def resolved_models(trials: list[AgenticTrial]) -> list[str]:
    """Every model identity a provider reported, over all turns of all trials."""
    return sorted({m for t in trials for m in [t.resolved_model, *t.turn_models] if m})


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
    # Abstention and localization are different skills; report them apart (audit 17).
    # positive_*: queries with a gold moment. no_answer_accuracy: share of no-answer
    # trials that explicitly abstained. false_positive_rate: share of no-answer trials
    # that returned a moment. abstention_on_positives: share of positive trials that
    # explicitly abstained. mean_tiou_positive excludes no-answer queries.
    positive_recall_at_1: float | None = None
    no_answer_accuracy: float | None = None
    false_positive_rate: float | None = None
    abstention_on_positives: float | None = None
    mean_tiou_positive: float | None = None
    # Net score: +1 for a correct answer (a hit, or an abstention on a no-answer query),
    # -1 for a wrong one (any submitted moment that is not a hit), 0 for abstaining or
    # not answering. The same rule as closed-book factuality net scores (correct minus
    # incorrect): a wrong answer costs more than saying nothing matches.
    net_score: float | None = None
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
    # True when titles were hidden and asset ids replaced with opaque aliases.
    anonymize_titles: bool = False
    # Number of assets the model could search (list_assets). 0 in runs saved before
    # this field existed.
    corpus_assets: int = 0
    # False for debugging runs on an incomplete or unverified corpus; such runs never
    # enter a results table (no-answer queries survive a partial corpus and inflate it).
    reportable: bool = True
    # asset_id -> sha256 of every corpus file the run could search, from verification.
    corpus_receipt: dict[str, str] = Field(default_factory=dict)
    # Commit, input hashes, provider, endpoint, packages and a compatibility key
    # (momentbench.agentic.identity).
    identity: dict[str, Any] = Field(default_factory=dict)
    # Trials left out when this run was rescored because their query text changed.
    rescore_excluded: int = 0
    budget: Budget
    pricing: Pricing | None
    timestamp: str
    summary: AgenticSummary
    trials: list[AgenticTrial]


IOU_THRESHOLDS = (0.3, 0.5, 0.7)


class MeteredCorpus:
    """The only corpus a custom run_episode gets: every operation is metered and capped.

    It exposes the four tools, execute, parse_answer and has_transcripts, and nothing
    else, so there is no unmetered path to frames or transcripts. Each operation
    counts as a tool call against the harness's own budget (whatever budget the
    entrant passes is ignored), frames count as drawn, and a call past either limit is
    refused. Calls are serialised, so parallel calls cannot overdraw. This bounds what
    an in-process entrant can see; untrusted code still needs process isolation.
    """

    __slots__ = ("_MeteredCorpus__corpus", "_MeteredCorpus__budget", "_MeteredCorpus__lock",
                 "usage")

    def __init__(self, corpus: LocalCorpus, budget: Budget):
        self.__corpus = corpus
        self.__budget = budget
        self.__lock = threading.Lock()
        self.usage = Usage()

    def execute(self, call: ToolCall, budget: Budget | None = None,
                usage: Usage | None = None) -> ToolResult:
        with self.__lock:
            before = (self.usage.tool_calls, self.usage.frames)
            result = self.__corpus.execute(call, self.__budget, self.usage)
        if usage is not None:
            usage.tool_calls += self.usage.tool_calls - before[0]
            usage.frames += self.usage.frames - before[1]
        return result

    def _tool(self, name: str, **arguments: Any) -> ToolResult:
        result = self.execute(ToolCall(name, name, arguments))
        if result.is_error:
            raise ValueError(result.text.removeprefix("error: "))
        return result

    def list_assets(self) -> str:
        return self._tool("list_assets").text

    def search_transcript(self, query: str, asset_id: str | None = None,
                          top_k: int = 5) -> str:
        return self._tool("search_transcript", query=query, asset_id=asset_id,
                          top_k=top_k).text

    def read_transcript(self, asset_id: str, t_start: float, t_end: float) -> str:
        return self._tool("read_transcript", asset_id=asset_id, t_start=t_start,
                          t_end=t_end).text

    def sample_frames(self, asset_id: str, t_start: float, t_end: float,
                      num_frames: int) -> tuple[str, list[bytes]]:
        r = self._tool("sample_frames", asset_id=asset_id, t_start=t_start, t_end=t_end,
                       num_frames=num_frames)
        return r.text, list(r.images_jpeg)

    def parse_answer(self, arguments: Any) -> list[Moment]:
        return self.__corpus.parse_answer(arguments)

    def has_transcripts(self) -> bool:
        return self.__corpus.has_transcripts()


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

    A run stores each trial's submitted moments, so a corrected span changes the score
    without rerunning the model. A changed request is different: the old answer was to
    a different question, so that trial is left out and counted in rescore_excluded.
    Runs from before trials recorded query_text_sha fall back to when the refinement
    was applied (applied_at, UTC like the run timestamp; else the day, conservatively).
    """
    trials, excluded = [], 0
    run_at = run.timestamp or ""
    for t in run.trials:
        q = queries.get(t.query_id)
        if q is None:
            continue
        if t.query_text_sha is not None:
            changed = t.query_text_sha != text_sha(q.text)
        else:
            refined = q.metadata.get("text_refined") or {}
            if refined.get("applied_at"):
                changed = run_at < refined["applied_at"]
            else:
                changed = bool(refined) and run_at[:10] <= refined.get("date", "")[:10]
        if changed:
            excluded += 1
            continue
        hit1, hit5, best, top1 = score_trial(q, t.moments, t.outcome)
        trials.append(t.model_copy(update={"hit_at_1": hit1, "hit_at_5": hit5,
                                           "best_temporal_iou": best, "top1_iou": top1,
                                           "tier": int(q.tier),
                                           "no_answer": q.is_no_answer}))
    return run.model_copy(update={"trials": trials, "rescore_excluded": excluded,
                                  "summary": summarize(trials, run.summary.num_trials)})


def trial_net(t: AgenticTrial) -> int:
    """+1 correct, -1 a submitted moment that is wrong, 0 abstained or no answer."""
    if t.hit_at_1:
        return 1
    return -1 if t.moments else 0


def _split_metrics(trials: list[AgenticTrial]) -> dict[str, float | None]:
    pos = [t for t in trials if not t.no_answer]
    neg = [t for t in trials if t.no_answer]

    def rate(ts, pred):
        return mean(1.0 if pred(t) else 0.0 for t in ts) if ts else None

    def abstained(t):
        return t.outcome == "submitted" and not t.moments

    return {
        "positive_recall_at_1": rate(pos, lambda t: t.hit_at_1),
        "no_answer_accuracy": rate(neg, abstained),
        "false_positive_rate": rate(neg, lambda t: t.outcome == "submitted" and bool(t.moments)),
        "abstention_on_positives": rate(pos, abstained),
        "mean_tiou_positive": mean(t.best_temporal_iou for t in pos) if pos else None,
        "net_score": mean(trial_net(t) for t in trials) if trials else None,
    }


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
        **_split_metrics(trials),
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
    identity: dict[str, Any] | None = None,
    prompt_condition: str = "standard",
) -> AgenticRun:
    """Run every query `trials` times and score the answers.

    A run is reportable only when `corpus` was built by prepare_corpus on a fully
    verified corpus; `reportable=False` can only lower that. `identity` adds
    environment fields (environment_identity); the task identity and its
    compatibility key are always computed here, from what the run was given.
    """
    budget = budget or Budget()
    system = system_prompt(prompt_condition)
    if hasattr(model, "run_episode"):  # entrants with their own loop read it from here
        model.system_prompt = system
    reportable = reportable and bool(getattr(corpus, "fully_verified", False))
    corpus_receipt = corpus_receipt or dict(getattr(corpus, "receipt", {}) or {})
    semaphore = asyncio.Semaphore(max_concurrency)
    # Cash guard (momentbench.spend): only API runs spend money; the Claude Code
    # harness runs on a subscription and the floors call nothing.
    billable = (spend.active() and getattr(model, "harness", "api") == "api"
                and not model.model_id.startswith("baseline-"))
    if billable and pricing is None:
        raise spend.SpendCapError(f"{model.model_id}: no pricing, so the spend ledger "
                                  "cannot enforce its cap; pass --pricing")
    # The most one trial can cost, reserved before it starts and settled after.
    trial_max = (pricing.cost_usd(Usage(
        input_tokens=spend.MAX_TRIAL_INPUT_TOKENS,
        output_tokens=budget.max_steps * spend.MAX_OUTPUT_TOKENS_PER_TURN))
                 if billable else 0.0)

    async def one(query: Query, trial: int) -> AgenticTrial:
        async with semaphore:
            reservation = None
            if billable:
                try:
                    reservation = spend.reserve(trial_max, f"agentic:{model.model_id}")
                except spend.SpendCapError as e:
                    raise ProviderUnavailableError(str(e)) from e
            start = time.perf_counter()
            error = None
            resolved = None
            turn_models: list[str | None] = []
            try:
                request = query.request()  # entrants never see gold, tier or negatives
                if hasattr(model, "run_episode"):  # entrants that run their own tool loop
                    metered = MeteredCorpus(corpus, budget)
                    episode = await model.run_episode(metered, request, budget)
                    # The harness owns tool accounting: replace whatever the entrant
                    # reported with what actually went through the corpus tools.
                    episode.usage.tool_calls = metered.usage.tool_calls
                    episode.usage.frames = metered.usage.frames
                else:
                    episode = await run_episode(model, corpus, request, budget, system)
                moments, usage, outcome, trace = (
                    episode.moments, episode.usage, episode.outcome, episode.trace
                )
                resolved = episode.resolved_model
                turn_models = list(getattr(episode, "turn_models", []) or [])
                # The same identity rule for every route, including entrants that run
                # their own loop: two reported models end the trial (audit 4, finding 1).
                if len({m for m in [resolved, *turn_models] if m}) > 1 \
                        and outcome != "model_changed":
                    moments, outcome = [], "model_changed"
                over = budget_violation(usage, budget)
                if over:
                    moments, outcome, error = [], "budget_exceeded", over
            except Exception as e:  # a failed trial scores zero and is counted, never dropped
                cause = e.cause if isinstance(e, EpisodeError) else e
                if is_fatal_provider_error(cause):
                    raise ProviderUnavailableError(f"{type(cause).__name__}: {cause}") from e
                usage = e.usage if isinstance(e, EpisodeError) else Usage()
                turn_models = list(getattr(e, "turn_models", []) or [])
                trace = e.trace if isinstance(e, EpisodeError) else []
                moments, outcome = [], "error"
                error = f"{type(cause).__name__}: {cause}"
            latency = time.perf_counter() - start
        cost = pricing.cost_usd(usage) if pricing else None
        if billable and outcome != "error":
            spend.settle(reservation, cost or 0.0, f"agentic:{model.model_id}",
                         query=query.query_id, trial=trial)
        # An errored trial (a timeout, a dropped connection) may have been billed for
        # more than it reported, so its reservation stays open at the maximum.
        hit1, hit5, best, top1 = score_trial(query, moments, outcome)
        return AgenticTrial(
            query_id=query.query_id,
            trial=trial,
            tier=int(query.tier),
            vertical=query.vertical,
            no_answer=query.is_no_answer,
            query_text_sha=text_sha(query.text),
            resolved_model=resolved,
            turn_models=turn_models,
            outcome=outcome,
            hit_at_1=hit1,
            hit_at_5=hit5,
            best_temporal_iou=best,
            top1_iou=top1,
            usage=usage,
            cost_usd=cost,
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
            partial = output_dir / f"{new_run_id(model.model_id)}.partial.json"
            partial.write_text(json.dumps(
                {"model_id": model.model_id, "complete": False,
                 "trials": [t.model_dump() for t in done]}, indent=2))
        raise
    run = AgenticRun(
        run_id=new_run_id(model.model_id),
        model_id=model.model_id,
        effort=model.effort,
        harness=getattr(model, "harness", "api"),
        modalities="frames+transcripts" if corpus.has_transcripts() else "frames",
        anonymize_titles=getattr(corpus, "anonymize_titles", False),
        corpus_assets=len(corpus.assets),
        reportable=reportable,
        corpus_receipt=corpus_receipt or {},
        identity={**(identity or {}),
                  **task_identity(
                      queries=queries, receipt=corpus_receipt,
                      transcripts=getattr(corpus, "transcripts", {}),
                      system_prompt=system, tools=TOOLS, budget=budget.model_dump(),
                      modalities="frames+transcripts" if corpus.has_transcripts() else "frames",
                      anonymize_titles=getattr(corpus, "anonymize_titles", False),
                      asset_view=corpus.list_assets(), frames=frame_condition(corpus),
                      prompt_condition=prompt_condition,
                      input_token_cap=spend.MAX_TRIAL_INPUT_TOKENS if billable else None),
                  "resolved_models": resolved_models(results)},
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
