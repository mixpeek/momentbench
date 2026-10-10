"""The agentic track as an Inspect AI task.

    inspect eval momentbench/agentic/inspect_task.py --model anthropic/claude-opus-5-5 \\
        -T corpus_dir=/path/to/videos --epochs 5

Same tools, budget and scoring rule as `momentbench agentic`. The model ends a
sample by calling `submit` with a JSON array of moments.
"""

from __future__ import annotations

import base64
import json
from pathlib import Path

from inspect_ai import Task, task
from inspect_ai.dataset import MemoryDataset, Sample
from inspect_ai.scorer import Score, Scorer, Target, mean, scorer, stderr
from inspect_ai.solver import TaskState, basic_agent, system_message
from inspect_ai.tool import ContentImage, ContentText, Tool, ToolError, tool
from inspect_ai.util import store

from momentbench.agentic.budget import Budget
from momentbench.agentic.environment import MAX_FRAMES_PER_CALL, LocalCorpus, parse_submission
from momentbench.agentic.episode import SYSTEM_PROMPT, build_task
from momentbench.eval.scoring import evaluate_prediction
from momentbench.runner import load_corpus_assets, load_queries
from momentbench.types import Prediction, Query

SUBMIT_DESCRIPTION = (
    "Submit the final answer and end the task. The answer is a JSON array of up to 5 "
    'moments, most likely first: [{"asset_id": "...", "t_in": 12.0, "t_out": 18.5}]. '
    "Times are in seconds."
)


def _charge_tool_call(budget: Budget) -> None:
    used = store().get("tool_calls", 0)
    if used >= budget.max_tool_calls:
        raise ToolError("tool-call budget exhausted; call submit now")
    store().set("tool_calls", used + 1)


def corpus_tools(corpus: LocalCorpus, budget: Budget) -> list[Tool]:
    @tool
    def list_assets() -> Tool:
        async def execute() -> str:
            """List every video in the corpus with its asset_id, title and duration."""
            _charge_tool_call(budget)
            return corpus.list_assets()

        return execute

    @tool
    def search_transcript() -> Tool:
        async def execute(query: str, asset_id: str | None = None, top_k: int = 5) -> str:
            """Keyword search over spoken-word transcripts.

            Returns matching segments with asset_id and timestamps in seconds.
            Videos without speech have no transcript and never appear here.

            Args:
                query: Words to look for.
                asset_id: Restrict the search to one video.
                top_k: Number of segments to return (1 to 20).
            """
            _charge_tool_call(budget)
            return corpus.search_transcript(query, asset_id, top_k)

        return execute

    @tool
    def read_transcript() -> Tool:
        async def execute(asset_id: str, t_start: float, t_end: float) -> str:
            """Read a video's transcript between two timestamps (at most 600 seconds).

            Args:
                asset_id: The video to read.
                t_start: Start time in seconds.
                t_end: End time in seconds.
            """
            _charge_tool_call(budget)
            try:
                return corpus.read_transcript(asset_id, t_start, t_end)
            except ValueError as e:
                raise ToolError(str(e)) from e

        return execute

    @tool
    def sample_frames() -> Tool:
        async def execute(
            asset_id: str, t_start: float, t_end: float, num_frames: int
        ) -> list[ContentText | ContentImage]:
            """Return evenly spaced frames from a video between two timestamps.

            Each frame counts against the frame budget.

            Args:
                asset_id: The video to sample.
                t_start: Start time in seconds.
                t_end: End time in seconds.
                num_frames: How many frames to return (1 to 8).
            """
            _charge_tool_call(budget)
            left = budget.max_frames - store().get("frames", 0)
            if left <= 0:
                raise ToolError("frame budget exhausted; call submit now")
            want = max(1, min(num_frames, MAX_FRAMES_PER_CALL, left))
            try:
                info, frames = corpus.sample_frames(asset_id, t_start, t_end, want)
            except (ValueError, RuntimeError) as e:
                raise ToolError(str(e)) from e
            store().set("frames", store().get("frames", 0) + len(frames))
            content: list[ContentText | ContentImage] = [ContentText(text=info)]
            for jpeg in frames:
                data = base64.standard_b64encode(jpeg).decode("ascii")
                content.append(ContentImage(image=f"data:image/jpeg;base64,{data}"))
            return content

        return execute

    return [list_assets(), search_transcript(), read_transcript(), sample_frames()]


def parse_answer(completion: str, known_assets: set[str]):
    """Parse a submitted JSON array of moments. Anything unparseable is no answer."""
    start, end = completion.find("["), completion.rfind("]")
    if start == -1 or end <= start:
        return []
    try:
        items = json.loads(completion[start : end + 1])
    except json.JSONDecodeError:
        return []
    return parse_submission({"moments": items}, known_assets) if isinstance(items, list) else []


@scorer(metrics=[mean(), stderr()])
def recall_at_1(known_assets: set[str]) -> Scorer:
    async def score(state: TaskState, target: Target) -> Score:
        query = Query.model_validate(state.metadata["query"])
        moments = parse_answer(state.output.completion or "", known_assets)
        result = evaluate_prediction(
            query,
            Prediction(query_id=query.query_id, ranked_moments=moments, model_id=str(state.model)),
        )
        return Score(
            value=1.0 if result.hit_at_1 else 0.0,
            answer=json.dumps([m.model_dump(exclude={"metadata"}) for m in moments]),
            metadata={
                "hit_at_5": result.hit_at_5,
                "best_temporal_iou": result.best_temporal_iou,
                "tier": int(query.tier),
                "vertical": query.vertical,
                "frames": store().get("frames", 0),
                "tool_calls": store().get("tool_calls", 0),
            },
        )

    return score


def build_task_from(queries: list[Query], corpus: LocalCorpus, budget: Budget) -> Task:
    samples = [
        Sample(
            id=q.query_id,
            input=build_task(q, budget),
            # Gold stays in metadata for the scorer. It is never shown to the model.
            metadata={"query": q.model_dump(mode="json")},
        )
        for q in queries
    ]
    return Task(
        dataset=MemoryDataset(samples),
        solver=basic_agent(
            init=system_message(SYSTEM_PROMPT),
            tools=corpus_tools(corpus, budget),
            max_attempts=1,
            # Each turn is an assistant message plus its tool results.
            message_limit=2 * budget.max_steps + 2,
            submit_description=SUBMIT_DESCRIPTION,
        ),
        scorer=recall_at_1(set(corpus.assets)),
    )


REPO_ROOT = Path(__file__).resolve().parents[2]


def _resolve(path: str) -> Path:
    """Inspect runs a task from the task file's directory, so try the repo root too."""
    p = Path(path)
    return p if p.is_absolute() or p.exists() else REPO_ROOT / p


@task
def momentbench_agentic(
    queries: str = "data/queries/sample.json",
    corpus: str = "data/corpus_manifests/sample_manifest.json",
    corpus_dir: str | None = None,
    max_steps: int = 30,
    max_frames: int = 64,
    max_tool_calls: int = 60,
) -> Task:
    """Corpus-scale video moment retrieval with tools. Use --epochs for repeated trials."""
    local = LocalCorpus(
        load_corpus_assets(_resolve(corpus)), _resolve(corpus_dir) if corpus_dir else None
    )
    budget = Budget(max_steps=max_steps, max_frames=max_frames, max_tool_calls=max_tool_calls)
    return build_task_from(load_queries(_resolve(queries)), local, budget)
