"""Evaluate your own model on MomentBench's agentic track.

    python examples/custom_agent.py --corpus-dir corpus/ --queries data/queries/film_v0.2.json
    python examples/custom_agent.py --corpus-dir corpus/ --limit 0   # a submission run

Replace `MyModelSession.step` with a call to your model. Each step receives the
results of the tool calls you made last turn (text, plus JPEG frames for
`sample_frames`) and returns the next tool calls. The harness runs the tools,
enforces the budget, scores the submission and writes a run file that
`scripts/results_table.py` can read. This stub samples frames from the first video
and submits a span, so the file runs end to end before you change anything.

The corpus is prepared the way `momentbench agentic` prepares it: every video is
hash-checked against the manifest and frames come only from the verified files. A
missing or altered video stops the run, unless you pass --partial, which runs on
what verified and marks the run not reportable. The default --limit 5 is a quick
diagnostic. A run you can submit uses --limit 0 (every query in the file) on a fully
verified corpus; `momentbench submit` checks both.

If your model is served behind an OpenAI-compatible endpoint (vLLM, SGLang,
Ollama), you do not need this file:

    momentbench agentic <model> --provider openai-compatible --base-url http://host:8000/v1 ...
"""

from __future__ import annotations

import argparse
import asyncio
import json
from pathlib import Path
from typing import Any

from momentbench.agentic.budget import Usage
from momentbench.agentic.identity import environment_identity
from momentbench.agentic.model import AgentModel, AgentSession, ModelTurn, ToolCall, ToolResult
from momentbench.agentic.prepare import CorpusNotVerifiedError, prepare_corpus
from momentbench.agentic.run import run_agentic_benchmark
from momentbench.runner import load_queries


class MyModelSession(AgentSession):
    def __init__(self, system: str, task: str, tools: list[dict[str, Any]]):
        # `system` is the fixed benchmark prompt, `task` names the moment to find and the
        # budget, `tools` are JSON-schema tool definitions. Keep your conversation here.
        self.history: list[Any] = [("system", system), ("user", task)]
        self.turn = 0

    async def step(self, tool_results: list[ToolResult] | None) -> ModelTurn:
        self.turn += 1
        if tool_results:
            for r in tool_results:
                # r.text is the tool's JSON or text output; r.images_jpeg holds frames.
                self.history.append(("tool", r.text, len(r.images_jpeg)))
        # --- replace from here with a call to your model --------------------------------
        if self.turn == 1:
            calls = [ToolCall("c1", "list_assets", {})]
        elif self.turn == 2:
            first = json.loads(tool_results[0].text)[0]
            self.asset = first["asset_id"]
            calls = [ToolCall("c2", "sample_frames", {"asset_id": self.asset, "t_start": 0,
                                                      "t_end": 30, "num_frames": 4})]
        else:
            calls = [ToolCall("c3", "submit_answer",
                              {"moments": [{"asset_id": self.asset, "t_in": 10, "t_out": 14}]})]
        # Report tokens so cost can be computed from configs/pricing.json.
        usage = Usage(input_tokens=0, output_tokens=0)
        # --- to here --------------------------------------------------------------------
        return ModelTurn(tool_calls=calls, usage=usage)


class MyModel(AgentModel):
    model_id = "my-model"
    effort = None

    def start(self, system: str, task: str, tools: list[dict[str, Any]]) -> AgentSession:
        return MyModelSession(system, task, tools)


async def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--corpus-dir", required=True, help="directory from corpus-download")
    p.add_argument("--queries", default="data/queries/film_v0.2.json")
    p.add_argument("--manifest", default="data/corpus_manifests/sample_manifest.json")
    p.add_argument("--transcripts-dir", default="data/transcripts")
    p.add_argument("--trials", type=int, default=1)
    p.add_argument("--limit", type=int, default=5,
                   help="queries to run (a diagnostic); 0 runs all of them, for submission")
    p.add_argument("--partial", action="store_true",
                   help="run on the videos that verify; the run is not reportable")
    p.add_argument("--output-dir", default="data/results/mine")
    args = p.parse_args(argv)

    queries = load_queries(Path(args.queries))
    if args.limit:
        queries = queries[: args.limit]
    try:
        corpus, failures = prepare_corpus(Path(args.manifest), Path(args.corpus_dir),
                                          transcripts_dir=Path(args.transcripts_dir),
                                          partial=args.partial)
    except CorpusNotVerifiedError as e:
        raise SystemExit(str(e).replace("prepare with partial=True", "pass --partial"))
    identity = environment_identity(queries_path=Path(args.queries),
                                    manifest_path=Path(args.manifest), provider="custom",
                                    query_filter=f"first {args.limit}" if args.limit else None)
    run = await run_agentic_benchmark(queries, MyModel(), corpus, trials=args.trials,
                                      output_dir=Path(args.output_dir), identity=identity)
    s = run.summary
    print(f"{run.run_id}: Recall@1 {s.recall_at_1:.1%} over {s.num_queries} queries, "
          f"by tier {json.dumps({k: round(v['recall_at_1'], 3) for k, v in s.by_tier.items()})}")
    if not run.reportable:
        print(f"Not reportable: {len(failures)} video(s) failed verification.")
    elif args.limit:
        print(f"Diagnostic run on {len(queries)} queries. Use --limit 0 for a submission run.")
    else:
        print(f"Submit with: momentbench submit -r {args.output_dir}/{run.run_id}.json "
              f"-q {args.queries} --model-name 'My model'")


if __name__ == "__main__":
    asyncio.run(main())
