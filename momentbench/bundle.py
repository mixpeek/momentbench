"""A result bundle: everything needed to report one validated run elsewhere.

    momentbench bundle -r RUN.json [-q QUERIES.json] [-o DIR]

From one run file that passes `momentbench submit`, writes:

- `result.json`: the submission entry (scores, intervals, net score, cost, conditions,
  compatibility key, release version) plus the query file's name and hash;
- `model_card.md`: a block to paste into a model card or README;
- `table.tex`: a one-row LaTeX table with the same numbers;
- `citation.bib`: how to cite MomentBench.

Every number comes from the validated entry, and the bundle names the release, the
query set, the condition and the run, so a reader can tell what was measured.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import momentbench
from momentbench.agentic.run import AgenticRun
from momentbench.submission import MANIFEST, agentic_entry, find_queries_file

BIBTEX = """@misc{{momentbench,
  title  = {{MomentBench: Corpus-Scale Video Moment Retrieval Benchmark}},
  author = {{{{Mixpeek}}}},
  year   = {{2026}},
  note   = {{Version {version}}},
  url    = {{https://github.com/mixpeek/momentbench}}
}}
"""


def condition(run: AgenticRun) -> str:
    bits = [run.modalities]
    if run.anonymize_titles:
        bits.append("anonymized titles")
    if run.identity.get("prompt_condition", "standard") != "standard":
        bits.append(f"{run.identity['prompt_condition']} prompt")
    if run.identity.get("frames"):
        bits.append(f"frames {run.identity['frames']['width']} px")
    return ", ".join(bits)


def make_bundle(run_path: Path, queries_path: Path | None = None,
                out_dir: Path | None = None, manifest: Path = MANIFEST) -> Path:
    run = AgenticRun.model_validate_json(Path(run_path).read_text())
    queries_path = Path(queries_path) if queries_path else find_queries_file(run)
    entry = agentic_entry(run, queries_path, manifest=manifest)  # raises if invalid
    out = Path(out_dir or f"bundle-{run.run_id}")
    out.mkdir(parents=True, exist_ok=True)
    qsha = hashlib.sha256(queries_path.read_bytes()).hexdigest()[:16]
    result = {**entry, "momentbench_version": momentbench.__version__,
              "queries_file": queries_path.name, "queries_file_sha256_16": qsha,
              "condition": condition(run), "run_id": run.run_id}
    (out / "result.json").write_text(json.dumps(result, indent=2) + "\n")

    lo, hi = (100 * x for x in entry["recall_at_1_ci95"])
    r1 = 100 * entry["recall_at_1"]
    net = entry.get("net_score")
    cost = entry.get("mean_cost_usd_per_query")
    cost_txt = f"${cost:.3f}" if cost is not None else "n/a"
    if cost is not None and run.harness == "claude-code":
        cost_txt += " (tokens at API list prices; run on a subscription)"
    net_txt = f"{net:+.2f}" if net is not None else "n/a"
    (out / "model_card.md").write_text(
        f"""### MomentBench {momentbench.__version__} ({queries_path.stem})

| Metric | Value |
|---|---|
| Recall@1 (tIoU 0.5) | {r1:.1f}% (95% CI {lo:.1f} to {hi:.1f}) |
| Net score (+1 correct, -1 wrong moment, 0 abstain) | {net_txt} |
| Queries x trials | {entry['num_queries']} x {entry['num_trials']} |
| Cost per query | {cost_txt} |
| Track and condition | {run.harness}; {condition(run)} |
| Compatibility key | `{entry['compatibility_key']}` |

Run `{run.run_id}` on `{queries_path.name}` (sha256 {qsha}). MomentBench:
https://github.com/mixpeek/momentbench
""")
    model = run.model_id.replace("_", "\\_")
    (out / "table.tex").write_text(
        "\\begin{tabular}{lrrrl}\n\\toprule\n"
        "System & R@1 & 95\\% CI & Net & Queries \\\\\n\\midrule\n"
        f"{model} & {r1:.1f} & {lo:.1f}--{hi:.1f} & {net_txt} & "
        f"{entry['num_queries']} \\\\\n\\bottomrule\n\\end{{tabular}}\n"
        f"% MomentBench {momentbench.__version__}, {queries_path.name}, {condition(run)}, "
        f"key {entry['compatibility_key']}\n")
    (out / "citation.bib").write_text(BIBTEX.format(version=momentbench.__version__))
    return out
