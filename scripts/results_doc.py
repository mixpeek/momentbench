"""Write docs/results.md from the v0.2 record runs.

    python scripts/results_doc.py > docs/results.md

Every number comes from a run file in data/results/agentic-v0.2/record/ or the
seed runs in data/results/agentic/, so the page cannot drift from the data.
Rerun it whenever a run is added.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from results_table import common_queries, load_run, table  # noqa: E402

from momentbench.agentic.stats import bootstrap_ci  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
RECORD = ROOT / "data/results/agentic-v0.2/record"
SEED = ROOT / "data/results/agentic"
QUERY_FILES = {"film": "film_v0.2.json", "aerospace": "aerospace_v0.2.json",
               "robotics": "robotics_v0.2.json"}
# Seed runs on identical query text (after the q-film-005 fix, before the q-film-001 rewording).
SEED_RUNS = ("agentic-claude-opus-5-5-e51b703b", "agentic-gpt-5.5-b893d59f",
             "agentic-gemini-3.1-pro-preview-a6ff462c", "agentic-gemma-4-31b-it-d23baa9c")
# Pairs of independent single-trial Opus runs on the same queries.
RETEST = [
    ("film", ROOT / "data/results/agentic-v0.2/agentic-claude-opus-5-5-65eb804b.json",
     RECORD / "agentic-claude-opus-5-5-da54af61.json"),
    ("aerospace",
     ROOT / "data/results/agentic-v0.2/superseded/agentic-claude-opus-5-5-2b5433a6.json",
     RECORD / "agentic-claude-opus-5-5-519357e4.json"),
]


def retest() -> str:
    lines = ["| Domain | Shared queries | Run A R@1 | Run B R@1 | Per-query agreement "
             "| Cohen's kappa | Runs |", "|---|---|---|---|---|---|---|"]
    for domain, pa_path, pb_path in RETEST:
        a, b = ({t.query_id: t.hit_at_1
                 for t in load_run(p).trials}
                for p in (pa_path, pb_path))
        ids = sorted(set(a) & set(b))
        agree = sum(a[i] == b[i] for i in ids) / len(ids)
        ra, rb = (sum(x[i] for i in ids) / len(ids) for x in (a, b))
        expected = ra * rb + (1 - ra) * (1 - rb)
        kappa = (agree - expected) / (1 - expected)
        lines.append(f"| {domain} | {len(ids)} | {100 * ra:.1f} | {100 * rb:.1f} | "
                     f"{100 * agree:.0f}% | {kappa:.2f} | `{pa_path.name}`, `{pb_path.name}` |")
    return ("Two independent single-trial Claude Opus 5.5 runs per domain, compared on the "
            "queries both covered. The aggregate moves by a point or two; individual queries "
            "flip more often, so per-query or small-slice comparisons need several trials.\n\n"
            + "\n".join(lines))


def runs_for(vertical: str) -> list[Path]:
    ids = {q["query_id"] for q in json.loads((ROOT / "data/queries" / QUERY_FILES[vertical])
                                             .read_text())}
    out = []
    for f in sorted(RECORD.glob("*.json")):
        if f.name.endswith(".partial.json"):
            continue
        run = load_run(f)
        ran = {t.query_id for t in run.trials}
        # A run belongs to this domain if nearly all of its queries are in the file;
        # queries removed later (review, cross-check) are then left out of its row.
        if ran and len(ran & ids) >= 0.9 * len(ran):
            out.append(f)
    return out


def split_rows(paths: list[Path]) -> str:
    queries = {q["query_id"]: q for q in json.loads(
        (ROOT / "data/queries" / QUERY_FILES["aerospace"]).read_text())}
    lines = ["| Model | Public R@1 (95% CI) | Rolling R@1 (95% CI) |", "|---|---|---|"]
    for f in paths:
        run = load_run(f)
        cells = [run.model_id]
        for split in ("public", "rolling"):
            per_query: dict[str, list[float]] = {}
            for t in run.trials:
                if t.query_id in queries and queries[t.query_id]["split"] == split:
                    per_query.setdefault(t.query_id, []).append(float(t.hit_at_1))
            # One value per query (mean over its trials): the interval resamples queries.
            hits = [sum(v) / len(v) for v in per_query.values()]
            lo, hi = bootstrap_ci(hits)
            cells.append(f"{100 * sum(hits) / len(hits):.1f} ({100 * lo:.1f} to {100 * hi:.1f}),"
                         f" n={len(hits)}")
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def physical_rows(paths: list[Path]) -> str:
    """Recall@1 on queries tagged metadata.physical_event, per run, resampled by query."""
    tagged = set()
    for name in QUERY_FILES.values():
        f = ROOT / "data/queries" / name
        if f.exists():
            tagged |= {q["query_id"] for q in json.loads(f.read_text())
                       if q["metadata"].get("physical_event")}
    lines = ["| Domain | Model | Physical-event R@1 (95% CI) | Run |", "|---|---|---|---|"]
    rows = []
    for f in paths:
        run = load_run(f)
        per_query: dict[str, list[float]] = {}
        for t in run.trials:
            if t.query_id in tagged:
                per_query.setdefault(t.query_id, []).append(float(t.hit_at_1))
        if not per_query:
            continue
        hits = [sum(v) / len(v) for v in per_query.values()]
        lo, hi = bootstrap_ci(hits)
        domain = run.trials[0].vertical
        mean_hit = sum(hits) / len(hits)
        model = run.model_id + (" (frames only)" if run.modalities == "frames" else "")
        rows.append((domain, -mean_hit, f"| {domain} | {model} | {100 * mean_hit:.1f} "
                     f"({100 * lo:.1f} to {100 * hi:.1f}), n={len(hits)} | `{f.name}` |"))
    lines += [r[2] for r in sorted(rows)]
    return "\n".join(lines) if len(lines) > 2 else ""


def section(paths: list[Path]) -> str:
    """A domain's table, every row computed on the queries all of its runs covered."""
    files = [str(p) for p in paths]
    current = set()
    for name in QUERY_FILES.values():
        f = ROOT / "data/queries" / name
        if f.exists():
            current |= {q["query_id"] for q in json.loads(f.read_text())}
    common = common_queries(files) & current
    covered = {len({t.query_id for t in load_run(p).trials})
               for p in paths}
    note = ("" if len(covered) <= 1 and max(covered, default=0) == len(common) else
            f"Every row is computed on the {len(common)} current queries that all of these "
            "runs covered (runs made on earlier versions of the file, or on queries since "
            "removed by review, are scored only on what remains).\n\n")
    return note + table(files, only=common)


def results_json(domains: dict[str, list[Path]]) -> dict:
    """Every row of the results tables as data, for leaderboards and model cards."""
    from results_table import corpus_size

    out = {"benchmark": "momentbench", "version": "0.2", "track": "agentic",
           "metric": "recall_at_1_tiou_0.5", "rows": []}
    for domain, paths in domains.items():
        files = [str(p) for p in paths]
        current = common_queries(files) & {
            q["query_id"] for name in QUERY_FILES.values()
            if (ROOT / "data/queries" / name).exists()
            for q in json.loads((ROOT / "data/queries" / name).read_text())}
        for f in paths:
            run = load_run(f)
            run.trials = [t for t in run.trials if t.query_id in current]
            from momentbench.agentic.run import summarize

            s = summarize(run.trials, run.summary.num_trials)
            out["rows"].append({
                "domain": domain, "model": run.model_id, "harness": run.harness,
                "modalities": run.modalities, "corpus_assets": corpus_size(run),
                "queries": s.num_queries, "trials": s.num_trials,
                "recall_at_1": round(s.recall_at_1, 4),
                "recall_at_1_ci95": [round(x, 4) for x in s.recall_at_1_ci95],
                "recall_at_1_by_iou": {k: round(v, 4) for k, v in s.recall_at_1_by_iou.items()},
                "by_tier": {k: {"queries": v["num_queries"],
                                "recall_at_1": round(v["recall_at_1"], 4)}
                            for k, v in s.by_tier.items()},
                "cost_usd_per_query": s.mean_cost_usd_per_query,
                "error_trials": s.error_trials, "run_file": Path(f).name})
    return out


def main() -> None:
    film, aero = runs_for("film"), runs_for("aerospace")
    (ROOT / "docs/results.json").write_text(json.dumps(
        results_json({"film": film, "aerospace": aero, "robotics": runs_for("robotics")}),
        indent=2) + "\n")
    robotics_file = ROOT / "data/queries" / QUERY_FILES["robotics"]
    robotics = runs_for("robotics") if robotics_file.exists() else []
    robotics_section = (f"""## Robotics ({QUERY_FILES['robotics']})

NASA rover, free-flyer and helicopter footage. Queries target physical events: contact,
wheels on slopes, lift-off and landing, cause then effect.

{section(robotics)}
""" if robotics else "")
    seed = [SEED / f"{r}.json" for r in SEED_RUNS]
    phys = physical_rows(aero + robotics)
    physical_section = (f"""## Physical-event subset

Queries tagged `metadata.physical_event`: a robot, vehicle or mechanism visibly moves or
makes contact (a wheel onto a ramp, rotors spinning up, a rocket lifting off). This is the
slice to report for physical understanding. Compare rows within a domain: runs made
before the aerospace physical-event queries were added cover none of them, so they do not
appear here.

{phys}

""" if phys else "")
    print(f"""# Results (v0.2)

Generated by `scripts/results_doc.py` from the run files named in each row. Read
[`docs/protocol.md`](protocol.md) for what the numbers mean. All runs use the agentic
track's default budget (30 turns, 64 frames, 60 tool calls), 1 trial per query, and the
27-asset v0.2 corpus. Rows marked "(20 assets)" were run before the 7 robotics videos
were added, on a smaller and so slightly easier corpus; they will be replaced by reruns on
all 27.

**Floors.** `baseline-random` submits a random asset and span (chance). `baseline-abstain`
always submits an empty answer, so it scores exactly the share of no-answer (tier 3)
queries. A system's overall Recall@1 only means something above both.

**Status.** The queries are model-annotated (Claude drafted and verified, Gemini
cross-checked) and not yet human-reviewed, so treat every number as provisional.
GPT-5.5 is not evaluated on v0.2: the OpenAI organization available to us has its billing
plan cancelled, and a first attempt that ran into it was discarded. Gemini 3.1 Pro has
film and robotics results; its aerospace run is waiting on budget.
Claude results come from the local Claude Code harness; their cost is what the tokens
would cost at API list prices, not money spent.

## Film ({QUERY_FILES['film']})

{section(film)}

## Aerospace ({QUERY_FILES['aerospace']})

{section(aero)}

### Public vs rolling (aerospace)

Rolling queries use NASA footage published after 2026-06-30, after current model
training cutoffs. A model whose public score came from memorised footage would score
lower on rolling. With 27 rolling queries the comparison can only catch a large gap, and
the rolling set is not matched for difficulty: much of it is NASA Minute footage with the
narration burned in as captions, which makes it easier, so a higher rolling score is not
evidence of anything by itself.

{split_rows(aero)}

{robotics_section}{physical_section}## Visual-only ablation

`--no-transcripts` rows (marked frames only) had no speech available to any tool. The gap
to the matching full run is how much of the score came from finding the moment through
what was said rather than what was seen.

## Run-to-run stability

{retest()}

## Against the pre-registered targets

The README pre-registered difficulty targets before any run: tier 1 above 80%, tier 2
50 to 70%, tier 3 below 35%, tier 4 below 25%. For Claude Opus 5.5, tier 1 is far harder
than targeted (whole-corpus search for one shot), tier 2 is easier (the transcript narrows
the search), tier 3 meets its target on film and misses it on aerospace, and tier 4 is
easier than targeted. The tiers are kept as defined; the targets were wrong, and this page
reports that rather than retuning the queries.

## Seed set (v0.1, 4 film queries x 5 trials)

Kept for comparison with the earlier runs, including the only open-weights entrant so
far (Gemma 4 31B, which the v0.2 set would take about 18 hours to run under its API rate
limit). Four queries cannot rank models: every interval overlaps.

{table([str(p) for p in seed])}
""")


if __name__ == "__main__":
    main()
