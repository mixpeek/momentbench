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

from results_table import (  # noqa: E402
    common_queries,
    condition,
    current_queries,
    is_standard,
    load_run,
    table,
)

from momentbench.agentic.run import trial_net  # noqa: E402
from momentbench.agentic.stats import (  # noqa: E402
    bootstrap_ci,
    cluster_bootstrap_ci,
    paired_difference_ci,
)

ROOT = Path(__file__).resolve().parents[1]
# Runs under the 0.2.1 contract (receipt, identity, compatibility key) come first. A
# model with no 0.2.1 run keeps its earlier row, labelled historical.
RECORD_025 = ROOT / "data/results/agentic-v0.2.5/record"  # scored set, anonymized titles
RECORD_021 = ROOT / "data/results/agentic-v0.2.1/record"
RECORD = ROOT / "data/results/agentic-v0.2/record"
SEED = ROOT / "data/results/agentic"
QUERY_FILES = {"film": "film_v0.2.json", "aerospace": "aerospace_v0.2.json",
               "robotics": "robotics_v0.2.json"}
# Seed runs on identical query text (after the q-film-005 fix, before the q-film-001 rewording).
SEED_RUNS = ("agentic-claude-opus-5-5-e51b703b", "agentic-gpt-5.5-b893d59f",
             "agentic-gemini-3.1-pro-preview-a6ff462c", "agentic-gemma-4-31b-it-d23baa9c")
# Pairs of independent single-trial Opus runs on the same queries: the earlier record
# run and its 0.2.1 rerun (as first run, before the reworded queries were merged in).
RETEST = [
    ("film", RECORD / "agentic-claude-opus-5-5-0d68625a.json",
     RECORD_021.parent / "superseded" / "agentic-claude-opus-5-5-381de39d.json"),
    ("aerospace", RECORD / "agentic-claude-opus-5-5-519357e4.json",
     RECORD_021.parent / "superseded" / "agentic-claude-opus-5-5-0ec085fb.json"),
    ("robotics", RECORD / "agentic-claude-opus-5-5-e7ca7bd1.json",
     RECORD_021.parent / "superseded" / "agentic-claude-opus-5-5-c5031ac2.json"),
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
    chosen: dict[tuple, Path] = {}
    for f in (sorted(RECORD_025.glob("*/*.json")) + sorted(RECORD_021.glob("*.json"))
              + sorted(RECORD.glob("*.json"))):
        if f.name.endswith(".partial.json"):
            continue
        run = load_run(f)
        if not run.reportable:  # debug runs on a partial corpus never enter a table
            continue
        ran = {t.query_id for t in run.trials}
        # A run belongs to this domain if nearly all of its queries are in the file;
        # queries removed later (review, cross-check) are then left out of its row.
        if not (ran and len(ran & ids) >= 0.9 * len(ran)):
            continue
        # One row per model and condition; the 0.2.1 directory is read first, so a
        # rerun replaces the earlier row.
        chosen.setdefault((run.model_id, run.harness, run.modalities, run.anonymize_titles,
                           run.identity.get("prompt_condition", "standard"),
                           json.dumps(run.identity.get("frames"))), f)
    return sorted(chosen.values())


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
        model = run.model_id + condition(run)
        rows.append((domain, -mean_hit, f"| {domain} | {model} | {100 * mean_hit:.1f} "
                     f"({100 * lo:.1f} to {100 * hi:.1f}), n={len(hits)} | `{f.name}` |"))
    lines += [r[2] for r in sorted(rows)]
    return "\n".join(lines) if len(lines) > 2 else ""


def source_video(query_id: str) -> str:
    q = current_queries()[query_id]
    if q.gold_moments:
        return q.gold_moments[0].asset_id
    return (q.metadata.get("altered_from") or {}).get("asset_id", query_id)


def abstain_floor(paths: list[Path]) -> float | None:
    """Net score of always abstaining on these queries: the share that are no-answer."""
    common = keyed_common(paths)
    for f in paths:
        run = load_run(f)
        if run.model_id == "baseline-abstain":
            ts = [t for t in run.trials if t.query_id in common]
            return sum(trial_net(t) for t in ts) / len(ts) if ts else None
    return None


def split_table(paths: list[Path]) -> str:
    """Localization on positive queries and abstention on no-answer queries, apart."""
    common = keyed_common(paths)
    lines = ["| Model | Positive R@1 (95% CI by video) | No-answer accuracy | "
             "False-positive rate | Abstains on positives | Net score |",
             "|---|---|---|---|---|---|"]
    for f in paths:
        run = load_run(f)
        if run.model_id.startswith("baseline-") or not is_standard(run):
            continue
        per_q: dict[str, list[float]] = {}
        for t in run.trials:
            if t.query_id in common and not t.no_answer:
                per_q.setdefault(t.query_id, []).append(float(t.hit_at_1))
        vals = [sum(v) / len(v) for v in per_q.values()]
        lo, hi = cluster_bootstrap_ci(vals, [source_video(k) for k in per_q])
        trials = [t for t in run.trials if t.query_id in common]
        neg = [t for t in trials if t.no_answer]
        pos = [t for t in trials if not t.no_answer]

        def pct(ts, pred):
            return f"{100 * sum(1 for t in ts if pred(t)) / len(ts):.1f}" if ts else "n/a"

        lines.append(
            f"| {run.model_id} | {100 * sum(vals) / max(len(vals), 1):.1f} ({100 * lo:.1f} to "
            f"{100 * hi:.1f}), n={len(vals)} | "
            f"{pct(neg, lambda t: t.outcome == 'submitted' and not t.moments)} | "
            f"{pct(neg, lambda t: t.outcome == 'submitted' and bool(t.moments))} | "
            f"{pct(pos, lambda t: t.outcome == 'submitted' and not t.moments)} | "
            f"{sum(trial_net(t) for t in trials) / max(len(trials), 1):+.2f} |")
    return "\n".join(lines)


def paired_table(paths: list[Path]) -> str:
    """Each model against the domain's best, on the queries both ran (paired)."""
    runs = [load_run(f) for f in paths]
    runs = [r for r in runs if not r.model_id.startswith("baseline-") and is_standard(r)]
    if len(runs) < 2:
        return ""
    common = set.intersection(*[{t.query_id for t in r.trials} for r in runs])

    def per_query(run):
        acc: dict[str, list[float]] = {}
        for t in run.trials:
            if t.query_id in common:
                acc.setdefault(t.query_id, []).append(float(t.hit_at_1))
        return {k: sum(v) / len(v) for k, v in acc.items()}

    scores = {r.model_id: per_query(r) for r in runs}
    best = max(scores, key=lambda m: sum(scores[m].values()))
    keys = sorted(common)
    lines = [f"| Model | Recall@1 difference vs {best} (paired 95% CI) |", "|---|---|"]
    for m in scores:
        if m == best:
            continue
        diff, (lo, hi) = paired_difference_ci([scores[m][k] for k in keys],
                                              [scores[best][k] for k in keys])
        lines.append(f"| {m} | {100 * diff:+.1f} ({100 * lo:+.1f} to {100 * hi:+.1f}), "
                     f"n={len(keys)} |")
    return "\n".join(lines)


def keyed_common(paths: list[Path]) -> set[str]:
    """Queries every run with a run identity covered (historical runs do not narrow it)."""
    keyed = [str(p) for p in paths if load_run(p).identity.get("compatibility_key")]
    return common_queries(keyed or [str(p) for p in paths])


def section(paths: list[Path]) -> str:
    """A domain's table: runs with a run identity on the queries they all covered,
    historical rows on the current queries they covered (their own row count)."""
    files = [str(p) for p in paths]
    current = set()
    for name in QUERY_FILES.values():
        f = ROOT / "data/queries" / name
        if f.exists():
            current |= {q["query_id"] for q in json.loads(f.read_text())}
    common = keyed_common(paths) & current
    covered = {len({t.query_id for t in load_run(p).trials} & common) for p in paths}
    note = ("" if covered == {len(common)} else
            f"Rows are computed on the {len(common)} current queries every 0.2.1 run covered. "
            "A historical row covers fewer (its Queries column says how many): it was run "
            "before some queries were reworded, and is scored only on the queries whose "
            "wording it saw.\n\n")
    paired = paired_table(paths)
    return (note + table(files, only=common)
            + "\n\nLocalization and abstention, separately (positive queries have a gold "
            "moment; no-answer queries are right only on an explicit empty answer). Net "
            "score counts +1 for a correct answer, -1 for a wrong moment and 0 for "
            "abstaining, so a wrong answer costs more than saying nothing matches; a span "
            "below tIoU 0.5 counts as wrong. Most of its penalty comes from wrong answers "
            "on positive queries (search and timing errors), and the current prompt asks "
            "for a best answer when the budget runs out, so read it as a localization "
            "plus abstention summary; the no-answer columns are the abstention measure."
            + (f" Never answering scores {floor:+.2f} here (`baseline-abstain`)."
               if (floor := abstain_floor(paths)) is not None else "")
            + "\n\n" + split_table(paths)
            + (f"\n\nPaired differences on the same queries:\n\n{paired}" if paired else "")
            + (f"\n\n{ctrl}" if (ctrl := control_table(paths)) else ""))


def control_table(paths: list[Path]) -> str:
    """Tier 2 split by whether a text-only control solved the query (audit 2 R12, audit 3).

    Two controls read only the transcript: `baseline-transcript` (the harness's keyword
    search, a run file) and BM25 over the same cues (text_controls.py, computed here).
    A model's score on queries either one solves can come from speech alone. The other
    column is only "not solved by these two text methods": a stronger text method could
    solve more, so it does not show that those queries need the picture.
    """
    from momentbench.agentic.text_controls import bm25_control, load_transcripts
    from momentbench.runner import load_queries
    from momentbench.submission import find_queries_file

    runs = [load_run(f) for f in paths]
    control = next((r for r in runs if r.model_id == "baseline-transcript"), None)
    if control is None:
        return ""
    t2 = {t.query_id: t.hit_at_1 for t in control.trials if t.tier == 2}
    keyword = {q for q, hit in t2.items() if hit}
    queries = [q for q in load_queries(find_queries_file(control)) if q.query_id in t2]
    manifest = json.loads((ROOT / "data/corpus_manifests/sample_manifest.json").read_text())
    transcripts = load_transcripts([a["asset_id"] for a in manifest], ROOT / "data/transcripts")
    bm25 = {q for q, hit in bm25_control(queries, transcripts).items() if hit}
    solved = keyword | bm25
    missed = set(t2) - solved
    lines = [f"Two text-only controls localize tier-2 queries from the transcript without "
             f"looking at a frame. `baseline-transcript` sends the request to the harness's "
             f"keyword search and submits the top cue: {len(keyword)} of {len(t2)}. BM25 "
             f"over the same cues (k1 1.2, b 0.75, top five cues; "
             f"`momentbench/agentic/text_controls.py`): {len(bm25)} of {len(t2)}. Either: "
             f"{len(solved)}. Tier-2 Recall@1 of each model on the queries a text control "
             "solves and on the rest. The second column is a lower bound on what speech alone "
             "can do; it does not show that those queries need the picture.", "",
             "| Model | Tier 2, solved by a text-only control | Tier 2, not solved by "
             "either text-only control |", "|---|---|---|"]

    def rate(run, ids):
        hits = [float(t.hit_at_1) for t in run.trials if t.query_id in ids]
        return f"{100 * sum(hits) / len(hits):.1f} (n={len(hits)})" if hits else "n/a"

    for run in runs:
        if run.model_id.startswith("baseline-") or not is_standard(run):
            continue
        lines.append(f"| {run.model_id} | {rate(run, solved)} | {rate(run, missed)} |")
    return "\n".join(lines)


def results_json(domains: dict[str, list[Path]]) -> dict:
    """Every row of the results tables as data, for leaderboards and model cards."""
    from results_table import corpus_size

    out = {"benchmark": "momentbench", "version": "0.2", "track": "agentic",
           "metric": "recall_at_1_tiou_0.5", "rows": []}
    for domain, paths in domains.items():
        current = keyed_common(paths) & {
            q["query_id"] for name in QUERY_FILES.values()
            if (ROOT / "data/queries" / name).exists()
            for q in json.loads((ROOT / "data/queries" / name).read_text())}
        for f in paths:
            run = load_run(f)
            run = run.model_copy(update={"trials": [t for t in run.trials
                                                    if t.query_id in current]})
            from momentbench.agentic.run import summarize

            s = summarize(run.trials, run.summary.num_trials)
            out["rows"].append({
                "domain": domain, "model": run.model_id, "harness": run.harness,
                "modalities": run.modalities, "anonymize_titles": run.anonymize_titles,
                "standard": is_standard(run), "condition": condition(run).strip(" ()"),
                "corpus_assets": corpus_size(run),
                "queries": s.num_queries, "trials": s.num_trials,
                "recall_at_1": round(s.recall_at_1, 4),
                "recall_at_1_ci95": [round(x, 4) for x in s.recall_at_1_ci95],
                "recall_at_1_by_iou": {k: round(v, 4) for k, v in s.recall_at_1_by_iou.items()},
                "by_tier": {k: {"queries": v["num_queries"],
                                "recall_at_1": round(v["recall_at_1"], 4)}
                            for k, v in s.by_tier.items()},
                "positive_recall_at_1": s.positive_recall_at_1,
                "no_answer_accuracy": s.no_answer_accuracy,
                "false_positive_rate": s.false_positive_rate,
                "net_score": s.net_score,
                "abstention_on_positives": s.abstention_on_positives,
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
    all_runs = [load_run(f) for f in {f for d in QUERY_FILES for f in runs_for(d)}]
    total = len(all_runs)
    historical = sum(1 for r in all_runs if not r.identity.get("compatibility_key"))
    reworded = {d: sum(1 for q in json.loads((ROOT / "data/queries" / f).read_text())
                       if q["metadata"].get("text_refined"))
                for d, f in QUERY_FILES.items() if (ROOT / "data/queries" / f).exists()}
    reworded_note = ", ".join(f"{n} {d}" for d, n in reworded.items() if n)
    print(f"""# Results (v0.2)

Generated by `scripts/results_doc.py` from the run files named in each row. Read
[`docs/protocol.md`](protocol.md) for what the numbers mean. All runs use the agentic
track's default budget (30 turns, 64 frames, 60 tool calls), 1 trial per query, and the
27-asset v0.2 corpus. Rows marked "(20 assets)" were run before the 7 robotics videos
were added, on a smaller and so slightly easier corpus; they will be replaced by reruns on
all 27.

**Floors.** `baseline-transcript` is a modality control: it answers from the keyword
transcript search alone and never looks at a frame. `baseline-random` submits a random
asset and span (chance). `baseline-abstain`
always submits an empty answer, so it scores exactly the share of no-answer (tier 3)
queries. A system's overall Recall@1 only means something above both.

**Status.** The queries are model-annotated (Claude drafted and verified, Gemini
cross-checked) and were then reviewed frame by frame by Claude on the maintainer's behalf
(59 corrected, 1 removed; `metadata.model_review`). They are not yet human-reviewed, so
treat every number as provisional. GPT-5.5 is not evaluated on v0.2: no OpenAI budget is
available for it. Gemini 3.1 Pro has
film and robotics results and no aerospace run, also for lack of API budget.
Claude results come from the local Claude Code harness; their cost is what the tokens
would cost at API list prices, not money spent.

**Historical rows.** Rows come from 0.2.1 runs, which record the verified corpus, the
exact queries and a compatibility key, except where a model has no 0.2.1 run yet.
{historical} of the {total} runs on this page are such earlier runs and are marked
"(historical)": their saved answers are rescored against the current gold, honestly but
without the 0.2.1 identity. The runs they replaced stay in
`data/results/agentic-v0.2/record`.

**Reworded queries.** Rows are rescored against the current gold, but a run is only
scored on queries whose wording it saw. Queries reworded after a run ({reworded_note})
are left out of that run's row, which is why the query counts are below the file's size.
New runs record a hash of each query's text, so this is exact for them.

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
