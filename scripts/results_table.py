"""Render saved agentic runs as a markdown results table.

    python scripts/results_table.py data/results/agentic-v0.2/*.json > docs/results.md

Every number is recomputed from the run's trials, so the table cannot drift from
the files it cites. Runs are grouped by query file (its name and query count), and
each row names the run file it came from.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from momentbench.agentic.run import AgenticRun, rescore, summarize
from momentbench.types import Query

QUERY_FILES = ("film_v0.2.json", "aerospace_v0.2.json", "robotics_v0.2.json", "sample.json")
_QUERIES: dict[str, Query] = {}


def current_queries() -> dict[str, Query]:
    """Every published query, by id, as the files stand now."""
    if not _QUERIES:
        root = Path(__file__).resolve().parents[1] / "data/queries"
        for name in QUERY_FILES:
            if (root / name).exists():
                for q in json.loads((root / name).read_text()):
                    _QUERIES[q["query_id"]] = Query(**q)
    return _QUERIES


def load_run(path: Path | str) -> AgenticRun:
    """A saved run, rescored against the current gold (see rescore)."""
    return rescore(AgenticRun.model_validate_json(Path(path).read_text()), current_queries())


def pct(x: float) -> str:
    return f"{100 * x:.1f}"


def common_queries(paths: list[str]) -> set[str]:
    """Query ids that every run in `paths` covered."""
    sets = [{t.query_id for t in load_run(f).trials}
            for f in paths if not str(f).endswith(".partial.json")]
    return set.intersection(*sets) if sets else set()


def table(paths: list[str], only: set[str] | None = None) -> str:
    """One markdown table, a row per run, best Recall@1 first.

    With `only`, every row is recomputed on just those query ids, so runs made on
    different versions of a query file are still compared on the same questions.
    """
    # .partial.json files are runs a provider cut off; they are never results.
    files = [Path(f) for f in paths if not str(f).endswith(".partial.json")]
    runs = [(f, load_run(f)) for f in files]
    if only is not None:
        for _, run in runs:
            run.trials = [t for t in run.trials if t.query_id in only]
    lines = []
    tiers = sorted({str(t.tier) for _, r in runs for t in r.trials})
    head = ["Model", "Harness", "Queries x trials", "Recall@1 (95% CI)", "R@1 tIoU 0.3 / 0.7"]
    head += [f"Tier {t}" for t in tiers] + ["Cost per query", "Run"]
    lines.append("| " + " | ".join(head) + " |")
    lines.append("|" + "---|" * len(head))
    for path, run in sorted(runs, key=lambda x: -summarize(x[1].trials, x[1].summary.num_trials)
                            .recall_at_1):
        s = summarize(run.trials, run.summary.num_trials)
        lo, hi = s.recall_at_1_ci95
        cells = [
            run.model_id + (f" ({run.effort})" if run.effort else "")
            + (" (frames only)" if run.modalities == "frames" else ""),
            run.harness,
            f"{s.num_queries} x {s.num_trials}",
            f"{pct(s.recall_at_1)} ({pct(lo)} to {pct(hi)})",
            # Runs saved before trials recorded top1_iou cannot give other thresholds.
            "n/a" if s.recall_at_1 and not any(t.top1_iou for t in run.trials) else
            f"{pct(s.recall_at_1_by_iou['0.3'])} / {pct(s.recall_at_1_by_iou['0.7'])}",
        ]
        for t in tiers:
            row = s.by_tier.get(t)
            cells.append(f"{pct(row['recall_at_1'])} (n={row['num_queries']})" if row else "")
        cost = s.mean_cost_usd_per_query
        note = " notional" if run.harness == "claude-code" else ""
        if run.model_id.startswith("baseline-"):
            cells.append("none (no model)")
        else:
            cells.append(f"${cost:.3f}{note}" if cost is not None else "not priced")
        errored = sum(1 for t in run.trials if t.outcome == "error")
        cells.append(f"`{path.name}`" + (f" ({errored} errored trials)" if errored else ""))
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("runs", nargs="+")
    print(table(p.parse_args().runs))


if __name__ == "__main__":
    main()
