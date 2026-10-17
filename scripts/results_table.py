"""Render saved agentic runs as a markdown results table.

    python scripts/results_table.py data/results/agentic-v0.2/*.json > docs/results.md

Every number is recomputed from the run's trials, so the table cannot drift from
the files it cites. Runs are grouped by query file (its name and query count), and
each row names the run file it came from.
"""

from __future__ import annotations

import argparse
import datetime
import functools
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


# The 7 robotics videos were written into the manifest at about this time; earlier runs,
# which predate the corpus_assets field, searched the 20 film and aerospace assets.
ROBOTICS_ADDED = "2026-10-10T23:50:00Z"


def corpus_size(run: AgenticRun) -> int:
    """Assets the run searched. Older runs store only their end time, so their start is
    estimated from the summed trial time at the concurrency of 3 they were run with."""
    if run.corpus_assets:
        return run.corpus_assets
    end = datetime.datetime.fromisoformat(run.timestamp.replace("Z", "+00:00"))
    start = end - datetime.timedelta(seconds=sum(t.latency_s for t in run.trials) / 3)
    added = datetime.datetime.fromisoformat(ROBOTICS_ADDED.replace("Z", "+00:00"))
    return 20 if start < added else 27


def load_run(path: Path | str) -> AgenticRun:
    """A saved run, rescored against the current gold (see rescore)."""
    return _load_run(str(Path(path).resolve()))


@functools.lru_cache(maxsize=None)
def _load_run(path: str) -> AgenticRun:
    # The generators load each run file many times; parse and rescore it once. Callers
    # must not modify the returned run (use model_copy), since it is shared.
    return rescore(AgenticRun.model_validate_json(Path(path).read_text()), current_queries())


def is_standard(run: AgenticRun) -> bool:
    """Full task: frames and transcripts, the standard prompt and frame size. Anonymized
    titles count as standard: from 0.2.5 they are the record condition."""
    return (run.modalities != "frames"
            and run.identity.get("prompt_condition", "standard") == "standard"
            and not run.identity.get("frames"))


def condition(run: AgenticRun) -> str:
    """Label for a non-standard run, appended to its model name ("" when standard)."""
    tags = (["frames only"] if run.modalities == "frames" else []) + (
        ["anonymized titles"] if run.anonymize_titles else []) + (
        ["abstain prompt"] if run.identity.get("prompt_condition") == "abstain" else []) + (
        [f"frames {run.identity['frames']['width']} px"] if run.identity.get("frames") else []) + (
        ["historical"] if not run.identity.get("compatibility_key") else [])
    return f" ({', '.join(tags)})" if tags else ""


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
    if only is not None:  # copies: load_run's results are cached and shared
        runs = [(f, r.model_copy(update={"trials": [t for t in r.trials if t.query_id in only]}))
                for f, r in runs]
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
            + condition(run)
            + (f" ({corpus_size(run)} assets)" if corpus_size(run) != 27 else ""),
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
