"""A 30-minute check that a new setup reproduces MomentBench exactly.

    momentbench corpus-download -m data/diagnostic/manifest.json --corpus-dir diag/
    momentbench diagnose --corpus-dir diag/

The diagnostic profile is three short NASA videos (about 50 MB) and the 15 scored
queries on them (`data/diagnostic/`). `diagnose` checks, in order: ffmpeg and ffprobe
are installed; every profile video matches its manifest hash and duration; a frame
comes out of each video; the two deterministic floors (`baseline-transcript`,
`baseline-abstain`) reproduce the stored per-query results exactly; and the
prediction-file scorer scores a known file to the stored result. Each step prints PASS
or FAIL with what to fix. No model is called and nothing is spent.

Passing means the harness, scoring and media pipeline on this machine match the
reference. It says nothing about a model's score.
"""

from __future__ import annotations

import asyncio
import json
import shutil
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DIAG = ROOT / "data/diagnostic"


def floor_hits(model: str, corpus_dir: Path) -> dict[str, list]:
    from momentbench.agentic.baselines import BaselineAgent
    from momentbench.agentic.prepare import prepare_corpus
    from momentbench.agentic.run import run_agentic_benchmark
    from momentbench.runner import load_queries

    corpus, _ = prepare_corpus(DIAG / "manifest.json", corpus_dir,
                               transcripts_dir=ROOT / "data/transcripts")
    queries = load_queries(DIAG / "queries.json")
    run = asyncio.run(run_agentic_benchmark(queries, BaselineAgent(model), corpus, trials=1))
    # The first answer itself, not just whether it hit: a stricter reproduction check.
    return {t.query_id: [t.hit_at_1, [t.moments[0].asset_id, round(t.moments[0].t_in, 2),
                                      round(t.moments[0].t_out, 2)] if t.moments else None]
            for t in run.trials}


def prediction_recall() -> float:
    from momentbench.predictions import score_predictions

    run = score_predictions(DIAG / "predictions.json", DIAG / "queries.json",
                            DIAG / "manifest.json")
    return round(run.summary.recall_at_1, 4)


def run_checks(corpus_dir: Path) -> list[tuple[str, bool, str]]:
    from momentbench.agentic.environment import ffmpeg_frame
    from momentbench.corpus.verify import verify_checks
    from momentbench.runner import load_corpus_assets

    expected = json.loads((DIAG / "expected.json").read_text())
    out: list[tuple[str, bool, str]] = []

    def step(name: str, fn):
        start = time.perf_counter()
        try:
            ok, detail = fn()
        except Exception as e:  # noqa: BLE001  a diagnostic reports every failure
            ok, detail = False, f"{type(e).__name__}: {e}"
        out.append((name, ok, f"{detail} ({time.perf_counter() - start:.1f} s)"))
        return ok

    def tools():
        missing = [t for t in ("ffmpeg", "ffprobe") if shutil.which(t) is None]
        return not missing, ("found" if not missing else
                             f"install {', '.join(missing)} (for example: brew install ffmpeg)")

    def corpus():
        checks = verify_checks(load_corpus_assets(DIAG / "manifest.json"), corpus_dir)
        bad = [f"{c.asset_id}: {'; '.join(c.problems)}" for c in checks if not c.ok]
        return not bad, ("3 of 3 videos verified" if not bad else
                         "; ".join(bad) + ". Run: momentbench corpus-download -m "
                         "data/diagnostic/manifest.json --corpus-dir " + str(corpus_dir))

    def frames():
        from momentbench.corpus.verify import resolve_video

        sizes = []
        for a in load_corpus_assets(DIAG / "manifest.json"):
            jpeg = ffmpeg_frame(resolve_video(a, corpus_dir), 1.0)
            sizes.append(len(jpeg))
        return all(s > 1000 for s in sizes), f"{len(sizes)} frames extracted"

    def floors():
        problems = []
        for model, want in expected["floors"].items():
            got = floor_hits(model, corpus_dir)
            diff = sorted(q for q in want if got.get(q) != want[q])
            if diff:
                problems.append(f"{model} differs on {', '.join(diff[:5])}")
        return not problems, ("both floors match the reference per query" if not problems
                              else "; ".join(problems))

    def predictions():
        got = prediction_recall()
        return got == expected["predictions_recall_at_1"], (
            f"Recall@1 {got} (reference {expected['predictions_recall_at_1']})")

    step("ffmpeg and ffprobe", tools)
    if step("profile videos verify", corpus):
        step("frame extraction", frames)
        step("floors reproduce the reference", floors)
    step("prediction-file scorer", predictions)
    return out


def build_profile(corpus_dir: Path, assets: list[str]) -> None:
    """Maintainers: write data/diagnostic/ from the scored queries on `assets`."""
    DIAG.mkdir(parents=True, exist_ok=True)
    manifest = [a for a in json.loads(
        (ROOT / "data/corpus_manifests/sample_manifest.json").read_text())
        if a["asset_id"] in assets]
    (DIAG / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    queries = []
    for d in ("film", "aerospace", "robotics"):
        for q in json.loads((ROOT / f"data/queries/scored/{d}_v0.2_scored.json").read_text()):
            src = q["gold_moments"][0] if q["gold_moments"] else q["metadata"]["altered_from"]
            if src["asset_id"] in assets:
                queries.append(q)
    (DIAG / "queries.json").write_text(json.dumps(queries, indent=2, ensure_ascii=False)
                                       + "\n")
    preds = [{"query_id": q["query_id"],
              "moments": [q["gold_moments"][0]] if q["gold_moments"] and i % 2 == 0 else []}
             for i, q in enumerate(queries)]
    (DIAG / "predictions.json").write_text(json.dumps(
        {"system": "diagnostic-reference", "predictions": preds}, indent=2) + "\n")
    expected = {"floors": {m: floor_hits(m, corpus_dir) for m in ("transcript", "abstain")},
                "predictions_recall_at_1": prediction_recall()}
    (DIAG / "expected.json").write_text(json.dumps(expected, indent=2) + "\n")


def main(corpus_dir: Path) -> int:
    results = run_checks(corpus_dir)
    for name, ok, detail in results:
        print(f"{'PASS' if ok else 'FAIL'}  {name}: {detail}")
    passed = all(ok for _, ok, _ in results) and len(results) == 5
    print("\nThis setup reproduces the reference." if passed else
          "\nFix the first FAIL above, then run diagnose again.")
    return 0 if passed else 1
