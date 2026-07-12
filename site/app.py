"""MomentBench leaderboard site — FastAPI + Jinja2."""

from __future__ import annotations

from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates

from momentbench.eval.metrics import compute_aggregate, compute_by_tier
from momentbench.runner import load_run

app = FastAPI(title="MomentBench Leaderboard")
templates = Jinja2Templates(directory=str(Path(__file__).parent / "templates"))

RESULTS_DIR = Path(__file__).parent.parent / "data" / "results"


def get_leaderboard_data() -> list[dict]:
    """Load all runs and compute leaderboard rankings."""
    if not RESULTS_DIR.exists():
        return []

    entries = []
    for f in RESULTS_DIR.glob("*.json"):
        try:
            run = load_run(f)
            overall = compute_aggregate(run)
            by_tier = compute_by_tier(run)
            entries.append({
                "model_id": run.model_id,
                "run_id": run.run_id,
                "recall_at_1": overall.recall_at_1,
                "recall_at_5": overall.recall_at_5,
                "mean_tiou": overall.mean_temporal_iou,
                "num_queries": overall.num_queries,
                "tier_breakdown": {
                    str(k.value): {
                        "recall_at_1": v.recall_at_1,
                        "recall_at_5": v.recall_at_5,
                        "mean_tiou": v.mean_temporal_iou,
                    }
                    for k, v in by_tier.items()
                },
                "timestamp": run.timestamp,
            })
        except Exception:
            continue

    entries.sort(key=lambda x: x["recall_at_1"], reverse=True)
    return entries


@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    leaderboard = get_leaderboard_data()
    return templates.TemplateResponse(
        "index.html", {"request": request, "leaderboard": leaderboard}
    )


@app.get("/api/leaderboard")
async def api_leaderboard():
    return get_leaderboard_data()


@app.get("/api/run/{run_id}")
async def api_run(run_id: str):
    for f in RESULTS_DIR.glob("*.json"):
        if run_id in f.stem:
            run = load_run(f)
            return run.model_dump()
    return {"error": "Run not found"}
