"""Model pre-review of queries, to focus a human reviewer's time.

    python scripts/prereview_queries.py data/queries/film_v0.2.json \\
        --review-dir REVIEW/film-public --out REVIEW/film-public/prereview.json

Reads the five frames that `review_queries.py build` wrote for each query (one second
before the gold span, its start, middle and end, one second after) and asks Claude
for accept, fix or reject with a reason, judging the text and the span boundaries the
way the human page asks. The output is shown on the review page next to each query,
with flagged queries first. It is never applied as a decision: only a human's
choices on the page are recorded as review (`review_queries.py apply`).
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import draft_queries as dq  # noqa: E402

from momentbench.agentic.environment import load_transcript  # noqa: E402

LABELS = ("before", "start", "middle", "end", "after")
SYSTEM = (
    "You review video retrieval queries before they enter a benchmark. Be strict and "
    "concrete. Reply with JSON only."
)
PROMPT = """Read the image {sheet}. It shows five frames, left to right: one second BEFORE
the gold span, the span's START, MIDDLE and END, and one second AFTER (the before and
after frames are outside the span).

Query (tier {tier}): "{text}"
Gold span: {asset} {t_in:.2f} to {t_out:.2f} s.
{extra}
Decide:
- accept: the text matches what happens inside the span, every detail is right, and the
  span starts and ends roughly where the described moment does.
- fix: the query is right in substance but one detail or a boundary is off by more than
  about a second; say exactly what to change.
- reject: the text is wrong, too vague to pin one moment, or does not match the span.
{t3}
Reply: {{"decision": "accept" | "fix" | "reject", "reason": "one sentence"}}"""
T3_RULE = ("This is a no-answer query: the frames show the real source shot, and the text "
           "deliberately changes one detail ({change}). Accept only if that change makes the "
           "text clearly false for these frames while the rest still fits.")


def sheet_for(review_dir: Path, qid: str, out: Path) -> Path | None:
    frames = [review_dir / "frames" / f"{qid}_{label}.jpg" for label in LABELS]
    if not all(f.exists() for f in frames):
        return None
    tiles = [Image.open(f).convert("RGB") for f in frames]
    w, h = tiles[0].size
    sheet = Image.new("RGB", (w * 5, h + 22), "black")
    draw = ImageDraw.Draw(sheet)
    font = ImageFont.load_default(size=16)
    for i, (tile, label) in enumerate(zip(tiles, LABELS)):
        sheet.paste(tile, (i * w, 22))
        draw.text((i * w + 4, 3), label, fill="yellow", font=font)
    sheet.save(out, quality=85)
    return out


async def review(q: dict, review_dir: Path, transcripts: Path, sem) -> tuple[str, dict]:
    qid = q["query_id"]
    span = q["gold_moments"][0] if q["gold_moments"] else q["metadata"].get("altered_from")
    work = review_dir / "prereview"
    work.mkdir(exist_ok=True)
    cache = work / f"{qid}.json"
    if cache.exists():
        return qid, json.loads(cache.read_text())
    sheet = sheet_for(review_dir, qid, work / f"{qid}.jpg")
    if span is None or sheet is None:
        return qid, {"decision": "fix", "reason": "no frames to review"}
    extra = ""
    if q["tier"] == 2:
        vtt = transcripts / f"{span['asset_id']}.vtt"
        said = [s["text"] for s in (load_transcript(vtt) if vtt.exists() else [])
                if s["t_out"] > span["t_in"] and s["t_in"] < span["t_out"]]
        extra = "Transcript during the span: " + (" ".join(said) or "(none)") + "\n"
    t3 = T3_RULE.format(change=span.get("change", "")) if not q["gold_moments"] else ""
    prompt = PROMPT.format(sheet=sheet, tier=q["tier"], text=q["text"], asset=span["asset_id"],
                           t_in=span["t_in"], t_out=span["t_out"], extra=extra, t3=t3)
    try:
        verdict = dq.parse_json(await dq.ask(SYSTEM, prompt, work, sem))
        result = {"decision": str(verdict.get("decision", "fix")).lower(),
                  "reason": str(verdict.get("reason", ""))[:300]}
    except Exception as e:  # one failure must not stop the pass
        result = {"decision": "fix", "reason": f"pre-review failed: {str(e)[:120]}"}
    cache.write_text(json.dumps(result))
    return qid, result


async def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("queries")
    p.add_argument("--review-dir", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--transcripts", default=str(ROOT / "data/transcripts"))
    p.add_argument("--concurrency", type=int, default=4)
    args = p.parse_args()

    queries = json.loads(Path(args.queries).read_text())
    sem = asyncio.Semaphore(args.concurrency)
    pairs = await asyncio.gather(*(review(q, Path(args.review_dir), Path(args.transcripts), sem)
                                   for q in queries))
    out = dict(pairs)
    Path(args.out).write_text(json.dumps(out, indent=1))
    counts: dict[str, int] = {}
    for v in out.values():
        counts[v["decision"]] = counts.get(v["decision"], 0) + 1
    print(f"{Path(args.queries).name}: {json.dumps(counts)}")


if __name__ == "__main__":
    asyncio.run(main())
