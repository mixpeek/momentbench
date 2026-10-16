"""Propose tighter gold spans for queries the pre-review flagged, and verify them.

    python scripts/refine_boundaries.py data/queries/film_v0.2.json \\
        --prereview REVIEW/film-public/prereview.json --corpus-dir CORPUS/videos \\
        --work REVIEW/film-public/refine --out REVIEW/film-public/refine.json
    python scripts/refine_boundaries.py ... --apply   # write accepted spans into the file

Scene detection misses dissolves and dark cuts, so some gold spans start or end inside a
neighbouring shot. For each query the pre-review marked `fix`, Claude sees frames every
half second or so from two seconds before the span to two seconds after, labelled with
their times, and picks where the described moment starts and ends. The new span is then
judged by the same pre-review prompt on fresh frames; only spans it accepts are proposed.
Applied spans record the old span in `metadata.boundary_refined`, and they are model-chosen,
not detected cuts, so human review still decides.
"""

from __future__ import annotations

import argparse
import asyncio
import datetime
import json
import subprocess
import sys
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

sys.path.insert(0, str(Path(__file__).resolve().parent))

import draft_queries as dq  # noqa: E402
import prereview_queries as pr  # noqa: E402

from momentbench.annotation import annotation_sha  # noqa: E402

PICK_PROMPT = """Read the image {sheet}. Its tiles are frames from one video, left to right and
top to bottom, each labelled with its time in seconds.

Query: "{text}"
Current gold span: {t_in:.2f} to {t_out:.2f} s. A reviewer said: {reason}

Using only the labelled frames, give the start and end of the moment the query describes:
start = the time of the first frame that shows it, end = the time of the last frame that
shows it. If the query text itself is wrong (no frame matches it), say so instead.
Reply: {{"start": seconds, "end": seconds}} or {{"unfixable": "one sentence"}}"""


EDIT_PROMPT = """A reviewer looked at the frames for this video retrieval query and said:
{reason}

Query: "{text}"

Rewrite the query with the smallest change that fixes exactly the problem the reviewer
named, keeping everything else the same (same moment, same style, one or two sentences).
Do not add new details you cannot be sure of.
Reply: {{"text": "..."}}"""


def video_end(video: Path) -> float:
    """Where the picture ends; matroska files report no stream duration, so fall back."""
    for entry in ("stream=duration", "format=duration"):
        cmd = ["ffprobe", "-v", "error", "-show_entries", entry, "-of", "csv=p=0", str(video)]
        if entry.startswith("stream"):
            cmd[3:3] = ["-select_streams", "v:0"]
        out = subprocess.run(cmd, capture_output=True, text=True).stdout.strip().strip(",")
        try:
            return float(out)
        except ValueError:
            continue
    return 1e9


def frames_around(video: Path, a: float, b: float, end: float, out: Path) -> list[tuple]:
    """Frames every `step` seconds over [a-2, b+2], as (time, path) pairs."""
    lo, hi = max(0.0, a - 2.0), min(end - 0.1, b + 2.0)
    step = max(0.5, (hi - lo) / 24)
    out.mkdir(parents=True, exist_ok=True)
    times, t = [], lo
    while t <= hi + 1e-6:
        times.append(round(t, 2))
        t += step
    pairs = []
    for t in times:
        f = out / f"t{t:08.2f}.jpg"
        dq.frame(video, t, f)
        pairs.append((t, f))
    return pairs, step


def sheet(pairs: list[tuple], out: Path) -> Path:
    font = ImageFont.load_default(size=16)
    tiles = []
    for t, f in pairs:
        img = Image.open(f).convert("RGB")
        img.thumbnail((240, 240))
        draw = ImageDraw.Draw(img)
        label = f"{t:.1f}s"
        draw.rectangle((0, 0, 8 * len(label) + 10, 22), fill="black")
        draw.text((4, 2), label, fill="yellow", font=font)
        tiles.append(img)
    return dq.grid(tiles, 6, out)


def five_frames(video: Path, a: float, b: float, end: float, out: Path) -> None:
    """Write the before/start/middle/end/after frames the pre-review reads."""
    (out / "frames").mkdir(parents=True, exist_ok=True)
    for label, t in (("before", a - 1.0), ("start", a + 0.2), ("middle", (a + b) / 2),
                     ("end", b - 0.2), ("after", b + 1.0)):
        path = out / "frames" / f"q_{label}.jpg"
        if 0 <= t < end - 0.05:
            dq.frame(video, t, path)
        else:  # never let an earlier span's frame stand in for one that does not exist
            path.unlink(missing_ok=True)


async def refine(q: dict, reason: str, corpus: Path, work: Path, sem) -> dict:
    g = q["gold_moments"][0]
    video = next(corpus.glob(f"{g['asset_id']}.*"))
    end = video_end(video)
    qdir = work / q["query_id"]
    cache = qdir / "result.json"
    if cache.exists() and (old := json.loads(cache.read_text())).get(
            "annotation_sha") == annotation_sha(q):
        return old
    pairs, step = frames_around(video, g["t_in"], g["t_out"], end, qdir / "dense")
    prompt = PICK_PROMPT.format(sheet=sheet(pairs, qdir / "dense.jpg"), text=q["text"],
                                t_in=g["t_in"], t_out=g["t_out"], reason=reason)
    try:
        pick = dq.parse_json(await dq.ask(dq.DRAFT_SYSTEM, prompt, qdir, sem))
    except Exception as e:
        return {"status": "error", "reason": str(e)[:200]}
    if "unfixable" in pick:
        result = {"status": "unfixable", "reason": str(pick["unfixable"])[:300]}
    else:
        a = round(max(0.0, float(pick["start"])), 2)
        b = round(min(end, float(pick["end"]) + step), 2)
        if b - a < 1.0:
            result = {"status": "too_short", "span": [a, b]}
        else:
            # Independent check on fresh frames of the proposed span.
            check = qdir / "check"
            five_frames(video, a, b, end, check)
            q2 = {**q, "query_id": "q", "gold_moments": [{**g, "t_in": a, "t_out": b}]}
            _, verdict = await pr.review(q2, check, pr.ROOT / "data/transcripts", sem)
            if verdict["decision"] == "error":  # a failed call, not a judgement: retry later
                return {"status": "error", "reason": verdict["reason"]}
            result = {"status": "accepted" if verdict["decision"] == "accept" else "not_verified",
                      "old": [g["t_in"], g["t_out"]], "new": [a, b], "check": verdict}
            if result["status"] == "not_verified" and verdict["decision"] == "fix":
                # The span may be right and a detail wrong: try the smallest text edit,
                # then check it again on the same frames with a fresh call.
                try:
                    edit = dq.parse_json(await dq.ask(dq.DRAFT_SYSTEM, EDIT_PROMPT.format(
                        reason=verdict["reason"], text=q["text"]), qdir, sem))
                    new_text = str(edit.get("text", "")).strip()
                except (ValueError, RuntimeError, json.JSONDecodeError) as e:
                    return {"status": "error", "reason": f"text edit failed: {str(e)[:120]}"}
                if new_text and new_text != q["text"]:
                    recheck = qdir / "recheck"
                    five_frames(video, a, b, end, recheck)
                    q3 = {**q2, "text": new_text}
                    _, v2 = await pr.review(q3, recheck, pr.ROOT / "data/transcripts", sem)
                    if v2["decision"] == "error":
                        return {"status": "error", "reason": v2["reason"]}
                    if v2["decision"] == "accept":
                        result.update(status="accepted", text=new_text, check=v2,
                                      first_check=verdict)
    qdir.mkdir(parents=True, exist_ok=True)
    cache.write_text(json.dumps({**result, "annotation_sha": annotation_sha(q)}, indent=1))
    return result


async def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("queries")
    p.add_argument("--prereview", required=True)
    p.add_argument("--corpus-dir", required=True)
    p.add_argument("--work", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--apply", action="store_true", help="write accepted spans into the file")
    p.add_argument("--concurrency", type=int, default=4)
    args = p.parse_args()

    path = Path(args.queries)
    queries = json.loads(path.read_text())
    pre = json.loads(Path(args.prereview).read_text())
    todo = [q for q in queries if q["gold_moments"]
            and pre.get(q["query_id"], {}).get("decision") == "fix"]
    sem = asyncio.Semaphore(args.concurrency)
    results = await asyncio.gather(*(refine(q, pre[q["query_id"]]["reason"],
                                            Path(args.corpus_dir), Path(args.work), sem)
                                     for q in todo))
    proposals = {q["query_id"]: r for q, r in zip(todo, results)}
    Path(args.out).write_text(json.dumps(proposals, indent=1))
    counts: dict[str, int] = {}
    for r in proposals.values():
        counts[r["status"]] = counts.get(r["status"], 0) + 1
    print(f"{path.name}: {len(todo)} flagged for fix, {json.dumps(counts)}")

    if args.apply:
        today = datetime.date.today().isoformat()
        # UTC instant, comparable with a run's timestamp: rescoring uses it to tell
        # whether a saved run saw the old wording or the new one.
        applied_at = datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
        for q in queries:
            r = proposals.get(q["query_id"])
            # Already applied on an earlier run: keep the record of what it replaced.
            if r and r["status"] == "accepted" and "boundary_refined" not in q["metadata"]:
                g = q["gold_moments"][0]
                q["metadata"]["boundary_refined"] = {
                    "from": r["old"], "to": r["new"], "date": today,
                    "by": f"{dq.MODEL} from frames, verified by an independent pre-review call",
                }
                g["t_in"], g["t_out"] = r["new"]
                if r.get("text"):
                    q["metadata"]["text_refined"] = {"from": q["text"], "date": today,
                                                     "applied_at": applied_at,
                                                     "reason": r["first_check"]["reason"]}
                    q["text"] = r["text"]
        path.write_text(json.dumps(queries, indent=2, ensure_ascii=False) + "\n")
        applied = sum(1 for q in queries
                      if q["metadata"].get("boundary_refined", {}).get("date") == today)
        print(f"{path.name}: {applied} queries carry a refinement applied today")


if __name__ == "__main__":
    asyncio.run(main())
