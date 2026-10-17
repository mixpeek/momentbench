"""Check every no-answer query against the whole corpus, not just three look-alikes.

    MOMENTBENCH_SPEND_LEDGER=... MOMENTBENCH_VERTEX_PROJECT=... MOMENTBENCH_GCLOUD_ACCOUNT=... \\
    OPENAI_API_KEY=... python scripts/absence_check.py --corpus-dir CORPUS/videos

A no-answer query is right only if nothing in the corpus matches it. Drafting checked
each one against three look-alike shots. Here the same two non-Claude judges as
scripts/adjudicate.py check the whole corpus:

1. Gemini watches each of the 27 videos once (a 1 fps, 320 px copy with audio) with all
   no-answer descriptions numbered, and lists every moment where all details of a
   description are true.
2. GPT checks each hit on a frame sheet around that moment.

A hit both judges confirm means the query has a match after all: the result goes to
`data/labels/adjudication.json` under the query's `absence` key, and label_sets.py
keeps it out of the scored set. Hits only one judge sees are recorded as unconfirmed.
"""

from __future__ import annotations

import argparse
import asyncio
import base64
import json
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

import adjudicate as adj  # noqa: E402

from momentbench import spend  # noqa: E402
from momentbench.annotation import annotation_sha  # noqa: E402
from momentbench.corpus.frame_cache import extract  # noqa: E402

SCAN = """Watch this whole video ({duration:.0f} s). Below are numbered descriptions of
moments. Most of them describe nothing in this video. For each description, list every
moment in this video where EVERY stated detail is true (colours, counts, objects,
actions, words). If a single detail is wrong, it is not a match. Be strict.

{items}

Reply as JSON: {{"matches": [{{"id": <number>, "t_start": <seconds>, "t_end": <seconds>,
"why": "one sentence"}}]}}. Reply {{"matches": []}} if nothing matches."""

CONFIRM = """These frames come from one video between {a:.1f} and {b:.1f} s. Is every
detail of this description true of what the frames show?

"{text}"

Reply as JSON: {{"match": true | false, "reason": "one sentence"}}"""


class IncompleteScanError(RuntimeError):
    """A scan reply could not be read, so absence was not checked."""


SCAN_OUT = 4_000  # max output tokens per scan call
SCAN_FPS = 1
SCAN_WIDTH = 320
SCAN_RESOLUTION = "MEDIA_RESOLUTION_LOW"
CHUNK = 100  # descriptions per call


def small_copy(video: Path, out: Path) -> bytes:
    if not out.exists():
        scale = f"fps={SCAN_FPS},scale={SCAN_WIDTH}:-2"
        subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", str(video), "-vf", scale,
                        "-c:v", "libx264", "-preset", "veryfast", "-crf", "32",
                        "-ac", "1", "-c:a", "aac", "-b:a", "32k", str(out)], check=True)
    return out.read_bytes()


async def scan(judges: adj.Judges, asset: dict, negs: list[dict], work: Path) -> list[dict]:
    out = []
    for i in range(0, len(negs), CHUNK):
        out += await scan_chunk(judges, asset, negs[i:i + CHUNK], work)
    return out


async def scan_chunk(judges: adj.Judges, asset: dict, negs: list[dict],
                     work: Path) -> list[dict]:
    from google.genai import types

    from momentbench.agentic.gemini_client import make_client

    judges._gemini = judges._gemini or make_client()
    data = small_copy(judges.video(asset["asset_id"]),
                      work / f"{asset['asset_id']}_{SCAN_FPS}_{SCAN_WIDTH}.mp4")
    items = "\n".join(f"{i}. {q['text']}" for i, q in enumerate(negs, 1))
    duration = float(asset.get("duration_s") or 600)
    # Video tokens per frame (low 70, medium 260, with headroom) plus 32 audio tokens a
    # second, plus the prompt; output is capped below at SCAN_OUT tokens.
    per_frame = 80 if SCAN_RESOLUTION.endswith("LOW") else 300
    tokens = int(duration * (SCAN_FPS * per_frame + 40)) + 10_000
    rid = spend.reserve(spend.token_cost_usd(adj.GEMINI, tokens, SCAN_OUT),
                        f"absence:{adj.GEMINI}")
    async with judges.sem:
        r = await judges._gemini.aio.models.generate_content(
            model=adj.GEMINI,
            contents=[types.Part(inline_data=types.Blob(data=data, mime_type="video/mp4"),
                                 video_metadata=types.VideoMetadata(fps=SCAN_FPS)),
                      SCAN.format(duration=duration, items=items)],
            config=types.GenerateContentConfig(
                response_mime_type="application/json", max_output_tokens=SCAN_OUT,
                media_resolution=getattr(types.MediaResolution, SCAN_RESOLUTION)))
    u = r.usage_metadata
    spend.settle(rid, spend.token_cost_usd(adj.GEMINI, u.prompt_token_count or 0,
                                      (u.candidates_token_count or 0)
                                      + (u.thoughts_token_count or 0)),
                 f"absence:{adj.GEMINI}", asset=asset["asset_id"])
    text = r.text or ""
    try:
        found = json.loads(text[text.find("{"):text.rfind("}") + 1])["matches"]
        if not isinstance(found, list):
            raise ValueError("matches is not a list")
    except (ValueError, KeyError, TypeError) as e:
        # A reply we cannot read is not "no match": the check stays incomplete.
        raise IncompleteScanError(f"{asset['asset_id']}: unreadable scan reply ({e})") from e
    out = []
    duration = float(asset.get("duration_s") or 0) or float("inf")
    for m in found:
        try:
            i = int(m["id"])
            a, b = float(m["t_start"]), float(m["t_end"])
            if not (1 <= i <= len(negs) and 0 <= a <= b <= duration + 1):
                raise ValueError(f"id {i} or span {a}-{b} out of range")
        except (KeyError, ValueError, TypeError) as e:
            # One malformed entry could hide a real match: the whole scan is incomplete.
            raise IncompleteScanError(f"{asset['asset_id']}: malformed match ({e})") from e
        out.append({"query_id": negs[i - 1]["query_id"], "asset_id": asset["asset_id"],
                    "t_start": a, "t_end": b, "why": str(m.get("why", ""))[:300]})
    return out


async def confirm(judges: adj.Judges, hit: dict, text: str, work: Path) -> dict:
    from openai import AsyncOpenAI

    judges._gpt = judges._gpt or AsyncOpenAI()
    a, b = hit["t_start"], max(hit["t_end"], hit["t_start"] + 0.5)
    times = [a + (b - a) * f for f in (0.0, 0.33, 0.66, 1.0)]
    frames = []
    for i, t in enumerate(times):
        p = extract(judges.video(hit["asset_id"]), t,
                    work / f"hit_{hit['query_id']}_{hit['asset_id']}_{a:.0f}_{i}.jpg", 512)
        if p is None:  # no frame means no confirmation: incomplete, not unconfirmed
            return {**hit, "confirmed": None, "incomplete": "frame extraction failed"}
        frames.append(p.read_bytes())
    content = [{"type": "text", "text": CONFIRM.format(a=a, b=b, text=text)}] + [
        {"type": "image_url", "image_url": {
            "url": "data:image/jpeg;base64," + base64.b64encode(f).decode()}} for f in frames]
    rid = spend.reserve(spend.token_cost_usd(adj.GPT, 8_000, 1_000), f"absence:{adj.GPT}")
    async with judges.sem:
        r = await judges._gpt.chat.completions.create(
            model=adj.GPT, messages=[{"role": "user", "content": content}],
            response_format={"type": "json_object"}, max_completion_tokens=1_000)
    spend.settle(rid, spend.token_cost_usd(adj.GPT, r.usage.prompt_tokens,
                                      r.usage.completion_tokens), f"absence:{adj.GPT}",
                 query=hit["query_id"])
    reply = r.choices[0].message.content or ""
    try:
        out = json.loads(reply[reply.find("{"):reply.rfind("}") + 1])
        if not isinstance(out.get("match"), bool):
            raise ValueError("no boolean match verdict")
    except ValueError as e:
        return {**hit, "confirmed": None, "incomplete": f"unreadable verdict ({e})"}
    return {**hit, "confirmed": out["match"], "gpt_reason": str(out.get("reason", ""))[:300]}


async def control(judges: adj.Judges, assets: list[dict], queries: list[dict],
                  negs: list[dict], work: Path, per_asset: int) -> dict:
    """How sensitive is the scan? Mix true descriptions (single-shot positives) from each
    video into the list and count how many the scan finds near their gold span."""
    found = total = 0
    misses = []
    for a in assets:
        pos = [q for q in queries if q["gold_moments"] and int(q["tier"]) == 1
               and q["gold_moments"][0]["asset_id"] == a["asset_id"]][:per_asset]
        if not pos:
            continue
        hits = await scan(judges, a, pos + negs, work)
        for q in pos:
            g = q["gold_moments"][0]
            ok = any(h["query_id"] == q["query_id"] and min(h["t_end"], g["t_out"] + 2)
                     > max(h["t_start"], g["t_in"] - 2) for h in hits)
            found, total = found + ok, total + 1
            if not ok:
                misses.append(q["query_id"])
    return {"videos": len(assets), "true_descriptions": total, "found": found,
            "recall": round(found / total, 3) if total else None, "missed": misses}


def is_source(hit: dict, q: dict) -> bool:
    """A hit on the shot the query was altered from is the intended near miss, which
    adjudicate.py already judged; only matches elsewhere test absence."""
    src = q["metadata"]["altered_from"]
    return hit["asset_id"] == src["asset_id"] and min(hit["t_end"], src["t_out"]) > max(
        hit["t_start"], src["t_in"]) - 2.0


async def run(args: argparse.Namespace) -> dict:
    queries = [q for d in adj.DOMAINS
               for q in json.loads((ROOT / f"data/queries/{d}_v0.2.json").read_text())]
    negs = [q for q in queries if not q["gold_moments"]]
    work = Path(tempfile.mkdtemp(prefix="momentbench-absence-"))
    judges = adj.Judges(Path(args.corpus_dir), ROOT / "data/transcripts", work,
                        args.concurrency)
    if args.control:
        picked = [a for a in judges.assets.values() if a["asset_id"] in args.control]
        result = await control(judges, picked, queries, negs, work, args.per_asset)
        log = json.loads(adj.LOG.read_text())
        log["summary"].setdefault("absence", {})["control"] = result
        adj.LOG.write_text(json.dumps(log, indent=1, ensure_ascii=False) + "\n")
        return result
    assets = list(judges.assets.values())
    if args.limit:
        assets = assets[: args.limit]
    spend.require_ledger("absence_check")
    results = await asyncio.gather(*(scan(judges, a, negs, work) for a in assets),
                                   return_exceptions=True)
    failed = [str(r) for r in results if isinstance(r, Exception)]
    scans = [r for r in results if not isinstance(r, Exception)]
    by_id = {q["query_id"]: q for q in negs}
    hits = [h for hs in scans for h in hs if not is_source(h, by_id[h["query_id"]])]
    checked = await asyncio.gather(*(confirm(judges, h, by_id[h["query_id"]]["text"], work)
                                     for h in hits))
    log = json.loads(adj.LOG.read_text())
    scanned = sorted(a["asset_id"] for a in assets)
    for q in negs:
        mine = [h for h in checked if h["query_id"] == q["query_id"]]
        log["queries"].setdefault(q["query_id"], {})["absence"] = {
            "annotation_sha": annotation_sha(q), "assets_scanned": len(scanned),
            "confirmed_matches": [h for h in mine if h["confirmed"] is True],
            "unconfirmed_hits": [h for h in mine if h["confirmed"] is False],
            "incomplete_hits": [h for h in mine if h["confirmed"] is None],
            "incomplete": failed,
            # Passed only when every video was scanned, every reply was readable and
            # every hit got an explicit "no match" verdict.
            "passed": len(scanned) == len(judges.assets) and not failed and all(
                h["confirmed"] is False for h in mine)}
    absent = sum(1 for q in negs if log["queries"][q["query_id"]]["absence"]["passed"])
    log["summary"]["absence"] = {"queries": len(negs), "assets": len(scanned),
                                 "hits_off_source": len(hits),
                                 "confirmed": sum(h["confirmed"] is True for h in checked),
                                 "incomplete": sum(h["confirmed"] is None for h in checked)
                                 + len(failed),
                                 "passed": absent}
    adj.LOG.write_text(json.dumps(log, indent=1, ensure_ascii=False) + "\n")
    return log["summary"]["absence"]


def main() -> None:
    global SCAN_FPS, SCAN_WIDTH, SCAN_RESOLUTION, CHUNK
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--corpus-dir", required=True)
    p.add_argument("--limit", type=int, default=0, help="scan only the first N videos")
    p.add_argument("--concurrency", type=int, default=4)
    p.add_argument("--control", nargs="*", help="asset ids for the sensitivity control")
    p.add_argument("--per-asset", type=int, default=4)
    p.add_argument("--fps", type=float, default=SCAN_FPS)
    p.add_argument("--width", type=int, default=SCAN_WIDTH)
    p.add_argument("--resolution", default=SCAN_RESOLUTION)
    p.add_argument("--chunk", type=int, default=CHUNK)
    args = p.parse_args()
    SCAN_FPS, SCAN_WIDTH = args.fps, args.width
    SCAN_RESOLUTION, CHUNK = args.resolution, args.chunk
    print(json.dumps(asyncio.run(run(args)), indent=1))


if __name__ == "__main__":
    main()
