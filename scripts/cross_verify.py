"""Re-check model-drafted queries with a second model family.

    python scripts/cross_verify.py data/queries/film_v0.2.json --work WORKDIR \\
        --corpus-dir CORPUS/videos --transcripts data/transcripts

Each query is shown to Gemini against the same candidates the drafting verifier
used (its gold span, or for a no-answer query its altered source shot, plus its
stored hard negatives), reshuffled. A query agrees when Gemini picks the gold with
no wrong detail, or, for a no-answer query, picks nothing for the altered text and
the source shot for the unaltered original. Queries that disagree are moved to a
`<name>.rejected.json` file in the work directory, so the drafter's own bias cannot carry a
query into the set on its own.

Needs GEMINI_API_KEY. Costs well under a dollar for a few hundred queries on Flash.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import draft_queries as dq  # noqa: E402

MODEL = "gemini-3.5-flash"


def draft_index(work: Path) -> dict[tuple[str, str], dict]:
    """Drafts by (asset_id, text), to recover each no-answer query's unaltered original."""
    index = {}
    for f in work.glob("*/draft*.json"):
        for d in json.loads(f.read_text()):
            index[(d["asset_id"], d["text"])] = d
    return index


async def ask_gemini(client, system: str, prompt: str, image: Path, sem, usage: dict) -> dict:
    from google.genai import types

    async with sem:
        for attempt in range(6):
            try:
                r = await client.aio.models.generate_content(
                    model=MODEL,
                    contents=[
                        types.Part.from_bytes(data=image.read_bytes(), mime_type="image/jpeg"),
                        prompt,
                    ],
                    config=types.GenerateContentConfig(
                        system_instruction=system, response_mime_type="application/json"),
                )
                break
            except Exception as e:
                if getattr(e, "code", None) not in (429, 500, 503) or attempt == 5:
                    raise
                await asyncio.sleep(10 * (attempt + 1))
    u = r.usage_metadata
    usage["in"] += u.prompt_token_count or 0
    usage["out"] += (u.candidates_token_count or 0) + (u.thoughts_token_count or 0)
    return dq.parse_json(r.text or "{}")


async def judge(films, q: dict, text: str, gold_span, sem, client, usage, tag: str) -> str:
    """Return which candidate Gemini picked for `text`: "gold", "neg", "none" or "wrong"."""
    asset = q["gold_moments"][0]["asset_id"] if q["gold_moments"] else \
        q["metadata"]["altered_from"]["asset_id"]
    film = films[asset]
    candidates = [(asset, gold_span, "gold")] + [
        (n["asset_id"], (n["t_in"], n["t_out"]), f"neg{i}")
        for i, n in enumerate(q.get("hard_negatives", [])) if n["asset_id"] in films
    ]
    random.Random(q["query_id"] + tag + "x").shuffle(candidates)
    labels = "ABCD"[: len(candidates)]
    tiles, dialogue = [], []
    for label, (fid, span, ctag) in zip(labels, candidates):
        key = f"x_{q['query_id']}_{ctag}"
        for p in films[fid].strip(span, key):
            tiles.append(dq.label_tile(p, label))
        if q["tier"] == 2:
            dialogue.append(f"{label} dialogue: {films[fid].dialogue(*span)}")
    sheet = dq.grid(tiles, 3, film.dir / f"xverify_{q['query_id']}_{tag}.jpg")
    said = "\n" + "\n".join(dialogue) + "\n" if dialogue else ""
    prompt = dq.VERIFY_PROMPT.format(sheet="attached", last_label=labels[-1], text=text,
                                     dialogue=said)
    reply = await ask_gemini(client, dq.VERIFY_SYSTEM, prompt, sheet, sem, usage)
    picked = dict(zip(labels, (c[2] for c in candidates))).get(
        str(reply.get("answer", "")).strip().upper(), "none")
    if picked == "gold" and reply.get("wrong_details"):
        return "wrong"
    return "neg" if picked.startswith("neg") else picked


async def cross_check(films, drafts, q: dict, sem, client, usage) -> dict:
    if q["gold_moments"]:
        g = q["gold_moments"][0]
        verdict = await judge(films, q, q["text"], (g["t_in"], g["t_out"]), sem, client,
                              usage, "t")
        return {"model": MODEL, "agreed": verdict == "gold", "picked": verdict}
    src = q["metadata"]["altered_from"]
    span = (src["t_in"], src["t_out"])
    altered = await judge(films, q, q["text"], span, sem, client, usage, "a")
    original = (drafts.get((src["asset_id"], q["text"])) or {}).get("original")
    orig_verdict = await judge(films, q, original, span, sem, client, usage, "o") \
        if original else "unavailable"
    agreed = altered == "none" and orig_verdict in ("gold", "unavailable")
    return {"model": MODEL, "agreed": agreed, "picked": altered, "original_picked": orig_verdict}


async def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("queries", nargs="+")
    p.add_argument("--manifest",
                   default=str(dq.ROOT / "data/corpus_manifests/sample_manifest.json"))
    p.add_argument("--corpus-dir", required=True)
    p.add_argument("--transcripts", default=str(dq.ROOT / "data/transcripts"))
    p.add_argument("--work", required=True)
    p.add_argument("--concurrency", type=int, default=8)
    args = p.parse_args()

    from google import genai

    client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])
    assets = json.loads(Path(args.manifest).read_text())
    corpus, work, tdir = Path(args.corpus_dir), Path(args.work), Path(args.transcripts)
    films = {}
    for a in assets:
        video = next(corpus.glob(f"{a['asset_id']}.*"), None)
        if video and (work / a["asset_id"] / "shots.json").exists():
            films[a["asset_id"]] = dq.Film(a, video, tdir / f"{a['asset_id']}.vtt", work)
    drafts = draft_index(work)
    sem = asyncio.Semaphore(args.concurrency)
    usage = {"in": 0, "out": 0}
    for path in map(Path, args.queries):
        queries = json.loads(path.read_text())
        results = await asyncio.gather(
            *(cross_check(films, drafts, q, sem, client, usage) for q in queries),
            return_exceptions=True,
        )
        kept, rejected = [], []
        for q, r in zip(queries, results):
            if isinstance(r, Exception):
                r = {"model": MODEL, "agreed": False, "error": str(r)[:200]}
            q["metadata"]["cross_check"] = r
            if r["agreed"]:
                q["metadata"]["source"] += f"; cross-checked by {MODEL}"
                kept.append(q)
            else:
                rejected.append(q)
        path.write_text(json.dumps(kept, indent=2, ensure_ascii=False) + "\n")
        out = work / f"{path.stem}.rejected.json"
        old = json.loads(out.read_text()) if out.exists() else []
        out.write_text(json.dumps(old + rejected, indent=2, ensure_ascii=False) + "\n")
        print(f"{path.name}: kept {len(kept)}, rejected {len(rejected)}")
    print(f"tokens: {usage['in']} in, {usage['out']} out")


if __name__ == "__main__":
    asyncio.run(main())
