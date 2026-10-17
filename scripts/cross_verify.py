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

Needs GEMINI_API_KEY, or MOMENTBENCH_VERTEX_PROJECT for Vertex AI
(momentbench/agentic/gemini_client.py). Costs well under a dollar for a few hundred
queries on Flash. With MOMENTBENCH_SPEND_LEDGER set, every call counts against the cap.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import draft_queries as dq  # noqa: E402

from momentbench import spend  # noqa: E402
from momentbench.annotation import annotation_sha  # noqa: E402
from momentbench.corpus.frame_cache import frame_inputs  # noqa: E402
from momentbench.corpus.verify import sha256_file  # noqa: E402

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

    # Each check is one image and a short prompt; reserve a generous ceiling for it.
    rid = spend.reserve(spend.token_cost_usd(MODEL, 4_000, 2_000), f"cross_verify:{MODEL}")
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
    spend.settle(rid, spend.token_cost_usd(MODEL, u.prompt_token_count or 0,
                                      (u.candidates_token_count or 0)
                                      + (u.thoughts_token_count or 0)),
                 f"cross_verify:{MODEL}")
    return dq.parse_json(r.text or "{}")


def config_sha() -> str:
    """The reviewer configuration: model, prompts and the sheet layout."""
    import hashlib

    blob = json.dumps([MODEL, dq.VERIFY_SYSTEM, dq.VERIFY_PROMPT, dq.TILE_W], sort_keys=True)
    return hashlib.sha256(blob.encode()).hexdigest()[:16]


async def judge(films, q: dict, text: str, gold_span, sem, client, usage, tag: str
                ) -> tuple[str, dict]:
    """Which candidate the checker picked for `text` ("gold", "neg", "none" or "wrong"),
    and the media it was shown: each candidate's span and the source of every frame
    (video sha256 and timestamp), the contact sheet's sha256 and the dialogue."""
    asset = q["gold_moments"][0]["asset_id"] if q["gold_moments"] else \
        q["metadata"]["altered_from"]["asset_id"]
    film = films[asset]
    candidates = [(asset, gold_span, "gold")] + [
        (n["asset_id"], (n["t_in"], n["t_out"]), f"neg{i}")
        for i, n in enumerate(q.get("hard_negatives", [])) if n["asset_id"] in films
    ]
    random.Random(q["query_id"] + tag + "x").shuffle(candidates)
    labels = "ABCD"[: len(candidates)]
    tiles, dialogue, shown = [], [], []
    for label, (fid, span, ctag) in zip(labels, candidates):
        key = f"x_{q['query_id']}_{ctag}"
        paths = films[fid].strip(span, key)
        for p in paths:
            tiles.append(dq.label_tile(p, label))
        shown.append({"label": label, "role": ctag, "asset_id": fid,
                      "span": [float(span[0]), float(span[1])],
                      "frames": frame_inputs(paths)})
        if q["tier"] == 2:
            dialogue.append(f"{label} dialogue: {films[fid].dialogue(*span)}")
    sheet = dq.grid(tiles, 3, film.dir / f"xverify_{q['query_id']}_{tag}.jpg")
    media = {"candidates": shown, "sheet_sha256": sha256_file(sheet),
             "dialogue": dialogue, "config_sha": config_sha()}
    said = "\n" + "\n".join(dialogue) + "\n" if dialogue else ""
    prompt = dq.VERIFY_PROMPT.format(sheet="attached", last_label=labels[-1], text=text,
                                     dialogue=said)
    if client is None:  # Claude cross-checks Gemini-drafted queries
        claude_prompt = dq.VERIFY_PROMPT.format(sheet=sheet, last_label=labels[-1], text=text,
                                                dialogue=said)
        reply = dq.parse_json(await dq.ask_claude(dq.VERIFY_SYSTEM, claude_prompt,
                                                  sheet.parent, sem))
    else:
        reply = await ask_gemini(client, dq.VERIFY_SYSTEM, prompt, sheet, sem, usage)
    picked = dict(zip(labels, (c[2] for c in candidates))).get(
        str(reply.get("answer", "")).strip().upper(), "none")
    if picked == "gold" and reply.get("wrong_details"):
        return "wrong", media
    return ("neg" if picked.startswith("neg") else picked), media


async def cross_check(films, drafts, q: dict, sem, client, usage) -> dict:
    if q["gold_moments"]:
        g = q["gold_moments"][0]
        verdict, media = await judge(films, q, q["text"], (g["t_in"], g["t_out"]), sem,
                                     client, usage, "t")
        return {"model": MODEL, "agreed": verdict == "gold", "picked": verdict,
                "media": {"text": media}}
    src = q["metadata"]["altered_from"]
    span = (src["t_in"], src["t_out"])
    # Both halves are required: the changed text matches nothing, and the unaltered
    # original matches the source shot. Without the original the check is incomplete,
    # never agreement (audit 4, finding 4).
    original = src.get("original_text") or \
        (drafts.get((src["asset_id"], q["text"])) or {}).get("original")
    if not original:
        return {"model": MODEL, "agreed": False, "complete": False,
                "picked": None, "original_picked": "unavailable"}
    altered, media_a = await judge(films, q, q["text"], span, sem, client, usage, "a")
    orig_verdict, media_o = await judge(films, q, original, span, sem, client, usage, "o")
    return {"model": MODEL, "agreed": altered == "none" and orig_verdict == "gold",
            "picked": altered, "original_picked": orig_verdict,
            "media": {"altered": media_a, "original": media_o}}


async def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("queries", nargs="+")
    p.add_argument("--manifest",
                   default=str(dq.ROOT / "data/corpus_manifests/sample_manifest.json"))
    p.add_argument("--corpus-dir", required=True)
    p.add_argument("--transcripts", default=str(dq.ROOT / "data/transcripts"))
    p.add_argument("--work", required=True)
    p.add_argument("--concurrency", type=int, default=8)
    p.add_argument("--checker", choices=["gemini", "claude"], default="gemini",
                   help="model family that cross-checks; use the one that did not draft")
    args = p.parse_args()

    global MODEL
    client = None
    if args.checker == "gemini":
        from momentbench.agentic.gemini_client import make_client

        client = make_client()
    else:
        MODEL = dq.MODEL
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
        async def keep_or_check(q: dict):
            # Already agreed on this exact annotation: keep the recorded result rather
            # than paying to check again. An edit since then needs a new check.
            done = q["metadata"].get("cross_check") or {}
            if done.get("agreed") and done.get("annotation_sha") == annotation_sha(q):
                return done
            result = await cross_check(films, drafts, q, sem, client, usage)
            return {**result, "annotation_sha": annotation_sha(q)}

        results = await asyncio.gather(*(keep_or_check(q) for q in queries),
                                       return_exceptions=True)
        kept, rejected, failed, incomplete = [], [], 0, 0
        for q, r in zip(queries, results):
            if isinstance(r, Exception):
                # A failed call is not a verdict: keep the query, unchanged, to retry.
                failed += 1
                kept.append(q)
                continue
            if r.get("complete") is False:
                # Not a verdict either: the no-answer original could not be checked.
                # Keep the query, unchanged, until its original text is recorded.
                incomplete += 1
                kept.append(q)
                continue
            already = q["metadata"].get("cross_check") is r
            q["metadata"]["cross_check"] = r
            if not r["agreed"]:
                rejected.append(q)
                continue
            if not already and f"cross-checked by {MODEL}" not in q["metadata"]["source"]:
                q["metadata"]["source"] += f"; cross-checked by {MODEL}"
            elif not already:
                q["metadata"]["source"] = q["metadata"]["source"].replace(
                    " before a later edit; new cross-check pending", "")
            kept.append(q)
        path.write_text(json.dumps(kept, indent=2, ensure_ascii=False) + "\n")
        out = work / f"{path.stem}.rejected.json"
        old = json.loads(out.read_text()) if out.exists() else []
        out.write_text(json.dumps(old + rejected, indent=2, ensure_ascii=False) + "\n")
        print(f"{path.name}: kept {len(kept)}, rejected {len(rejected)}, "
              f"failed calls left unchanged {failed}, incomplete checks left unchanged "
              f"{incomplete}")
    print(f"tokens: {usage['in']} in, {usage['out']} out")


if __name__ == "__main__":
    asyncio.run(main())
