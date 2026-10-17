"""Adjudicate published query labels with two model families other than Claude.

    MOMENTBENCH_SPEND_LEDGER=~/.momentbench/spend.jsonl \\
    MOMENTBENCH_VERTEX_PROJECT=... MOMENTBENCH_GCLOUD_ACCOUNT=... OPENAI_API_KEY=... \\
        python scripts/adjudicate.py --corpus-dir CORPUS/videos [--limit 10]

Claude drafted, verified and reviewed these labels, so Claude does not vote here.
Two other families each judge every published query against its current annotation:

- **Gemini** watches the clip itself (native video with audio): the gold span plus 2 s
  on each side. For a no-answer query it watches the source shot the query was
  altered from, and checks that the unaltered original matches while the altered text
  does not.
- **GPT** reads a five-frame sheet (before, start, middle, end, after) of the same span,
  plus the transcript lines inside it for dialogue queries.

A query is **scored** when both families accept it, and **provisional** otherwise. The
decisions, each judge's reason and the hashes of what it was shown go to
`data/labels/adjudication.json`, the public label log. A query whose
`annotation_sha` and judge configuration are unchanged is not judged again. Every
call reserves and records its cost in the spend ledger.
"""

from __future__ import annotations

import argparse
import asyncio
import base64
import hashlib
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from momentbench import spend  # noqa: E402
from momentbench.agentic.environment import load_transcript  # noqa: E402
from momentbench.annotation import annotation_sha  # noqa: E402
from momentbench.corpus.frame_cache import extract  # noqa: E402
from momentbench.corpus.verify import resolve_video  # noqa: E402

GEMINI = os.environ.get("MOMENTBENCH_ADJ_GEMINI", "gemini-3.8-flash")
GPT = os.environ.get("MOMENTBENCH_ADJ_GPT", "gpt-5.4-mini")
DOMAINS = ("film", "aerospace", "robotics")
LOG = ROOT / "data/labels/adjudication.json"
PAD_S = 2.0

POSITIVE = """You are auditing one label in a video retrieval benchmark. Be strict.
The {media} covers source-video seconds {cs:.2f} to {ce:.2f}. The label says this
description matches the span {a:.2f} to {b:.2f} s{rel}:

"{text}"
{said}
Decide:
- accept: every stated detail is true inside that span, and the described moment starts
  and ends within about one second of the span.
- wrong_span: the description is accurate, but the moment's boundaries differ from the
  span by more than about one second.
- reject: a stated detail is false, or the description does not fit the span.
Reply as JSON: {{"verdict": "accept" | "wrong_span" | "reject", "reason": "one sentence"}}"""

NEGATIVE = """You are auditing one label in a video retrieval benchmark. Be strict.
The {media} covers source-video seconds {cs:.2f} to {ce:.2f}; the shot of interest is
{a:.2f} to {b:.2f} s{rel}.

Description A: "{original}"
Description B: "{text}"
B differs from A in one detail: {change}
{said}
Answer three questions about the shot of interest:
1. a_true: is every detail of description A true of the shot?
2. b_true: is every detail of description B true of the shot?
3. detail_visible: can you see or hear the part of the shot that the changed detail is
   about, clearly enough to tell A and B apart?
Reply as JSON: {{"a_true": true | false, "b_true": true | false,
"detail_visible": true | false, "reason": "one sentence"}}"""


def sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()[:16]


CONFIG_SHA = sha(json.dumps([GEMINI, GPT, POSITIVE, NEGATIVE, PAD_S]).encode())


def span_of(q: dict) -> tuple[str, float, float]:
    g = q["gold_moments"][0] if q["gold_moments"] else q["metadata"]["altered_from"]
    return g["asset_id"], float(g["t_in"]), float(g["t_out"])


def prompt(q: dict, media: str, cs: float, ce: float, rel: str, said: str) -> str:
    _, a, b = span_of(q)
    said = f"\nSpeech during the span: {said}\n" if said else ""
    if q["gold_moments"]:
        return POSITIVE.format(media=media, cs=cs, ce=ce, a=a, b=b, rel=rel, text=q["text"],
                               said=said)
    src = q["metadata"]["altered_from"]
    return NEGATIVE.format(media=media, cs=cs, ce=ce, a=a, b=b, rel=rel, text=q["text"],
                           original=src.get("original_text", ""),
                           change=src.get("change", "one detail was changed"), said=said)


def parse(text: str) -> dict:
    """The judge's verdict. For a no-answer query it is derived here from three yes/no
    answers, so the rubric's wording cannot flip it: accept only when the original is
    true, the altered text is false and the changed detail can be seen."""
    text = (text or "").strip()
    start, end = text.find("{"), text.rfind("}")
    try:
        out = json.loads(text[start:end + 1])
    except ValueError:
        return {"verdict": "error", "reason": f"unparseable reply: {text[:120]}"}
    reason = str(out.get("reason", ""))[:400]
    if "a_true" in out:
        answers = {k: out.get(k) is True for k in ("a_true", "b_true", "detail_visible")}
        ok = answers["a_true"] and not answers["b_true"] and answers["detail_visible"]
        return {"verdict": "accept" if ok else "reject", "reason": reason, **answers}
    return {"verdict": str(out.get("verdict", "error")).lower(), "reason": reason}


class Judges:
    def __init__(self, corpus: Path, transcripts: Path, work: Path, concurrency: int):
        self.assets = {a["asset_id"]: a for a in json.loads(
            (ROOT / "data/corpus_manifests/sample_manifest.json").read_text())}
        self.corpus, self.transcripts, self.work = corpus, transcripts, work
        self.sem = asyncio.Semaphore(concurrency)
        self._gemini = self._gpt = None

    def video(self, asset_id: str) -> Path:
        path = resolve_video(self.assets[asset_id], self.corpus)
        if path is None:
            raise FileNotFoundError(f"no video for {asset_id} under {self.corpus}")
        return path

    def said(self, q: dict) -> str:
        if int(q["tier"]) != 2:
            return ""
        aid, a, b = span_of(q)
        for ext in (".transcript.json", ".vtt"):
            if (f := self.transcripts / f"{aid}{ext}").exists():
                return " ".join(s["text"] for s in load_transcript(f)
                                if s["t_out"] > a and s["t_in"] < b)
        return ""

    def clip(self, q: dict) -> tuple[bytes, float, float]:
        aid, a, b = span_of(q)
        duration = float(self.assets[aid].get("duration_s") or b + PAD_S)
        cs, ce = max(0.0, a - PAD_S), min(duration, b + PAD_S)
        out = self.work / f"{q['query_id']}_{cs:.2f}_{ce:.2f}.mp4"
        if not out.exists():
            subprocess.run(["ffmpeg", "-v", "error", "-y", "-ss", f"{cs:.2f}", "-to",
                            f"{ce:.2f}", "-i", str(self.video(aid)), "-vf", "scale=640:-2",
                            "-c:v", "libx264", "-preset", "veryfast", "-crf", "28",
                            "-c:a", "aac", "-b:a", "64k", str(out)], check=True)
        return out.read_bytes(), cs, ce

    def frames(self, q: dict) -> tuple[list[bytes], float, float]:
        aid, a, b = span_of(q)
        duration = float(self.assets[aid].get("duration_s") or b + 1)
        times = [max(0.0, a - 1.0), a + min(0.3, (b - a) / 4), (a + b) / 2,
                 b - min(0.3, (b - a) / 4), min(duration - 0.3, b + 1.0)]
        out = []
        for i, t in enumerate(times):
            path = extract(self.video(aid), t, self.work / f"{q['query_id']}_f{i}.jpg", 512)
            if path is not None:
                out.append(path.read_bytes())
        return out, times[0], times[-1]

    async def gemini(self, q: dict) -> dict:
        from google.genai import types

        from momentbench.agentic.gemini_client import make_client

        self._gemini = self._gemini or make_client()
        data, cs, ce = self.clip(q)
        _, a, b = span_of(q)
        text = prompt(q, "clip", cs, ce, f" (clip time {a - cs:.2f} to {b - cs:.2f} s)",
                      self.said(q))
        spend.reserve(spend.token_cost_usd(GEMINI, 12_000, 2_000), f"adjudicate:{GEMINI}")
        async with self.sem:
            part = types.Part(inline_data=types.Blob(data=data, mime_type="video/mp4"),
                              video_metadata=types.VideoMetadata(fps=4))
            r = await self._gemini.aio.models.generate_content(
                model=GEMINI, contents=[part, text],
                config=types.GenerateContentConfig(response_mime_type="application/json"))
        u = r.usage_metadata
        spend.record(spend.token_cost_usd(GEMINI, u.prompt_token_count or 0,
                                          (u.candidates_token_count or 0)
                                          + (u.thoughts_token_count or 0)),
                     f"adjudicate:{GEMINI}", query=q["query_id"])
        return {**parse(r.text), "model": GEMINI, "media_sha": sha(data),
                "clip": [round(cs, 2), round(ce, 2)]}

    async def gpt(self, q: dict) -> dict:
        from openai import AsyncOpenAI

        self._gpt = self._gpt or AsyncOpenAI()
        frames, cs, ce = self.frames(q)
        _, a, b = span_of(q)
        labels = "before the span, at its start, its middle, its end, and after it"
        text = prompt(q, f"five-frame sheet ({labels})", cs, ce, "", self.said(q))
        content = [{"type": "text", "text": text}] + [
            {"type": "image_url", "image_url": {
                "url": "data:image/jpeg;base64," + base64.b64encode(f).decode()}}
            for f in frames]
        spend.reserve(spend.token_cost_usd(GPT, 6_000, 1_500), f"adjudicate:{GPT}")
        async with self.sem:
            r = await self._gpt.chat.completions.create(
                model=GPT, messages=[{"role": "user", "content": content}],
                response_format={"type": "json_object"})
        u = r.usage
        spend.record(spend.token_cost_usd(GPT, u.prompt_tokens, u.completion_tokens),
                     f"adjudicate:{GPT}", query=q["query_id"])
        return {**parse(r.choices[0].message.content), "model": GPT,
                "media_sha": sha(b"".join(frames))}


def decide(gemini: dict, gpt: dict) -> str:
    return "scored" if gemini.get("verdict") == "accept" and gpt.get("verdict") == "accept" \
        else "provisional"


async def run(args: argparse.Namespace) -> dict:
    log = json.loads(LOG.read_text()) if LOG.exists() else {"queries": {}}
    entries = log["queries"]
    queries = [q for d in DOMAINS
               for q in json.loads((ROOT / f"data/queries/{d}_v0.2.json").read_text())]
    todo = [q for q in queries if entries.get(q["query_id"], {}).get("annotation_sha")
            != annotation_sha(q) or entries[q["query_id"]].get("config_sha") != CONFIG_SHA
            or "error" in (entries[q["query_id"]]["gemini"]["verdict"],
                           entries[q["query_id"]]["gpt"]["verdict"])]
    if args.only:
        todo = [q for q in todo if q["query_id"] in set(args.only)]
    if args.limit:
        todo = todo[: args.limit]
    work = Path(tempfile.mkdtemp(prefix="momentbench-adj-"))
    judges = Judges(Path(args.corpus_dir), ROOT / "data/transcripts", work, args.concurrency)

    async def one(q: dict) -> None:
        results = await asyncio.gather(judges.gemini(q), judges.gpt(q), return_exceptions=True)
        if any(isinstance(r, spend.SpendCapError) for r in results):
            raise next(r for r in results if isinstance(r, spend.SpendCapError))
        gem, gpt = ({"verdict": "error", "reason": f"{type(r).__name__}: {r}"[:300]}
                    if isinstance(r, Exception) else r for r in results)
        entries[q["query_id"]] = {
            "annotation_sha": annotation_sha(q), "config_sha": CONFIG_SHA,
            "tier": int(q["tier"]), "gemini": gem, "gpt": gpt, "decision": decide(gem, gpt)}

    try:
        await asyncio.gather(*(one(q) for q in todo))
    finally:
        live = {q["query_id"] for q in queries}
        log["queries"] = {k: v for k, v in sorted(entries.items()) if k in live}
        counts: dict[str, int] = {}
        for v in log["queries"].values():
            counts[v["decision"]] = counts.get(v["decision"], 0) + 1
        log["summary"] = {"judges": [GEMINI, GPT], "config_sha": CONFIG_SHA,
                          "queries": len(log["queries"]), **counts}
        LOG.parent.mkdir(parents=True, exist_ok=True)
        LOG.write_text(json.dumps(log, indent=1, ensure_ascii=False) + "\n")
    return log["summary"]


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--corpus-dir", required=True)
    p.add_argument("--limit", type=int, default=0)
    p.add_argument("--only", nargs="*", help="query ids to (re)judge")
    p.add_argument("--concurrency", type=int, default=6)
    args = p.parse_args()
    print(json.dumps(asyncio.run(run(args)), indent=1))


if __name__ == "__main__":
    main()
