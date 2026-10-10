"""Draft and verify film queries with Claude, for later human review.

    python scripts/draft_queries.py --corpus-dir CORPUS/videos \\
        --transcripts CORPUS/transcripts --work WORKDIR --out data/queries/film_draft.json

Pipeline, all cached under --work so a rerun resumes:

1. Shots: ffmpeg scene cuts per film. A gold span is always one shot, or a run of
   consecutive shots, so its boundaries are cuts rather than guesses.
2. Draft: Claude sees a labelled sheet of ~16 shots plus the dialogue in that
   window and writes queries of every tier, each pointing at shot ids.
3. Verify: a separate Claude call sees the gold candidate shuffled among the
   three most visually similar shots elsewhere in the corpus (by thumbnail
   distance), and must pick the match or "none". A draft survives only if the
   verifier picks the gold (or "none" for a no-answer query) and reports no wrong
   detail. The three look-alikes are stored as hard negatives.

Claude runs through the local Claude Code login (claude-agent-sdk), so there is
no API spend. Every surviving query is marked pending human review. Because the
drafter and verifier are both Claude, the set may favour Claude; human review and
cross-model checks are the remedy, and the metadata says so.
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import random
import re
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from momentbench.agentic.environment import load_transcript  # noqa: E402
from momentbench.canary import CANARY  # noqa: E402

MODEL = "claude-opus-5-5"
SCENE_THRESHOLD = 0.3
SHOTS_PER_WINDOW = 16
MIN_GOLD_S = 1.5
MAX_GOLD_S = 20.0
TILE_W = 320
HELD_OUT_FRACTION = 0.25

DRAFT_SYSTEM = (
    "You write evaluation queries for a video moment retrieval benchmark. A system "
    "must find the exact moment from your text alone, among many videos. Write what a "
    "viewer can see or hear, specific enough that only the cited shots match. Never "
    "use character or people's names unless the name is spoken in the dialogue shown, "
    "never cite timestamps, credits or the video's title, and never describe anything you "
    "cannot see in the frames. Reply with JSON only."
)

DRAFT_PROMPT = """Read the image {sheet}. It shows shots {first} to {last} of one video.
Each tile is labelled with its shot id and time range; shots longer than 4 s have two
tiles (early and late).

Shots:
{shots}

Dialogue in this window (may be empty or imperfect):
{dialogue}

Write up to 6 queries as a JSON list. Each item:
{{"tier": 1|2|3|4, "text": "...", "shots": [ids], "altered": "...", "original": "..."}}

- tier 1 (2 or 3 of these): one shot, described visually. Pick shots whose content is distinctive.
- tier 2 (one, only if the dialogue clearly belongs to a shot you can see): needs both what is said
  and what is seen, e.g. "while the old man says he will not go back, the camera ...". Quote or
  paraphrase the line.
- tier 4 (one): two or three consecutive shots as an ordered event, "after X, Y". shots lists
  them in order.
- tier 3 (one): a no-answer query. Take one shot, change one concrete detail so the text no
  longer matches it (a different object, colour, action or count), and put the shot id in shots
  and what you changed in "altered". Put the accurate, unchanged description of that shot in
  "original"; it must match the shot exactly. The changed text must still sound plausible.

Skip shots shorter than 1.5 s for tiers 1 and 2. Each query is one or two sentences."""

DIALOGUE_PROMPT = """Read the image {sheet}. It shows shots {first} to {last} of one video.
Each tile is labelled with its shot id and time range; shots longer than 4 s have two
tiles (early and late).

Shots:
{shots}

Dialogue in this window, with times (machine transcribed, may be imperfect):
{dialogue}

Write up to 3 tier 2 queries as a JSON list: {{"tier": 2, "text": "...", "shots": [id]}}.
A tier 2 query needs both what is said and what is seen: a system must use the dialogue
and the picture together to find it, e.g. "while a woman says she will not go back, the
camera holds on her hands gripping a rail". Use only lines whose times fall inside the
cited shot, quote or closely paraphrase them, and describe something visible in that shot.
One shot per query, at least 1.5 s long. Return [] if no line fits a shot."""

VERIFY_SYSTEM = (
    "You check video retrieval queries against candidate clips. Be strict: a candidate "
    "matches only if every detail in the query is visible or audible in it. Reply with "
    "JSON only."
)

VERIFY_PROMPT = """Read the image {sheet}. Each row is one candidate clip, labelled A to
{last_label}, shown as frames from its start, middle and end.
{dialogue}
Query: "{text}"

Which candidate does the query describe? Answer "none" if no candidate matches every detail.
Reply: {{"answer": "A".."{last_label}" or "none", "wrong_details": ["..."], "reason": "..."}}
wrong_details lists any detail of the query that is false for your chosen candidate
(empty if none)."""


# ---------------------------------------------------------------- media helpers


def detect_shots(video: Path, duration: float) -> list[tuple[float, float]]:
    out = subprocess.run(
        ["ffprobe", "-v", "error", "-f", "lavfi", "-i",
         f"movie={video},select=gt(scene\\,{SCENE_THRESHOLD})",
         "-show_entries", "frame=pts_time", "-of", "csv=p=0"],
        capture_output=True, text=True, check=True,
    ).stdout
    cuts = sorted(float(x.strip(",")) for x in out.split() if x.strip(","))
    bounds = [0.0, *[c for c in cuts if 0.2 < c < duration - 0.2], duration]
    return [(round(a, 2), round(b, 2)) for a, b in zip(bounds, bounds[1:]) if b - a > 0.2]


def frame(video: Path, t: float, out: Path) -> Path:
    # A seek in the last fraction of a second can land past the final frame, so step back.
    for back in (0.0, 0.5, 1.0, 2.0):
        if out.exists():
            break
        subprocess.run(
            ["ffmpeg", "-v", "error", "-y", "-ss", f"{max(0.0, t - back):.2f}", "-i", str(video),
             "-frames:v", "1", "-vf", f"scale={TILE_W}:-2", str(out)],
            check=False, capture_output=True,
        )
    if not out.exists():
        raise RuntimeError(f"no frame at {t:.2f}s in {video.name}")
    return out


def label_tile(path: Path, label: str) -> Image.Image:
    img = Image.open(path).convert("RGB")
    draw = ImageDraw.Draw(img)
    font = ImageFont.load_default(size=16)
    box = draw.textbbox((0, 0), label, font=font)
    draw.rectangle((0, 0, box[2] + 8, box[3] + 6), fill=(0, 0, 0))
    draw.text((4, 2), label, fill=(255, 255, 0), font=font)
    return img


def grid(tiles: list[Image.Image], cols: int, out: Path) -> Path:
    w = max(t.width for t in tiles)
    h = max(t.height for t in tiles)
    rows = (len(tiles) + cols - 1) // cols
    sheet = Image.new("RGB", (cols * w, rows * h))
    for i, t in enumerate(tiles):
        sheet.paste(t, ((i % cols) * w, (i // cols) * h))
    sheet.save(out, quality=85)
    return out


def signature(path: Path) -> list[float]:
    img = Image.open(path).convert("RGB").resize((16, 9))
    return [c / 255 for px in img.getdata() for c in px]


# ---------------------------------------------------------------- claude


async def ask(system: str, prompt: str, cwd: Path, sem: asyncio.Semaphore) -> str:
    from claude_agent_sdk import ClaudeAgentOptions, ResultMessage, query

    options = ClaudeAgentOptions(
        model=MODEL, system_prompt=system, tools=["Read"], allowed_tools=["Read"],
        permission_mode="dontAsk", setting_sources=[], strict_mcp_config=True,
        max_turns=6, cwd=str(cwd),
    )
    async with sem:
        result = ""
        async for message in query(prompt=prompt, options=options):
            if isinstance(message, ResultMessage):
                if message.is_error:
                    raise RuntimeError(f"claude: {message.subtype} {message.errors}")
                result = message.result or ""
        return result


def parse_json(text: str):
    fenced = re.search(r"```(?:json)?\s*(.*?)```", text, re.S)
    body = fenced.group(1) if fenced else text
    start = min((i for i in (body.find("["), body.find("{")) if i >= 0), default=-1)
    if start < 0:
        raise ValueError(f"no JSON in reply: {text[:200]}")
    return json.JSONDecoder().raw_decode(body[start:])[0]


# ---------------------------------------------------------------- pipeline


class Film:
    def __init__(self, asset: dict, video: Path, transcript: Path | None, work: Path):
        self.id = asset["asset_id"]
        self.vertical = asset.get("vertical", "film")
        # Footage published after current model training cutoffs goes to the rolling split.
        self.rolling = bool(asset.get("metadata", {}).get("rolling"))
        self.duration = float(asset["duration_s"])
        self.video = video
        self.lines = load_transcript(transcript) if transcript and transcript.exists() else []
        self.dir = work / self.id
        self.dir.mkdir(parents=True, exist_ok=True)
        # The picture can end before the audio (Elephants Dream: 653.79 s vs 658.32 s).
        probe = subprocess.run(
            ["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries",
             "stream=duration", "-of", "csv=p=0", str(video)],
            capture_output=True, text=True,
        ).stdout.strip()
        self.video_end = float(probe) if probe.replace(".", "", 1).isdigit() else self.duration
        cache = self.dir / "shots.json"
        if cache.exists():
            self.shots = [tuple(s) for s in json.loads(cache.read_text())]
        else:
            self.shots = detect_shots(video, self.duration)
            cache.write_text(json.dumps(self.shots))

    def at(self, t: float) -> float:
        return min(t, self.video_end - 0.2)

    def mid_frame(self, i: int) -> Path:
        a, b = self.shots[i]
        return frame(self.video, self.at((a + b) / 2), self.dir / f"s{i:04d}_mid.jpg")

    def tiles(self, i: int) -> list[tuple[Path, str]]:
        a, b = self.shots[i]
        label = f"S{i} {a:.1f}-{b:.1f}s"
        if b - a <= 4:
            return [(self.mid_frame(i), label)]
        return [(frame(self.video, self.at(a + (b - a) * f), self.dir / f"s{i:04d}_{k}.jpg"),
                 f"{label} {k}")
                for f, k in ((0.25, "early"), (0.75, "late"))]

    def strip(self, span: tuple[float, float], key: str) -> list[Path]:
        a, b = span
        pad = min(0.3, (b - a) / 4)
        return [frame(self.video, self.at(t), self.dir / f"c_{key}_{n}.jpg")
                for n, t in enumerate((a + pad, (a + b) / 2, b - pad))]

    def dialogue(self, a: float, b: float) -> str:
        said = [f"[{s['t_in']:.1f}-{s['t_out']:.1f}s] {s['text']}"
                for s in self.lines if s["t_out"] > a and s["t_in"] < b]
        return "\n".join(said) or "(none)"


async def draft_window(film: Film, first: int, last: int, sem, dialogue_only=False) -> list[dict]:
    cache = film.dir / f"draft{'2' if dialogue_only else ''}_{first:04d}.json"
    if cache.exists():
        return json.loads(cache.read_text())
    tiles = [label_tile(p, label) for i in range(first, last + 1) for p, label in film.tiles(i)]
    sheet = grid(tiles, 4, film.dir / f"sheet_{first:04d}.jpg")
    shots = "\n".join(f"S{i}: {film.shots[i][0]:.1f}-{film.shots[i][1]:.1f}s "
                      f"({film.shots[i][1] - film.shots[i][0]:.1f}s)"
                      for i in range(first, last + 1))
    a, b = film.shots[first][0], film.shots[last][1]
    said = film.dialogue(a, b)
    if dialogue_only and said == "(none)":
        cache.write_text("[]")
        return []
    template = DIALOGUE_PROMPT if dialogue_only else DRAFT_PROMPT
    prompt = template.format(sheet=sheet, first=f"S{first}", last=f"S{last}",
                             shots=shots, dialogue=said)
    try:
        drafts = parse_json(await ask(DRAFT_SYSTEM, prompt, film.dir, sem))
    except Exception as e:  # one bad window must not stop the run
        print(f"  draft {film.id} S{first}: {e}", file=sys.stderr)
        return []
    keep = []
    for d in drafts if isinstance(drafts, list) else []:
        ids = [int(str(s).lstrip("S")) for s in d.get("shots", []) if str(s).lstrip("S").isdigit()]
        if not ids or any(i < first or i > last for i in ids) or not d.get("text"):
            continue
        keep.append({"asset_id": film.id, "tier": int(d.get("tier", 1)), "text": d["text"].strip(),
                     "shots": sorted(ids), "altered": d.get("altered"),
                     "original": d.get("original")})
    cache.write_text(json.dumps(keep, indent=1))
    return keep


def span_of(film: Film, ids: list[int]) -> tuple[float, float]:
    return film.shots[ids[0]][0], film.shots[ids[-1]][1]


def well_formed(film: Film, d: dict) -> bool:
    ids = d["shots"]
    if ids != list(range(ids[0], ids[0] + len(ids))):
        return False
    a, b = span_of(film, ids)
    if d["tier"] in (1, 2) and (len(ids) != 1 or not MIN_GOLD_S <= b - a <= MAX_GOLD_S):
        return False
    if d["tier"] == 4 and not (2 <= len(ids) <= 3 and b - a <= 2 * MAX_GOLD_S):
        return False
    return not (d["tier"] == 3 and (len(ids) != 1 or not d.get("altered")
                                    or not d.get("original")))


def lookalikes(films: dict[str, Film], sigs: dict, d: dict, k: int = 3) -> list[tuple[str, int]]:
    """The k shots elsewhere in the corpus that look most like the draft's first shot."""
    film = films[d["asset_id"]]
    target = sigs[(film.id, d["shots"][0])]
    near = set(range(d["shots"][0] - 1, d["shots"][-1] + 2))
    scored = []
    for (fid, i), sig in sigs.items():
        if fid == film.id and i in near:
            continue
        a, b = films[fid].shots[i]
        if not MIN_GOLD_S <= b - a <= 2 * MAX_GOLD_S or a >= films[fid].video_end:
            continue
        scored.append((sum((x - y) ** 2 for x, y in zip(target, sig)), fid, i))
    scored.sort()
    return [(fid, i) for _, fid, i in scored[:k]]


async def check(films, sigs, d: dict, text: str, sem) -> dict:
    """Show `text` against the draft's span and its 3 look-alikes; report what was picked."""
    film = films[d["asset_id"]]
    key = hashlib.sha1(json.dumps([d, text], sort_keys=True).encode()).hexdigest()[:12]
    cache = film.dir / f"verify_{key}.json"
    if cache.exists():
        return json.loads(cache.read_text())
    gold = span_of(film, d["shots"])
    others = lookalikes(films, sigs, d)
    candidates = [(film.id, gold, "gold")] + [
        (fid, films[fid].shots[i], f"neg{j}") for j, (fid, i) in enumerate(others)
    ]
    random.Random(key).shuffle(candidates)
    labels = "ABCD"[: len(candidates)]
    tiles, dialogue = [], []
    for label, (fid, span, tag) in zip(labels, candidates):
        for p in films[fid].strip(span, f"{key}_{tag}"):
            tiles.append(label_tile(p, label))
        if d["tier"] == 2:
            dialogue.append(f"{label} dialogue: {films[fid].dialogue(*span)}")
    sheet = grid(tiles, 3, film.dir / f"verify_{key}.jpg")
    said = "\n" + "\n".join(dialogue) + "\n" if dialogue else ""
    prompt = VERIFY_PROMPT.format(sheet=sheet, last_label=labels[-1], text=text, dialogue=said)
    try:
        reply = parse_json(await ask(VERIFY_SYSTEM, prompt, film.dir, sem))
    except Exception as e:
        print(f"  verify {key}: {e}", file=sys.stderr)
        return {"picked": "error", "error": str(e)}
    answer = str(reply.get("answer", "")).strip().strip('"').upper()
    out = {"picked": dict(zip(labels, (c[2] for c in candidates))).get(answer, "none"),
           "wrong": bool(reply.get("wrong_details")), "reply": reply,
           "negatives": [{"asset_id": fid, "t_in": films[fid].shots[i][0],
                          "t_out": films[fid].shots[i][1]} for fid, i in others]}
    cache.write_text(json.dumps(out, indent=1))
    return out


async def verify(films, sigs, d: dict, sem) -> dict:
    """A positive must pick out its gold with no wrong detail. A no-answer query must
    match nothing, while its unaltered original picks out the source shot cleanly, so
    the query is a near miss rather than a description that was wrong to begin with."""
    if d["tier"] != 3:
        r = await check(films, sigs, d, d["text"], sem)
        return {**r, "ok": r["picked"] == "gold" and not r.get("wrong")}
    original = await check(films, sigs, d, d.get("original") or "", sem)
    altered = await check(films, sigs, d, d["text"], sem)
    ok = original["picked"] == "gold" and not original.get("wrong") and altered["picked"] == "none"
    return {**altered, "ok": ok, "original_check": original.get("reply")}


def to_query(film: Film, d: dict, v: dict, n: int) -> dict:
    a, b = span_of(film, d["shots"])
    qid = f"q-{film.vertical}-{n:04d}"
    held = int(hashlib.sha256(qid.encode()).hexdigest(), 16) % 100 < HELD_OUT_FRACTION * 100
    meta = {
        "source": f"model-annotated: drafted and independently verified by {MODEL} via Claude "
                  "Code; pending human review",
        "canary": CANARY,
        "verification": "gold span is a scene-cut-bounded shot (or consecutive shots); the "
                        "verifier picked it out of it plus 3 corpus look-alikes with no wrong "
                        "detail. Not checked against every shot in the corpus.",
        "bias_note": "drafter and verifier are the same model family as one entrant",
    }
    if d["tier"] == 3:
        meta["altered_from"] = {"asset_id": film.id, "t_in": a, "t_out": b,
                                "change": d.get("altered")}
        meta["verification"] = ("no-answer: the verifier rejected the altered source shot and "
                                "its 3 look-alikes. Absence from the whole corpus is not proven.")
    return {
        "query_id": qid,
        "text": d["text"],
        "tier": d["tier"],
        "vertical": film.vertical,
        "gold_moments": [] if d["tier"] == 3 else [{"asset_id": film.id, "t_in": a, "t_out": b}],
        "hard_negatives": v.get("negatives", []),
        "is_public": not held,
        "split": "held_out" if held else ("rolling" if film.rolling else "public"),
        "metadata": meta,
    }


async def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--manifest", default=str(ROOT / "data/corpus_manifests/sample_manifest.json"))
    p.add_argument("--corpus-dir", required=True)
    p.add_argument("--transcripts", default=None, help="directory of <asset_id>.vtt")
    p.add_argument("--work", required=True)
    p.add_argument("--out", required=True, help="public queries JSON")
    p.add_argument("--held-out", required=True, help="held-out queries JSON, keep out of git")
    p.add_argument("--assets", nargs="*", help="limit to these asset ids")
    p.add_argument("--max-windows", type=int, default=None, help="per film, for a trial run")
    p.add_argument("--concurrency", type=int, default=4)
    p.add_argument("--vertical", default="film", help='draft for this vertical, or "all"')
    p.add_argument("--dialogue-only", action="store_true",
                   help="draft only tier 2 (dialogue plus picture) queries")
    p.add_argument("--offset", type=int, default=0,
                   help="start windows this many shots in, for a second pass over new groupings")
    p.add_argument("--first-id", type=int, default=100)
    p.add_argument("--existing", nargs="*", default=[str(ROOT / "data/queries/sample.json")],
                   help="query files whose gold spans new drafts must not duplicate")
    args = p.parse_args()

    manifest = json.loads(Path(args.manifest).read_text())
    assets = manifest if isinstance(manifest, list) else manifest["assets"]
    corpus, work = Path(args.corpus_dir), Path(args.work)
    tdir = Path(args.transcripts) if args.transcripts else corpus
    films: dict[str, Film] = {}
    for asset in assets:
        wanted = not args.assets or asset["asset_id"] in args.assets
        if args.vertical not in ("all", asset.get("vertical")) or not wanted:
            continue
        video = next(corpus.glob(f"{asset['asset_id']}.*"), None)
        if video is None:
            continue
        films[asset["asset_id"]] = Film(asset, video, tdir / f"{asset['asset_id']}.vtt", work)
        print(f"{asset['asset_id']}: {len(films[asset['asset_id']].shots)} shots")

    with ThreadPoolExecutor(8) as pool:
        keys = [(f.id, i) for f in films.values() for i in range(len(f.shots))]
        mids = list(pool.map(lambda k: films[k[0]].mid_frame(k[1]), keys))
    sigs = {k: signature(m) for k, m in zip(keys, mids)}

    sem = asyncio.Semaphore(args.concurrency)
    jobs = []
    for film in films.values():
        windows = [(s, min(s + SHOTS_PER_WINDOW, len(film.shots)) - 1)
                   for s in range(args.offset, len(film.shots), SHOTS_PER_WINDOW)]
        jobs += [draft_window(film, a, b, sem, args.dialogue_only)
                 for a, b in windows[: args.max_windows]]
    drafts = [d for batch in await asyncio.gather(*jobs) for d in batch]
    taken = [(g["asset_id"], g["t_in"], g["t_out"]) for f in args.existing
             for q in json.loads(Path(f).read_text()) for g in q.get("gold_moments", [])]

    def duplicate(d: dict) -> bool:
        a, b = span_of(films[d["asset_id"]], d["shots"])
        return any(fid == d["asset_id"] and min(b, t1) - max(a, t0) > 0.5 * min(b - a, t1 - t0)
                   for fid, t0, t1 in taken)

    formed = [d for d in drafts if well_formed(films[d["asset_id"]], d) and not duplicate(d)]
    print(f"drafted {len(drafts)}, well-formed {len(formed)}")

    checks = await asyncio.gather(*(verify(films, sigs, d, sem) for d in formed))
    public, held = [], []
    n = args.first_id
    for d, v in zip(formed, checks):
        if not v.get("ok"):
            continue
        q = to_query(films[d["asset_id"]], d, v, n)
        n += 1
        (held if q["split"] == "held_out" else public).append(q)
    tiers = {t: sum(1 for q in public + held if q["tier"] == t) for t in (1, 2, 3, 4)}
    print(f"verified {len(public) + len(held)} of {len(formed)} "
          f"(public {len(public)}, held out {len(held)}), by tier {tiers}")
    Path(args.out).write_text(json.dumps(public, indent=2, ensure_ascii=False) + "\n")
    Path(args.held_out).write_text(json.dumps(held, indent=2, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    asyncio.run(main())
