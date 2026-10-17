"""The corpus as a set of tools: list assets, search and read transcripts, sample frames.

Deliberately has no embedding index. Retrieval quality is what the benchmark
measures, so the environment only offers primitives every entrant can get
without a vendor: a lexical transcript search and raw frames.
"""

from __future__ import annotations

import functools
import hashlib
import json
import math
import re
import subprocess
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any

from momentbench.agentic.budget import Budget, BudgetExceededError, Usage
from momentbench.agentic.model import ToolCall, ToolResult
from momentbench.corpus.verify import VIDEO_SUFFIXES, resolve_video  # noqa: F401
from momentbench.types import Moment

MAX_FRAMES_PER_CALL = 8
MAX_TRANSCRIPT_WINDOW_S = 600.0
FRAME_WIDTH_PX = 512

FrameExtractor = Callable[[Path, float], bytes]

TOOLS: list[dict[str, Any]] = [
    {
        "name": "list_assets",
        "description": "List every video in the corpus with its asset_id, title and duration.",
        "input_schema": {"type": "object", "properties": {}, "additionalProperties": False},
    },
    {
        "name": "search_transcript",
        "description": (
            "Keyword search over spoken-word transcripts. Returns the best matching "
            "transcript segments with asset_id and timestamps in seconds. Videos "
            "without speech have no transcript and never appear here."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "query": {"type": "string"},
                "asset_id": {"type": "string", "description": "Restrict to one video."},
                "top_k": {"type": "integer", "minimum": 1, "maximum": 20},
            },
            "required": ["query"],
            "additionalProperties": False,
        },
    },
    {
        "name": "read_transcript",
        "description": (
            f"Read a video's transcript between two timestamps (at most "
            f"{int(MAX_TRANSCRIPT_WINDOW_S)} seconds per call)."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "asset_id": {"type": "string"},
                "t_start": {"type": "number", "minimum": 0},
                "t_end": {"type": "number", "minimum": 0},
            },
            "required": ["asset_id", "t_start", "t_end"],
            "additionalProperties": False,
        },
    },
    {
        "name": "sample_frames",
        "description": (
            f"Return up to {MAX_FRAMES_PER_CALL} evenly spaced frames from a video "
            "between two timestamps. Each frame counts against the frame budget."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "asset_id": {"type": "string"},
                "t_start": {"type": "number", "minimum": 0},
                "t_end": {"type": "number", "minimum": 0},
                "num_frames": {"type": "integer", "minimum": 1, "maximum": MAX_FRAMES_PER_CALL},
            },
            "required": ["asset_id", "t_start", "t_end", "num_frames"],
            "additionalProperties": False,
        },
    },
    {
        "name": "submit_answer",
        "description": (
            "Submit the final answer and end the task. Give up to 5 moments, most "
            "likely first. Each moment is one video and a start and end time in seconds."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "moments": {
                    "type": "array",
                    "maxItems": 5,
                    "items": {
                        "type": "object",
                        "properties": {
                            "asset_id": {"type": "string"},
                            "t_in": {"type": "number", "minimum": 0},
                            "t_out": {"type": "number", "minimum": 0},
                        },
                        "required": ["asset_id", "t_in", "t_out"],
                        "additionalProperties": False,
                    },
                }
            },
            "required": ["moments"],
            "additionalProperties": False,
        },
    },
]


# Seeking to the last instant of a stream decodes nothing; sample this far inside the end.
END_MARGIN_S = 0.25
FFMPEG_TIMEOUT_S = 60


@functools.lru_cache(maxsize=256)
def video_stream_end(video: Path) -> float | None:
    """End time of the first video stream, which can fall seconds before the container's.

    Elephants Dream's picture ends at 653.8 s while its audio runs to 658.3 s; a seek
    into that gap returns no frame at all.
    """
    try:
        out = subprocess.run(
            ["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries",
             "stream=duration:format=duration", "-of", "json", str(video)],
            capture_output=True, text=True, check=False, timeout=FFMPEG_TIMEOUT_S).stdout
        info = json.loads(out or "{}")
    except (OSError, subprocess.SubprocessError, ValueError):
        return None
    for value in ([s.get("duration") for s in info.get("streams", [])]
                  + [info.get("format", {}).get("duration")]):
        try:
            return float(value)
        except (TypeError, ValueError):
            continue
    return None


def ffmpeg_frame(video: Path, t: float, width: int = FRAME_WIDTH_PX,
                 jpeg_quality: int | None = None) -> bytes:
    """Extract one JPEG frame at time t, scaled to `width` px wide.

    `jpeg_quality` is ffmpeg's `-q:v` (2 is near lossless); None keeps ffmpeg's default,
    which is what every run before the resolution ablation used.

    t is clamped to just inside the end of the video stream, so a request at the
    asset's final second returns its last frame rather than an error.
    """
    end = video_stream_end(video)
    if end is not None:
        t = min(t, max(0.0, end - END_MARGIN_S))
    proc = subprocess.run(
        [
            "ffmpeg", "-v", "error", "-ss", f"{t:.3f}", "-i", str(video),
            "-frames:v", "1", "-vf", f"scale={width}:-2",
            *(["-q:v", str(jpeg_quality)] if jpeg_quality else []),
            "-f", "image2pipe", "-vcodec", "mjpeg", "-",
        ],
        capture_output=True,
        check=False,
        timeout=FFMPEG_TIMEOUT_S,
    )
    if proc.returncode != 0 or not proc.stdout:
        raise RuntimeError(f"ffmpeg failed at t={t}: {proc.stderr.decode(errors='replace')[:200]}")
    return proc.stdout


def _tokens(text: str) -> list[str]:
    return re.findall(r"[a-z0-9']+", text.lower())


def load_transcript(path: Path) -> list[dict[str, Any]]:
    """Load a transcript as [{t_in, t_out, text}], from .json or .vtt."""
    if path.suffix == ".json":
        return json.loads(path.read_text())
    segments = []
    # WebVTT allows the hours field to be omitted (mm:ss.ttt), and whisper writes it that way.
    stamp = r"(?:(\d+):)?(\d\d):(\d\d)[.,](\d{3})"
    cue = re.compile(stamp + r" --> " + stamp)
    lines = path.read_text().splitlines()
    i = 0
    while i < len(lines):
        m = cue.search(lines[i])
        if not m:
            i += 1
            continue
        h1, m1, s1, ms1, h2, m2, s2, ms2 = (int(g or 0) for g in m.groups())
        text = []
        i += 1
        while i < len(lines) and lines[i].strip():
            text.append(lines[i].strip())
            i += 1
        segments.append(
            {
                "t_in": h1 * 3600 + m1 * 60 + s1 + ms1 / 1000,
                "t_out": h2 * 3600 + m2 * 60 + s2 + ms2 / 1000,
                "text": " ".join(text),
            }
        )
    return segments


class LocalCorpus:
    """Corpus backed by files on disk.

    Videos resolve from the manifest's `storage_path`, else
    `<corpus_dir>/<asset_id><suffix>`. Transcripts are optional and resolve from
    `<asset_id>.transcript.json` or `<asset_id>.vtt` in `corpus_dir`, then in
    `transcripts_dir` (the repo ships one, data/transcripts).
    """

    def __init__(
        self,
        assets: list[dict[str, Any]],
        corpus_dir: Path | None = None,
        frame_extractor: FrameExtractor = ffmpeg_frame,
        transcripts: dict[str, list[dict[str, Any]]] | None = None,
        transcripts_dir: Path | None = None,
        anonymize_titles: bool = False,
        frame_width: int = FRAME_WIDTH_PX,
        jpeg_quality: int | None = None,
    ):
        self.assets = {a["asset_id"]: a for a in assets}
        # Frame size and JPEG quality are task conditions: a non-default value changes
        # the compatibility key (the resolution ablation).
        self.frame_width, self.jpeg_quality = frame_width, jpeg_quality
        # Titles and descriptive asset ids ("nasa-artemis-ii-launch-recap") name what a
        # video shows. Anonymized, the model sees "video-07" and no title; answers are
        # mapped back to the real ids before scoring. The order is a hash, not the name.
        self.anonymize_titles = anonymize_titles
        self._pinned: dict[str, tuple[Path, int, int]] | None = None
        # asset_id -> sha256 of the verified file, set by pin_files.
        self.receipt: dict[str, str] = {}
        order = sorted(self.assets, key=lambda a: hashlib.sha256(a.encode()).hexdigest())
        self._alias = ({real: f"video-{i:02d}" for i, real in enumerate(order, 1)}
                       if anonymize_titles else {})
        self._real = {shown: real for real, shown in self._alias.items()}
        self.corpus_dir = corpus_dir
        self._ffmpeg = frame_extractor is ffmpeg_frame
        if self._ffmpeg and (frame_width, jpeg_quality) != (FRAME_WIDTH_PX, None):
            frame_extractor = functools.partial(ffmpeg_frame, width=frame_width,
                                                jpeg_quality=jpeg_quality)
        self._extract = frame_extractor
        self._transcripts = dict(transcripts or {})
        search = [d for d in (corpus_dir, transcripts_dir) if d]
        if transcripts is None:
            for asset_id in self.assets:
                found = [d / f"{asset_id}{ext}" for d in search
                         for ext in (".transcript.json", ".vtt")]
                path = next((f for f in found if f.exists()), None)
                if path:
                    self._transcripts[asset_id] = load_transcript(path)

    def video_path(self, asset_id: str) -> Path | None:
        if self._pinned is not None:
            pin = self._pinned.get(asset_id)
            if pin is None:
                return None  # not verified, so never shown
            path, size, mtime_ns = pin
            st = path.stat() if path.exists() else None
            if st is None or (st.st_size, st.st_mtime_ns) != (size, mtime_ns):
                raise RuntimeError(f"{asset_id}: {path} changed after verification")
            return path
        return resolve_video(self.assets[asset_id], self.corpus_dir)

    def pin_files(self, verified: dict[str, Path],
                  receipt: dict[str, str] | None = None) -> None:
        """Serve only these verified files from now on (asset_id -> the path hashed).

        An asset outside the map has no video, and a pinned file whose size or mtime
        changes is refused, so every frame comes from the bytes in the receipt."""
        self._pinned = {a: (p, p.stat().st_size, p.stat().st_mtime_ns)
                        for a, p in verified.items()}
        self.receipt = dict(receipt or {})

    @property
    def pinned(self) -> bool:
        return self._pinned is not None

    @property
    def fully_verified(self) -> bool:
        """Every manifest asset was hash-verified and is pinned."""
        return self.pinned and set(self.receipt) == set(self.assets)

    @property
    def transcripts(self) -> dict[str, list[dict[str, Any]]]:
        return self._transcripts

    def has_transcripts(self) -> bool:
        return bool(self._transcripts)

    def missing_videos(self) -> list[str]:
        return [a for a in self.assets if self.video_path(a) is None]

    # --- tools -----------------------------------------------------------

    def shown_id(self, asset_id: str) -> str:
        """The id the model sees for a real asset id."""
        return self._alias.get(asset_id, asset_id)

    def real_id(self, shown: str) -> str:
        """The real asset id behind an id the model used."""
        if self._alias:
            if shown not in self._real:
                raise ValueError(f"unknown asset_id {shown!r}; call list_assets")
            return self._real[shown]
        return shown

    def parse_answer(self, arguments: Mapping[str, Any]) -> list[Moment]:
        """Validate a submit_answer call against the ids the model saw; return real ids."""
        shown = {self.shown_id(a): meta for a, meta in self.assets.items()}
        return [m.model_copy(update={"asset_id": self.real_id(m.asset_id)})
                for m in parse_submission(arguments, shown)]

    def list_assets(self) -> str:
        rows = []
        assets = list(self.assets.values())
        if self.anonymize_titles:  # by alias, so manifest order cannot hint at the domain
            assets.sort(key=lambda a: self.shown_id(a["asset_id"]))
        for a in assets:
            row = {"asset_id": self.shown_id(a["asset_id"])}
            if not self.anonymize_titles:
                row["title"] = a.get("title", "")
            row |= {"duration_s": a.get("duration_s"),
                    "has_transcript": a["asset_id"] in self._transcripts}
            rows.append(row)
        return json.dumps(rows)

    def search_transcript(self, query: str, asset_id: str | None = None, top_k: int = 5) -> str:
        terms = set(_tokens(query))
        asset_id = self.real_id(asset_id) if asset_id else None
        scored = []
        for aid, segments in self._transcripts.items():
            if asset_id and aid != asset_id:
                continue
            for seg in segments:
                overlap = len(terms & set(_tokens(seg["text"])))
                if overlap:
                    scored.append((overlap, aid, seg))
        scored.sort(key=lambda s: (-s[0], s[1], s[2]["t_in"]))
        hits = [
            {"asset_id": self.shown_id(aid), "t_in": seg["t_in"], "t_out": seg["t_out"],
             "text": seg["text"]}
            for _, aid, seg in scored[: max(1, min(top_k, 20))]
        ]
        return json.dumps(hits)

    def read_transcript(self, asset_id: str, t_start: float, t_end: float) -> str:
        asset_id = self.real_id(asset_id)
        self._require_asset(asset_id)
        t_end = min(t_end, t_start + MAX_TRANSCRIPT_WINDOW_S)
        segments = [
            s for s in self._transcripts.get(asset_id, [])
            if s["t_out"] > t_start and s["t_in"] < t_end
        ]
        return json.dumps(segments)

    def sample_frames(
        self, asset_id: str, t_start: float, t_end: float, num_frames: int
    ) -> tuple[str, list[bytes]]:
        asset_id = self.real_id(asset_id)
        self._require_asset(asset_id)
        video = self.video_path(asset_id)
        if video is None:
            raise ValueError(f"no video file available for {asset_id}")
        duration = self.assets[asset_id].get("duration_s")
        if t_end < t_start:
            raise ValueError("t_end must be >= t_start")
        ends = [float(duration)] if duration else []
        if self._ffmpeg and (stream_end := video_stream_end(video)):
            ends.append(stream_end)  # so frame_times_s reports what was decoded
        if ends:
            t_end = min(t_end, max(0.0, min(ends) - END_MARGIN_S))
            t_start = min(t_start, t_end)
        n = max(1, min(num_frames, MAX_FRAMES_PER_CALL))
        if n == 1:
            times = [(t_start + t_end) / 2]
        else:
            times = [t_start + (t_end - t_start) * i / (n - 1) for i in range(n)]
        frames = [self._extract(video, t) for t in times]
        info = {"asset_id": self.shown_id(asset_id), "frame_times_s": [round(t, 2) for t in times]}
        return json.dumps(info), frames

    def _require_asset(self, asset_id: str) -> None:
        if asset_id not in self.assets:
            raise ValueError(f"unknown asset_id {asset_id!r}; call list_assets")

    # --- dispatch --------------------------------------------------------

    def execute(self, call: ToolCall, budget: Budget, usage: Usage) -> ToolResult:
        """Run one non-terminal tool call, charging the budget. Errors go back to the model."""
        try:
            if usage.tool_calls >= budget.max_tool_calls:
                raise BudgetExceededError("tool-call budget exhausted; call submit_answer now")
            usage.tool_calls += 1
            args = call.arguments
            if call.name == "list_assets":
                return ToolResult(call.call_id, self.list_assets())
            if call.name == "search_transcript":
                return ToolResult(
                    call.call_id,
                    self.search_transcript(
                        str(args["query"]), args.get("asset_id"), int(args.get("top_k", 5))
                    ),
                )
            if call.name == "read_transcript":
                return ToolResult(
                    call.call_id,
                    self.read_transcript(
                        str(args["asset_id"]), float(args["t_start"]), float(args["t_end"])
                    ),
                )
            if call.name == "sample_frames":
                want = max(1, min(int(args["num_frames"]), MAX_FRAMES_PER_CALL))
                left = budget.max_frames - usage.frames
                if left <= 0:
                    raise BudgetExceededError("frame budget exhausted; call submit_answer now")
                text, frames = self.sample_frames(
                    str(args["asset_id"]), float(args["t_start"]), float(args["t_end"]),
                    min(want, left),
                )
                usage.frames += len(frames)
                return ToolResult(call.call_id, text, images_jpeg=frames)
            raise ValueError(f"unknown tool {call.name!r}")
        except (BudgetExceededError, ValueError, KeyError, TypeError, RuntimeError) as e:
            return ToolResult(call.call_id, f"error: {e}", is_error=True)


class InvalidSubmissionError(ValueError):
    """A submit_answer call that is neither a valid answer nor an explicit abstention."""


def parse_submission(arguments: Any, known_assets: Mapping[str, Any] | set[str]
                     ) -> list[Moment]:
    """Turn a submit_answer call into Moments.

    `{"moments": []}` is an explicit abstention and returns []. Anything else must
    contain at least one valid moment (a known asset, finite times, 0 <= t_in < t_out,
    inside the video); otherwise InvalidSubmissionError is raised, so malformed output
    can never be scored as a correct "no answer". An invalid entry anywhere in the
    ranking rejects the whole submission: dropping it would move the entries below it
    up, and Recall@1 scores whatever is first. The harness loops return the error to
    the model, which may resubmit.
    """
    raw = arguments.get("moments") if isinstance(arguments, dict) else None
    if not isinstance(raw, list):
        raise InvalidSubmissionError("moments must be a list; submit [] if nothing matches")
    if not raw:
        return []
    durations = (
        {k: float((v or {}).get("duration_s") or "inf") for k, v in known_assets.items()}
        if isinstance(known_assets, Mapping) else {k: math.inf for k in known_assets}
    )
    moments, problems = [], []
    for item in raw[:5]:
        try:
            asset_id = str(item["asset_id"])
            t_in, t_out = float(item["t_in"]), float(item["t_out"])
        except (KeyError, TypeError, ValueError):
            problems.append(f"malformed entry {item!r}")
            continue
        if asset_id not in durations:
            problems.append(f"unknown asset_id {asset_id!r}")
        elif not (math.isfinite(t_in) and math.isfinite(t_out)) or not 0 <= t_in < t_out:
            problems.append(f"bad times {t_in}..{t_out} for {asset_id}")
        elif t_out > durations[asset_id] + 0.5:
            problems.append(f"{asset_id} ends at {durations[asset_id]:.1f}s, got t_out {t_out}")
        else:
            moments.append(Moment(asset_id=asset_id, t_in=t_in, t_out=t_out))
    if problems:
        raise InvalidSubmissionError("invalid submission: " + "; ".join(problems))
    return moments
