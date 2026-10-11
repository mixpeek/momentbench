"""The corpus as a set of tools: list assets, search and read transcripts, sample frames.

Deliberately has no embedding index. Retrieval quality is what the benchmark
measures, so the environment only offers primitives every entrant can get
without a vendor: a lexical transcript search and raw frames.
"""

from __future__ import annotations

import json
import re
import subprocess
from collections.abc import Callable
from pathlib import Path
from typing import Any

from momentbench.agentic.budget import Budget, BudgetExceededError, Usage
from momentbench.agentic.model import ToolCall, ToolResult
from momentbench.types import Moment

VIDEO_SUFFIXES = (".mp4", ".mkv", ".webm", ".mov")
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


def ffmpeg_frame(video: Path, t: float) -> bytes:
    """Extract one JPEG frame at time t, scaled to FRAME_WIDTH_PX wide."""
    proc = subprocess.run(
        [
            "ffmpeg", "-v", "error", "-ss", f"{t:.3f}", "-i", str(video),
            "-frames:v", "1", "-vf", f"scale={FRAME_WIDTH_PX}:-2",
            "-f", "image2pipe", "-vcodec", "mjpeg", "-",
        ],
        capture_output=True,
        check=False,
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
    ):
        self.assets = {a["asset_id"]: a for a in assets}
        self.corpus_dir = corpus_dir
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
        stored = self.assets[asset_id].get("storage_path")
        if stored and Path(stored).exists():
            return Path(stored)
        if self.corpus_dir:
            for suffix in VIDEO_SUFFIXES:
                candidate = self.corpus_dir / f"{asset_id}{suffix}"
                if candidate.exists():
                    return candidate
        return None

    def has_transcripts(self) -> bool:
        return bool(self._transcripts)

    def missing_videos(self) -> list[str]:
        return [a for a in self.assets if self.video_path(a) is None]

    # --- tools -----------------------------------------------------------

    def list_assets(self) -> str:
        rows = [
            {
                "asset_id": a["asset_id"],
                "title": a.get("title", ""),
                "duration_s": a.get("duration_s"),
                "has_transcript": a["asset_id"] in self._transcripts,
            }
            for a in self.assets.values()
        ]
        return json.dumps(rows)

    def search_transcript(self, query: str, asset_id: str | None = None, top_k: int = 5) -> str:
        terms = set(_tokens(query))
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
            {"asset_id": aid, "t_in": seg["t_in"], "t_out": seg["t_out"], "text": seg["text"]}
            for _, aid, seg in scored[: max(1, min(top_k, 20))]
        ]
        return json.dumps(hits)

    def read_transcript(self, asset_id: str, t_start: float, t_end: float) -> str:
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
        self._require_asset(asset_id)
        video = self.video_path(asset_id)
        if video is None:
            raise ValueError(f"no video file available for {asset_id}")
        duration = self.assets[asset_id].get("duration_s")
        if duration:
            t_end = min(t_end, float(duration))
        if t_end < t_start:
            raise ValueError("t_end must be >= t_start")
        n = max(1, min(num_frames, MAX_FRAMES_PER_CALL))
        if n == 1:
            times = [(t_start + t_end) / 2]
        else:
            times = [t_start + (t_end - t_start) * i / (n - 1) for i in range(n)]
        frames = [self._extract(video, t) for t in times]
        info = {"asset_id": asset_id, "frame_times_s": [round(t, 2) for t in times]}
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


def parse_submission(arguments: dict[str, Any], known_assets: set[str]) -> list[Moment]:
    """Turn a submit_answer call into Moments, dropping malformed entries."""
    moments = []
    for item in (arguments.get("moments") or [])[:5]:
        try:
            asset_id = str(item["asset_id"])
            t_in, t_out = float(item["t_in"]), float(item["t_out"])
        except (KeyError, TypeError, ValueError):
            continue
        if asset_id in known_assets and 0 <= t_in < t_out:
            moments.append(Moment(asset_id=asset_id, t_in=t_in, t_out=t_out))
    return moments
