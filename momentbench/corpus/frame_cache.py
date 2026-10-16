"""Cached review frames that know which video bytes and timestamp they came from.

The annotation tools (draft, cross-check, refine, pre-review, review pages) cache JPEGs
under names like `c_<query>_gold_1.jpg`. A name says nothing about the span, so after
an edit a recheck could reuse a frame from the old span (audit 4, finding 3). Each
cached frame now carries a sidecar `<frame>.src.json` with the video's sha256, the
timestamp and the extraction settings, and is reused only when all three match.
"""

from __future__ import annotations

import functools
import json
import subprocess
from pathlib import Path
from typing import Any

from momentbench.corpus.verify import sha256_file


@functools.lru_cache(maxsize=256)
def _video_sha(path: str, size: int, mtime_ns: int) -> str:
    return sha256_file(Path(path))


def video_sha(video: Path) -> str:
    """sha256 of the video's bytes, hashed once per (path, size, mtime)."""
    st = video.stat()
    return _video_sha(str(video.resolve()), st.st_size, st.st_mtime_ns)


def source(video: Path, t: float, **settings: Any) -> dict[str, Any]:
    return {"video_sha256": video_sha(video), "t": round(float(t), 3), **settings}


def _sidecar(out: Path) -> Path:
    return out.with_name(out.name + ".src.json")


def is_fresh(out: Path, src: dict[str, Any]) -> bool:
    side = _sidecar(out)
    if not (out.exists() and side.exists()):
        return False
    try:
        return json.loads(side.read_text()) == src
    except ValueError:
        return False


def extract(video: Path, t: float, out: Path, width: int,
            step_back: tuple[float, ...] = (0.0,)) -> Path | None:
    """Write the frame at `t` to `out`, reusing it only if it came from these bytes at
    this time and width. Returns None when ffmpeg produced nothing."""
    src = source(video, t, width=width)
    if is_fresh(out, src):
        return out
    out.unlink(missing_ok=True)
    _sidecar(out).unlink(missing_ok=True)
    out.parent.mkdir(parents=True, exist_ok=True)
    # A seek in the last fraction of a second can land past the final frame, so step back.
    for back in step_back:
        subprocess.run(
            ["ffmpeg", "-v", "error", "-y", "-ss", f"{max(0.0, t - back):.2f}", "-i", str(video),
             "-frames:v", "1", "-vf", f"scale={width}:-2", str(out)],
            check=False, capture_output=True,
        )
        if out.exists():
            _sidecar(out).write_text(json.dumps(src))
            return out
    return None


def frame_inputs(paths: list[Path]) -> list[dict[str, Any]]:
    """The recorded source of each frame, for a review's media record."""
    out = []
    for p in paths:
        side = _sidecar(p)
        out.append(json.loads(side.read_text()) if side.exists() else {"missing": p.name})
    return out
