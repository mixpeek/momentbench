"""Check corpus files on disk against the manifest: presence, sha256 and duration."""

from __future__ import annotations

import hashlib
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any

VIDEO_SUFFIXES = (".mp4", ".mkv", ".webm", ".mov")
DURATION_TOLERANCE_S = 1.0


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def probe_duration(path: Path) -> float:
    out = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration",
         "-of", "default=noprint_wrappers=1:nokey=1", str(path)],
        capture_output=True, text=True, check=True,
    )
    return float(out.stdout.strip())


@dataclass
class AssetCheck:
    asset_id: str
    path: Path | None
    sha256: str | None = None
    duration_s: float | None = None
    problems: tuple[str, ...] = ()

    @property
    def ok(self) -> bool:
        return self.path is not None and not self.problems


def find_video(asset_id: str, corpus_dir: Path) -> Path | None:
    for suffix in VIDEO_SUFFIXES:
        candidate = corpus_dir / f"{asset_id}{suffix}"
        if candidate.exists():
            return candidate
    return None


def check_asset(
    asset: dict[str, Any], corpus_dir: Path, hasher=sha256_file, prober=probe_duration
) -> AssetCheck:
    asset_id = asset["asset_id"]
    path = find_video(asset_id, corpus_dir)
    if path is None:
        return AssetCheck(asset_id, None, problems=("no file on disk",))
    digest, duration = hasher(path), prober(path)
    problems = []
    expected = (asset.get("file") or {}).get("sha256")
    if expected is None:
        problems.append("manifest has no sha256 for this asset")
    elif expected != digest:
        problems.append(f"sha256 mismatch: manifest {expected[:12]}, file {digest[:12]}")
    if abs(float(asset["duration_s"]) - duration) > DURATION_TOLERANCE_S:
        problems.append(f"duration mismatch: manifest {asset['duration_s']}s, file {duration:.2f}s")
    return AssetCheck(asset_id, path, digest, duration, tuple(problems))


def verify_corpus(
    assets: list[dict[str, Any]], corpus_dir: Path, cache_path: Path | None = None
) -> tuple[dict[str, str], list[AssetCheck]]:
    """Check every manifest asset. Returns (receipt of asset_id -> sha256, failures).

    Hashes are cached by (file path, size, mtime) in `cache_path`, so a second run over
    an unchanged corpus does not re-read 2.6 GB.
    """
    import json

    cache: dict[str, Any] = {}
    if cache_path and cache_path.exists():
        try:
            cache = json.loads(cache_path.read_text())
        except ValueError:
            cache = {}

    def cached_hash(path: Path) -> str:
        st = path.stat()
        key = f"{path.resolve()}|{st.st_size}|{st.st_mtime_ns}"
        if key not in cache:
            cache[key] = {"sha256": sha256_file(path), "duration_s": probe_duration(path)}
        return cache[key]["sha256"]

    def cached_duration(path: Path) -> float:
        st = path.stat()
        return cache[f"{path.resolve()}|{st.st_size}|{st.st_mtime_ns}"]["duration_s"]

    checks = [check_asset(a, corpus_dir, hasher=cached_hash, prober=cached_duration)
              for a in assets]
    if cache_path:
        try:
            cache_path.write_text(json.dumps(cache))
        except OSError:
            pass
    receipt = {c.asset_id: c.sha256 for c in checks if c.ok}
    return receipt, [c for c in checks if not c.ok]
