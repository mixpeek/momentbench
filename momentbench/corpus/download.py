"""Fetch corpus files listed in the manifest's `file` blocks into a directory."""

from __future__ import annotations

import shutil
import time
import urllib.error
import urllib.request
import zipfile
from pathlib import Path
from typing import Any

from momentbench.corpus.verify import sha256_file

# download.blender.org answers urllib's default User-Agent with 403.
USER_AGENT = "momentbench-corpus-download/0.1 (+https://github.com/mixpeek/momentbench)"
# Seconds a connect or a single read may stall before the attempt is abandoned.
TIMEOUT_S = 60
# Waits before the second and third attempts.
RETRY_WAITS_S = (5, 20)


def _retryable(e: OSError) -> bool:
    """Network faults and 5xx/429 are worth retrying; other HTTP errors are not."""
    if isinstance(e, urllib.error.HTTPError):
        return e.code == 429 or e.code >= 500
    return True


def download_asset(asset: dict[str, Any], corpus_dir: Path,
                   sleep: Any = time.sleep) -> str:
    """Download one asset to <corpus_dir>/<file.name>. Returns a one-line status."""
    spec = asset.get("file")
    if not spec:
        return f"skip {asset['asset_id']}: manifest has no file block"
    target = corpus_dir / spec["name"]
    if target.exists() and sha256_file(target) == spec["sha256"]:
        return f"have {asset['asset_id']}"
    url = spec["download_url"]
    tmp = corpus_dir / (spec["name"] + ".part")
    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    for attempt, wait in enumerate((*RETRY_WAITS_S, None), start=1):
        try:
            _fetch(request, tmp)
            break
        except OSError as e:  # URLError, HTTPError and socket timeouts are all OSError
            tmp.unlink(missing_ok=True)
            if wait is None or not _retryable(e):
                return f"FAIL {asset['asset_id']}: {url}: {e} (attempt {attempt})"
            sleep(wait)
    return _finish(asset, spec, url, tmp, target)


def _fetch(request: urllib.request.Request, tmp: Path) -> None:
    with (urllib.request.urlopen(request, timeout=TIMEOUT_S) as resp,  # noqa: S310
          tmp.open("wb") as out):
        shutil.copyfileobj(resp, out, length=1 << 20)


def _finish(asset: dict[str, Any], spec: dict[str, Any], url: str, tmp: Path,
            target: Path) -> str:
    if url.endswith(".zip"):
        with zipfile.ZipFile(tmp) as z:
            videos = [n for n in z.namelist() if not n.endswith("/")]
            if len(videos) != 1:
                raise ValueError(f"{url}: expected one file in the zip, found {len(videos)}")
            with z.open(videos[0]) as src, target.open("wb") as out:
                shutil.copyfileobj(src, out, length=1 << 20)
        tmp.unlink()
    else:
        tmp.replace(target)
    digest = sha256_file(target)
    if digest != spec["sha256"]:
        return f"FAIL {asset['asset_id']}: sha256 {digest[:12]} does not match the manifest"
    return f"got  {asset['asset_id']}"
