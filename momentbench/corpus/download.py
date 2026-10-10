"""Fetch corpus files listed in the manifest's `file` blocks into a directory."""

from __future__ import annotations

import shutil
import urllib.request
import zipfile
from pathlib import Path
from typing import Any

from momentbench.corpus.verify import sha256_file


def download_asset(asset: dict[str, Any], corpus_dir: Path) -> str:
    """Download one asset to <corpus_dir>/<file.name>. Returns a one-line status."""
    spec = asset.get("file")
    if not spec:
        return f"skip {asset['asset_id']}: manifest has no file block"
    target = corpus_dir / spec["name"]
    if target.exists() and sha256_file(target) == spec["sha256"]:
        return f"have {asset['asset_id']}"
    url = spec["download_url"]
    tmp = corpus_dir / (spec["name"] + ".part")
    with urllib.request.urlopen(url) as resp, tmp.open("wb") as out:  # noqa: S310
        shutil.copyfileobj(resp, out, length=1 << 20)
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
