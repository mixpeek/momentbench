"""Upload MomentBench to Hugging Face as Mixpeek/momentbench.

    python scripts/upload_to_hf.py --videos CORPUS/videos --dry-run
    HF_TOKEN=... python scripts/upload_to_hf.py --videos CORPUS/videos

Uploads the dataset card, query files, manifest, transcripts, the 0.2.1 record runs
and, with --videos, every video whose manifest entry has license_status "confirmed".
A video with a pending licence is never uploaded; it stays fetchable from its source
with `momentbench corpus-download`. Each video's sha256 is checked against the manifest
before upload, and everything goes up in one commit.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from momentbench.corpus.verify import resolve_video, sha256_file  # noqa: E402

REPO_ID = "Mixpeek/momentbench"
REPO_TYPE = "dataset"


def plan(args: argparse.Namespace) -> list[tuple[Path, str]]:
    uploads: list[tuple[Path, str]] = [(ROOT / "scripts/hf_dataset_card.yaml", "README.md")]
    uploads += [(f, f"queries/{f.name}") for f in sorted((ROOT / "data/queries").glob("*.json"))]
    manifest = ROOT / "data/corpus_manifests/sample_manifest.json"
    uploads.append((manifest, f"corpus_manifests/{manifest.name}"))
    # Every entrant's transcript tools read these, so they ship with the queries.
    uploads += [(f, f"transcripts/{f.name}") for f in sorted((ROOT / "data/transcripts").glob("*"))]
    uploads += [(f, f"runs/{f.name}")
                for f in sorted((ROOT / "data/results/agentic-v0.2.1/record").glob("*.json"))]
    for name in ("ATTRIBUTION.md", "LICENSE", "GOVERNANCE.md"):
        if (ROOT / name).exists():
            uploads.append((ROOT / name, name))
    if args.videos:
        for asset in json.loads(manifest.read_text()):
            if asset.get("license_status") != "confirmed":
                print(f"  skip {asset['asset_id']}: licence {asset.get('license_status')}")
                continue
            path = resolve_video(asset, Path(args.videos))
            if path is None:
                raise SystemExit(f"{asset['asset_id']}: no video under {args.videos}")
            if sha256_file(path) != asset["file"]["sha256"]:
                raise SystemExit(f"{asset['asset_id']}: sha256 does not match the manifest")
            uploads.append((path, f"videos/{asset['file']['name']}"))
    return uploads


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--videos", help="corpus directory; uploads videos with confirmed licences")
    p.add_argument("--private", action="store_true", help="create as a private dataset")
    p.add_argument("--dry-run", action="store_true", help="list what would be uploaded")
    args = p.parse_args()

    uploads = plan(args)
    size = sum(f.stat().st_size for f, _ in uploads)
    print(f"{len(uploads)} files, {size / 1e9:.2f} GB -> {REPO_ID}")
    if args.dry_run:
        for local, remote in uploads:
            print(f"  {remote}  <-  {local}")
        return

    from huggingface_hub import HfApi

    api = HfApi(token=os.environ.get("HF_TOKEN"))
    api.create_repo(repo_id=REPO_ID, repo_type=REPO_TYPE, private=args.private,
                    exist_ok=True)
    with tempfile.TemporaryDirectory() as tmp:
        for local, remote in uploads:  # stage with hard links, then one commit
            dest = Path(tmp) / remote
            dest.parent.mkdir(parents=True, exist_ok=True)
            try:
                os.link(local, dest)
            except OSError:
                shutil.copy2(local, dest)
        api.upload_folder(folder_path=tmp, repo_id=REPO_ID, repo_type=REPO_TYPE,
                          commit_message="MomentBench v0.2.2: queries, transcripts, "
                                         "runs and licence-confirmed videos")
    print(f"Done: https://huggingface.co/datasets/{REPO_ID}")


if __name__ == "__main__":
    main()
