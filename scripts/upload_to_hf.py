"""Upload MomentBench dataset to Hugging Face as mixpeek/momentbench."""

import argparse
from pathlib import Path

from huggingface_hub import HfApi

REPO_ID = "Mixpeek/momentbench"
REPO_TYPE = "dataset"


def main():
    parser = argparse.ArgumentParser(description="Upload MomentBench to Hugging Face")
    parser.add_argument(
        "--queries", type=str, default="data/queries",
        help="Path to queries directory",
    )
    parser.add_argument(
        "--manifests", type=str, default="data/corpus_manifests",
        help="Path to corpus manifests directory",
    )
    parser.add_argument(
        "--results", type=str, default="data/results/agentic-v0.2/record",
        help="Path to results directory (baseline runs)",
    )
    parser.add_argument(
        "--dataset-card", type=str, default="scripts/hf_dataset_card.yaml",
        help="Path to dataset card YAML",
    )
    parser.add_argument("--private", action="store_true", help="Create as private dataset")
    parser.add_argument("--dry-run", action="store_true", help="Show what would be uploaded")
    args = parser.parse_args()

    uploads = []

    card_path = Path(args.dataset_card)
    if card_path.exists():
        uploads.append((card_path, "README.md"))

    queries_dir = Path(args.queries)
    if queries_dir.exists():
        for f in queries_dir.glob("*.json"):
            uploads.append((f, f"queries/{f.name}"))

    manifests_dir = Path(args.manifests)
    if manifests_dir.exists():
        for f in manifests_dir.glob("*.json"):
            uploads.append((f, f"corpus_manifests/{f.name}"))

    # Every entrant's transcript tools read these, so they ship with the queries.
    for f in sorted(Path("data/transcripts").glob("*")):
        uploads.append((f, f"transcripts/{f.name}"))

    results_dir = Path(args.results)
    if results_dir.exists():
        for f in results_dir.glob("*.json"):
            uploads.append((f, f"baselines/{f.name}"))

    if not uploads:
        print("No data files found to upload.")
        return

    if args.dry_run:
        print(f"Would upload {len(uploads)} files to {REPO_ID}:")
        for local, remote in uploads:
            print(f"  {local} -> {remote}")
        return

    api = HfApi()

    api.create_repo(
        repo_id=REPO_ID,
        repo_type=REPO_TYPE,
        private=args.private,
        exist_ok=True,
    )

    for local_path, repo_path in uploads:
        api.upload_file(
            path_or_fileobj=str(local_path),
            path_in_repo=repo_path,
            repo_id=REPO_ID,
            repo_type=REPO_TYPE,
        )
        print(f"  {local_path} -> {repo_path}")

    print(f"\nDone. Dataset at: https://huggingface.co/datasets/{REPO_ID}")


if __name__ == "__main__":
    main()
