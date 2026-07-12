"""Ingest corpus assets into Mixpeek buckets and collections."""

from __future__ import annotations

from typing import Any

from momentbench.corpus.manifest import CorpusManifest

VERTICAL_TO_BUCKET = {
    "civic": "momentbench-civic",
    "aerospace": "momentbench-aerospace",
    "film": "momentbench-film",
}


async def ingest_corpus(
    manifest: CorpusManifest,
    namespace: str = "momentbench",
) -> dict[str, Any]:
    """Ingest all corpus assets into Mixpeek, organized by vertical buckets."""
    from mixpeek import Mixpeek

    client = Mixpeek()
    results = {"ingested": 0, "skipped": 0, "errors": []}

    for entry in manifest.entries:
        try:
            bucket = VERTICAL_TO_BUCKET.get(entry.vertical)
            if not bucket:
                results["errors"].append(f"Unknown vertical: {entry.vertical}")
                continue

            response = client.assets.create(
                url=entry.source_url,
                collection=f"momentbench-{entry.vertical}",
                metadata={
                    "source_id": entry.source_id,
                    "title": entry.title,
                    "vertical": entry.vertical,
                    "license": entry.license.value,
                    "attribution": entry.attribution,
                    "duration_s": entry.duration_s,
                    "is_sellable": entry.is_sellable,
                },
            )
            entry.mixpeek_asset_id = response.asset_id
            results["ingested"] += 1
        except Exception as e:
            results["errors"].append(f"{entry.asset_id}: {e}")

    return results


async def ensure_collections(namespace: str = "momentbench") -> None:
    """Create Mixpeek collections for each vertical if they don't exist."""
    from mixpeek import Mixpeek

    client = Mixpeek()
    for vertical in VERTICAL_TO_BUCKET:
        collection_name = f"momentbench-{vertical}"
        try:
            client.collections.create(
                name=collection_name,
                metadata={"vertical": vertical, "benchmark": "momentbench"},
            )
        except Exception:
            pass  # Collection already exists
