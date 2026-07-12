"""Run Mixpeek decomposition pipelines on ingested assets."""

from __future__ import annotations

from typing import Any

from momentbench.corpus.manifest import CorpusManifest

PIPELINE_CONFIG = {
    "scene_segmentation": {
        "enabled": True,
        "max_scene_duration_s": 5.0,
    },
    "transcription": {
        "enabled": True,
        "word_level_timestamps": True,
        "speaker_diarization": True,
    },
    "visual_embeddings": {
        "enabled": True,
        "keyframe_sampling": True,
    },
    "ocr": {
        "enabled": True,
    },
    "audio_events": {
        "enabled": True,
        "categories": ["applause", "gavel", "engine", "crowd", "silence", "music"],
    },
    "object_detection": {
        "enabled": True,
        "face_reid": True,
    },
}


async def run_decomposition(
    manifest: CorpusManifest,
    pipeline_config: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Run feature extraction pipelines on all assets in the manifest."""
    from mixpeek import Mixpeek

    client = Mixpeek()
    config = pipeline_config or PIPELINE_CONFIG
    results = {"processed": 0, "errors": []}

    for entry in manifest.entries:
        if not entry.mixpeek_asset_id:
            results["errors"].append(f"{entry.asset_id}: not ingested yet")
            continue

        try:
            client.assets.process(
                asset_id=entry.mixpeek_asset_id,
                pipeline=config,
            )
            results["processed"] += 1
        except Exception as e:
            results["errors"].append(f"{entry.asset_id}: {e}")

    return results
