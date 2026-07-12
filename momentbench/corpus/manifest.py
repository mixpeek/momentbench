"""Corpus manifest management and attribution tracking."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from momentbench.corpus.sources import LicenseType


@dataclass
class ManifestEntry:
    asset_id: str
    source_id: str
    title: str
    vertical: str
    license: LicenseType
    attribution: str
    duration_s: float
    source_url: str
    is_sellable: bool
    storage_path: str | None = None
    mixpeek_asset_id: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class CorpusManifest:
    entries: list[ManifestEntry] = field(default_factory=list)

    @property
    def total_duration_hours(self) -> float:
        return sum(e.duration_s for e in self.entries) / 3600

    @property
    def public_entries(self) -> list[ManifestEntry]:
        return self.entries

    @property
    def sellable_entries(self) -> list[ManifestEntry]:
        return [e for e in self.entries if e.is_sellable]

    def by_vertical(self, vertical: str) -> list[ManifestEntry]:
        return [e for e in self.entries if e.vertical == vertical]

    def add(self, entry: ManifestEntry) -> None:
        self.entries.append(entry)

    def save(self, path: Path) -> None:
        data = []
        for e in self.entries:
            data.append({
                "asset_id": e.asset_id,
                "source_id": e.source_id,
                "title": e.title,
                "vertical": e.vertical,
                "license": e.license.value,
                "attribution": e.attribution,
                "duration_s": e.duration_s,
                "source_url": e.source_url,
                "is_sellable": e.is_sellable,
                "storage_path": e.storage_path,
                "mixpeek_asset_id": e.mixpeek_asset_id,
                "metadata": e.metadata,
            })
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(data, indent=2))

    def generate_attribution_md(self) -> str:
        """Generate ATTRIBUTION.md content."""
        lines = ["# MomentBench Attribution", "", ""]
        by_license: dict[str, list[ManifestEntry]] = {}
        for e in self.entries:
            by_license.setdefault(e.license.value, []).append(e)

        for license_type, entries in sorted(by_license.items()):
            lines.append(f"## {license_type.replace('_', ' ').title()}")
            lines.append("")
            for e in entries:
                lines.append(f"- **{e.title}** ({e.vertical}): {e.attribution}")
                lines.append(f"  Source: {e.source_url}")
            lines.append("")

        return "\n".join(lines)


def load_manifest(path: Path) -> CorpusManifest:
    """Load a corpus manifest from JSON."""
    data = json.loads(path.read_text())
    manifest = CorpusManifest()
    for item in data:
        manifest.add(ManifestEntry(
            asset_id=item["asset_id"],
            source_id=item["source_id"],
            title=item["title"],
            vertical=item["vertical"],
            license=LicenseType(item["license"]),
            attribution=item["attribution"],
            duration_s=item["duration_s"],
            source_url=item["source_url"],
            is_sellable=item["is_sellable"],
            storage_path=item.get("storage_path"),
            mixpeek_asset_id=item.get("mixpeek_asset_id"),
            metadata=item.get("metadata", {}),
        ))
    return manifest
