"""Corpus manifest management and attribution tracking."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from momentbench.corpus.sources import LicenseType

LICENSE_STATUSES = {"confirmed", "pending"}
FIELDS = ("asset_id", "source_id", "title", "vertical", "license", "license_status",
          "license_note", "attribution", "duration_s", "source_url", "is_sellable",
          "storage_path", "mixpeek_asset_id", "metadata")


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
    # "confirmed" or "pending". A pending licence is never sellable.
    license_status: str = "confirmed"
    license_note: str | None = None
    # Every other key in the manifest entry (the `file` block with its sha256 and
    # download URL, among others), kept so load then save loses nothing.
    extra: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.license_status not in LICENSE_STATUSES:
            raise ValueError(f"{self.asset_id}: license_status must be one of "
                             f"{sorted(LICENSE_STATUSES)}, got {self.license_status!r}")
        if self.license_status == "pending" and self.is_sellable:
            raise ValueError(f"{self.asset_id}: a pending licence cannot be sellable")


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
            item = {k: getattr(e, k) for k in FIELDS}
            item["license"] = e.license.value
            if item["license_note"] is None:
                del item["license_note"]
            data.append({**item, **e.extra})
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n")

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
        known = {k: item[k] for k in FIELDS if k in item}
        known["license"] = LicenseType(item["license"])
        known.setdefault("metadata", {})
        manifest.add(ManifestEntry(
            **known, extra={k: v for k, v in item.items() if k not in FIELDS}))
    return manifest
