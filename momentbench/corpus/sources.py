"""Corpus source definitions and download utilities."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any

import httpx


class LicenseType(str, Enum):
    PUBLIC_DOMAIN = "public_domain"
    CC_BY = "cc_by"
    CC_BY_SA = "cc_by_sa"
    CC0 = "cc0"


@dataclass
class CorpusSource:
    source_id: str
    title: str
    url: str
    license: LicenseType
    vertical: str
    attribution: str
    duration_s: float | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def is_sellable(self) -> bool:
        """CC BY-SA is public-only; everything else can go in the private set."""
        return self.license != LicenseType.CC_BY_SA


# Pre-defined source registries per vertical
CIVIC_SOURCES: list[dict[str, Any]] = [
    {
        "source_id": "house-gov-hearing",
        "base_url": "https://www.house.gov/feature-stories/watch-live",
        "license": LicenseType.PUBLIC_DOMAIN,
        "attribution": "U.S. House of Representatives (public domain, 17 U.S.C. § 105)",
    },
    {
        "source_id": "senate-gov-hearing",
        "base_url": "https://www.senate.gov/committees",
        "license": LicenseType.PUBLIC_DOMAIN,
        "attribution": "U.S. Senate (public domain, 17 U.S.C. § 105)",
    },
    {
        "source_id": "dvids-briefing",
        "base_url": "https://www.dvidshub.net",
        "license": LicenseType.PUBLIC_DOMAIN,
        "attribution": "Defense Visual Information Distribution Service (DVIDS)",
    },
]

AEROSPACE_SOURCES: list[dict[str, Any]] = [
    {
        "source_id": "nasa-launch",
        "base_url": "https://www.nasa.gov/multimedia/",
        "license": LicenseType.PUBLIC_DOMAIN,
        "attribution": "NASA (public domain)",
    },
    {
        "source_id": "noaa-field",
        "base_url": "https://www.noaa.gov/media-advisory",
        "license": LicenseType.PUBLIC_DOMAIN,
        "attribution": "NOAA (public domain)",
    },
]

FILM_SOURCES: list[dict[str, Any]] = [
    {
        "source_id": "blender-open-movies",
        "base_url": "https://studio.blender.org/films/",
        "license": LicenseType.CC_BY,
        "attribution": "Blender Foundation (CC BY)",
    },
    {
        "source_id": "prelinger-archives",
        "base_url": "https://archive.org/details/prelinger",
        "license": LicenseType.PUBLIC_DOMAIN,
        "attribution": "Prelinger Archives (public domain)",
    },
    {
        "source_id": "national-archives",
        "base_url": "https://www.archives.gov/research/catalog",
        "license": LicenseType.PUBLIC_DOMAIN,
        "attribution": "U.S. National Archives (public domain)",
    },
]


async def download_source(
    source: CorpusSource, output_dir: Path, overwrite: bool = False
) -> Path:
    """Download a source video to the output directory."""
    output_path = output_dir / f"{source.source_id}.mp4"
    if output_path.exists() and not overwrite:
        return output_path

    async with httpx.AsyncClient(follow_redirects=True, timeout=300) as client:
        async with client.stream("GET", source.url) as response:
            response.raise_for_status()
            output_path.parent.mkdir(parents=True, exist_ok=True)
            with open(output_path, "wb") as f:
                async for chunk in response.aiter_bytes(chunk_size=8192):
                    f.write(chunk)

    return output_path
