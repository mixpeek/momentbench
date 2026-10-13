"""Release facts stated in prose must match the files (audit 2026-10-13, findings 10, 25).

Each check reads the number out of the document, so a stale count fails here
instead of reaching a reader.
"""

import json
import re
from collections import Counter
from pathlib import Path

import pytest

import momentbench
from momentbench.corpus.manifest import load_manifest

ROOT = Path(__file__).resolve().parents[1]
DOMAINS = ("film", "aerospace", "robotics")
MANIFEST = ROOT / "data/corpus_manifests/sample_manifest.json"


def queries(domain: str) -> list[dict]:
    return json.loads((ROOT / f"data/queries/{domain}_v0.2.json").read_text())


def read(rel: str) -> str:
    return (ROOT / rel).read_text()


PUBLISHED = sum(len(queries(d)) for d in DOMAINS)
TIERS = Counter(q["tier"] for d in DOMAINS for q in queries(d))


def test_package_version_matches_pyproject():
    # A regex, since tomllib needs Python 3.11 and CI also runs 3.10.
    version = re.search(r'^version = "([^"]+)"', read("pyproject.toml"), re.M)[1]
    assert momentbench.__version__ == version


def test_dataset_card_tier_table_matches_the_query_files():
    card = read("scripts/hf_dataset_card.yaml")
    rows = dict(re.findall(r"^\| (\d) \|.*\| (\d+) \|$", card, re.M))
    assert {int(k): int(v) for k, v in rows.items()} == dict(TIERS)
    assert re.search(rf"\b{PUBLISHED} are published here", card)
    for d in DOMAINS:
        assert f"queries/{d}_v0.2.json" in card, d


def test_held_out_and_total_counts_agree_everywhere():
    card, readme = read("scripts/hf_dataset_card.yaml"), read("README.md")
    total = int(re.search(r"In total (\d+) queries", readme)[1])
    assert int(re.search(r"(\d+) in this repository", readme)[1]) == PUBLISHED
    held = total - PUBLISHED
    assert re.search(rf"\b{held} are held out", card)
    assert f"The {held}-query private held-out set" in read("SUBMISSION.md")
    assert f"{PUBLISHED} are published, {held} are held out" in read("docs/report.md")
    per_domain = dict(re.findall(r"data/queries/(\w+)_v0\.2\.json\)?: (\d+) ", readme))
    for d in DOMAINS:
        assert int(per_domain[d]) == len(queries(d)), d
    assert sum(int(n) for n in re.findall(r"(\d+) more are held out", readme)) == held


def test_corpus_minutes_agree():
    minutes = round(sum(a["duration_s"] for a in json.loads(MANIFEST.read_text())) / 60)
    assert f"about {minutes} minutes of footage" in read("scripts/hf_dataset_card.yaml")
    assert f"({minutes} minutes of" in read("docs/report.md")


class TestManifest:  # finding 25 (licence) and 10 (round trip)
    def test_load_then_save_keeps_every_field(self, tmp_path):
        out = tmp_path / "m.json"
        load_manifest(MANIFEST).save(out)
        assert out.read_text() == MANIFEST.read_text()

    def test_licence_status_is_machine_readable(self):
        for e in load_manifest(MANIFEST).entries:
            assert e.license_status in {"confirmed", "pending"}
            if e.license_status == "pending":
                assert not e.is_sellable and e.license_note, e.asset_id

    def test_a_pending_licence_cannot_be_sellable(self, tmp_path):
        data = json.loads(MANIFEST.read_text())
        data[0].update(license_status="pending", is_sellable=True)
        bad = tmp_path / "m.json"
        bad.write_text(json.dumps(data))
        with pytest.raises(ValueError, match="pending licence"):
            load_manifest(bad)

    def test_attribution_names_every_pending_asset(self):
        attribution = read("ATTRIBUTION.md")
        for e in load_manifest(MANIFEST).entries:
            if e.license_status == "pending":
                assert e.title.split(":")[0] in attribution.split("Open licence questions")[1]
