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


def test_every_review_names_the_annotation_it_saw():
    """Audit 3, finding 5: a cross-check stands for the current annotation only when it
    checked that exact version; a review's result is the annotation that ships."""
    from momentbench.annotation import annotation_sha

    for f in sorted((ROOT / "data/queries").glob("*_v0.2.json")):
        for q in json.loads(f.read_text()):
            meta, sha = q["metadata"], annotation_sha(q)
            if "cross_check" in meta:
                assert meta["cross_check"]["annotation_sha"] == sha, q["query_id"]
            else:
                assert "new cross-check pending" in meta["source"], q["query_id"]
            for field in ("model_review", "human_review"):
                if field in meta:
                    assert meta[field]["result_sha"] == sha, (q["query_id"], field)


def test_citation_names_the_current_release():
    cff = " ".join(read("CITATION.cff").split())
    assert f'version: "{momentbench.__version__}"' in cff
    assert f"{PUBLISHED} queries are published" in cff


def test_scored_files_match_the_label_log():
    log = json.loads(read("data/labels/adjudication.json"))["queries"]
    for d in DOMAINS:
        path = f"data/queries/scored/{d}_v0.2_scored.json"
        scored = {q["query_id"] for q in json.loads(read(path))}
        decided = {q["query_id"] for q in queries(d)
                   if q["metadata"]["adjudication"]["decision"] == "scored"}
        assert scored == decided
        assert all(log[qid]["decision"] == "scored" for qid in scored)


def test_leaderboard_groups_never_mix_keys():
    import importlib.util

    spec = importlib.util.spec_from_file_location(
        "leaderboard_data", ROOT / "scripts/leaderboard_data.py")
    lb = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(lb)
    data = lb.build()
    keys = [g["compatibility_key"] for g in data["groups"] if g["compatibility_key"]]
    assert len(keys) == len(set(keys))
    for g in data["groups"]:
        if g["rows"]:
            assert {r["num_queries"] for r in g["rows"]} == {g["rows"][0]["num_queries"]}
        else:
            assert g["compatibility_key"] is None and g["not_run"]
    assert set(data["primary_keys"].values()) <= set(keys)


def test_every_leaderboard_number_rescores_from_its_run_file():
    """Codex round 3: rescore every packaged run and compare with the exported row."""
    from momentbench.agentic.run import AgenticRun
    from momentbench.submission import agentic_entry

    board = json.loads(read("docs/leaderboard.json"))
    for g in board["groups"]:
        for row in g["rows"]:
            run = AgenticRun.model_validate_json(read(row["file"]))
            qfile = next(p for p in (ROOT / "data/queries/scored" / g["queries_file"],
                                     ROOT / "data/queries" / g["queries_file"]) if p.exists())
            entry = agentic_entry(run, qfile)
            assert entry["compatibility_key"] == g["compatibility_key"], row["file"]
            for k in ("recall_at_1", "positive_recall_at_1", "net_score"):
                assert entry[k] == row[k], (row["file"], k)


def test_site_headlines_come_from_the_release_primary_groups():
    """The page's headline numbers, version and data links match docs/leaderboard.json,
    never the legacy results tables."""
    page = (ROOT / "site/index.html").read_text()
    board = json.loads((ROOT / "docs/leaderboard.json").read_text())
    rows = [r for g in board["groups"] if g["compatibility_key"] in
            board["primary_keys"].values() for r in g["rows"]]
    best_pos = max(r["positive_recall_at_1"] for r in rows)
    best_na = max(r["no_answer_accuracy"] for r in rows if r["no_answer_accuracy"] is not None)
    assert f"<dt>{100 * best_pos:.0f}%</dt><dd>best positive-query Recall@1" in page
    assert f"<dt>{100 * best_na:.0f}%</dt><dd>best no-answer accuracy, exploratory" in page
    assert f'content="{board["generated_at_commit"]}"' in page
    assert f"MomentBench v{momentbench.__version__} &middot;" in page
    assert "docs/results.json" not in page
    assert not re.search(r"v0\.2\.0 &middot;|Version 0\.2\.0", page)
    # The free start path is on the page, and crawlers can find it.
    assert 'id="start"' in page and "momentbench diagnose" in page
    assert "https://momentbench.com/" in (ROOT / "site/sitemap.xml").read_text()
