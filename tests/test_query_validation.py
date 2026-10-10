"""The shipped query set must pass static validation, and the validator must catch errors."""

import json
from pathlib import Path

from momentbench.canary import CANARY
from momentbench.runner import load_queries
from momentbench.taxonomy.validate import check_queries
from momentbench.types import Moment, Query, QueryTier

MANIFEST = Path("data/corpus_manifests/sample_manifest.json")


def test_shipped_queries_are_clean():
    assets = json.loads(MANIFEST.read_text())
    for f in sorted(Path("data/queries").glob("*.json")):
        assert check_queries(load_queries(f), assets) == [], f


def test_validator_catches_each_problem():
    assets = [{"asset_id": "v1", "duration_s": 100}]
    good = {"canary": CANARY, "source": "test"}
    queries = [
        Query(query_id="a", text="t", tier=QueryTier.SINGLE_VISUAL, vertical="film",
              gold_moments=[Moment(asset_id="v1", t_in=90, t_out=120)], metadata=good),
        Query(query_id="a", text="t", tier=QueryTier.SINGLE_VISUAL, vertical="film",
              gold_moments=[Moment(asset_id="ghost", t_in=1, t_out=2)], metadata={}),
    ]
    problems = "\n".join(check_queries(queries, assets))
    for expected in ("past the end", "duplicate", "missing canary", "metadata.source",
                     "unknown asset"):
        assert expected in problems
