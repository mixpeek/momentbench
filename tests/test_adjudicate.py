"""scripts/adjudicate.py: verdicts derived from the judges' answers, and the decision rule."""

import importlib.util
from pathlib import Path

spec = importlib.util.spec_from_file_location(
    "adjudicate", Path(__file__).resolve().parents[1] / "scripts/adjudicate.py")
adj = importlib.util.module_from_spec(spec)
spec.loader.exec_module(adj)


def test_a_no_answer_verdict_comes_from_the_three_answers():
    ok = adj.parse('{"a_true": true, "b_true": false, "detail_visible": true, "reason": "r"}')
    assert ok["verdict"] == "accept"
    for bad in ('{"a_true": false, "b_true": false, "detail_visible": true}',
                '{"a_true": true, "b_true": true, "detail_visible": true}',
                '{"a_true": true, "b_true": false, "detail_visible": false}'):
        assert adj.parse(bad)["verdict"] == "reject"


def test_positive_verdicts_and_garbage():
    assert adj.parse('```json\n{"verdict": "Accept", "reason": "fine"}\n```')["verdict"] == "accept"
    assert adj.parse("no json here")["verdict"] == "error"


def test_scored_needs_both_judges():
    yes, no = {"verdict": "accept"}, {"verdict": "wrong_span"}
    assert adj.decide(yes, yes) == "scored"
    assert adj.decide(yes, no) == adj.decide(no, yes) == "provisional"
    assert adj.decide({"verdict": "error"}, yes) == "provisional"


def test_absence_ignores_hits_on_the_source_shot():
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
    spec = importlib.util.spec_from_file_location(
        "absence_check", Path(__file__).resolve().parents[1] / "scripts/absence_check.py")
    ac = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(ac)
    q = {"metadata": {"altered_from": {"asset_id": "v", "t_in": 10.0, "t_out": 14.0}}}
    assert ac.is_source({"asset_id": "v", "t_start": 12, "t_end": 13}, q)
    assert not ac.is_source({"asset_id": "v", "t_start": 40, "t_end": 44}, q)
    assert not ac.is_source({"asset_id": "w", "t_start": 12, "t_end": 13}, q)
