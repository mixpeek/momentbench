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


def test_heldout_commitments_detect_a_changed_answer(tmp_path, monkeypatch):
    import json

    spec = importlib.util.spec_from_file_location(
        "heldout_commit", Path(__file__).resolve().parents[1] / "scripts/heldout_commit.py")
    hc = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(hc)
    monkeypatch.setattr(hc, "OUT", tmp_path / "commitments.json")
    q = {"query_id": "h1", "text": "a red door", "tier": 1,
         "gold_moments": [{"asset_id": "v", "t_in": 1.0, "t_out": 3.0}]}
    for name in hc.FILES:
        (tmp_path / name).write_text(json.dumps([q] if name.startswith("film") else []))
    hc.commit(tmp_path)
    assert hc.verify(tmp_path) == 0
    q["gold_moments"][0]["t_out"] = 4.0
    (tmp_path / "film_v0.2_held_out.json").write_text(json.dumps([q]))
    assert hc.verify(tmp_path) == 1


def test_an_unreadable_absence_reply_is_incomplete(tmp_path, monkeypatch):
    import asyncio
    import sys
    from types import SimpleNamespace

    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
    spec = importlib.util.spec_from_file_location(
        "absence_check2", Path(__file__).resolve().parents[1] / "scripts/absence_check.py")
    ac = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(ac)
    monkeypatch.setenv("MOMENTBENCH_SPEND_LEDGER", str(tmp_path / "l.jsonl"))
    monkeypatch.setattr(ac, "small_copy", lambda v, o: b"video")

    class FakeModels:
        reply = ""

        async def generate_content(self, **kw):
            return SimpleNamespace(text=self.reply, usage_metadata=SimpleNamespace(
                prompt_token_count=10, candidates_token_count=5, thoughts_token_count=0))

    judges = SimpleNamespace(_gemini=SimpleNamespace(aio=SimpleNamespace(models=FakeModels())),
                             video=lambda aid: tmp_path / "v.mp4", sem=asyncio.Semaphore(1))
    asset = {"asset_id": "v", "duration_s": 10}
    for reply in ("I could not decide", '{"matches": [{"id": 7, "t_start": 1, "t_end": 2}]}',
                  '{"matches": [{"id": 1, "t_start": "soon"}]}'):
        FakeModels.reply = reply
        try:
            asyncio.run(ac.scan_chunk(judges, asset, [{"query_id": "na", "text": "x"}],
                                      tmp_path))
        except ac.IncompleteScanError:
            continue
        raise AssertionError(f"{reply!r} must leave the scan incomplete")
