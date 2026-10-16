"""scripts/review_queries.py apply: who reviewed is recorded truthfully."""

import json
import subprocess
import sys
from pathlib import Path

SCRIPT = Path(__file__).resolve().parents[1] / "scripts/review_queries.py"


def q(qid, text="a rover on sand"):
    return {"query_id": qid, "text": text, "tier": 1, "vertical": "robotics",
            "gold_moments": [{"asset_id": "v", "t_in": 1.0, "t_out": 5.0}],
            "metadata": {"source": "Claude drafted; pending human review"}}


def run(tmp_path, decisions, kind):
    qf, df = tmp_path / "q.json", tmp_path / "d.json"
    qf.write_text(json.dumps([q("a"), q("b"), q("c")]))
    df.write_text(json.dumps(decisions))
    subprocess.run([sys.executable, str(SCRIPT), "apply", str(qf), str(df),
                    "--reviewer", "R", "--kind", kind], check=True, capture_output=True)
    return {x["query_id"]: x for x in json.loads(qf.read_text())}


def test_a_model_review_never_reads_as_human(tmp_path):
    out = run(tmp_path, {"a": {"decision": "accept", "note": "ok"},
                         "b": {"decision": "fix", "note": "span ends early",
                               "fix": {"t_in": 1.0, "t_out": 6.5, "text": "a rover on red sand"}},
                         "c": {"decision": "reject", "note": "matches many shots"}}, "model")
    assert set(out) == {"a", "b"}
    for x in out.values():
        assert "human_review" not in x["metadata"]
        assert x["metadata"]["model_review"]["reviewer"] == "R"
        assert "pending human review" in x["metadata"]["source"]
    b = out["b"]
    assert b["gold_moments"][0]["t_out"] == 6.5 and b["text"] == "a rover on red sand"
    assert b["metadata"]["boundary_refined"]["from"] == [1.0, 5.0]
    assert b["metadata"]["text_refined"]["applied_at"].endswith("Z")


def test_a_human_review_marks_accepted_queries(tmp_path):
    out = run(tmp_path, {"a": {"decision": "accept", "note": ""}}, "human")
    assert out["a"]["metadata"]["human_review"]["decision"] == "accept"
    assert "human-reviewed by R" in out["a"]["metadata"]["source"]


def test_a_fix_supersedes_the_cross_check_and_reviews_are_bound(tmp_path):
    sys.path.insert(0, str(SCRIPT.parents[1]))
    from momentbench.annotation import annotation_sha

    qf, df = tmp_path / "q.json", tmp_path / "d.json"
    rows = [q("a"), q("b")]
    for r in rows:
        r["metadata"]["source"] += "; cross-checked by gemini"
        r["metadata"]["cross_check"] = {"model": "gemini", "agreed": True, "picked": "gold",
                                        "annotation_sha": annotation_sha(r)}
    qf.write_text(json.dumps(rows))
    df.write_text(json.dumps({"a": {"decision": "accept", "note": "ok"},
                              "b": {"decision": "fix", "note": "late",
                                    "fix": {"t_in": 1.5, "t_out": 5.0}}}))
    subprocess.run([sys.executable, str(SCRIPT), "apply", str(qf), str(df),
                    "--reviewer", "R", "--kind", "model"], check=True, capture_output=True)
    out = {x["query_id"]: x for x in json.loads(qf.read_text())}
    a, b = out["a"]["metadata"], out["b"]["metadata"]
    assert a["cross_check"]["annotation_sha"] == annotation_sha(out["a"])
    assert a["model_review"]["annotation_sha"] == a["model_review"]["result_sha"]
    assert "cross_check" not in b and b["cross_check_history"][0]["agreed"] is True
    assert b["model_review"]["result_sha"] == annotation_sha(out["b"])
    assert b["model_review"]["annotation_sha"] != b["model_review"]["result_sha"]
    assert "new cross-check pending" in b["source"]
