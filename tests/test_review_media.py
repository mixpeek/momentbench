"""Annotation rechecks see the current span's frames (audit 4, findings 3 and 4).

A real eight-second video, red for four seconds then blue. A cross-check of 0-2 s and
then, for the same query ID, of 5-7 s must show red and then blue frames.
"""

import asyncio
import importlib
import shutil
import subprocess
import sys
from pathlib import Path

import pytest
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
pytestmark = pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="needs ffmpeg")


def scripts():
    sys.path.insert(0, str(ROOT / "scripts"))
    return importlib.import_module("draft_queries"), importlib.import_module("cross_verify")


@pytest.fixture
def red_blue(tmp_path):
    video = tmp_path / "rb.mp4"
    subprocess.run(["ffmpeg", "-v", "error", "-y",
                    "-f", "lavfi", "-i", "color=c=red:s=64x36:d=4:r=10",
                    "-f", "lavfi", "-i", "color=c=blue:s=64x36:d=4:r=10",
                    "-filter_complex", "[0][1]concat=n=2:v=1", "-pix_fmt", "yuv420p",
                    str(video)], check=True)
    return video


def colour(path: Path) -> str:
    r, g, b = Image.open(path).convert("RGB").resize((1, 1)).getpixel((0, 0))
    return "red" if r > b else "blue"


def test_a_cached_frame_is_not_reused_for_another_time(red_blue, tmp_path):
    dq, _ = scripts()
    out = tmp_path / "c_q1_gold_1.jpg"
    assert colour(dq.frame(red_blue, 1.0, out)) == "red"
    assert colour(dq.frame(red_blue, 6.0, out)) == "blue"


class StubFilm:
    """The parts of draft_queries.Film that judge() uses, on a real video."""

    def __init__(self, dq, video, work):
        self.video, self.dir, self.video_end = video, work, 8.0
        self.strip = dq.Film.strip.__get__(self)
        self.at = dq.Film.at.__get__(self)

    def dialogue(self, a, b):
        return "(none)"


def test_a_recheck_after_a_span_edit_sends_the_new_frames(red_blue, tmp_path, monkeypatch):
    dq, xv = scripts()
    film = StubFilm(dq, red_blue, tmp_path)
    seen = []

    async def fake_gemini(client, system, prompt, image, sem, usage):
        seen.append(image)
        return {"answer": "A", "wrong_details": []}

    monkeypatch.setattr(xv, "ask_gemini", fake_gemini)
    q = {"query_id": "q1", "tier": 1, "hard_negatives": [], "metadata": {},
         "gold_moments": [{"asset_id": "rb", "t_in": 0.0, "t_out": 2.0}]}

    async def check(span):
        return await xv.judge({"rb": film}, q, "a colour", span, asyncio.Semaphore(1),
                              object(), {}, "t")

    _, first = asyncio.run(check((0.0, 2.0)))
    _, second = asyncio.run(check((5.0, 7.0)))
    assert first["sheet_sha256"] != second["sheet_sha256"]
    [gold] = second["candidates"]
    assert gold["span"] == [5.0, 7.0]
    assert [f["t"] for f in gold["frames"]] == [5.3, 6.0, 6.7]
    assert gold["frames"][0]["video_sha256"] == first["candidates"][0]["frames"][0][
        "video_sha256"]
    middle = tmp_path / "c_x_q1_gold_1.jpg"
    assert colour(middle) == "blue"


def test_a_no_answer_check_needs_its_original(monkeypatch):
    _, xv = scripts()
    calls = []

    async def fake_judge(*args, **kwargs):
        calls.append(args[2])
        return "none", {}

    monkeypatch.setattr(xv, "judge", fake_judge)
    q = {"query_id": "na", "tier": 3, "text": "a green card", "gold_moments": [],
         "metadata": {"altered_from": {"asset_id": "rb", "t_in": 1.0, "t_out": 2.0}}}
    result = asyncio.run(xv.cross_check({}, {}, q, None, None, {}))
    assert result["agreed"] is False and result["complete"] is False and calls == []
    q["metadata"]["altered_from"]["original_text"] = "a red card"
    result = asyncio.run(xv.cross_check({}, {}, q, None, None, {}))
    assert calls == ["a green card", "a red card"]
    assert result["agreed"] is False  # the original was answered "none", not gold
