"""examples/custom_agent.py on real media: a complete run packages, wrong media is refused.

Audit 3, finding 2: the example built LocalCorpus directly, so its runs were never
reportable. It now prepares the corpus the way `momentbench agentic` does.
"""

import importlib.util
import json
import shutil
import subprocess
from pathlib import Path

import pytest

from momentbench.agentic.run import AgenticRun
from momentbench.corpus.verify import probe_duration, sha256_file
from momentbench.submission import agentic_entry
from momentbench.types import Moment, Query, QueryTier

ROOT = Path(__file__).resolve().parents[1]
pytestmark = pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="needs ffmpeg")


def load_example():
    spec = importlib.util.spec_from_file_location("custom_agent",
                                                  ROOT / "examples/custom_agent.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def video(path: Path, seconds: int, colour: str) -> Path:
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-f", "lavfi", "-i",
                    f"color=c={colour}:s=64x36:d={seconds}", "-r", "2",
                    "-pix_fmt", "yuv420p", str(path)], check=True)
    return path


@pytest.fixture
def release(tmp_path):
    corpus = tmp_path / "corpus"
    corpus.mkdir()
    assets = []
    for aid, colour in (("clip-a", "red"), ("clip-b", "blue")):
        f = video(corpus / f"{aid}.mp4", 32, colour)
        assets.append({"asset_id": aid, "title": aid, "vertical": "film",
                       "duration_s": round(probe_duration(f), 2),
                       "file": {"name": f.name, "sha256": sha256_file(f)}})
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps(assets))
    tdir = tmp_path / "transcripts"
    tdir.mkdir()
    transcripts = {"clip-a": [{"t_in": 10.0, "t_out": 14.0, "text": "the red card"}]}
    for aid, segs in transcripts.items():
        (tdir / f"{aid}.transcript.json").write_text(json.dumps(segs))
    queries = [
        Query(query_id="hit", text="the red card", vertical="film",
              tier=QueryTier.SINGLE_VISUAL,
              gold_moments=[Moment(asset_id="clip-a", t_in=10.0, t_out=14.0)]),
        Query(query_id="miss", text="the blue card", vertical="film",
              tier=QueryTier.SINGLE_VISUAL,
              gold_moments=[Moment(asset_id="clip-b", t_in=2.0, t_out=6.0)]),
        Query(query_id="none", text="a green card", vertical="film",
              tier=QueryTier.CORPUS_HARD_NEGATIVE, gold_moments=[]),
    ]
    qfile = tmp_path / "film_v0.2.json"
    qfile.write_text(json.dumps([q.model_dump(mode="json") for q in queries]))
    return {"corpus": corpus, "manifest": manifest, "tdir": tdir, "queries": qfile,
            "transcripts": transcripts, "out": tmp_path / "runs"}


def args(r, *extra):
    return ["--corpus-dir", str(r["corpus"]), "--manifest", str(r["manifest"]),
            "--transcripts-dir", str(r["tdir"]), "--queries", str(r["queries"]),
            "--output-dir", str(r["out"]), *extra]


async def test_a_complete_run_on_verified_media_packages(release):
    await load_example().main(args(release, "--limit", "0"))
    [path] = release["out"].glob("*.json")
    run = AgenticRun.model_validate_json(path.read_text())
    assert run.reportable and set(run.corpus_receipt) == {"clip-a", "clip-b"}
    assert all(t.usage.frames == 4 for t in run.trials)  # real frames were extracted
    entry = agentic_entry(run, release["queries"], manifest=release["manifest"],
                          transcripts=release["transcripts"])
    assert entry["num_queries"] == 3 and entry["recall_at_1"] == pytest.approx(1 / 3, abs=1e-3)


async def test_wrong_media_is_refused(release):
    video(release["corpus"] / "clip-b.mp4", 32, "green")  # same length, other bytes
    with pytest.raises(SystemExit, match="clip-b"):
        await load_example().main(args(release, "--limit", "0"))
    assert not release["out"].exists()


async def test_partial_and_diagnostic_runs_are_labelled(release, capsys):
    (release["corpus"] / "clip-b.mp4").unlink()
    await load_example().main(args(release, "--limit", "0", "--partial"))
    assert "Not reportable" in capsys.readouterr().out
    shutil.rmtree(release["out"])
    video(release["corpus"] / "clip-b.mp4", 32, "blue")
    release["manifest"].write_text(json.dumps(
        [{**a, "file": {**a["file"], "sha256": sha256_file(release["corpus"] / a["file"]["name"])}}
         for a in json.loads(release["manifest"].read_text())]))
    await load_example().main(args(release, "--limit", "2"))
    assert "Diagnostic run on 2 queries" in capsys.readouterr().out


def test_frame_width_and_quality_reach_ffmpeg(release):
    import io

    from PIL import Image

    from momentbench.agentic.environment import ffmpeg_frame

    video = release["corpus"] / "clip-a.mp4"
    small = ffmpeg_frame(video, 5.0)
    wide = ffmpeg_frame(video, 5.0, width=1024, jpeg_quality=2)
    assert Image.open(io.BytesIO(small)).size[0] == 512
    assert Image.open(io.BytesIO(wide)).size[0] == 1024
