"""Tests for corpus verification (hashing and probing are injected)."""

from momentbench.corpus.verify import check_asset, sha256_file


def asset(**over):
    base = {"asset_id": "v1", "duration_s": 100.0, "file": {"sha256": "abc"}}
    return {**base, **over}


def check(tmp_path, a, digest="abc", duration=100.2):
    (tmp_path / "v1.mp4").write_bytes(b"x")
    return check_asset(a, tmp_path, hasher=lambda p: digest, prober=lambda p: duration)


def test_matching_file_is_ok(tmp_path):
    assert check(tmp_path, asset()).ok


def test_missing_file_fails(tmp_path):
    result = check_asset(asset(), tmp_path)
    assert not result.ok and result.problems == ("no file on disk",)


def test_hash_mismatch_fails(tmp_path):
    result = check(tmp_path, asset(), digest="zzz")
    assert not result.ok and "sha256 mismatch" in result.problems[0]


def test_manifest_without_hash_fails(tmp_path):
    result = check(tmp_path, asset(file=None))
    assert not result.ok and "no sha256" in result.problems[0]


def test_duration_off_by_more_than_a_second_fails(tmp_path):
    result = check(tmp_path, asset(), duration=113.8)
    assert not result.ok and "duration mismatch" in result.problems[0]


def test_sha256_file_hashes_contents(tmp_path):
    p = tmp_path / "f"
    p.write_bytes(b"abc")
    assert sha256_file(p) == "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"


def test_download_skips_assets_without_a_file_block(tmp_path):
    from momentbench.corpus.download import download_asset

    assert download_asset({"asset_id": "x"}, tmp_path).startswith("skip")


def test_download_unzips_and_checks_hash(tmp_path):
    import zipfile

    from momentbench.corpus.download import download_asset

    src = tmp_path / "src.zip"
    with zipfile.ZipFile(src, "w") as z:
        z.writestr("movie.mov", b"abc")
    asset = {"asset_id": "v1", "file": {
        "name": "v1.mov", "download_url": src.as_uri(),
        "sha256": "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"}}
    out = tmp_path / "out"
    out.mkdir()
    assert download_asset(asset, out) == "got  v1"
    assert (out / "v1.mov").read_bytes() == b"abc"
    assert download_asset(asset, out) == "have v1"
