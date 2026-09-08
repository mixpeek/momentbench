"""Tests for the Mixpeek adapter's server-build gate (mixpeek board MO-3163)."""

import pytest

from momentbench.models.mixpeek_adapter import (
    MIN_SUPPORTED_BUILD_SHA,
    MixpeekAdapter,
    MixpeekBuildTooOldError,
    check_server_build,
)


class TestCheckServerBuild:
    def test_missing_build_raises(self):
        with pytest.raises(MixpeekBuildTooOldError, match="server_build is required"):
            check_server_build(None)

    def test_empty_string_raises(self):
        with pytest.raises(MixpeekBuildTooOldError, match="server_build is required"):
            check_server_build("")

    def test_exact_min_build_is_accepted(self):
        check_server_build(MIN_SUPPORTED_BUILD_SHA)

    def test_build_after_min_is_accepted(self):
        check_server_build("zzzzzzzzzz")

    def test_build_before_min_raises(self):
        with pytest.raises(MixpeekBuildTooOldError, match="predates"):
            check_server_build("00000000")


class TestMixpeekAdapterConstruction:
    def test_construction_without_build_raises(self):
        with pytest.raises(MixpeekBuildTooOldError):
            MixpeekAdapter(namespace="ns_test", collection="col_test")

    def test_construction_with_old_build_raises(self):
        with pytest.raises(MixpeekBuildTooOldError):
            MixpeekAdapter(server_build="00000000")

    def test_construction_with_current_build_succeeds(self):
        adapter = MixpeekAdapter(server_build=MIN_SUPPORTED_BUILD_SHA)
        assert adapter._server_build == MIN_SUPPORTED_BUILD_SHA
