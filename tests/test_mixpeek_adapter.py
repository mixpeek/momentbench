"""Tests for the Mixpeek adapter (mixpeek board MO-3163, MO-3164)."""

import pytest

from momentbench.models.mixpeek_adapter import (
    MIN_SUPPORTED_BUILD_SHA,
    REQUIRED_TIME_FIELD,
    MixpeekAdapter,
    MixpeekBuildTooOldError,
    _build_stages,
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
            MixpeekAdapter(namespace="ns_test", collection="col_test", feature_uri="mixpeek://x@v1/e")

    def test_construction_with_old_build_raises(self):
        with pytest.raises(MixpeekBuildTooOldError):
            MixpeekAdapter(server_build="00000000", feature_uri="mixpeek://x@v1/e")

    def test_construction_without_feature_uri_or_retriever_id_raises(self):
        with pytest.raises(ValueError, match="feature_uri"):
            MixpeekAdapter(server_build=MIN_SUPPORTED_BUILD_SHA)

    def test_construction_with_retriever_id_but_no_feature_uri_succeeds(self):
        adapter = MixpeekAdapter(server_build=MIN_SUPPORTED_BUILD_SHA, retriever_id="ret_existing")
        assert adapter._retriever_id == "ret_existing"

    def test_construction_with_current_build_succeeds(self):
        adapter = MixpeekAdapter(
            server_build=MIN_SUPPORTED_BUILD_SHA, feature_uri="mixpeek://text_extractor@v1/embedding"
        )
        assert adapter._server_build == MIN_SUPPORTED_BUILD_SHA


class TestBuildStages:
    """Verified live 2026-09-08 against a throwaway standalone namespace — see module docstring."""

    def test_stage_schema_shape(self):
        stages = _build_stages("mixpeek://text_extractor@v1/embedding", "source_object_id", 2000)
        assert len(stages) == 2
        for stage in stages:
            assert "stage_name" in stage
            assert "stage_id" in stage["config"]
            assert "parameters" in stage["config"]

    def test_feature_search_stage_carries_feature_uri_and_query_template(self):
        stages = _build_stages("mixpeek://text_extractor@v1/embedding", "source_object_id", 2000)
        params = stages[0]["config"]["parameters"]
        assert params["searches"][0]["feature_uri"] == "mixpeek://text_extractor@v1/embedding"
        assert params["searches"][0]["query"] == {"input_mode": "text", "text": "{{INPUT.query}}"}

    def test_moment_group_stage_uses_required_time_field(self):
        stages = _build_stages("mixpeek://text_extractor@v1/embedding", "source_object_id", 2000)
        params = stages[1]["config"]["parameters"]
        assert params["time_field"] == REQUIRED_TIME_FIELD == "start_time"
        assert params["output_mode"] == "moments_only"

    def test_moment_group_stage_uses_given_parent_field_and_tolerance(self):
        stages = _build_stages("mixpeek://text_extractor@v1/embedding", "video_id", 500)
        params = stages[1]["config"]["parameters"]
        assert params["parent_field"] == "video_id"
        assert params["merge_tolerance_ms"] == 500
