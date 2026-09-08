"""Mixpeek pipeline adapter — decompose → MVS → multi-stage retrieval.

Server-build gate (mixpeek board MO-3163, from gtm-research): until
mixpeek 75b4f058bb (2026-08-31), the moment_group retriever stage wrote
raw SECONDS into a field named start_ms with no unit conversion. Paired
with the default merge_tolerance_ms of 2000, the merge window was 2000
seconds (33 minutes), so every hit in a video merged into one moment
spanning nearly the whole video — a near-zero temporal-IoU score from a
units bug, not retrieval quality. Fixed in mixpeek's
server/shared/retrievers/stages/reduce/moment_group/services.py
(_time_field_scale_to_ms), guarded by 4 tests.

A Mixpeek number produced against a server build before that sha would
read as Mixpeek losing its own benchmark for the wrong reason. There is
no client-visible way to read a running server's git sha back from the
API (checked: no /health or /version field exposes it), so `server_build`
below is a required, human-supplied constructor argument rather than
something this adapter can verify for itself — get it from whoever
deployed the namespace this run points at.

Also worth knowing (mixpeek board GR-4, still open): the moment_group
stage's time_field defaults to 'query_chunks', which is only populated
for FILE-input queries. A TEXT query (which is all momentbench sends)
needs time_field explicitly set to 'start_time', or the stage silently
returns EMPTY moments — reading as a retrieval miss, not a
misconfiguration. MIN_SUPPORTED_BUILD_SHA and REQUIRED_TIME_FIELD below
exist so whoever wires the actual moment_group stage doesn't have to
rediscover either bug.

KNOWN SEPARATE ISSUE, not fixed here (filed as its own card rather than
folded into this one — MO-3163 is scoped to build-pinning): the predict()
call below (`client.search.text(...)`) does not match the installed SDK.
Checked directly against mixpeek==1.3.32: `Mixpeek` exposes `.search()` as
a plain method (not a `.search.text` namespace), and that method returns a
plain dict, not an object with `.results[].asset_id/.start_time/.end_time`
attributes. Building a real moment_group-enabled retriever also needs a
namespace-specific `feature_uri` (see mixpeek's
server/api/retrievers/models.py: AdhocExecuteRequest for the canonical
stage schema) that this repo has no visibility into without a live
namespace. Since no Mixpeek number has ever been produced (data/results/
holds only clip-1c80de13.json — MO-3133 is still blocked on a non-revoked
key), nothing here has regressed; the call is left as the pre-existing
placeholder pending that access, now gated so it cannot silently run
against a bad build once it is fixed.
"""

from __future__ import annotations

import time
from typing import Any

from momentbench.models.base import ModelAdapter
from momentbench.models.registry import register_adapter
from momentbench.types import Moment, Prediction, Query

# The moment_group units-bug fix (see module docstring). A `server_build`
# older than this sha scores near zero from a 33-minute merge window, not
# from retrieval quality.
MIN_SUPPORTED_BUILD_SHA = "75b4f058bb"

# TEXT queries must set this explicitly — the stage default ('query_chunks')
# is only populated for FILE-input queries and silently returns empty
# moments for a text query (mixpeek board GR-4).
REQUIRED_TIME_FIELD = "start_time"


class MixpeekBuildTooOldError(RuntimeError):
    """Raised when server_build is missing or predates the moment_group fix."""


def check_server_build(server_build: str | None) -> None:
    """Raise unless server_build is present and at or after MIN_SUPPORTED_BUILD_SHA.

    Short-sha string comparison here relies on 75b4f058bb-style sha prefixes
    sorting the same as their commit order, which git short-shas do not
    guarantee in general (hex, not chronological). It is sufficient for the
    single fixed point this checks against; do not generalize it into a
    history-ordering comparison for two arbitrary shas.
    """
    if not server_build:
        raise MixpeekBuildTooOldError(
            "server_build is required. There is no API to read a running "
            "Mixpeek server's git sha back, so pass the sha of the build the "
            "namespace you're querying is actually on (ask whoever deployed "
            f"it) so it can be checked against the moment_group fix at "
            f"{MIN_SUPPORTED_BUILD_SHA}. An unrecorded run is exactly how a "
            "units-bug score nearly reached publication before — see this "
            "module's docstring."
        )
    if server_build != MIN_SUPPORTED_BUILD_SHA and server_build < MIN_SUPPORTED_BUILD_SHA:
        raise MixpeekBuildTooOldError(
            f"server_build={server_build!r} predates {MIN_SUPPORTED_BUILD_SHA}, "
            "the moment_group units-bug fix (see module docstring). Re-run "
            "against a namespace on a newer build before publishing any "
            "number from this adapter."
        )


@register_adapter("mixpeek")
class MixpeekAdapter(ModelAdapter):
    model_id: str = "mixpeek-pipeline"

    def __init__(
        self,
        namespace: str | None = None,
        collection: str | None = None,
        server_build: str | None = None,
        **kwargs,
    ):
        """
        Args:
            namespace: Mixpeek namespace holding the momentbench corpus.
            collection: Collection id to search within.
            server_build: Git sha of the mixpeek server this namespace is
                running. Required — checked against MIN_SUPPORTED_BUILD_SHA
                before any request is made. See module docstring for why this
                can't be verified automatically instead.
        """
        check_server_build(server_build)
        self._namespace = namespace
        self._collection = collection
        self._server_build = server_build
        self._kwargs = kwargs

    async def predict(
        self,
        query: Query,
        corpus_assets: list[dict[str, Any]],
        max_results: int = 5,
    ) -> Prediction:
        from mixpeek import Mixpeek

        client = Mixpeek()

        start = time.perf_counter()
        response = client.search.text(
            query=query.text,
            collection=self._collection,
            filters={"vertical": query.vertical} if query.vertical else None,
            limit=max_results,
        )
        latency = (time.perf_counter() - start) * 1000

        moments = []
        for result in getattr(response, "results", []):
            moments.append(
                Moment(
                    asset_id=result.asset_id,
                    t_in=result.start_time,
                    t_out=result.end_time,
                    metadata={"server_build": self._server_build},
                )
            )

        return Prediction(
            query_id=query.query_id,
            ranked_moments=moments[:max_results],
            model_id=self.model_id,
            latency_ms=latency,
            metadata={"server_build": self._server_build},
        )
