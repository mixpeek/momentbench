"""Mixpeek pipeline adapter — feature_search → moment_group retrieval.

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
misconfiguration.

RETRIEVER MECHANICS (mixpeek board MO-3164) — verified live 2026-09-08
against a throwaway standalone namespace (created, probed, and deleted in
the same session; no momentbench data touched), not guessed from source
reading alone:

- The stage schema is {"stage_name": <label>, "config": {"stage_id": <type>,
  "parameters": {...}}} — confirmed against mixpeek's
  server/api/retrievers/models.py:AdhocExecuteRequest docstring AND by
  actually creating a retriever with it (ret_24680fedcf9efa, deleted after).
- moment_group with time_field="start_time" ALSO requires a same-named
  "end_time" field on each document: the stage derives the end field by
  literal string-replacing "start" with "end" in time_field
  (services.py: `end_field = time_field.replace("start", "end")`), and
  drops any document missing either field. A first probe with only
  start_time silently returned 0 moments from 4 input documents — the
  same "reads as a retrieval miss" trap GR-4 already describes, just from
  a second, undocumented cause. Missed this from reading the source alone;
  only caught it by actually running a query.
- With output_mode="moments_only", each response document has exactly one
  entry in `moments`, keyed by `parent_id` (not `asset_id`), with
  `start_ms`/`end_ms` in MILLISECONDS (divide by 1000 for t_in/t_out).
- The response's result list is under the `documents` key. `results` is a
  deprecated alias that this API version no longer populates by default
  (confirmed via the live response's own `warnings` field).

NOT verified (blocked on MO-3133 — no real momentbench corpus is ingested
anywhere yet, so there is no real collection to point at): the actual
`feature_uri` a live momentbench collection would use for a text query
(e.g. "mixpeek://text_extractor@v1/embedding" for a plain text extractor,
or a multimodal extractor's URI if that's what ingestion picks for video
content — this repo has no visibility into that choice). `feature_uri`
below is therefore a required constructor argument, not a hardcoded guess.
"""

from __future__ import annotations

import os
import subprocess
import time
from pathlib import Path
from typing import Any

from momentbench.models.base import ModelAdapter
from momentbench.models.registry import register_adapter
from momentbench.types import Moment, Prediction, Query

# The moment_group units-bug fix (see module docstring). A `server_build`
# older than this sha scores near zero from a 33-minute merge window, not
# from retrieval quality.
MIN_SUPPORTED_BUILD_SHA = "75b4f058bb"
# Commit time of that sha on mixpeek main, for runs without a mixpeek checkout.
MIN_SUPPORTED_BUILD_TIME = "2026-08-31T20:56:05Z"

# TEXT queries must set this explicitly — the stage default ('query_chunks')
# is only populated for FILE-input queries and silently returns empty
# moments for a text query (mixpeek board GR-4). A matching end_time field
# is also required on every document (see module docstring).
REQUIRED_TIME_FIELD = "start_time"

DEFAULT_PARENT_FIELD = "source_object_id"
DEFAULT_MERGE_TOLERANCE_MS = 2000


class MixpeekBuildTooOldError(RuntimeError):
    """Raised when server_build is missing or predates the moment_group fix."""


def _is_ancestor(repo: Path, older: str, newer: str) -> bool | None:
    """True/False from git ancestry in `repo`; None when git cannot resolve either sha."""
    def git(*args: str) -> subprocess.CompletedProcess:
        return subprocess.run(["git", "-C", str(repo), *args], capture_output=True,
                              text=True, timeout=30)

    try:
        for sha in (older, newer):
            if git("cat-file", "-e", f"{sha}^{{commit}}").returncode != 0:
                return None
        return git("merge-base", "--is-ancestor", older, newer).returncode == 0
    except (OSError, subprocess.SubprocessError):
        return None


def check_server_build(server_build: str | None, server_build_time: str | None = None,
                       mixpeek_repo: str | Path | None = None) -> str:
    """Raise unless server_build includes the moment_group fix; return how that was checked.

    Shas carry no order, so the check is git ancestry in a mixpeek checkout
    (`mixpeek_repo` or $MIXPEEK_REPO) when one can resolve the sha. Without one,
    `server_build_time` (the build commit's ISO time, UTC "Z" or a date) is compared
    with the fix's commit time, and a date on the fix's own day is refused as ambiguous.
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
    too_old = (f"server_build={server_build!r} predates {MIN_SUPPORTED_BUILD_SHA}, "
               "the moment_group units-bug fix (see module docstring). Re-run "
               "against a namespace on a newer build before publishing any "
               "number from this adapter.")
    if server_build.startswith(MIN_SUPPORTED_BUILD_SHA):
        return "exact"
    repo = mixpeek_repo or os.environ.get("MIXPEEK_REPO")
    if repo:
        verdict = _is_ancestor(Path(repo).expanduser(), MIN_SUPPORTED_BUILD_SHA, server_build)
        if verdict is True:
            return "ancestry"
        if verdict is False:
            raise MixpeekBuildTooOldError(too_old)
    if not server_build_time:
        raise MixpeekBuildTooOldError(
            f"cannot order server_build={server_build!r} against "
            f"{MIN_SUPPORTED_BUILD_SHA}: git shas have no order. Set MIXPEEK_REPO to a "
            "mixpeek checkout containing it, or pass server_build_time (its commit "
            "time, e.g. 2026-09-02T14:00:00Z).")
    when = server_build_time.strip()
    if len(when) == 10:  # a bare date
        if when == MIN_SUPPORTED_BUILD_TIME[:10]:
            raise MixpeekBuildTooOldError(
                f"server_build_time {when} is the day of the fix; pass the full commit "
                "time or set MIXPEEK_REPO so the order is certain.")
        if when < MIN_SUPPORTED_BUILD_TIME[:10]:
            raise MixpeekBuildTooOldError(too_old)
        return "commit-date"
    if not when.endswith("Z"):
        raise MixpeekBuildTooOldError("server_build_time must be UTC, ending in Z")
    if when < MIN_SUPPORTED_BUILD_TIME:
        raise MixpeekBuildTooOldError(too_old)
    return "commit-time"


def _build_stages(
    feature_uri: str, parent_field: str, merge_tolerance_ms: int
) -> list[dict[str, Any]]:
    """The verified feature_search -> moment_group stage pair (see module docstring)."""
    return [
        {
            "stage_name": "feature_search",
            "config": {
                "stage_id": "feature_search",
                "parameters": {
                    "searches": [
                        {
                            "feature_uri": feature_uri,
                            "query": {"input_mode": "text", "text": "{{INPUT.query}}"},
                            "top_k": 100,
                        }
                    ],
                    "final_top_k": 100,
                },
            },
        },
        {
            "stage_name": "moment_group",
            "config": {
                "stage_id": "moment_group",
                "parameters": {
                    "parent_field": parent_field,
                    "time_field": REQUIRED_TIME_FIELD,
                    "merge_tolerance_ms": merge_tolerance_ms,
                    "output_mode": "moments_only",
                },
            },
        },
    ]


@register_adapter("mixpeek")
class MixpeekAdapter(ModelAdapter):
    model_id: str = "mixpeek-pipeline"
    input_kind = "indexed-platform"

    def __init__(
        self,
        namespace: str | None = None,
        collection: str | None = None,
        server_build: str | None = None,
        server_build_time: str | None = None,
        feature_uri: str | None = None,
        retriever_id: str | None = None,
        parent_field: str = DEFAULT_PARENT_FIELD,
        merge_tolerance_ms: int = DEFAULT_MERGE_TOLERANCE_MS,
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
            server_build_time: Commit time of server_build (UTC, ending in Z),
                used to order it against the fix when no mixpeek checkout is
                available via $MIXPEEK_REPO.
            feature_uri: The text extractor's feature URI on the target
                collection (e.g. "mixpeek://text_extractor@v1/embedding").
                Required — depends on which extractor the corpus was actually
                ingested with, which this repo has no way to know in advance.
                See module docstring.
            retriever_id: Reuse an existing feature_search+moment_group
                retriever instead of creating one on first use.
            parent_field: Document field identifying the parent video, used
                to group hits into moments. Defaults to Mixpeek's own default
                for this stage ('source_object_id').
            merge_tolerance_ms: Gap tolerance for merging nearby hits into one
                moment. Defaults to the stage's own default (2000ms/2s).
        """
        self._build_check = check_server_build(server_build, server_build_time)
        if not feature_uri and not retriever_id:
            raise ValueError(
                "feature_uri (or a pre-built retriever_id) is required — see "
                "this module's docstring for why it can't be guessed."
            )
        self._namespace = namespace
        self._collection = collection
        self._server_build = server_build
        self._feature_uri = feature_uri
        self._retriever_id = retriever_id
        self._parent_field = parent_field
        self._merge_tolerance_ms = merge_tolerance_ms
        self._kwargs = kwargs

    def _ensure_retriever(self, client: Any) -> str:
        if self._retriever_id:
            return self._retriever_id
        created = client.retrievers.create(
            retriever_name=f"momentbench-{self._collection or 'default'}",
            collection_identifiers=[self._collection] if self._collection else [],
            input_schema={"query": {"type": "string", "required": True}},
            stages=_build_stages(
                self._feature_uri, self._parent_field, self._merge_tolerance_ms
            ),
            namespace=self._namespace,
        )
        self._retriever_id = created["retriever_id"]
        return self._retriever_id

    async def predict(
        self,
        query: Query,
        corpus_assets: list[dict[str, Any]],
        max_results: int = 5,
    ) -> Prediction:
        from mixpeek import Mixpeek

        client = Mixpeek(namespace=self._namespace)
        retriever_id = self._ensure_retriever(client)

        start = time.perf_counter()
        response = client.retrievers.execute(
            retriever_id,
            inputs={"query": query.text},
        )
        latency = (time.perf_counter() - start) * 1000

        moments = []
        for doc in response.get("documents", []):
            for m in doc.get("moments") or []:
                moments.append(
                    Moment(
                        asset_id=str(doc.get("parent_id", "")),
                        t_in=m["start_ms"] / 1000.0,
                        t_out=m["end_ms"] / 1000.0,
                        metadata={"score": doc.get("score"), "server_build": self._server_build},
                    )
                )
        moments.sort(key=lambda m: m.metadata.get("score") or 0.0, reverse=True)

        return Prediction(
            query_id=query.query_id,
            ranked_moments=moments[:max_results],
            model_id=self.model_id,
            latency_ms=latency,
            metadata={"server_build": self._server_build, "build_check": self._build_check,
                      "retriever_id": retriever_id},
        )
