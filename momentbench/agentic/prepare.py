"""One corpus preparation step under every entry point (CLI, Python API, Inspect).

It loads the manifest, hashes every asset at the path frames will be read from, and
pins the corpus to those files. A run is reportable only on a fully verified corpus,
whichever route started it (audit 2, findings R3, R5, R6).
"""

from __future__ import annotations

from pathlib import Path

from momentbench.agentic.environment import LocalCorpus
from momentbench.corpus.verify import AssetCheck, verify_checks
from momentbench.runner import load_corpus_assets


class CorpusNotVerifiedError(RuntimeError):
    """Some manifest asset is missing or does not match its hash."""

    def __init__(self, failures: list[AssetCheck], total: int):
        self.failures = failures
        detail = "; ".join(f"{c.asset_id}: {', '.join(c.problems)}" for c in failures[:5])
        super().__init__(
            f"{len(failures)} of {total} assets failed verification ({detail}). "
            "Run corpus-download, or prepare with partial=True for a non-reportable "
            "debug run.")


def prepare_corpus(manifest: Path, corpus_dir: Path | None, *,
                   transcripts_dir: Path | None = None, no_transcripts: bool = False,
                   anonymize_titles: bool = False,
                   partial: bool = False) -> tuple[LocalCorpus, list[AssetCheck]]:
    """Build a LocalCorpus pinned to its verified files. Returns (corpus, failures).

    Raises CorpusNotVerifiedError when any asset fails, unless partial is set; a
    partial corpus serves only its verified assets and its runs are not reportable.
    """
    assets = load_corpus_assets(Path(manifest))
    corpus = LocalCorpus(assets, corpus_dir, transcripts={} if no_transcripts else None,
                         transcripts_dir=transcripts_dir, anonymize_titles=anonymize_titles)
    if corpus_dir is None:
        failures = [AssetCheck(a["asset_id"], None, problems=("no corpus_dir given",))
                    for a in assets]
    else:
        checks = verify_checks(assets, Path(corpus_dir),
                               Path(corpus_dir) / ".momentbench-verify-cache.json")
        failures = [c for c in checks if not c.ok]
        ok = [c for c in checks if c.ok]
        corpus.pin_files({c.asset_id: c.path for c in ok},
                         {c.asset_id: c.sha256 for c in ok})
    if failures and not partial:
        raise CorpusNotVerifiedError(failures, len(assets))
    if corpus_dir is None:
        corpus.pin_files({}, {})
    return corpus, failures
