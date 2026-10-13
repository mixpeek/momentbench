"""What a run was, precisely enough to tell whether two runs can be compared.

Recorded on every AgenticRun (audit finding 24). `compatibility_key` hashes the inputs
that define the task (the selected queries and their text, the verified corpus
bytes, transcripts, prompt, tools, budget and conditions); two runs with the same
key ran the same benchmark.
"""

from __future__ import annotations

import hashlib
import json
import subprocess
from importlib import metadata
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
PACKAGES = ("momentbench", "anthropic", "openai", "google-genai", "claude-agent-sdk",
            "inspect-ai", "pillow")


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def text_sha(text: str) -> str:
    return sha256_bytes(text.encode("utf-8"))[:16]


def file_sha(path: Path | None) -> str | None:
    return sha256_bytes(path.read_bytes()) if path and path.is_file() else None


def transcripts_sha(transcripts: dict[str, list[dict[str, Any]]]) -> str:
    return sha256_bytes(json.dumps(transcripts, sort_keys=True).encode())[:16]


def git_state() -> dict[str, Any]:
    def git(*args: str) -> str:
        try:
            return subprocess.run(["git", *args], cwd=ROOT, capture_output=True, text=True,
                                  timeout=10).stdout.strip()
        except (OSError, subprocess.SubprocessError):
            return ""

    return {"commit": git("rev-parse", "HEAD") or None,
            "dirty": bool(git("status", "--porcelain", "--untracked-files=no"))}


def package_versions() -> dict[str, str]:
    out = {}
    for name in PACKAGES:
        try:
            out[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            continue
    return out


def queries_sha(queries: list[Any]) -> str:
    """Hash of exactly the queries a run was given: ids and request text, in id order."""
    rows = sorted((q.query_id, q.text) for q in queries)
    return text_sha(json.dumps(rows))


def task_identity(*, queries: list[Any], receipt: dict[str, str],
                  transcripts: dict[str, list[dict[str, Any]]], system_prompt: str,
                  tools: list[dict[str, Any]], budget: dict[str, Any], modalities: str,
                  anonymize_titles: bool = False) -> dict[str, Any]:
    """What defines the task. Two runs with the same compatibility_key ran the same
    benchmark: the same selected queries and wording, the same verified video bytes,
    transcripts, prompt, tools, budget and conditions."""
    task = {
        "queries_sha": queries_sha(queries),
        "num_queries": len(queries),
        "corpus_sha": text_sha(json.dumps(sorted(receipt.items()))),
        "transcripts_sha": transcripts_sha(transcripts),
        "system_prompt_sha": text_sha(system_prompt),
        "tools_sha": text_sha(json.dumps(tools, sort_keys=True)),
        "budget": budget,
        "modalities": modalities,
        "anonymize_titles": anonymize_titles,
    }
    return {**task, "compatibility_key": text_sha(json.dumps(task, sort_keys=True))}


def environment_identity(*, queries_path: Path | None = None,
                         manifest_path: Path | None = None, provider: str | None = None,
                         endpoint: str | None = None, pricing_as_of: str | None = None,
                         query_filter: str | None = None) -> dict[str, Any]:
    """Where a run came from: source files, provider, code and packages."""
    return {
        "queries_file_sha256": file_sha(queries_path),
        "manifest_sha256": file_sha(manifest_path),
        "query_filter": query_filter,
        "provider": provider,
        "endpoint": endpoint,
        "pricing_as_of": pricing_as_of,
        "git": git_state(),
        "packages": package_versions(),
    }
