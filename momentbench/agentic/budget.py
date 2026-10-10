"""Per-query budget and usage accounting for the agentic track."""

from __future__ import annotations

import json
from pathlib import Path

from pydantic import BaseModel


class BudgetExceededError(Exception):
    """Raised when a tool call would exceed the per-query budget."""


class Budget(BaseModel):
    """Hard limits for one query. The harness enforces these, the model is told them."""

    max_steps: int = 30
    max_frames: int = 64
    max_tool_calls: int = 60


class Usage(BaseModel):
    steps: int = 0
    tool_calls: int = 0
    frames: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    cache_read_tokens: int = 0
    cache_write_tokens: int = 0

    def add(self, other: Usage) -> None:
        for name in type(self).model_fields:
            setattr(self, name, getattr(self, name) + getattr(other, name))


class Pricing(BaseModel):
    """USD per million tokens. Loaded from a file so prices are dated and auditable."""

    input_per_mtok: float
    output_per_mtok: float
    cache_read_per_mtok: float | None = None
    cache_write_per_mtok: float | None = None

    def cost_usd(self, usage: Usage) -> float:
        cache_read = self.cache_read_per_mtok
        cache_write = self.cache_write_per_mtok
        return (
            usage.input_tokens * self.input_per_mtok
            + usage.output_tokens * self.output_per_mtok
            + usage.cache_read_tokens
            * (self.input_per_mtok if cache_read is None else cache_read)
            + usage.cache_write_tokens
            * (self.input_per_mtok if cache_write is None else cache_write)
        ) / 1_000_000


def load_pricing(path: Path, model_id: str) -> Pricing | None:
    """Look up a model's prices. Returns None when the file does not list the model."""
    table = json.loads(path.read_text()).get("models", {})
    entry = table.get(model_id)
    return Pricing.model_validate(entry) if entry else None
