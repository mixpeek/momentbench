"""Core data types for MomentBench."""

from __future__ import annotations

from enum import IntEnum
from typing import Any

from pydantic import BaseModel, Field


class QueryTier(IntEnum):
    SINGLE_VISUAL = 1
    CROSS_MODAL = 2
    CORPUS_HARD_NEGATIVE = 3
    TEMPORAL_REASONING = 4


class Vertical(str):
    CIVIC = "civic"
    AEROSPACE = "aerospace"
    FILM = "film"


class Moment(BaseModel):
    asset_id: str
    t_in: float = Field(ge=0, description="Start time in seconds")
    t_out: float = Field(gt=0, description="End time in seconds")
    metadata: dict[str, Any] = Field(default_factory=dict)

    @property
    def duration(self) -> float:
        return self.t_out - self.t_in


class Query(BaseModel):
    query_id: str
    text: str
    tier: QueryTier
    vertical: str
    gold_moments: list[Moment] = Field(min_length=1)
    hard_negatives: list[Moment] = Field(default_factory=list)
    is_public: bool = True
    metadata: dict[str, Any] = Field(default_factory=dict)


class Prediction(BaseModel):
    query_id: str
    ranked_moments: list[Moment]
    model_id: str
    latency_ms: float | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class EvalResult(BaseModel):
    query_id: str
    model_id: str
    hit_at_1: bool
    hit_at_5: bool
    best_temporal_iou: float
    tier: QueryTier
    vertical: str


class BenchmarkRun(BaseModel):
    run_id: str
    model_id: str
    results: list[EvalResult]
    config: dict[str, Any] = Field(default_factory=dict)
    timestamp: str | None = None
