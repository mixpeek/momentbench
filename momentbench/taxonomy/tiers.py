"""4-tier query taxonomy for MomentBench."""

from __future__ import annotations

from dataclasses import dataclass

from momentbench.types import QueryTier


@dataclass
class TierDefinition:
    tier: QueryTier
    name: str
    description: str
    examples: list[str]
    expected_model_performance: str
    target_percentage: float


TIER_DEFINITIONS: list[TierDefinition] = [
    TierDefinition(
        tier=QueryTier.SINGLE_VISUAL,
        name="Single-condition visual",
        description=(
            "Queries requiring identification of a single visual event or object. "
            "Models should perform reasonably well on these — a bench where everyone "
            "scores <10% looks broken."
        ),
        examples=[
            "A red vehicle enters the frame",
            "A person stands at a podium",
            "The rocket lifts off the pad",
        ],
        expected_model_performance=">80% for frontier models",
        target_percentage=0.40,
    ),
    TierDefinition(
        tier=QueryTier.CROSS_MODAL,
        name="Cross-modal joint",
        description=(
            "Queries requiring joint reasoning across modalities — visual + audio, "
            "visual + text (OCR/ASR), or audio + temporal context."
        ),
        examples=[
            "The room goes quiet just before the chairman bangs the gavel",
            "The speaker says 'objection' while pointing at the screen",
            "Engine ignition sound followed by visible thrust plume",
        ],
        expected_model_performance="50-70% for frontier models",
        target_percentage=0.35,
    ),
    TierDefinition(
        tier=QueryTier.CORPUS_HARD_NEGATIVE,
        name="Corpus-scale hard negatives",
        description=(
            "Queries where the target moment is visually/semantically similar to many "
            "other moments in the corpus. Requires corpus-level discrimination — "
            "the same official testifies in 14 hearings; exactly one satisfies the query."
        ),
        examples=[
            "The second time Senator Johnson asks about budget allocation in the 2024 hearings",
            "The Soyuz launch (not the Atlas V launch from the same week)",
            "The scene where the robot reaches toward the girl (not the similar sequel scene)",
        ],
        expected_model_performance="<35% for frontier models",
        target_percentage=0.15,
    ),
    TierDefinition(
        tier=QueryTier.TEMPORAL_REASONING,
        name="Temporal reasoning",
        description=(
            "Queries requiring understanding of temporal ordering, relative positioning, "
            "or sequence reasoning across the corpus."
        ),
        examples=[
            "The question immediately following the second recess",
            "The last speaker before the vote is called",
            "The moment 30 seconds after stage separation",
        ],
        expected_model_performance="<25% for frontier models",
        target_percentage=0.10,
    ),
]


def classify_query_tier(
    has_multi_modal: bool = False,
    has_hard_negatives: bool = False,
    has_temporal_reasoning: bool = False,
) -> QueryTier:
    """Classify a query into the appropriate tier based on its characteristics."""
    if has_temporal_reasoning:
        return QueryTier.TEMPORAL_REASONING
    if has_hard_negatives:
        return QueryTier.CORPUS_HARD_NEGATIVE
    if has_multi_modal:
        return QueryTier.CROSS_MODAL
    return QueryTier.SINGLE_VISUAL
