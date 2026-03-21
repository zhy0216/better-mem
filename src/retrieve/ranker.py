import math
from datetime import datetime, timezone
from uuid import UUID

from src.models.proposition import ScoredProposition

# Weights per §8 of proposition_plan.md
W_SEMANTIC_RELEVANCE = 0.40
W_BELIEF_CONFIDENCE = 0.25
W_UTILITY_IMPORTANCE = 0.20
W_FRESHNESS = 0.10
W_ACCESS_BOOST = 0.05


def reciprocal_rank_fusion(
    vector_results: list[ScoredProposition],
    keyword_results: list[ScoredProposition],
    k: int = 60,
) -> list[ScoredProposition]:
    """Merge two ranked lists using RRF, then re-rank with weighted-sum scoring."""
    rrf_scores: dict[UUID, float] = {}
    prop_map: dict[UUID, ScoredProposition] = {}

    for rank, prop in enumerate(vector_results):
        rrf_scores[prop.id] = rrf_scores.get(prop.id, 0.0) + 1.0 / (k + rank + 1)
        prop_map[prop.id] = prop

    for rank, prop in enumerate(keyword_results):
        rrf_scores[prop.id] = rrf_scores.get(prop.id, 0.0) + 1.0 / (k + rank + 1)
        if prop.id not in prop_map:
            prop_map[prop.id] = prop
        else:
            existing = prop_map[prop.id]
            if prop.source != existing.source:
                combined = ScoredProposition(
                    id=existing.id,
                    canonical_text=existing.canonical_text,
                    proposition_type=existing.proposition_type,
                    semantic_key=existing.semantic_key,
                    confidence=existing.confidence,
                    utility_importance=existing.utility_importance,
                    freshness_decay=existing.freshness_decay,
                    access_count=existing.access_count,
                    belief_status=existing.belief_status,
                    first_observed_at=existing.first_observed_at,
                    last_observed_at=existing.last_observed_at,
                    metadata=existing.metadata,
                    tags=existing.tags,
                    score=rrf_scores[prop.id],
                    source="hybrid",
                )
                prop_map[prop.id] = combined

    now = datetime.now(tz=timezone.utc)
    # Normalize RRF scores to 0-1
    max_rrf = max(rrf_scores.values()) if rrf_scores else 1.0
    if max_rrf == 0:
        max_rrf = 1.0

    merged = []
    for pid, rrf_score in sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True):
        prop = prop_map[pid]
        semantic_relevance = rrf_score / max_rrf

        final_score = compute_retrieval_score(
            semantic_relevance=semantic_relevance,
            confidence=prop.confidence,
            utility_importance=prop.utility_importance,
            freshness_decay=prop.freshness_decay,
            last_observed_at=prop.last_observed_at or prop.first_observed_at,
            access_count=prop.access_count,
            now=now,
        )
        merged.append(
            ScoredProposition(
                id=prop.id,
                canonical_text=prop.canonical_text,
                proposition_type=prop.proposition_type,
                semantic_key=prop.semantic_key,
                confidence=prop.confidence,
                utility_importance=prop.utility_importance,
                freshness_decay=prop.freshness_decay,
                access_count=prop.access_count,
                belief_status=prop.belief_status,
                first_observed_at=prop.first_observed_at,
                last_observed_at=prop.last_observed_at,
                metadata=prop.metadata,
                tags=prop.tags,
                score=final_score,
                source=prop.source,
            )
        )
    merged.sort(key=lambda p: p.score, reverse=True)
    return merged


def compute_retrieval_score(
    semantic_relevance: float,
    confidence: float,
    utility_importance: float,
    freshness_decay: float,
    last_observed_at: datetime | None,
    access_count: int,
    now: datetime | None = None,
) -> float:
    """Weighted-sum retrieval score per §8."""
    if now is None:
        now = datetime.now(tz=timezone.utc)

    # Freshness factor
    if last_observed_at is not None:
        if last_observed_at.tzinfo is None:
            last_observed_at = last_observed_at.replace(tzinfo=timezone.utc)
        age_days = (now - last_observed_at).total_seconds() / 86400
        freshness_factor = math.exp(-freshness_decay * age_days)
    else:
        freshness_factor = 0.5

    # Access boost: 1.0 + 0.1*log1p(n), normalized to ~0-1 range
    raw_access = 1.0 + 0.1 * math.log1p(access_count)
    access_boost = min(raw_access / 2.0, 1.0)  # rough normalization

    return (
        W_SEMANTIC_RELEVANCE * semantic_relevance
        + W_BELIEF_CONFIDENCE * confidence
        + W_UTILITY_IMPORTANCE * utility_importance
        + W_FRESHNESS * freshness_factor
        + W_ACCESS_BOOST * access_boost
    )
