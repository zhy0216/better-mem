import math
from datetime import datetime, timezone
from uuid import UUID

from src.models.fact import ScoredFact


def reciprocal_rank_fusion(
    vector_results: list[ScoredFact],
    keyword_results: list[ScoredFact],
    k: int = 60,
) -> list[ScoredFact]:
    """Merge two ranked lists using Reciprocal Rank Fusion, then re-rank by relevance."""
    scores: dict[UUID, float] = {}
    fact_map: dict[UUID, ScoredFact] = {}

    for rank, fact in enumerate(vector_results):
        scores[fact.id] = scores.get(fact.id, 0.0) + 1.0 / (k + rank + 1)
        fact_map[fact.id] = fact

    for rank, fact in enumerate(keyword_results):
        scores[fact.id] = scores.get(fact.id, 0.0) + 1.0 / (k + rank + 1)
        if fact.id not in fact_map:
            fact_map[fact.id] = fact
        else:
            existing = fact_map[fact.id]
            if fact.source != existing.source:
                combined = ScoredFact(
                    id=existing.id,
                    content=existing.content,
                    fact_type=existing.fact_type,
                    occurred_at=existing.occurred_at,
                    importance=existing.importance,
                    decay_rate=existing.decay_rate,
                    access_count=existing.access_count,
                    metadata=existing.metadata,
                    tags=existing.tags,
                    score=scores[fact.id],
                    source="hybrid",
                )
                fact_map[fact.id] = combined

    now = datetime.now(tz=timezone.utc)
    merged = []
    for fid, rrf_score in sorted(scores.items(), key=lambda x: x[1], reverse=True):
        fact = fact_map[fid]
        relevance = compute_relevance_score(
            importance=fact.importance,
            decay_rate=fact.decay_rate,
            occurred_at=fact.occurred_at,
            access_count=fact.access_count,
            now=now,
        )
        merged.append(
            ScoredFact(
                id=fact.id,
                content=fact.content,
                fact_type=fact.fact_type,
                occurred_at=fact.occurred_at,
                importance=fact.importance,
                decay_rate=fact.decay_rate,
                access_count=fact.access_count,
                metadata=fact.metadata,
                tags=fact.tags,
                score=rrf_score * relevance,
                source=fact.source,
            )
        )
    merged.sort(key=lambda f: f.score, reverse=True)
    return merged


def compute_relevance_score(
    importance: float,
    decay_rate: float,
    occurred_at: datetime,
    access_count: int,
    now: datetime | None = None,
) -> float:
    if now is None:
        now = datetime.now(tz=timezone.utc)
    if occurred_at.tzinfo is None:
        occurred_at = occurred_at.replace(tzinfo=timezone.utc)
    age_days = (now - occurred_at).total_seconds() / 86400
    recency_factor = math.exp(-decay_rate * age_days)
    access_boost = 1.0 + 0.1 * math.log1p(access_count)
    return importance * recency_factor * access_boost
