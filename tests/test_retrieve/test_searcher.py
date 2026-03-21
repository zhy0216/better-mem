from datetime import datetime, timezone
from unittest.mock import AsyncMock, patch
from uuid import uuid4

import pytest

from src.models.proposition import ScoredProposition
from src.retrieve.searcher import hybrid_search


def make_scored_prop(text: str, score: float = 0.9, source: str = "vector") -> ScoredProposition:
    return ScoredProposition(
        id=uuid4(),
        canonical_text=text,
        proposition_type="observation",
        confidence=0.8,
        utility_importance=0.5,
        freshness_decay=0.01,
        access_count=0,
        belief_status="active",
        first_observed_at=datetime(2024, 3, 14, tzinfo=timezone.utc),
        last_observed_at=datetime(2024, 3, 14, tzinfo=timezone.utc),
        metadata={},
        tags=[],
        score=score,
        source=source,
    )


@pytest.mark.asyncio
async def test_hybrid_search_merges_results():
    vec_props = [make_scored_prop("Tokyo trip plan", 0.95, "vector")]
    kw_props = [make_scored_prop("travel budget", 0.8, "keyword")]

    with (
        patch("src.retrieve.searcher.embedding_service.embed", AsyncMock(return_value=[0.1] * 1024)),
        patch("src.retrieve.searcher.proposition_store.vector_search", AsyncMock(return_value=vec_props)),
        patch("src.retrieve.searcher.proposition_store.keyword_search", AsyncMock(return_value=kw_props)),
    ):
        results, elapsed_ms = await hybrid_search(
            query="Tokyo travel plans",
            user_id="user_001",
        )

    assert len(results) == 2
    assert elapsed_ms >= 0


@pytest.mark.asyncio
async def test_hybrid_search_deduplicates():
    shared_id = uuid4()
    vec_prop = ScoredProposition(
        id=shared_id, canonical_text="shared proposition", proposition_type="observation",
        confidence=0.8, utility_importance=0.5, freshness_decay=0.01,
        access_count=0, belief_status="active",
        first_observed_at=datetime(2024, 3, 14, tzinfo=timezone.utc),
        last_observed_at=datetime(2024, 3, 14, tzinfo=timezone.utc),
        metadata={}, tags=[], score=0.9, source="vector",
    )
    kw_prop = ScoredProposition(
        id=shared_id, canonical_text="shared proposition", proposition_type="observation",
        confidence=0.8, utility_importance=0.5, freshness_decay=0.01,
        access_count=0, belief_status="active",
        first_observed_at=datetime(2024, 3, 14, tzinfo=timezone.utc),
        last_observed_at=datetime(2024, 3, 14, tzinfo=timezone.utc),
        metadata={}, tags=[], score=0.8, source="keyword",
    )

    with (
        patch("src.retrieve.searcher.embedding_service.embed", AsyncMock(return_value=[0.1] * 1024)),
        patch("src.retrieve.searcher.proposition_store.vector_search", AsyncMock(return_value=[vec_prop])),
        patch("src.retrieve.searcher.proposition_store.keyword_search", AsyncMock(return_value=[kw_prop])),
    ):
        results, _ = await hybrid_search(query="shared proposition", user_id="user_001")

    assert len(results) == 1
    assert results[0].id == shared_id
