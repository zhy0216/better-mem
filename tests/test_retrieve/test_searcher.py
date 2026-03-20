from datetime import datetime, timezone
from unittest.mock import AsyncMock, patch
from uuid import uuid4

import pytest

from src.models.fact import ScoredFact
from src.retrieve.searcher import hybrid_search


def make_scored_fact(content: str, score: float = 0.9, source: str = "vector") -> ScoredFact:
    return ScoredFact(
        id=uuid4(),
        content=content,
        fact_type="observation",
        occurred_at=datetime(2024, 3, 14, tzinfo=timezone.utc),
        importance=0.5,
        metadata={},
        tags=[],
        score=score,
        source=source,
    )


@pytest.mark.asyncio
async def test_hybrid_search_merges_results():
    vec_facts = [make_scored_fact("Tokyo trip plan", 0.95, "vector")]
    kw_facts = [make_scored_fact("travel budget", 0.8, "keyword")]

    with (
        patch("src.retrieve.searcher.embedding_service.embed", AsyncMock(return_value=[0.1] * 1024)),
        patch("src.retrieve.searcher.fact_store.vector_search", AsyncMock(return_value=vec_facts)),
        patch("src.retrieve.searcher.fact_store.keyword_search", AsyncMock(return_value=kw_facts)),
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
    vec_fact = ScoredFact(
        id=shared_id, content="shared fact", fact_type="observation",
        occurred_at=datetime(2024, 3, 14, tzinfo=timezone.utc),
        importance=0.5, metadata={}, tags=[], score=0.9, source="vector",
    )
    kw_fact = ScoredFact(
        id=shared_id, content="shared fact", fact_type="observation",
        occurred_at=datetime(2024, 3, 14, tzinfo=timezone.utc),
        importance=0.5, metadata={}, tags=[], score=0.8, source="keyword",
    )

    with (
        patch("src.retrieve.searcher.embedding_service.embed", AsyncMock(return_value=[0.1] * 1024)),
        patch("src.retrieve.searcher.fact_store.vector_search", AsyncMock(return_value=[vec_fact])),
        patch("src.retrieve.searcher.fact_store.keyword_search", AsyncMock(return_value=[kw_fact])),
    ):
        results, _ = await hybrid_search(query="shared fact", user_id="user_001")

    assert len(results) == 1
    assert results[0].id == shared_id
