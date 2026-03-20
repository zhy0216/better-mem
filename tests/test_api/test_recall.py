from datetime import datetime, timezone
from unittest.mock import AsyncMock, patch
from uuid import uuid4

import pytest

from src.models.fact import ScoredFact


def make_scored_fact(content: str = "some fact") -> ScoredFact:
    return ScoredFact(
        id=uuid4(),
        content=content,
        fact_type="observation",
        occurred_at=datetime(2024, 3, 14, tzinfo=timezone.utc),
        importance=0.5,
        metadata={},
        tags=[],
        score=0.9,
        source="vector",
    )


@pytest.mark.asyncio
async def test_recall_raw(client):
    candidates = [make_scored_fact("Zhang San likes Tokyo.")]

    with (
        patch("src.api.recall.cache.get_recall_cache", AsyncMock(return_value=None)),
        patch("src.api.recall.cache.set_recall_cache", AsyncMock()),
        patch("src.api.recall.searcher.hybrid_search", AsyncMock(return_value=(candidates, 42.0))),
        patch("src.api.recall.fact_store.track_access", AsyncMock()),
        patch("src.api.recall.profile_store.get", AsyncMock(return_value=None)),
    ):
        resp = await client.post(
            "/v1/recall",
            json={
                "user_id": "user_001",
                "query": "Tokyo plans",
                "assemble": False,
                "include_profile": False,
            },
        )

    assert resp.status_code == 200
    data = resp.json()
    assert "facts" in data
    assert data["total_candidates"] == 1
    assert data["search_time_ms"] == 42.0


@pytest.mark.asyncio
async def test_recall_assembled(client):
    from src.models.api import AssembledContext

    candidates = [make_scored_fact("Zhang San plans Tokyo trip.")]
    assembled = AssembledContext(
        context="Zhang San is planning a Tokyo trip.",
        selected_fact_ids=[str(candidates[0].id)],
        confidence=0.9,
        information_gaps=[],
    )

    with (
        patch("src.api.recall.cache.get_recall_cache", AsyncMock(return_value=None)),
        patch("src.api.recall.cache.set_recall_cache", AsyncMock()),
        patch("src.api.recall.searcher.hybrid_search", AsyncMock(return_value=(candidates, 30.0))),
        patch("src.api.recall.fact_store.track_access", AsyncMock()),
        patch("src.api.recall.profile_store.get", AsyncMock(return_value=None)),
        patch("src.api.recall.assembler.assemble_context", AsyncMock(return_value=assembled)),
    ):
        resp = await client.post(
            "/v1/recall",
            json={
                "user_id": "user_001",
                "query": "Tokyo plans",
                "assemble": True,
                "include_profile": True,
            },
        )

    assert resp.status_code == 200
    data = resp.json()
    assert data["context"] == "Zhang San is planning a Tokyo trip."
    assert data["total_candidates"] == 1


@pytest.mark.asyncio
async def test_recall_returns_cache(client):
    cached = {
        "facts": [],
        "total_candidates": 0,
        "search_time_ms": 10.0,
    }

    with patch("src.api.recall.cache.get_recall_cache", AsyncMock(return_value=cached)):
        resp = await client.post(
            "/v1/recall",
            json={
                "user_id": "user_001",
                "query": "anything",
                "assemble": False,
            },
        )

    assert resp.status_code == 200
    assert resp.json()["total_candidates"] == 0
