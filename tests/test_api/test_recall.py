from datetime import datetime, timezone
from unittest.mock import AsyncMock, patch
from uuid import uuid4

import pytest

from src.models.proposition import ScoredProposition


def make_scored_prop(text: str = "some proposition") -> ScoredProposition:
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
        score=0.9,
        source="vector",
    )


@pytest.mark.asyncio
async def test_recall_raw(client):
    candidates = [make_scored_prop("Zhang San likes Tokyo.")]

    with (
        patch("src.api.recall.cache.get_recall_cache", AsyncMock(return_value=None)),
        patch("src.api.recall.cache.set_recall_cache", AsyncMock()),
        patch("src.api.recall.searcher.hybrid_search", AsyncMock(return_value=(candidates, 42.0))),
        patch("src.api.recall.proposition_store.track_access", AsyncMock()),
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
    assert "propositions" in data
    assert data["total_candidates"] == 1
    assert data["search_time_ms"] == 42.0


@pytest.mark.asyncio
async def test_recall_assembled(client):
    from src.models.proposition import AssembledContext

    candidates = [make_scored_prop("Zhang San plans Tokyo trip.")]
    assembled = AssembledContext(
        context="Zhang San is planning a Tokyo trip.",
        selected_proposition_ids=[str(candidates[0].id)],
        confidence=0.9,
        information_gaps=[],
    )

    with (
        patch("src.api.recall.cache.get_recall_cache", AsyncMock(return_value=None)),
        patch("src.api.recall.cache.set_recall_cache", AsyncMock()),
        patch("src.api.recall.searcher.hybrid_search", AsyncMock(return_value=(candidates, 30.0))),
        patch("src.api.recall.proposition_store.track_access", AsyncMock()),
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
        "propositions": [],
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
