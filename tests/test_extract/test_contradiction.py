from unittest.mock import AsyncMock, patch
from uuid import uuid4

import pytest

from src.extract.contradiction import detect_contradictions
from src.models.fact import FactCreate, ScoredFact
from datetime import datetime, timezone


def make_fact_create(content: str, embedding: list[float] | None = None) -> FactCreate:
    return FactCreate(
        content=content,
        fact_type="observation",
        importance=0.5,
        embedding=embedding or [0.1] * 1024,
    )


def make_scored_fact(content: str, fact_id=None) -> ScoredFact:
    return ScoredFact(
        id=fact_id or uuid4(),
        content=content,
        fact_type="observation",
        occurred_at=datetime(2024, 1, 1, tzinfo=timezone.utc),
        importance=0.5,
        metadata={},
        tags=[],
        score=0.9,
        source="vector",
    )


@pytest.mark.asyncio
async def test_detect_contradictions_found():
    old_id = uuid4()
    new_fact = make_fact_create("Zhang San hates Python.")
    similar = [make_scored_fact("Zhang San prefers Python.", fact_id=old_id)]

    llm_response = {
        "results": [
            {"existing_fact_id": str(old_id), "relationship": "contradicts", "reason": "opposite sentiment"}
        ]
    }

    with (
        patch("src.extract.contradiction.fact_store.search_similar", AsyncMock(return_value=similar)),
        patch("src.extract.contradiction.llm_service.complete_json", AsyncMock(return_value=llm_response)),
    ):
        pairs = await detect_contradictions([new_fact], user_id="user_001")

    assert len(pairs) == 1
    assert pairs[0].old_fact_id == old_id
    assert pairs[0].relation == "supersedes"


@pytest.mark.asyncio
async def test_detect_contradictions_none_found():
    new_fact = make_fact_create("Zhang San likes coffee.")

    with patch("src.extract.contradiction.fact_store.search_similar", AsyncMock(return_value=[])):
        pairs = await detect_contradictions([new_fact], user_id="user_001")

    assert pairs == []


@pytest.mark.asyncio
async def test_detect_contradictions_skips_no_embedding():
    new_fact = FactCreate(content="Zhang San likes tea.", embedding=None)
    with patch("src.extract.contradiction.fact_store.search_similar", AsyncMock(return_value=[])) as mock_search:
        pairs = await detect_contradictions([new_fact], user_id="user_001")
    mock_search.assert_not_called()
    assert pairs == []
