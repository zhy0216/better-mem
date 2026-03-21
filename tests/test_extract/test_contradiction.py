from unittest.mock import AsyncMock, patch
from uuid import uuid4

import pytest

from src.extract.belief_updater import update_beliefs
from src.models.proposition import PropositionCreate, ScoredProposition
from datetime import datetime, timezone


def make_prop_create(text: str, embedding: list[float] | None = None) -> PropositionCreate:
    return PropositionCreate(
        canonical_text=text,
        proposition_type="observation",
        importance=0.5,
        embedding=embedding or [0.1] * 1024,
    )


def make_scored_prop(text: str, prop_id=None) -> ScoredProposition:
    return ScoredProposition(
        id=prop_id or uuid4(),
        canonical_text=text,
        proposition_type="observation",
        confidence=0.8,
        utility_importance=0.5,
        freshness_decay=0.02,
        access_count=0,
        belief_status="active",
        first_observed_at=datetime(2024, 1, 1, tzinfo=timezone.utc),
        last_observed_at=datetime(2024, 1, 1, tzinfo=timezone.utc),
        metadata={},
        tags=[],
        score=0.9,
        source="vector",
    )


@pytest.mark.asyncio
async def test_update_beliefs_contradiction_returns_new():
    old_id = uuid4()
    new_prop = make_prop_create("Zhang San hates Python.")
    similar = [make_scored_prop("Zhang San prefers Python.", prop_id=old_id)]

    llm_response = {
        "results": [
            {"existing_proposition_id": str(old_id), "relationship": "contradicts", "reason": "opposite sentiment"}
        ]
    }

    with (
        patch("src.extract.belief_updater.proposition_store.search_by_semantic_key", AsyncMock(return_value=[])),
        patch("src.extract.belief_updater.proposition_store.search_similar", AsyncMock(return_value=similar)),
        patch("src.extract.belief_updater.llm_service.complete_json", AsyncMock(return_value=llm_response)),
        patch("src.extract.belief_updater.proposition_store.add_evidence", AsyncMock()),
    ):
        genuinely_new = await update_beliefs([new_prop], user_id="user_001")

    # Contradictions still produce genuinely new propositions (the new one gets its own row)
    assert len(genuinely_new) == 1
    assert genuinely_new[0].canonical_text == "Zhang San hates Python."


@pytest.mark.asyncio
async def test_update_beliefs_none_found():
    new_prop = make_prop_create("Zhang San likes coffee.")

    with (
        patch("src.extract.belief_updater.proposition_store.search_by_semantic_key", AsyncMock(return_value=[])),
        patch("src.extract.belief_updater.proposition_store.search_similar", AsyncMock(return_value=[])),
    ):
        genuinely_new = await update_beliefs([new_prop], user_id="user_001")

    assert len(genuinely_new) == 1


@pytest.mark.asyncio
async def test_update_beliefs_skips_no_embedding():
    new_prop = PropositionCreate(canonical_text="Zhang San likes tea.", embedding=None)
    with (
        patch("src.extract.belief_updater.proposition_store.search_by_semantic_key", AsyncMock(return_value=[])),
        patch("src.extract.belief_updater.proposition_store.search_similar", AsyncMock(return_value=[])) as mock_search,
    ):
        genuinely_new = await update_beliefs([new_prop], user_id="user_001")
    mock_search.assert_not_called()
    assert len(genuinely_new) == 1
