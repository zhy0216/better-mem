from datetime import datetime, timezone
from unittest.mock import AsyncMock, patch
from uuid import uuid4

import pytest

from src.models.proposition import AssembledContext, ScoredProposition
from src.retrieve.assembler import assemble_context


def make_candidates(n: int = 2) -> list[ScoredProposition]:
    return [
        ScoredProposition(
            id=uuid4(),
            canonical_text=f"Proposition number {i}",
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
            score=0.9 - i * 0.05,
            source="vector",
        )
        for i in range(n)
    ]


@pytest.mark.asyncio
async def test_assemble_context_returns_assembled():
    candidates = make_candidates(2)
    ids = [str(c.id) for c in candidates]
    llm_response = {
        "context": "Zhang San is planning a Tokyo trip.",
        "selected_proposition_ids": ids,
        "confidence": 0.9,
        "information_gaps": [],
    }

    with patch("src.retrieve.assembler.llm_service.complete_json", AsyncMock(return_value=llm_response)):
        result = await assemble_context("Tokyo plans", candidates)

    assert isinstance(result, AssembledContext)
    assert result.context == "Zhang San is planning a Tokyo trip."
    assert result.confidence == 0.9


@pytest.mark.asyncio
async def test_assemble_context_falls_back_on_llm_error():
    candidates = make_candidates(3)

    with patch(
        "src.retrieve.assembler.llm_service.complete_json",
        AsyncMock(side_effect=RuntimeError("LLM down")),
    ):
        result = await assemble_context("Tokyo plans", candidates)

    assert result.confidence == 0.0
    assert len(result.information_gaps) > 0
    assert result.context != ""
