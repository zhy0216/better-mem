from datetime import datetime, timezone
from unittest.mock import AsyncMock, patch
from uuid import uuid4

import pytest

from src.models.api import AssembledContext
from src.models.fact import ScoredFact
from src.retrieve.assembler import assemble_context


def make_candidates(n: int = 2) -> list[ScoredFact]:
    return [
        ScoredFact(
            id=uuid4(),
            content=f"Fact number {i}",
            fact_type="observation",
            occurred_at=datetime(2024, 3, 14, tzinfo=timezone.utc),
            importance=0.5,
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
        "selected_fact_ids": ids,
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
