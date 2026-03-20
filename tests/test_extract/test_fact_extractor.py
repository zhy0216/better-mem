from datetime import datetime, timezone
from unittest.mock import AsyncMock, patch
from uuid import uuid4

import pytest

from src.extract.fact_extractor import extract_facts
from src.models.message import MessageBuffer


def make_buffer(content_text: str, speaker_name: str = "Zhang San") -> MessageBuffer:
    ts = datetime.now(tz=timezone.utc)
    return MessageBuffer(
        id=uuid4(),
        tenant_id="default",
        group_id="group_abc",
        user_id="user_001",
        content={
            "role": "user",
            "speaker_id": "user_001",
            "speaker_name": speaker_name,
            "content": content_text,
            "timestamp": ts.isoformat(),
        },
        status="pending",
        batch_id=None,
        created_at=ts,
    )


@pytest.mark.asyncio
async def test_extract_facts_returns_list():
    messages = [
        make_buffer("I'm planning a trip to Tokyo next month."),
        make_buffer("My budget is about $3000."),
    ]
    mock_llm_response = {
        "facts": [
            {
                "content": "Zhang San plans to visit Tokyo next month.",
                "fact_type": "plan",
                "importance": 0.7,
                "tags": ["travel", "tokyo"],
            }
        ]
    }
    with (
        patch("src.extract.fact_extractor.llm_service.complete_json", AsyncMock(return_value=mock_llm_response)),
        patch("src.extract.fact_extractor.embedding_service.embed_batch", AsyncMock(return_value=[[0.1] * 1024])),
    ):
        facts = await extract_facts(messages, user_id="user_001")

    assert len(facts) == 1
    assert facts[0].fact_type == "plan"
    assert facts[0].importance == 0.7
    assert facts[0].embedding is not None
    assert len(facts[0].embedding) == 1024


@pytest.mark.asyncio
async def test_extract_facts_empty_on_llm_failure():
    messages = [make_buffer("Hello there.")]
    with patch(
        "src.extract.fact_extractor.llm_service.complete_json",
        AsyncMock(side_effect=RuntimeError("LLM unavailable")),
    ):
        facts = await extract_facts(messages, user_id="user_001")

    assert facts == []
