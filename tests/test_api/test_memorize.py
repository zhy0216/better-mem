from datetime import datetime, timezone
from unittest.mock import AsyncMock, patch
from uuid import uuid4

import pytest


@pytest.mark.asyncio
async def test_memorize_async_mode(client):
    batch_id = uuid4()
    payload = {
        "tenant_id": "default",
        "group_id": "group_abc",
        "messages": [
            {
                "role": "user",
                "speaker_id": "user_001",
                "speaker_name": "Zhang San",
                "content": "I plan to visit Tokyo next month.",
                "timestamp": "2024-03-14T10:30:00Z",
            }
        ],
        "extract_mode": "async",
    }

    fake_buffer = AsyncMock()
    fake_buffer.id = uuid4()

    with (
        patch("src.api.memorize.buffer_store.insert_messages", AsyncMock(return_value=[fake_buffer])),
        patch("src.api.memorize.cache.push_message_buffer", AsyncMock()),
    ):
        resp = await client.post("/v1/memorize", json=payload)

    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "accepted"
    assert data["message_count"] == 1


@pytest.mark.asyncio
async def test_memorize_sync_mode(client):
    from src.models.fact import Fact

    fake_fact = Fact(
        id=uuid4(),
        tenant_id="default",
        user_id="user_001",
        group_id="group_abc",
        content="Zhang San plans to visit Tokyo.",
        fact_type="plan",
        occurred_at=datetime(2024, 3, 14, 10, 30, tzinfo=timezone.utc),
        valid_from=None,
        valid_until=None,
        superseded_by=None,
        supersedes=None,
        status="active",
        importance=0.7,
        access_count=0,
        last_accessed=None,
        decay_rate=0.01,
        source_type="conversation",
        source_id=None,
        source_meta=None,
        tags=["travel"],
        metadata={},
        created_at=datetime(2024, 3, 14, tzinfo=timezone.utc),
        updated_at=datetime(2024, 3, 14, tzinfo=timezone.utc),
    )

    fake_buffer = AsyncMock()
    fake_buffer.id = uuid4()

    payload = {
        "tenant_id": "default",
        "group_id": "group_abc",
        "messages": [
            {
                "role": "user",
                "speaker_id": "user_001",
                "speaker_name": "Zhang San",
                "content": "I plan to visit Tokyo next month.",
                "timestamp": "2024-03-14T10:30:00Z",
            }
        ],
        "extract_mode": "sync",
    }

    with (
        patch("src.api.memorize.buffer_store.insert_messages", AsyncMock(return_value=[fake_buffer])),
        patch("src.api.memorize.buffer_store.mark_processing", AsyncMock()),
        patch("src.api.memorize.buffer_store.mark_consumed", AsyncMock()),
        patch("src.api.memorize.extract_facts", AsyncMock(return_value=[])),
        patch("src.api.memorize.detect_contradictions", AsyncMock(return_value=[])),
        patch("src.api.memorize.store_facts_with_contradictions", AsyncMock(return_value=[fake_fact])),
    ):
        resp = await client.post("/v1/memorize", json=payload)

    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "completed"
    assert len(data["facts"]) == 1
