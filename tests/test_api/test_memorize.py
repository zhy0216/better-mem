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

    with patch("src.api.memorize.buffer_store.insert_messages", AsyncMock(return_value=[fake_buffer])):
        resp = await client.post("/v1/memorize", json=payload)

    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "accepted"
    assert data["message_count"] == 1


@pytest.mark.asyncio
async def test_memorize_sync_mode(client):
    from src.models.proposition import Proposition

    fake_prop = Proposition(
        id=uuid4(),
        tenant_id="default",
        user_id="user_001",
        group_id="group_abc",
        subject_id=None,
        canonical_text="Zhang San plans to visit Tokyo.",
        proposition_type="plan",
        semantic_key="trip_tokyo_2024",
        valid_from=None,
        valid_until=None,
        first_observed_at=datetime(2024, 3, 14, 10, 30, tzinfo=timezone.utc),
        last_observed_at=datetime(2024, 3, 14, 10, 30, tzinfo=timezone.utc),
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
        patch("src.api.memorize.memorize_service.process_buffered_messages", AsyncMock(return_value=[fake_prop])),
    ):
        resp = await client.post("/v1/memorize", json=payload)

    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "completed"
    assert len(data["propositions"]) == 1
