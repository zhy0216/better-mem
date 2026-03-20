from datetime import datetime, timedelta, timezone
from unittest.mock import AsyncMock, patch
from uuid import uuid4

import pytest

from src.extract.session_detector import SessionDetector
from src.models.message import MessageBuffer


def make_buffer(content_text: str, minutes_ago: int = 0) -> MessageBuffer:
    ts = datetime.now(tz=timezone.utc) - timedelta(minutes=minutes_ago)
    return MessageBuffer(
        id=uuid4(),
        tenant_id="default",
        group_id="group_abc",
        user_id="user_001",
        content={"role": "user", "content": content_text, "timestamp": ts.isoformat()},
        status="pending",
        batch_id=None,
        created_at=ts,
    )


@pytest.mark.asyncio
async def test_too_few_messages():
    detector = SessionDetector()
    single = [make_buffer("hello")]
    with patch("src.extract.session_detector.buffer_store.get_pending", AsyncMock(return_value=single)):
        result = await detector.should_extract("group_abc")
    assert result is False


@pytest.mark.asyncio
async def test_time_gap_triggers():
    detector = SessionDetector()
    old_messages = [make_buffer("hello", minutes_ago=35), make_buffer("world", minutes_ago=35)]
    with patch("src.extract.session_detector.buffer_store.get_pending", AsyncMock(return_value=old_messages)):
        result = await detector.should_extract("group_abc")
    assert result is True


@pytest.mark.asyncio
async def test_max_messages_triggers():
    detector = SessionDetector()
    messages = [make_buffer(f"msg {i}", minutes_ago=1) for i in range(51)]
    with patch("src.extract.session_detector.buffer_store.get_pending", AsyncMock(return_value=messages)):
        result = await detector.should_extract("group_abc")
    assert result is True


@pytest.mark.asyncio
async def test_no_trigger_recent_few_messages():
    detector = SessionDetector()
    messages = [make_buffer(f"msg {i}", minutes_ago=1) for i in range(5)]
    with patch("src.extract.session_detector.buffer_store.get_pending", AsyncMock(return_value=messages)):
        result = await detector.should_extract("group_abc")
    assert result is False
