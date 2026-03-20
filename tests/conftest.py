import asyncio
from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock
from uuid import uuid4

import pytest
from httpx import AsyncClient, ASGITransport

from src.main import app


@pytest.fixture(scope="session")
def event_loop_policy():
    return asyncio.DefaultEventLoopPolicy()


@pytest.fixture
def fake_fact():
    from src.models.fact import Fact
    return Fact(
        id=uuid4(),
        tenant_id="default",
        user_id="user_001",
        group_id="group_abc",
        content="Zhang San plans to visit Tokyo in April 2024.",
        fact_type="plan",
        occurred_at=datetime(2024, 3, 14, 10, 30, tzinfo=timezone.utc),
        valid_from=datetime(2024, 4, 1, tzinfo=timezone.utc),
        valid_until=datetime(2024, 4, 30, tzinfo=timezone.utc),
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
        tags=["travel", "tokyo"],
        metadata={},
        created_at=datetime(2024, 3, 14, 10, 30, tzinfo=timezone.utc),
        updated_at=datetime(2024, 3, 14, 10, 30, tzinfo=timezone.utc),
    )


@pytest.fixture
def fake_scored_fact(fake_fact):
    from src.models.fact import ScoredFact
    return ScoredFact(
        id=fake_fact.id,
        content=fake_fact.content,
        fact_type=fake_fact.fact_type,
        occurred_at=fake_fact.occurred_at,
        importance=fake_fact.importance,
        metadata=fake_fact.metadata,
        tags=fake_fact.tags,
        score=0.92,
        source="vector",
    )


@pytest.fixture
def mock_db(monkeypatch):
    pool = MagicMock()
    monkeypatch.setattr("src.store.database._pool", pool)
    return pool


@pytest.fixture
def mock_redis(monkeypatch):
    r = AsyncMock()
    r.get = AsyncMock(return_value=None)
    r.set = AsyncMock(return_value=True)
    r.delete = AsyncMock(return_value=1)
    r.rpush = AsyncMock(return_value=1)
    monkeypatch.setattr("src.store.cache._redis", r)
    return r


@pytest.fixture
async def client(mock_db, mock_redis):
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as c:
        yield c
