import asyncio
from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock, patch
from uuid import uuid4

import pytest
import pytest_asyncio
from httpx import ASGITransport, AsyncClient

from src.models.proposition import Proposition, ScoredProposition


@pytest.fixture(scope="session")
def event_loop():
    loop = asyncio.new_event_loop()
    yield loop
    loop.close()


@pytest.fixture
def fake_proposition() -> Proposition:
    return Proposition(
        id=uuid4(),
        tenant_id="default",
        user_id="user_001",
        group_id="group_abc",
        subject_id=None,
        canonical_text="Zhang San likes traveling.",
        proposition_type="preference",
        semantic_key="travel_preference",
        valid_from=None,
        valid_until=None,
        first_observed_at=datetime(2024, 3, 14, tzinfo=timezone.utc),
        last_observed_at=datetime(2024, 3, 14, tzinfo=timezone.utc),
        tags=["travel"],
        metadata={},
        created_at=datetime(2024, 3, 14, tzinfo=timezone.utc),
        updated_at=datetime(2024, 3, 14, tzinfo=timezone.utc),
    )


@pytest.fixture
def fake_scored_proposition() -> ScoredProposition:
    return ScoredProposition(
        id=uuid4(),
        canonical_text="Zhang San likes traveling.",
        proposition_type="preference",
        semantic_key="travel_preference",
        confidence=0.8,
        utility_importance=0.5,
        freshness_decay=0.005,
        access_count=0,
        belief_status="active",
        first_observed_at=datetime(2024, 3, 14, tzinfo=timezone.utc),
        last_observed_at=datetime(2024, 3, 14, tzinfo=timezone.utc),
        metadata={},
        tags=["travel"],
        score=0.85,
        source="vector",
    )


@pytest.fixture
def mock_db():
    conn = AsyncMock()
    pool = MagicMock()
    pool.acquire = MagicMock(
        return_value=AsyncMock(
            __aenter__=AsyncMock(return_value=conn),
            __aexit__=AsyncMock(return_value=False),
        )
    )
    return pool, conn


@pytest.fixture
def mock_redis():
    return AsyncMock()


@pytest_asyncio.fixture
async def client():
    from src.main import app
    with (
        patch("src.store.database.create_pool"),
        patch("src.store.cache.create_redis"),
    ):
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as c:
            yield c
