from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock, patch
from uuid import uuid4

import pytest

from src.store import proposition_store


def make_proposition_row(prop_id=None):
    pid = prop_id or uuid4()
    return {
        "id": pid,
        "tenant_id": "default",
        "user_id": "user_001",
        "group_id": "group_abc",
        "subject_id": None,
        "canonical_text": "Zhang San plans to visit Tokyo.",
        "proposition_type": "plan",
        "semantic_key": "trip_tokyo_2024",
        "valid_from": None,
        "valid_until": None,
        "first_observed_at": datetime(2024, 3, 14, tzinfo=timezone.utc),
        "last_observed_at": datetime(2024, 3, 14, tzinfo=timezone.utc),
        "tags": ["travel", "tokyo"],
        "metadata": {},
        "created_at": datetime(2024, 3, 14, tzinfo=timezone.utc),
        "updated_at": datetime(2024, 3, 14, tzinfo=timezone.utc),
    }


def make_mock_pool(fetchrow_return=None, fetch_return=None, execute_return="UPDATE 1"):
    conn = AsyncMock()
    conn.fetchrow = AsyncMock(return_value=fetchrow_return)
    conn.fetch = AsyncMock(return_value=fetch_return or [])
    conn.execute = AsyncMock(return_value=execute_return)
    pool = MagicMock()
    pool.acquire = MagicMock(return_value=AsyncMock(__aenter__=AsyncMock(return_value=conn), __aexit__=AsyncMock(return_value=False)))
    return pool, conn


@pytest.mark.asyncio
async def test_get_by_id_found():
    row = make_proposition_row()
    pool, _ = make_mock_pool(fetchrow_return=row)
    with patch("src.store.proposition_store.get_pool", return_value=pool):
        result = await proposition_store.get_by_id(row["id"])
    assert result is not None
    assert result.canonical_text == "Zhang San plans to visit Tokyo."


@pytest.mark.asyncio
async def test_get_by_id_not_found():
    pool, _ = make_mock_pool(fetchrow_return=None)
    with patch("src.store.proposition_store.get_pool", return_value=pool):
        result = await proposition_store.get_by_id(uuid4())
    assert result is None


@pytest.mark.asyncio
async def test_soft_delete_success():
    pool, _ = make_mock_pool(execute_return="UPDATE 1")
    with patch("src.store.proposition_store.get_pool", return_value=pool):
        ok = await proposition_store.soft_delete(uuid4())
    assert ok is True


@pytest.mark.asyncio
async def test_soft_delete_not_found():
    pool, _ = make_mock_pool(execute_return="UPDATE 0")
    with patch("src.store.proposition_store.get_pool", return_value=pool):
        ok = await proposition_store.soft_delete(uuid4())
    assert ok is False


@pytest.mark.asyncio
async def test_list_propositions():
    rows = [make_proposition_row(), make_proposition_row()]
    pool, _ = make_mock_pool(fetch_return=rows)
    with patch("src.store.proposition_store.get_pool", return_value=pool):
        props = await proposition_store.list_propositions(user_id="user_001")
    assert len(props) == 2
