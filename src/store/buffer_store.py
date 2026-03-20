import json
from datetime import datetime
from uuid import UUID

import asyncpg
import structlog

from src.models.message import Message, MessageBuffer
from src.store.database import get_pool

logger = structlog.get_logger(__name__)


def _row_to_buffer(row: asyncpg.Record) -> MessageBuffer:
    content = row["content"]
    if isinstance(content, str):
        content = json.loads(content)
    return MessageBuffer(
        id=row["id"],
        tenant_id=row["tenant_id"],
        group_id=row["group_id"],
        user_id=row["user_id"],
        content=dict(content),
        status=row["status"],
        batch_id=row["batch_id"],
        created_at=row["created_at"],
    )


async def insert_messages(
    messages: list[Message],
    group_id: str,
    tenant_id: str = "default",
    batch_id: UUID | None = None,
    user_id: str = "unknown",
) -> list[MessageBuffer]:
    pool = get_pool()
    saved = []
    async with pool.acquire() as conn:
        for msg in messages:
            effective_user_id = msg.speaker_id or user_id
            row = await conn.fetchrow(
                """
                INSERT INTO message_buffer (tenant_id, group_id, user_id, content, batch_id)
                VALUES ($1, $2, $3, $4::jsonb, $5)
                RETURNING *
                """,
                tenant_id, group_id, effective_user_id,
                json.dumps(msg.model_dump(mode="json")),
                batch_id,
            )
            saved.append(_row_to_buffer(row))
    return saved


async def get_pending(
    group_id: str,
    tenant_id: str = "default",
) -> list[MessageBuffer]:
    pool = get_pool()
    async with pool.acquire() as conn:
        rows = await conn.fetch(
            """
            SELECT * FROM message_buffer
            WHERE tenant_id = $1 AND group_id = $2 AND status = 'pending'
            ORDER BY created_at ASC
            """,
            tenant_id, group_id,
        )
    return [_row_to_buffer(r) for r in rows]


async def mark_processing(ids: list[UUID]) -> None:
    pool = get_pool()
    async with pool.acquire() as conn:
        await conn.execute(
            "UPDATE message_buffer SET status = 'processing' WHERE id = ANY($1)",
            ids,
        )


async def mark_consumed(ids: list[UUID]) -> None:
    pool = get_pool()
    async with pool.acquire() as conn:
        await conn.execute(
            "UPDATE message_buffer SET status = 'consumed' WHERE id = ANY($1)",
            ids,
        )


async def mark_pending(ids: list[UUID]) -> None:
    pool = get_pool()
    async with pool.acquire() as conn:
        await conn.execute(
            "UPDATE message_buffer SET status = 'pending' WHERE id = ANY($1)",
            ids,
        )


async def get_active_group_ids() -> list[tuple[str, str]]:
    """Return (tenant_id, group_id) for all groups with pending messages."""
    pool = get_pool()
    async with pool.acquire() as conn:
        rows = await conn.fetch(
            """
            SELECT DISTINCT tenant_id, group_id FROM message_buffer
            WHERE status = 'pending'
            """
        )
    return [(r["tenant_id"], r["group_id"]) for r in rows]


async def recover_stuck_processing(stuck_minutes: int = 10) -> int:
    """Reset messages stuck in 'processing' state back to 'pending'."""
    pool = get_pool()
    async with pool.acquire() as conn:
        result = await conn.execute(
            """
            UPDATE message_buffer
            SET status = 'pending'
            WHERE status = 'processing'
              AND created_at < now() - ($1 * interval '1 minute')
            """,
            stuck_minutes,
        )
    count = int(result.split()[-1]) if result else 0
    if count:
        import structlog
        structlog.get_logger(__name__).warning(
            "recovered_stuck_messages", count=count
        )
    return count
