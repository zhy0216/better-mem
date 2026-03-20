from datetime import datetime, timezone
from uuid import UUID

import asyncpg
import structlog

from src.models.fact import Fact, FactCreate, FactUpdate, ScoredFact, SearchFilters
from src.store.database import get_pool

logger = structlog.get_logger(__name__)


def _row_to_fact(row: asyncpg.Record) -> Fact:
    return Fact(
        id=row["id"],
        tenant_id=row["tenant_id"],
        user_id=row["user_id"],
        group_id=row["group_id"],
        content=row["content"],
        fact_type=row["fact_type"],
        occurred_at=row["occurred_at"],
        valid_from=row["valid_from"],
        valid_until=row["valid_until"],
        superseded_by=row["superseded_by"],
        supersedes=row["supersedes"],
        status=row["status"],
        importance=row["importance"],
        access_count=row["access_count"],
        last_accessed=row["last_accessed"],
        decay_rate=row["decay_rate"],
        source_type=row["source_type"],
        source_id=row["source_id"],
        source_meta=dict(row["source_meta"]) if row["source_meta"] else None,
        tags=list(row["tags"]) if row["tags"] else [],
        metadata=dict(row["metadata"]) if row["metadata"] else {},
        created_at=row["created_at"],
        updated_at=row["updated_at"],
    )


async def insert_batch(
    facts: list[FactCreate],
    user_id: str,
    tenant_id: str = "default",
    group_id: str | None = None,
    conn=None,
) -> list[Fact]:
    async def _do_insert(c) -> list[Fact]:
        saved = []
        for fact in facts:
            occurred_at = fact.occurred_at or datetime.now(tz=timezone.utc)
            effective_user_id = fact.speaker_id or user_id
            row = await c.fetchrow(
                """
                INSERT INTO facts (
                    tenant_id, user_id, group_id, content, fact_type,
                    occurred_at, valid_from, valid_until,
                    importance, decay_rate, source_type, source_id, source_meta,
                    tags, metadata, embedding
                ) VALUES ($1,$2,$3,$4,$5,$6,$7,$8,$9,$10,$11,$12,$13,$14,$15,$16)
                RETURNING *
                """,
                tenant_id, effective_user_id, group_id, fact.content, fact.fact_type,
                occurred_at, fact.valid_from, fact.valid_until,
                fact.importance, fact.decay_rate, fact.source_type,
                fact.source_id,
                fact.source_meta,
                fact.tags or [],
                fact.metadata or {},
                f"[{','.join(str(v) for v in fact.embedding)}]" if fact.embedding else None,
            )
            saved.append(_row_to_fact(row))
        return saved

    if conn is not None:
        return await _do_insert(conn)
    pool = get_pool()
    async with pool.acquire() as c:
        return await _do_insert(c)


async def get_by_id(fact_id: UUID, tenant_id: str = "default") -> Fact | None:
    pool = get_pool()
    async with pool.acquire() as conn:
        row = await conn.fetchrow(
            "SELECT * FROM facts WHERE id = $1 AND tenant_id = $2",
            fact_id, tenant_id,
        )
    return _row_to_fact(row) if row else None


async def update(fact_id: UUID, update: FactUpdate, tenant_id: str = "default") -> Fact | None:
    pool = get_pool()
    fields, params, idx = [], [], 1
    if update.tags is not None:
        fields.append(f"tags = ${idx}")
        params.append(update.tags)
        idx += 1
    if update.metadata is not None:
        fields.append(f"metadata = ${idx}")
        params.append(update.metadata)
        idx += 1
    if update.importance is not None:
        fields.append(f"importance = ${idx}")
        params.append(update.importance)
        idx += 1
    if not fields:
        return await get_by_id(fact_id, tenant_id)
    fields.append(f"updated_at = ${idx}")
    params.append(datetime.now(tz=timezone.utc))
    idx += 1
    params.extend([fact_id, tenant_id])
    sql = f"UPDATE facts SET {', '.join(fields)} WHERE id = ${idx} AND tenant_id = ${idx+1} RETURNING *"
    async with pool.acquire() as conn:
        row = await conn.fetchrow(sql, *params)
    return _row_to_fact(row) if row else None


async def soft_delete(fact_id: UUID, tenant_id: str = "default") -> bool:
    pool = get_pool()
    async with pool.acquire() as conn:
        result = await conn.execute(
            "UPDATE facts SET status = 'deleted', updated_at = now() WHERE id = $1 AND tenant_id = $2",
            fact_id, tenant_id,
        )
    return result == "UPDATE 1"


async def mark_superseded(old_fact_id: UUID, superseded_by: UUID, conn=None) -> None:
    async def _do(c) -> None:
        await c.execute(
            """
            UPDATE facts
            SET status = 'superseded', superseded_by = $2, updated_at = now()
            WHERE id = $1
            """,
            old_fact_id, superseded_by,
        )
        await c.execute(
            """
            UPDATE facts
            SET supersedes = $2, updated_at = now()
            WHERE id = $1
            """,
            superseded_by, old_fact_id,
        )

    if conn is not None:
        await _do(conn)
        return
    pool = get_pool()
    async with pool.acquire() as c:
        await _do(c)


async def vector_search(
    embedding: list[float],
    user_id: str,
    tenant_id: str = "default",
    top_k: int = 50,
    filters: SearchFilters | None = None,
) -> list[ScoredFact]:
    pool = get_pool()
    f = filters or SearchFilters()
    time_start = f.time_range.get("start") if f.time_range else None
    time_end = f.time_range.get("end") if f.time_range else None
    vec_str = f"[{','.join(str(v) for v in embedding)}]"
    sql = """
        SELECT id, content, fact_type, occurred_at, importance, decay_rate, access_count,
               metadata, tags,
               1 - (embedding <=> $1::vector) AS score
        FROM facts
        WHERE tenant_id = $2
          AND user_id = $3
          AND status = ANY($4)
          AND ($5::text IS NULL OR group_id = $5)
          AND ($6::timestamptz IS NULL OR occurred_at >= $6)
          AND ($7::timestamptz IS NULL OR occurred_at <= $7)
          AND ($8::text[] IS NULL OR fact_type = ANY($8))
        ORDER BY embedding <=> $1::vector
        LIMIT $9
    """
    async with pool.acquire() as conn:
        rows = await conn.fetch(
            sql, vec_str, tenant_id, user_id,
            f.status or ["active"],
            f.group_id, time_start, time_end, f.fact_types, top_k,
        )
    return [
        ScoredFact(
            id=r["id"], content=r["content"], fact_type=r["fact_type"],
            occurred_at=r["occurred_at"], importance=r["importance"],
            decay_rate=float(r["decay_rate"]), access_count=int(r["access_count"]),
            metadata=dict(r["metadata"]) if r["metadata"] else {},
            tags=list(r["tags"]) if r["tags"] else [],
            score=float(r["score"]), source="vector",
        )
        for r in rows
    ]


async def keyword_search(
    query: str,
    user_id: str,
    tenant_id: str = "default",
    top_k: int = 50,
    filters: SearchFilters | None = None,
) -> list[ScoredFact]:
    pool = get_pool()
    f = filters or SearchFilters()
    time_start = f.time_range.get("start") if f.time_range else None
    time_end = f.time_range.get("end") if f.time_range else None
    sql = """
        SELECT id, content, fact_type, occurred_at, importance, decay_rate, access_count,
               metadata, tags,
               GREATEST(
                   similarity(content, $1),
                   CASE WHEN tsv @@ plainto_tsquery('simple', $1)
                        THEN ts_rank(tsv, plainto_tsquery('simple', $1))
                        ELSE 0
                   END
               ) AS score
        FROM facts
        WHERE tenant_id = $2
          AND user_id = $3
          AND status = ANY($4)
          AND (
              content ILIKE '%' || $1 || '%'
              OR similarity(content, $1) > 0.1
              OR tsv @@ plainto_tsquery('simple', $1)
          )
          AND ($5::text IS NULL OR group_id = $5)
          AND ($6::timestamptz IS NULL OR occurred_at >= $6)
          AND ($7::timestamptz IS NULL OR occurred_at <= $7)
        ORDER BY score DESC
        LIMIT $8
    """
    async with pool.acquire() as conn:
        rows = await conn.fetch(
            sql, query, tenant_id, user_id,
            f.status or ["active"],
            f.group_id, time_start, time_end, top_k,
        )
    return [
        ScoredFact(
            id=r["id"], content=r["content"], fact_type=r["fact_type"],
            occurred_at=r["occurred_at"], importance=r["importance"],
            decay_rate=float(r["decay_rate"]), access_count=int(r["access_count"]),
            metadata=dict(r["metadata"]) if r["metadata"] else {},
            tags=list(r["tags"]) if r["tags"] else [],
            score=float(r["score"]), source="keyword",
        )
        for r in rows
    ]


async def search_similar(
    embedding: list[float],
    user_id: str,
    tenant_id: str = "default",
    top_k: int = 5,
    score_threshold: float = 0.85,
) -> list[ScoredFact]:
    results = await vector_search(embedding, user_id, tenant_id, top_k)
    return [r for r in results if r.score >= score_threshold]


async def list_facts(
    user_id: str,
    tenant_id: str = "default",
    fact_type: str | None = None,
    status: str | None = "active",
    limit: int = 50,
    offset: int = 0,
) -> list[Fact]:
    pool = get_pool()
    sql = """
        SELECT * FROM facts
        WHERE tenant_id = $1 AND user_id = $2
          AND ($3::text IS NULL OR fact_type = $3)
          AND ($4::text IS NULL OR status = $4)
        ORDER BY occurred_at DESC
        LIMIT $5 OFFSET $6
    """
    async with pool.acquire() as conn:
        rows = await conn.fetch(sql, tenant_id, user_id, fact_type, status, limit, offset)
    return [_row_to_fact(r) for r in rows]


async def count_since(user_id: str, since: datetime, tenant_id: str = "default") -> int:
    pool = get_pool()
    async with pool.acquire() as conn:
        row = await conn.fetchrow(
            "SELECT COUNT(*) FROM facts WHERE tenant_id=$1 AND user_id=$2 AND created_at > $3 AND status='active'",
            tenant_id, user_id, since,
        )
    return int(row[0]) if row else 0


async def track_access(fact_ids: list[UUID]) -> None:
    pool = get_pool()
    async with pool.acquire() as conn:
        await conn.execute(
            """
            UPDATE facts
            SET access_count = access_count + 1,
                last_accessed = now(),
                updated_at = now()
            WHERE id = ANY($1)
            """,
            fact_ids,
        )
