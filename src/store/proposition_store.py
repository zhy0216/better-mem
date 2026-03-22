import math
from datetime import datetime, timezone
from uuid import UUID

import asyncpg
import structlog

from src.models.proposition import (
    Belief,
    Evidence,
    Proposition,
    PropositionCreate,
    PropositionUpdate,
    ScoredProposition,
    SearchFilters,
    get_decay_rate,
    get_evidence_weight,
    get_prior,
)
from src.store.database import get_pool

logger = structlog.get_logger(__name__)

MAX_DECAY_CAP = 2.0


# ---------------------------------------------------------------------------
# Row mappers
# ---------------------------------------------------------------------------

def _row_to_proposition(row: asyncpg.Record) -> Proposition:
    return Proposition(
        id=row["id"],
        tenant_id=row["tenant_id"],
        user_id=row["user_id"],
        group_id=row["group_id"],
        subject_id=row["subject_id"],
        canonical_text=row["canonical_text"],
        proposition_type=row["proposition_type"],
        semantic_key=row["semantic_key"],
        valid_from=row["valid_from"],
        valid_until=row["valid_until"],
        first_observed_at=row["first_observed_at"],
        last_observed_at=row["last_observed_at"],
        tags=list(row["tags"]) if row["tags"] else [],
        metadata=dict(row["metadata"]) if row["metadata"] else {},
        created_at=row["created_at"],
        updated_at=row["updated_at"],
    )


def _row_to_scored(row: asyncpg.Record, source: str = "vector") -> ScoredProposition:
    return ScoredProposition(
        id=row["id"],
        canonical_text=row["canonical_text"],
        proposition_type=row["proposition_type"],
        semantic_key=row.get("semantic_key"),
        confidence=float(row.get("confidence", 0.5)),
        utility_importance=float(row.get("utility_importance", 0.5)),
        freshness_decay=float(row.get("freshness_decay", 0.01)),
        access_count=int(row.get("access_count", 0)),
        belief_status=row.get("status", "active"),
        first_observed_at=row.get("first_observed_at"),
        last_observed_at=row.get("last_observed_at"),
        metadata=dict(row["metadata"]) if row.get("metadata") else {},
        tags=list(row["tags"]) if row.get("tags") else [],
        score=float(row.get("score", 0.0)),
        source=source,
    )


# ---------------------------------------------------------------------------
# Confidence computation (logit-space update per §7.2)
# ---------------------------------------------------------------------------

def compute_confidence(
    prior: float,
    evidences: list[dict],
    freshness_decay: float,
    age_days: float,
) -> float:
    """Compute posterior confidence in logit space."""
    prior = max(0.01, min(prior, 0.99))
    logit_prior = math.log(prior / (1.0 - prior))

    support_sum = sum(e["weight"] for e in evidences if e["direction"] == "support")
    contradict_sum = sum(e["weight"] for e in evidences if e["direction"] == "contradict")

    time_decay = min(freshness_decay * age_days, MAX_DECAY_CAP)

    logit_posterior = logit_prior + support_sum - contradict_sum - time_decay
    return 1.0 / (1.0 + math.exp(-logit_posterior))


# ---------------------------------------------------------------------------
# Write operations
# ---------------------------------------------------------------------------

async def _with_conn(conn, fn):
    """Run fn(conn) inside a transaction, reusing conn if provided."""
    if conn is not None:
        return await fn(conn)
    pool = get_pool()
    async with pool.acquire() as c:
        async with c.transaction():
            return await fn(c)


async def insert_proposition(
    create: PropositionCreate,
    user_id: str,
    tenant_id: str = "default",
    group_id: str | None = None,
    conn=None,
) -> Proposition:
    """Insert a proposition with its initial belief and first evidence atomically."""
    effective_user_id = create.speaker_id or user_id
    now = datetime.now(tz=timezone.utc)
    observed_at = create.observed_at or now
    decay = get_decay_rate(create.proposition_type)
    is_self = (create.speaker_id is None) or (create.speaker_id == user_id)
    prior = create.prior or get_prior(create.proposition_type, create.evidence_type)
    weight = get_evidence_weight(create.evidence_type, is_self=is_self)
    initial_confidence = compute_confidence(
        prior=prior,
        evidences=[{"direction": "support", "weight": weight}],
        freshness_decay=decay,
        age_days=0.0,
    )

    vec_str = (
        f"[{','.join(str(v) for v in create.embedding)}]" if create.embedding else None
    )

    async def _do(c) -> Proposition:
        # 1. Insert proposition
        p_row = await c.fetchrow(
            """
            INSERT INTO propositions (
                tenant_id, user_id, group_id, subject_id,
                canonical_text, proposition_type, semantic_key,
                valid_from, valid_until,
                first_observed_at, last_observed_at,
                embedding, tags, metadata
            ) VALUES ($1,$2,$3,$4,$5,$6,$7,$8,$9,$10,$10,$11::vector,$12,$13)
            RETURNING *
            """,
            tenant_id, effective_user_id, group_id, create.subject_id,
            create.canonical_text, create.proposition_type, create.semantic_key,
            create.valid_from, create.valid_until,
            observed_at,
            vec_str,
            create.tags or [],
            create.metadata or {},
        )

        prop_id = p_row["id"]

        # 2. Insert belief
        await c.execute(
            """
            INSERT INTO beliefs (
                proposition_id, confidence, prior, source_reliability,
                utility_importance, freshness_decay,
                support_count, contradiction_count,
                status
            ) VALUES ($1,$2,$3,$4,$5,$6,$7,$8,$9)
            """,
            prop_id, initial_confidence, prior, 0.8,
            create.importance, decay,
            1, 0,
            "active",
        )

        # 3. Insert first evidence
        await c.execute(
            """
            INSERT INTO evidence (
                proposition_id, evidence_type, direction,
                source_type, source_id, source_meta,
                speaker_id, quoted_text, observed_at,
                weight
            ) VALUES ($1,$2,$3,$4,$5,$6,$7,$8,$9,$10)
            """,
            prop_id, create.evidence_type, "support",
            create.source_type, create.source_id, create.source_meta,
            create.speaker_id, create.quoted_text or create.canonical_text, observed_at,
            weight,
        )

        return _row_to_proposition(p_row)

    return await _with_conn(conn, _do)


async def insert_batch(
    creates: list[PropositionCreate],
    user_id: str,
    tenant_id: str = "default",
    group_id: str | None = None,
    conn=None,
) -> list[Proposition]:
    async def _do(c) -> list[Proposition]:
        saved = []
        for create in creates:
            prop = await insert_proposition(create, user_id, tenant_id, group_id, conn=c)
            saved.append(prop)
        return saved

    return await _with_conn(conn, _do)


async def add_evidence(
    proposition_id: UUID,
    evidence_type: str,
    direction: str,
    source_type: str,
    weight: float,
    speaker_id: str | None = None,
    source_id: str | None = None,
    source_meta: dict | None = None,
    quoted_text: str | None = None,
    observed_at: datetime | None = None,
    conn=None,
) -> None:
    """Add evidence to a proposition and recompute its belief confidence."""
    now = datetime.now(tz=timezone.utc)
    obs = observed_at or now

    async def _do(c) -> None:
        await c.execute(
            """
            INSERT INTO evidence (
                proposition_id, evidence_type, direction,
                source_type, source_id, source_meta,
                speaker_id, quoted_text, observed_at, weight
            ) VALUES ($1,$2,$3,$4,$5,$6,$7,$8,$9,$10)
            """,
            proposition_id, evidence_type, direction,
            source_type, source_id, source_meta,
            speaker_id, quoted_text, obs, weight,
        )

        # Update proposition timestamps
        await c.execute(
            """
            UPDATE propositions SET last_observed_at = GREATEST(last_observed_at, $2), updated_at = now()
            WHERE id = $1
            """,
            proposition_id, obs,
        )

        # Recompute confidence
        await _recompute_belief(c, proposition_id)

    await _with_conn(conn, _do)


async def _recompute_belief(conn, proposition_id: UUID) -> None:
    """Recompute belief confidence from all evidence."""
    belief_row = await conn.fetchrow(
        "SELECT prior, freshness_decay FROM beliefs WHERE proposition_id = $1",
        proposition_id,
    )
    if not belief_row:
        return

    prop_row = await conn.fetchrow(
        "SELECT first_observed_at FROM propositions WHERE id = $1",
        proposition_id,
    )
    now = datetime.now(tz=timezone.utc)
    first_obs = prop_row["first_observed_at"] or now
    if first_obs.tzinfo is None:
        first_obs = first_obs.replace(tzinfo=timezone.utc)
    age_days = (now - first_obs).total_seconds() / 86400

    ev_rows = await conn.fetch(
        "SELECT direction, weight FROM evidence WHERE proposition_id = $1",
        proposition_id,
    )
    evidences = [{"direction": r["direction"], "weight": float(r["weight"])} for r in ev_rows]

    confidence = compute_confidence(
        prior=float(belief_row["prior"]),
        evidences=evidences,
        freshness_decay=float(belief_row["freshness_decay"]),
        age_days=age_days,
    )
    support_count = sum(1 for e in evidences if e["direction"] == "support")
    contradiction_count = sum(1 for e in evidences if e["direction"] == "contradict")

    status = "active"
    if confidence < 0.2:
        status = "deprecated"
    elif confidence < 0.4:
        status = "uncertain"

    await conn.execute(
        """
        UPDATE beliefs SET
            confidence = $2, support_count = $3, contradiction_count = $4,
            status = $5, updated_at = now()
        WHERE proposition_id = $1
        """,
        proposition_id, confidence, support_count, contradiction_count, status,
    )


async def update_belief(
    proposition_id: UUID,
    confidence: float | None = None,
    utility_importance: float | None = None,
    status: str | None = None,
    tenant_id: str = "default",
) -> None:
    """Manual belief correction."""
    pool = get_pool()
    fields, params, idx = [], [], 1
    if confidence is not None:
        fields.append(f"confidence = ${idx}")
        params.append(confidence)
        idx += 1
    if utility_importance is not None:
        fields.append(f"utility_importance = ${idx}")
        params.append(utility_importance)
        idx += 1
    if status is not None:
        fields.append(f"status = ${idx}")
        params.append(status)
        idx += 1
    if not fields:
        return
    fields.append(f"updated_at = ${idx}")
    params.append(datetime.now(tz=timezone.utc))
    idx += 1
    params.append(proposition_id)
    sql = f"UPDATE beliefs SET {', '.join(fields)} WHERE proposition_id = ${idx} RETURNING *"
    async with pool.acquire() as conn:
        await conn.fetchrow(sql, *params)


# ---------------------------------------------------------------------------
# Read operations
# ---------------------------------------------------------------------------

async def get_by_id(proposition_id: UUID, tenant_id: str = "default") -> Proposition | None:
    pool = get_pool()
    async with pool.acquire() as conn:
        row = await conn.fetchrow(
            "SELECT * FROM propositions WHERE id = $1 AND tenant_id = $2",
            proposition_id, tenant_id,
        )
    return _row_to_proposition(row) if row else None


async def soft_delete(proposition_id: UUID, tenant_id: str = "default") -> bool:
    pool = get_pool()
    async with pool.acquire() as conn:
        result = await conn.execute(
            """
            UPDATE beliefs SET status = 'deprecated', updated_at = now()
            WHERE proposition_id = $1
              AND proposition_id IN (SELECT id FROM propositions WHERE tenant_id = $2)
            """,
            proposition_id, tenant_id,
        )
    return result == "UPDATE 1"


async def list_propositions(
    user_id: str,
    tenant_id: str = "default",
    proposition_type: str | None = None,
    status: str | None = "active",
    limit: int = 50,
    offset: int = 0,
) -> list[Proposition]:
    pool = get_pool()
    sql = """
        SELECT p.* FROM propositions p
        JOIN beliefs b ON b.proposition_id = p.id
        WHERE p.tenant_id = $1 AND p.user_id = $2
          AND ($3::text IS NULL OR p.proposition_type = $3)
          AND ($4::text IS NULL OR b.status = $4)
        ORDER BY p.last_observed_at DESC NULLS LAST
        LIMIT $5 OFFSET $6
    """
    async with pool.acquire() as conn:
        rows = await conn.fetch(sql, tenant_id, user_id, proposition_type, status, limit, offset)
    return [_row_to_proposition(r) for r in rows]


async def count_since(user_id: str, since: datetime, tenant_id: str = "default") -> int:
    pool = get_pool()
    async with pool.acquire() as conn:
        row = await conn.fetchrow(
            """
            SELECT COUNT(*) FROM propositions p
            JOIN beliefs b ON b.proposition_id = p.id
            WHERE p.tenant_id=$1 AND p.user_id=$2 AND p.created_at > $3 AND b.status='active'
            """,
            tenant_id, user_id, since,
        )
    return int(row[0]) if row else 0


# ---------------------------------------------------------------------------
# Search operations
# ---------------------------------------------------------------------------

_JOINED_SELECT = """
    SELECT p.id, p.canonical_text, p.proposition_type, p.semantic_key,
           p.first_observed_at, p.last_observed_at,
           p.metadata, p.tags,
           b.confidence, b.utility_importance, b.freshness_decay,
           b.access_count, b.status,
"""

_FILTER_WHERE = """
      AND b.status = ANY(${{s}})
      AND (${{g}}::text IS NULL OR p.group_id = ${{g}})
      AND (${{ts}}::timestamptz IS NULL OR p.first_observed_at >= ${{ts}})
      AND (${{te}}::timestamptz IS NULL OR p.first_observed_at <= ${{te}})
      AND (${{pt}}::text[] IS NULL OR p.proposition_type = ANY(${{pt}}))
      AND (${{tg}}::text[] IS NULL OR p.tags @> ${{tg}})
      AND (p.valid_until IS NULL OR p.valid_until > now())
      AND (${{mc}}::float IS NULL OR b.confidence >= ${{mc}})
"""


def _resolve_filters(filters: SearchFilters | None) -> tuple:
    """Return (filters_obj, time_start, time_end) from a SearchFilters."""
    f = filters or SearchFilters()
    time_start = f.time_range.get("start") if f.time_range else None
    time_end = f.time_range.get("end") if f.time_range else None
    return f, time_start, time_end


def _filter_params(f: SearchFilters, time_start, time_end) -> tuple:
    """Return the common filter parameter tuple for search queries."""
    return (
        f.status or ["active"],
        f.group_id, time_start, time_end,
        f.proposition_types, f.tags,
        f.min_confidence,
    )


async def vector_search(
    embedding: list[float],
    user_id: str,
    tenant_id: str = "default",
    top_k: int = 50,
    filters: SearchFilters | None = None,
) -> list[ScoredProposition]:
    pool = get_pool()
    f, time_start, time_end = _resolve_filters(filters)
    vec_str = f"[{','.join(str(v) for v in embedding)}]"
    sql = _JOINED_SELECT + """
           1 - (p.embedding <=> $1::vector) AS score
    FROM propositions p
    JOIN beliefs b ON b.proposition_id = p.id
    WHERE p.tenant_id = $2
      AND p.user_id = $3
      AND b.status = ANY($4)
      AND ($5::text IS NULL OR p.group_id = $5)
      AND ($6::timestamptz IS NULL OR p.first_observed_at >= $6)
      AND ($7::timestamptz IS NULL OR p.first_observed_at <= $7)
      AND ($8::text[] IS NULL OR p.proposition_type = ANY($8))
      AND ($9::text[] IS NULL OR p.tags @> $9)
      AND (p.valid_until IS NULL OR p.valid_until > now())
      AND ($10::float IS NULL OR b.confidence >= $10)
    ORDER BY p.embedding <=> $1::vector
    LIMIT $11
    """
    async with pool.acquire() as conn:
        rows = await conn.fetch(
            sql, vec_str, tenant_id, user_id,
            *_filter_params(f, time_start, time_end),
            top_k,
        )
    return [_row_to_scored(r, "vector") for r in rows]


async def keyword_search(
    query: str,
    user_id: str,
    tenant_id: str = "default",
    top_k: int = 50,
    filters: SearchFilters | None = None,
) -> list[ScoredProposition]:
    pool = get_pool()
    f, time_start, time_end = _resolve_filters(filters)
    sql = _JOINED_SELECT + """
           GREATEST(
               similarity(p.canonical_text, $1),
               CASE WHEN p.tsv @@ plainto_tsquery('simple', $1)
                    THEN ts_rank(p.tsv, plainto_tsquery('simple', $1))
                    ELSE 0
               END
           ) AS score
    FROM propositions p
    JOIN beliefs b ON b.proposition_id = p.id
    WHERE p.tenant_id = $2
      AND p.user_id = $3
      AND b.status = ANY($4)
      AND (
          p.canonical_text ILIKE '%' || $1 || '%'
          OR similarity(p.canonical_text, $1) > 0.1
          OR p.tsv @@ plainto_tsquery('simple', $1)
      )
      AND ($5::text IS NULL OR p.group_id = $5)
      AND ($6::timestamptz IS NULL OR p.first_observed_at >= $6)
      AND ($7::timestamptz IS NULL OR p.first_observed_at <= $7)
      AND ($8::text[] IS NULL OR p.proposition_type = ANY($8))
      AND ($9::text[] IS NULL OR p.tags @> $9)
      AND (p.valid_until IS NULL OR p.valid_until > now())
      AND ($10::float IS NULL OR b.confidence >= $10)
    ORDER BY score DESC
    LIMIT $11
    """
    async with pool.acquire() as conn:
        rows = await conn.fetch(
            sql, query, tenant_id, user_id,
            *_filter_params(f, time_start, time_end),
            top_k,
        )
    return [_row_to_scored(r, "keyword") for r in rows]


async def search_similar(
    embedding: list[float],
    user_id: str,
    tenant_id: str = "default",
    top_k: int = 5,
    score_threshold: float = 0.85,
) -> list[ScoredProposition]:
    results = await vector_search(embedding, user_id, tenant_id, top_k)
    return [r for r in results if r.score >= score_threshold]


async def search_by_semantic_key(
    semantic_key: str,
    user_id: str,
    tenant_id: str = "default",
) -> list[ScoredProposition]:
    """Find all propositions competing in the same semantic slot."""
    pool = get_pool()
    sql = _JOINED_SELECT + """
           b.confidence AS score
    FROM propositions p
    JOIN beliefs b ON b.proposition_id = p.id
    WHERE p.tenant_id = $1
      AND p.user_id = $2
      AND p.semantic_key = $3
      AND b.status IN ('active', 'uncertain')
    ORDER BY b.confidence DESC
    """
    async with pool.acquire() as conn:
        rows = await conn.fetch(sql, tenant_id, user_id, semantic_key)
    return [_row_to_scored(r, "semantic_key") for r in rows]


async def get_existing_semantic_keys(
    user_id: str,
    tenant_id: str = "default",
) -> list[str]:
    """Get all existing semantic keys for a user (for key normalization)."""
    pool = get_pool()
    async with pool.acquire() as conn:
        rows = await conn.fetch(
            """
            SELECT DISTINCT semantic_key FROM propositions
            WHERE tenant_id = $1 AND user_id = $2 AND semantic_key IS NOT NULL
            """,
            tenant_id, user_id,
        )
    return [r["semantic_key"] for r in rows]


# ---------------------------------------------------------------------------
# Access tracking
# ---------------------------------------------------------------------------

async def track_access(proposition_ids: list[UUID]) -> None:
    pool = get_pool()
    async with pool.acquire() as conn:
        await conn.execute(
            """
            UPDATE beliefs
            SET access_count = access_count + 1,
                last_accessed = now(),
                updated_at = now()
            WHERE proposition_id = ANY($1)
            """,
            proposition_ids,
        )


# ---------------------------------------------------------------------------
# Evidence retrieval
# ---------------------------------------------------------------------------

async def get_evidence(proposition_id: UUID, limit: int = 10) -> list[Evidence]:
    pool = get_pool()
    async with pool.acquire() as conn:
        rows = await conn.fetch(
            "SELECT * FROM evidence WHERE proposition_id = $1 ORDER BY observed_at DESC LIMIT $2",
            proposition_id, limit,
        )
    return [
        Evidence(
            id=r["id"],
            proposition_id=r["proposition_id"],
            evidence_type=r["evidence_type"],
            direction=r["direction"],
            source_type=r["source_type"],
            source_id=r["source_id"],
            source_meta=dict(r["source_meta"]) if r["source_meta"] else None,
            speaker_id=r["speaker_id"],
            quoted_text=r["quoted_text"],
            observed_at=r["observed_at"],
            weight=float(r["weight"]),
            metadata=dict(r["metadata"]) if r["metadata"] else {},
            created_at=r["created_at"],
        )
        for r in rows
    ]
