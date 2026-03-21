from uuid import UUID

import asyncpg
import structlog

from src.models.profile import Profile, ProfileData
from src.store.database import get_pool

logger = structlog.get_logger(__name__)


def _row_to_profile(row: asyncpg.Record) -> Profile:
    pd = row["profile_data"]
    if isinstance(pd, str):
        import json
        pd = json.loads(pd)
    return Profile(
        id=row["id"],
        tenant_id=row["tenant_id"],
        user_id=row["user_id"],
        scope=row["scope"],
        group_id=row["group_id"],
        profile_data=ProfileData(**pd) if pd else ProfileData(),
        version=row["version"],
        fact_count=row["fact_count"],
        last_fact_id=row["last_fact_id"],
        created_at=row["created_at"],
        updated_at=row["updated_at"],
    )


async def get(
    user_id: str,
    tenant_id: str = "default",
    scope: str = "global",
    group_id: str | None = None,
) -> Profile | None:
    pool = get_pool()
    async with pool.acquire() as conn:
        row = await conn.fetchrow(
            """
            SELECT * FROM profiles
            WHERE tenant_id = $1 AND user_id = $2 AND scope = $3
              AND (($4::text IS NULL AND group_id IS NULL) OR group_id = $4)
            """,
            tenant_id, user_id, scope, group_id,
        )
    return _row_to_profile(row) if row else None


async def upsert(
    user_id: str,
    profile_data: ProfileData,
    tenant_id: str = "default",
    scope: str = "global",
    group_id: str | None = None,
    last_fact_id: UUID | None = None,
    fact_count: int = 1,
) -> Profile:
    import json
    pool = get_pool()
    pd_json = json.dumps(profile_data.model_dump())
    async with pool.acquire() as conn:
        if group_id is None:
            row = await conn.fetchrow(
                """
                INSERT INTO profiles (tenant_id, user_id, scope, group_id, profile_data, last_fact_id, fact_count)
                VALUES ($1, $2, $3, NULL, $4::jsonb, $5, $6)
                ON CONFLICT (tenant_id, user_id, scope) WHERE group_id IS NULL
                DO UPDATE SET
                    profile_data = EXCLUDED.profile_data,
                    version = profiles.version + 1,
                    fact_count = profiles.fact_count + $6,
                    last_fact_id = EXCLUDED.last_fact_id,
                    updated_at = now()
                RETURNING *
                """,
                tenant_id, user_id, scope, pd_json, last_fact_id, fact_count,
            )
        else:
            row = await conn.fetchrow(
                """
                INSERT INTO profiles (tenant_id, user_id, scope, group_id, profile_data, last_fact_id, fact_count)
                VALUES ($1, $2, $3, $4, $5::jsonb, $6, $7)
                ON CONFLICT (tenant_id, user_id, scope, group_id) WHERE group_id IS NOT NULL
                DO UPDATE SET
                    profile_data = EXCLUDED.profile_data,
                    version = profiles.version + 1,
                    fact_count = profiles.fact_count + $7,
                    last_fact_id = EXCLUDED.last_fact_id,
                    updated_at = now()
                RETURNING *
                """,
                tenant_id, user_id, scope, group_id, pd_json, last_fact_id, fact_count,
            )
    return _row_to_profile(row)
