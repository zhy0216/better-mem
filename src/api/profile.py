import structlog
from fastapi import APIRouter, HTTPException, Query

from src.store import cache, profile_store

router = APIRouter()
logger = structlog.get_logger(__name__)


@router.get("/v1/profile/{user_id}")
async def get_profile(
    user_id: str,
    tenant_id: str = Query(default="default"),
    scope: str = Query(default="global"),
    group_id: str | None = Query(default=None),
) -> dict:
    cached = await cache.get_profile_cache(tenant_id, user_id, scope, group_id)
    if cached:
        return cached

    profile = await profile_store.get(user_id, tenant_id, scope, group_id)
    if not profile:
        raise HTTPException(status_code=404, detail="Profile not found")

    result = {
        "user_id": profile.user_id,
        "scope": profile.scope,
        "version": profile.version,
        "fact_count": profile.fact_count,
        "profile_data": profile.profile_data.model_dump(),
        "updated_at": profile.updated_at.isoformat(),
    }
    await cache.set_profile_cache(tenant_id, user_id, result, scope, group_id)
    return result
