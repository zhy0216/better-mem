import redis.asyncio as aioredis
import structlog

from src.config import settings

logger = structlog.get_logger(__name__)

_redis: aioredis.Redis | None = None


async def create_redis() -> aioredis.Redis:
    global _redis
    _redis = aioredis.from_url(settings.REDIS_URL, decode_responses=True)
    logger.info("redis_connected", url=settings.REDIS_URL)
    return _redis


async def close_redis() -> None:
    global _redis
    if _redis:
        await _redis.aclose()
        _redis = None
        logger.info("redis_closed")


def get_redis() -> aioredis.Redis:
    if _redis is None:
        raise RuntimeError("Redis not initialized")
    return _redis


def _profile_cache_key(tenant_id: str, user_id: str, scope: str, group_id: str | None) -> str:
    gid = group_id or "__none__"
    return f"cache:profile:{tenant_id}:{user_id}:{scope}:{gid}"


async def invalidate_profile_cache(
    tenant_id: str, user_id: str, scope: str = "global", group_id: str | None = None
) -> None:
    try:
        r = get_redis()
        key = _profile_cache_key(tenant_id, user_id, scope, group_id)
        await r.delete(key)
    except Exception:
        pass


async def acquire_extract_lock(tenant_id: str, group_id: str, ttl: int = 120) -> bool:
    try:
        r = get_redis()
        key = f"lock:extract:{tenant_id}:{group_id}"
        result = await r.set(key, "1", ex=ttl, nx=True)
        return result is True
    except Exception:
        return False


async def release_extract_lock(tenant_id: str, group_id: str) -> None:
    try:
        r = get_redis()
        key = f"lock:extract:{tenant_id}:{group_id}"
        await r.delete(key)
    except Exception:
        pass
