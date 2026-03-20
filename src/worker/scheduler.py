from datetime import timedelta

from arq.connections import RedisSettings

from src.config import settings
from src.worker.tasks import decay_sweep, scan_groups


async def startup(ctx: dict) -> None:
    from src.store.cache import create_redis
    from src.store.database import create_pool
    await create_pool()
    await create_redis()


async def shutdown(ctx: dict) -> None:
    from src.store.cache import close_redis
    from src.store.database import close_pool
    await close_pool()
    await close_redis()


cron_jobs = [
    {
        "coroutine": scan_groups,
        "name": "scan_groups",
        "second": {i for i in range(0, 60, settings.EXTRACT_SCAN_INTERVAL_SECONDS)},
    },
    {
        "coroutine": decay_sweep,
        "name": "decay_sweep",
        "hour": {i for i in range(0, 24, settings.DECAY_SWEEP_INTERVAL_HOURS)},
        "minute": {0},
    },
]
