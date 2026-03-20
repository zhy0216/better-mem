from arq.connections import RedisSettings

from src.config import settings
from src.worker.scheduler import cron_jobs, shutdown, startup
from src.worker.tasks import decay_sweep, process_group, scan_groups


class WorkerSettings:
    functions = [process_group, scan_groups, decay_sweep]
    cron_jobs = cron_jobs
    on_startup = startup
    on_shutdown = shutdown
    redis_settings = RedisSettings.from_dsn(settings.REDIS_URL)
    max_jobs = 10
    job_timeout = 300
