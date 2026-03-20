from contextlib import asynccontextmanager

import structlog
from fastapi import FastAPI

from src.api import facts, health, memorize, profile, recall
from src.config import settings
from src.services.embedding import get_model
from src.store.cache import close_redis, create_redis
from src.store.database import close_pool, create_pool

logger = structlog.get_logger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("starting_up")
    await create_pool()
    await create_redis()
    get_model()
    logger.info("startup_complete")
    yield
    logger.info("shutting_down")
    await close_pool()
    await close_redis()
    logger.info("shutdown_complete")


app = FastAPI(
    title="Memory Service",
    description="Fact-centric long-term memory system for conversational AI agents",
    version="0.1.0",
    lifespan=lifespan,
)

app.include_router(health.router)
app.include_router(memorize.router)
app.include_router(recall.router)
app.include_router(profile.router)
app.include_router(facts.router)
