import asyncio

import structlog
from sentence_transformers import SentenceTransformer

from src.config import settings

logger = structlog.get_logger(__name__)

_model: SentenceTransformer | None = None


def get_model() -> SentenceTransformer:
    global _model
    if _model is None:
        logger.info("loading_embedding_model", model=settings.EMBEDDING_MODEL)
        _model = SentenceTransformer(settings.EMBEDDING_MODEL, device=settings.EMBEDDING_DEVICE)
        logger.info("embedding_model_loaded")
    return _model


async def embed(text: str) -> list[float]:
    model = get_model()
    loop = asyncio.get_event_loop()
    vec = await loop.run_in_executor(None, model.encode, text)
    return vec.tolist()


async def embed_batch(texts: list[str]) -> list[list[float]]:
    model = get_model()
    loop = asyncio.get_event_loop()
    vecs = await loop.run_in_executor(None, model.encode, texts)
    return [v.tolist() for v in vecs]
