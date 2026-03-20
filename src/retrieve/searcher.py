import asyncio
import time

import structlog

from src.models.api import SearchFilters
from src.models.fact import ScoredFact
from src.retrieve.ranker import reciprocal_rank_fusion
from src.services import embedding as embedding_service
from src.store import fact_store

logger = structlog.get_logger(__name__)


async def hybrid_search(
    query: str,
    user_id: str,
    tenant_id: str = "default",
    top_k: int = 50,
    filters: SearchFilters | None = None,
) -> tuple[list[ScoredFact], float]:
    start = time.monotonic()

    query_embedding = await embedding_service.embed(query)

    vector_results, keyword_results = await asyncio.gather(
        fact_store.vector_search(
            embedding=query_embedding,
            user_id=user_id,
            tenant_id=tenant_id,
            top_k=top_k,
            filters=filters,
        ),
        fact_store.keyword_search(
            query=query,
            user_id=user_id,
            tenant_id=tenant_id,
            top_k=top_k,
            filters=filters,
        ),
    )

    merged = reciprocal_rank_fusion(vector_results, keyword_results)
    elapsed_ms = (time.monotonic() - start) * 1000

    logger.info(
        "hybrid_search_done",
        user_id=user_id,
        vector_count=len(vector_results),
        keyword_count=len(keyword_results),
        merged_count=len(merged),
        elapsed_ms=round(elapsed_ms, 1),
    )

    return merged, elapsed_ms
