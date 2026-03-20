import structlog
from fastapi import APIRouter

from src.models.api import (
    ProfileSnippet,
    RecallAssembledResponse,
    RecallRawResponse,
    RecallRequest,
)
from src.retrieve import assembler, searcher
from src.store import cache, fact_store, profile_store

router = APIRouter()
logger = structlog.get_logger(__name__)


@router.post("/v1/recall")
async def recall(req: RecallRequest) -> RecallAssembledResponse | RecallRawResponse:
    filters = req.filters
    cache_hash = cache.make_recall_hash(
        req.tenant_id, req.user_id, req.query, filters.model_dump()
    )
    cached = await cache.get_recall_cache(cache_hash)
    if cached:
        logger.info("recall_cache_hit", user_id=req.user_id)
        if req.assemble:
            return RecallAssembledResponse(**cached)
        return RecallRawResponse(**cached)

    candidates, elapsed_ms = await searcher.hybrid_search(
        query=req.query,
        user_id=req.user_id,
        tenant_id=req.tenant_id,
        top_k=req.top_k,
        filters=filters,
    )

    hit_ids = [f.id for f in candidates]
    if hit_ids:
        try:
            await fact_store.track_access(hit_ids)
        except Exception:
            pass

    profile = None
    if req.include_profile:
        profile = await profile_store.get(req.user_id, req.tenant_id)

    if req.assemble:
        assembled = await assembler.assemble_context(req.query, candidates, profile)

        selected_ids = set(assembled.selected_fact_ids)
        filtered_facts = [f for f in candidates if str(f.id) in selected_ids] or candidates

        profile_snippet = None
        if profile:
            relevant_traits = [
                p["trait"] for p in profile.profile_data.personality[:3]
            ]
            profile_snippet = ProfileSnippet(
                summary=profile.profile_data.summary,
                relevant_traits=relevant_traits,
            )

        response = RecallAssembledResponse(
            context=assembled.context,
            facts=[
                {
                    "id": str(f.id),
                    "content": f.content,
                    "score": round(f.score, 4),
                    "source": f.source,
                }
                for f in filtered_facts
            ],
            profile_snippet=profile_snippet,
            total_candidates=len(candidates),
            search_time_ms=round(elapsed_ms, 1),
        )
        await cache.set_recall_cache(cache_hash, response.model_dump())
        return response

    response = RecallRawResponse(
        facts=[
            {
                "id": str(f.id),
                "content": f.content,
                "fact_type": f.fact_type,
                "occurred_at": f.occurred_at.isoformat(),
                "score": round(f.score, 4),
                "source": f.source,
                "importance": f.importance,
                "metadata": f.metadata,
            }
            for f in candidates
        ],
        total_candidates=len(candidates),
        search_time_ms=round(elapsed_ms, 1),
    )
    await cache.set_recall_cache(cache_hash, response.model_dump())
    return response
