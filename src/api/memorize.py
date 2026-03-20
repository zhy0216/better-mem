import uuid

import structlog
from fastapi import APIRouter, HTTPException

from src.extract.contradiction import detect_contradictions, store_facts_with_contradictions
from src.extract.fact_extractor import extract_facts
from src.models.api import MemorizeAsyncResponse, MemorizeRequest, MemorizeSyncResponse
from src.store import buffer_store, cache

router = APIRouter()
logger = structlog.get_logger(__name__)


@router.post("/v1/memorize")
async def memorize(req: MemorizeRequest) -> MemorizeAsyncResponse | MemorizeSyncResponse:
    batch_id = uuid.uuid4()

    user_ids = list({m.speaker_id for m in req.messages if m.speaker_id})
    primary_user_id = user_ids[0] if user_ids else "unknown"

    if req.extract_mode == "sync":
        buffers = await buffer_store.insert_messages(
            messages=req.messages,
            group_id=req.group_id,
            user_id=primary_user_id,
            tenant_id=req.tenant_id,
            batch_id=batch_id,
        )
        await buffer_store.mark_processing([b.id for b in buffers])

        new_facts = await extract_facts(
            messages=buffers,
            user_id=primary_user_id,
        )

        contradictions = []
        try:
            contradictions = await detect_contradictions(new_facts, primary_user_id, req.tenant_id)
        except Exception as e:
            logger.warning("contradiction_detection_skipped", error=str(e))

        saved = await store_facts_with_contradictions(
            new_facts=new_facts,
            contradictions=contradictions,
            user_id=primary_user_id,
            tenant_id=req.tenant_id,
            group_id=req.group_id,
        )
        await buffer_store.mark_consumed([b.id for b in buffers])

        return MemorizeSyncResponse(
            status="completed",
            facts=[
                {
                    "id": str(f.id),
                    "content": f.content,
                    "fact_type": f.fact_type,
                    "occurred_at": f.occurred_at.isoformat(),
                    "importance": f.importance,
                    "tags": f.tags,
                }
                for f in saved
            ],
        )

    buffers = await buffer_store.insert_messages(
        messages=req.messages,
        group_id=req.group_id,
        user_id=primary_user_id,
        tenant_id=req.tenant_id,
        batch_id=batch_id,
    )

    for msg in req.messages:
        await cache.push_message_buffer(req.tenant_id, req.group_id, msg.model_dump(mode="json"))

    logger.info(
        "memorize_accepted",
        batch_id=str(batch_id),
        message_count=len(req.messages),
        group_id=req.group_id,
    )
    return MemorizeAsyncResponse(
        status="accepted",
        batch_id=batch_id,
        message_count=len(req.messages),
    )
