import uuid

import structlog
from fastapi import APIRouter

from src.models.api import MemorizeAsyncResponse, MemorizeRequest, MemorizeSyncResponse
from src.services import memorize_service
from src.store import buffer_store

router = APIRouter()
logger = structlog.get_logger(__name__)


@router.post("/v1/memorize")
async def memorize(req: MemorizeRequest) -> MemorizeAsyncResponse | MemorizeSyncResponse:
    batch_id = uuid.uuid4()

    if req.extract_mode == "sync":
        buffers = await buffer_store.insert_messages(
            messages=req.messages,
            group_id=req.group_id,
            tenant_id=req.tenant_id,
            batch_id=batch_id,
        )
        await buffer_store.mark_processing([b.id for b in buffers])

        saved = await memorize_service.process_buffered_messages(
            buffers=buffers,
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
        tenant_id=req.tenant_id,
        batch_id=batch_id,
    )

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
