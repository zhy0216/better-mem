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
            user_id=req.user_id,
        )
        await buffer_store.mark_processing([b.id for b in buffers])

        try:
            saved = await memorize_service.process_buffered_messages(
                buffers=buffers,
                tenant_id=req.tenant_id,
                group_id=req.group_id,
                user_id=req.user_id,
            )
            await buffer_store.mark_consumed([b.id for b in buffers])
        except Exception:
            await buffer_store.mark_pending([b.id for b in buffers])
            raise

        return MemorizeSyncResponse(
            status="completed",
            propositions=[
                {
                    "id": str(p.id),
                    "canonical_text": p.canonical_text,
                    "proposition_type": p.proposition_type,
                    "semantic_key": p.semantic_key,
                    "tags": p.tags,
                }
                for p in saved
            ],
        )

    buffers = await buffer_store.insert_messages(
        messages=req.messages,
        group_id=req.group_id,
        tenant_id=req.tenant_id,
        batch_id=batch_id,
        user_id=req.user_id,
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
