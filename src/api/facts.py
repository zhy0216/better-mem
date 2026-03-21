from uuid import UUID

import structlog
from fastapi import APIRouter, HTTPException, Query

from src.models.fact import FactUpdate
from src.services import embedding as embedding_service
from src.store import fact_store

router = APIRouter()
logger = structlog.get_logger(__name__)


@router.get("/v1/facts")
async def list_facts(
    user_id: str = Query(...),
    tenant_id: str = Query(default="default"),
    fact_type: str | None = Query(default=None),
    status: str | None = Query(default="active"),
    limit: int = Query(default=50, le=200),
    offset: int = Query(default=0),
) -> dict:
    facts = await fact_store.list_facts(
        user_id=user_id,
        tenant_id=tenant_id,
        fact_type=fact_type,
        status=status,
        limit=limit,
        offset=offset,
    )
    return {
        "facts": [
            {
                "id": str(f.id),
                "content": f.content,
                "fact_type": f.fact_type,
                "occurred_at": f.occurred_at.isoformat(),
                "status": f.status,
                "importance": f.importance,
                "tags": f.tags,
                "metadata": f.metadata,
            }
            for f in facts
        ],
        "count": len(facts),
    }


@router.delete("/v1/facts/{fact_id}")
async def delete_fact(
    fact_id: UUID,
    tenant_id: str = Query(default="default"),
) -> dict:
    ok = await fact_store.soft_delete(fact_id, tenant_id)
    if not ok:
        raise HTTPException(status_code=404, detail="Fact not found")
    return {"status": "deleted", "id": str(fact_id)}


@router.patch("/v1/facts/{fact_id}")
async def patch_fact(
    fact_id: UUID,
    update: FactUpdate,
    tenant_id: str = Query(default="default"),
) -> dict:
    new_embedding = None
    if update.content is not None:
        new_embedding = await embedding_service.embed(update.content)
    fact = await fact_store.update(fact_id, update, tenant_id, embedding=new_embedding)
    if not fact:
        raise HTTPException(status_code=404, detail="Fact not found")
    return {
        "id": str(fact.id),
        "content": fact.content,
        "tags": fact.tags,
        "metadata": fact.metadata,
        "importance": fact.importance,
        "updated_at": fact.updated_at.isoformat(),
    }


@router.post("/v1/facts/{fact_id}/supersede")
async def supersede_fact(
    fact_id: UUID,
    body: dict,
    tenant_id: str = Query(default="default"),
) -> dict:
    superseded_by = body.get("superseded_by")
    if not superseded_by:
        raise HTTPException(status_code=422, detail="superseded_by is required")
    await fact_store.mark_superseded(fact_id, UUID(superseded_by), tenant_id=tenant_id)
    return {"status": "superseded", "id": str(fact_id), "superseded_by": superseded_by}
