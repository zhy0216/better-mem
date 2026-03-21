from uuid import UUID

import structlog
from fastapi import APIRouter, HTTPException, Query

from src.models.proposition import PropositionUpdate
from src.services import embedding as embedding_service
from src.store import proposition_store

router = APIRouter()
logger = structlog.get_logger(__name__)


@router.get("/v1/propositions")
async def list_propositions(
    user_id: str = Query(...),
    tenant_id: str = Query(default="default"),
    proposition_type: str | None = Query(default=None),
    status: str | None = Query(default="active"),
    limit: int = Query(default=50, le=200),
    offset: int = Query(default=0),
) -> dict:
    props = await proposition_store.list_propositions(
        user_id=user_id,
        tenant_id=tenant_id,
        proposition_type=proposition_type,
        status=status,
        limit=limit,
        offset=offset,
    )
    return {
        "propositions": [
            {
                "id": str(p.id),
                "canonical_text": p.canonical_text,
                "proposition_type": p.proposition_type,
                "semantic_key": p.semantic_key,
                "first_observed_at": p.first_observed_at.isoformat() if p.first_observed_at else None,
                "last_observed_at": p.last_observed_at.isoformat() if p.last_observed_at else None,
                "tags": p.tags,
                "metadata": p.metadata,
            }
            for p in props
        ],
        "count": len(props),
    }


@router.delete("/v1/propositions/{proposition_id}")
async def delete_proposition(
    proposition_id: UUID,
    tenant_id: str = Query(default="default"),
) -> dict:
    ok = await proposition_store.soft_delete(proposition_id, tenant_id)
    if not ok:
        raise HTTPException(status_code=404, detail="Proposition not found")
    return {"status": "deleted", "id": str(proposition_id)}


@router.post("/v1/propositions/{proposition_id}/evidence")
async def add_evidence(
    proposition_id: UUID,
    body: dict,
    tenant_id: str = Query(default="default"),
) -> dict:
    direction = body.get("direction", "support")
    evidence_type = body.get("evidence_type", "utterance")
    weight = float(body.get("weight", 1.0))
    source_type = body.get("source_type", "manual")

    await proposition_store.add_evidence(
        proposition_id=proposition_id,
        evidence_type=evidence_type,
        direction=direction,
        source_type=source_type,
        weight=weight,
        speaker_id=body.get("speaker_id"),
        source_id=body.get("source_id"),
        quoted_text=body.get("quoted_text"),
    )
    return {"status": "evidence_added", "proposition_id": str(proposition_id)}


@router.patch("/v1/beliefs/{proposition_id}")
async def patch_belief(
    proposition_id: UUID,
    body: dict,
    tenant_id: str = Query(default="default"),
) -> dict:
    await proposition_store.update_belief(
        proposition_id=proposition_id,
        confidence=body.get("confidence"),
        utility_importance=body.get("utility_importance"),
        status=body.get("status"),
        tenant_id=tenant_id,
    )
    return {"status": "updated", "proposition_id": str(proposition_id)}


@router.get("/v1/propositions/{proposition_id}/evidence")
async def get_evidence(
    proposition_id: UUID,
    limit: int = Query(default=10, le=50),
) -> dict:
    evidences = await proposition_store.get_evidence(proposition_id, limit=limit)
    return {
        "evidence": [
            {
                "id": str(e.id),
                "evidence_type": e.evidence_type,
                "direction": e.direction,
                "source_type": e.source_type,
                "speaker_id": e.speaker_id,
                "quoted_text": e.quoted_text,
                "observed_at": e.observed_at.isoformat() if e.observed_at else None,
                "weight": e.weight,
                "created_at": e.created_at.isoformat(),
            }
            for e in evidences
        ],
        "count": len(evidences),
    }
