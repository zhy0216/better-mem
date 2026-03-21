from uuid import UUID

import structlog

from src.extract.prompts import BELIEF_UPDATE_PROMPT
from src.models.proposition import Proposition, PropositionCreate, ScoredProposition, get_evidence_weight
from src.services import llm as llm_service
from src.store import proposition_store
from src.store.database import get_pool

logger = structlog.get_logger(__name__)


async def update_beliefs(
    new_propositions: list[PropositionCreate],
    user_id: str,
    tenant_id: str = "default",
) -> list[PropositionCreate]:
    """For each new proposition, find candidates and add evidence to update beliefs.

    Returns the list of propositions that are genuinely new (not merged into existing ones).
    """
    genuinely_new: list[PropositionCreate] = []

    for new_prop in new_propositions:
        merged = False

        # 1. Try semantic_key exact match first
        if new_prop.semantic_key:
            candidates = await proposition_store.search_by_semantic_key(
                semantic_key=new_prop.semantic_key,
                user_id=new_prop.speaker_id or user_id,
                tenant_id=tenant_id,
            )
            if candidates:
                merged = await _process_candidates(
                    new_prop, candidates, user_id, tenant_id
                )

        # 2. Fall back to embedding similarity
        if not merged and new_prop.embedding:
            effective_user_id = new_prop.speaker_id or user_id
            try:
                similar = await proposition_store.search_similar(
                    embedding=new_prop.embedding,
                    user_id=effective_user_id,
                    tenant_id=tenant_id,
                    top_k=5,
                    score_threshold=0.85,
                )
            except Exception as e:
                logger.warning("belief_update_search_failed", error=str(e))
                similar = []

            if similar:
                merged = await _process_candidates(
                    new_prop, similar, user_id, tenant_id
                )

        if not merged:
            genuinely_new.append(new_prop)

    return genuinely_new


async def _process_candidates(
    new_prop: PropositionCreate,
    candidates: list[ScoredProposition],
    user_id: str,
    tenant_id: str,
) -> bool:
    """Use LLM to determine relationships, then add evidence accordingly.

    Returns True if the new proposition was merged into an existing one.
    """
    existing_text = "\n".join(
        f"[{c.id}] {c.canonical_text}" for c in candidates
    )
    prompt = BELIEF_UPDATE_PROMPT.format(
        new_proposition=new_prop.canonical_text,
        existing_propositions=existing_text,
    )

    try:
        data = await llm_service.complete_json(system_prompt=prompt)
    except Exception as e:
        logger.warning("belief_update_llm_failed", error=str(e))
        return False

    merged = False
    is_self = (new_prop.speaker_id is None) or (new_prop.speaker_id == user_id)
    evidence_type = new_prop.evidence_type or "utterance"
    weight = get_evidence_weight(evidence_type, is_self=is_self)

    for result in data.get("results", []):
        relationship = result.get("relationship", "unrelated")
        if relationship == "unrelated":
            continue

        existing_id = result.get("existing_proposition_id")
        if not existing_id:
            continue

        try:
            prop_id = UUID(existing_id)
        except (ValueError, TypeError):
            continue

        if relationship in ("contradicts", "updates"):
            # Add contradicting evidence to the OLD proposition
            try:
                await proposition_store.add_evidence(
                    proposition_id=prop_id,
                    evidence_type=evidence_type,
                    direction="contradict",
                    source_type=new_prop.source_type,
                    weight=weight,
                    speaker_id=new_prop.speaker_id,
                    source_id=new_prop.source_id,
                    source_meta=new_prop.source_meta,
                    quoted_text=new_prop.quoted_text or new_prop.canonical_text,
                    observed_at=new_prop.observed_at,
                )
                logger.info(
                    "belief_contradicted",
                    relationship=relationship,
                    old_proposition_id=existing_id,
                )
            except Exception as e:
                logger.warning("add_contradict_evidence_failed", error=str(e))

            # The new proposition itself is still genuinely new
            # (it will be inserted separately with supporting evidence)
            # Don't set merged=True here — the new prop needs its own row

        elif relationship == "supports":
            # Add supporting evidence to the existing proposition
            try:
                await proposition_store.add_evidence(
                    proposition_id=prop_id,
                    evidence_type=evidence_type,
                    direction="support",
                    source_type=new_prop.source_type,
                    weight=weight,
                    speaker_id=new_prop.speaker_id,
                    source_id=new_prop.source_id,
                    source_meta=new_prop.source_meta,
                    quoted_text=new_prop.quoted_text or new_prop.canonical_text,
                    observed_at=new_prop.observed_at,
                )
                logger.info(
                    "belief_supported",
                    existing_proposition_id=existing_id,
                )
                merged = True
            except Exception as e:
                logger.warning("add_support_evidence_failed", error=str(e))

    return merged


async def store_propositions_with_belief_update(
    new_propositions: list[PropositionCreate],
    genuinely_new: list[PropositionCreate],
    user_id: str,
    tenant_id: str = "default",
    group_id: str | None = None,
) -> list[Proposition]:
    """Insert genuinely new propositions (those not merged into existing ones)."""
    if not genuinely_new:
        return []

    pool = get_pool()
    async with pool.acquire() as conn:
        async with conn.transaction():
            saved = await proposition_store.insert_batch(
                genuinely_new, user_id, tenant_id, group_id, conn=conn
            )
    return saved
