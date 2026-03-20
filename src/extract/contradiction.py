from uuid import UUID

import structlog

from src.extract.prompts import CONTRADICTION_CHECK_PROMPT
from src.models.fact import ContradictionPair, Fact, FactCreate
from src.services import llm as llm_service
from src.store import fact_store

logger = structlog.get_logger(__name__)


async def detect_contradictions(
    new_facts: list[FactCreate],
    user_id: str,
    tenant_id: str = "default",
) -> list[ContradictionPair]:
    contradictions: list[ContradictionPair] = []

    for idx, new_fact in enumerate(new_facts):
        if not new_fact.embedding:
            continue

        try:
            similar = await fact_store.search_similar(
                embedding=new_fact.embedding,
                user_id=user_id,
                tenant_id=tenant_id,
                top_k=5,
                score_threshold=0.85,
            )
        except Exception as e:
            logger.warning("contradiction_search_failed", error=str(e))
            continue

        if not similar:
            continue

        existing_facts_text = "\n".join(
            f"[{f.id}] {f.content}" for f in similar
        )
        prompt = CONTRADICTION_CHECK_PROMPT.format(
            new_fact=new_fact.content,
            existing_facts=existing_facts_text,
        )

        try:
            data = await llm_service.complete_json(system_prompt=prompt)
        except Exception as e:
            logger.warning("contradiction_llm_failed", error=str(e))
            continue

        for result in data.get("results", []):
            relationship = result.get("relationship", "unrelated")
            if relationship in ("contradicts", "updates"):
                contradictions.append(
                    ContradictionPair(
                        old_fact_id=UUID(result["existing_fact_id"]),
                        new_fact=new_fact,
                        new_fact_index=idx,
                        relation="supersedes",
                    )
                )
                logger.info(
                    "contradiction_detected",
                    relationship=relationship,
                    old_fact_id=result["existing_fact_id"],
                )

    return contradictions


async def store_facts_with_contradictions(
    new_facts: list[FactCreate],
    contradictions: list[ContradictionPair],
    user_id: str,
    tenant_id: str = "default",
    group_id: str | None = None,
) -> list[Fact]:
    saved = await fact_store.insert_batch(new_facts, user_id, tenant_id, group_id)

    for contradiction in contradictions:
        if contradiction.new_fact_index < len(saved):
            new_id = saved[contradiction.new_fact_index].id
            try:
                await fact_store.mark_superseded(
                    old_fact_id=contradiction.old_fact_id,
                    superseded_by=new_id,
                )
            except Exception as e:
                logger.warning("mark_superseded_failed", error=str(e))

    return saved
