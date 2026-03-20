from datetime import datetime, timezone

import structlog

from src.extract.prompts import FACT_EXTRACTION_PROMPT
from src.models.fact import FactCreate
from src.models.message import MessageBuffer
from src.services import embedding as embedding_service
from src.services import llm as llm_service
from src.config import settings

logger = structlog.get_logger(__name__)


def _format_messages(messages: list[MessageBuffer]) -> str:
    lines = []
    for m in messages:
        content = m.content
        role = content.get("role", "user")
        name = content.get("speaker_name") or content.get("speaker_id", "unknown")
        text = content.get("content", "")
        ts = content.get("timestamp", "")
        lines.append(f"[{ts}] {name} ({role}): {text}")
    return "\n".join(lines)


def _extract_participants(messages: list[MessageBuffer]) -> list[str]:
    seen: set[str] = set()
    names = []
    for m in messages:
        content = m.content
        name = content.get("speaker_name") or content.get("speaker_id")
        if name and name not in seen:
            seen.add(name)
            names.append(name)
    return names


async def extract_facts(
    messages: list[MessageBuffer],
    user_id: str,
    timestamp: str | None = None,
) -> list[FactCreate]:
    if not timestamp:
        timestamp = datetime.now(tz=timezone.utc).isoformat()

    conversation = _format_messages(messages)
    participants = _extract_participants(messages)

    prompt = FACT_EXTRACTION_PROMPT.format(
        timestamp=timestamp,
        participants=", ".join(participants),
        conversation=conversation,
    )

    try:
        data = await llm_service.complete_json(
            system_prompt=prompt,
            model=settings.EXTRACT_MODEL,
            temperature=0.1,
        )
    except Exception as e:
        logger.error("fact_extraction_failed", error=str(e))
        return []

    raw_facts = data.get("facts", [])
    logger.info("facts_extracted", count=len(raw_facts))

    fact_creates: list[FactCreate] = []
    for f in raw_facts:
        fc = FactCreate(
            content=f["content"],
            fact_type=f.get("fact_type", "observation"),
            importance=float(f.get("importance", 0.5)),
            valid_from=f.get("valid_from"),
            valid_until=f.get("valid_until"),
            tags=f.get("tags", []),
            source_type="conversation",
        )
        fact_creates.append(fc)

    if fact_creates:
        texts = [fc.content for fc in fact_creates]
        try:
            embeddings = await embedding_service.embed_batch(texts)
            for fc, emb in zip(fact_creates, embeddings):
                fc.embedding = emb
        except Exception as e:
            logger.error("embedding_failed", error=str(e))

    return fact_creates
