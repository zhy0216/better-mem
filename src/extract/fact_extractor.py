from datetime import datetime, timezone
from dateutil import parser as dateutil_parser

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


_DECAY_RATES: dict[str, float] = {
    "plan": 0.05,
    "observation": 0.02,
    "declaration": 0.005,
    "preference": 0.005,
    "relation": 0.003,
}


def _extract_participant_map(messages: list[MessageBuffer]) -> dict[str, str]:
    """Return {speaker_name: speaker_id} for all participants."""
    result: dict[str, str] = {}
    for m in messages:
        content = m.content
        name = content.get("speaker_name")
        sid = content.get("speaker_id")
        if name and sid and name not in result:
            result[name] = sid
    return result


def _parse_datetime(value: str | None, fallback: str | None = None) -> datetime | None:
    """Parse an ISO-8601 string into a timezone-aware datetime, with optional fallback."""
    for raw in (value, fallback):
        if not raw:
            continue
        try:
            dt = dateutil_parser.parse(raw)
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=timezone.utc)
            return dt
        except Exception:
            continue
    return None


async def extract_facts(
    messages: list[MessageBuffer],
    user_id: str,
    timestamp: str | None = None,
) -> list[FactCreate]:
    if not timestamp:
        timestamp = datetime.now(tz=timezone.utc).isoformat()

    conversation = _format_messages(messages)
    participant_map = _extract_participant_map(messages)
    participants_str = ", ".join(f"{name} -> {sid}" for name, sid in participant_map.items())

    prompt = FACT_EXTRACTION_PROMPT.format(
        timestamp=timestamp,
        participants=participants_str or user_id,
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
        fact_type = f.get("fact_type", "observation")
        occurred_at = _parse_datetime(f.get("occurred_at"), fallback=timestamp)
        fc = FactCreate(
            content=f["content"],
            fact_type=fact_type,
            occurred_at=occurred_at,
            importance=float(f.get("importance", 0.5)),
            decay_rate=_DECAY_RATES.get(fact_type, 0.01),
            valid_from=f.get("valid_from"),
            valid_until=f.get("valid_until"),
            tags=f.get("tags", []),
            source_type="conversation",
            speaker_id=f.get("speaker_id") or None,
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
