from datetime import datetime, timezone
from dateutil import parser as dateutil_parser

import structlog

from src.extract.prompts import PROPOSITION_EXTRACTION_PROMPT
from src.models.proposition import (
    DECAY_RATES,
    PropositionCreate,
    get_prior,
)
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


def _normalize_semantic_key(key: str | None) -> str | None:
    """Post-process semantic_key: lowercase, strip, normalize separators."""
    if not key:
        return None
    key = key.strip().lower()
    key = key.replace(" ", "_").replace("-", "_").replace("/", ".")
    return key or None


async def extract_propositions(
    messages: list[MessageBuffer],
    user_id: str,
    timestamp: str | None = None,
) -> list[PropositionCreate]:
    if not timestamp:
        timestamp = datetime.now(tz=timezone.utc).isoformat()

    conversation = _format_messages(messages)
    participant_map = _extract_participant_map(messages)
    participants_str = ", ".join(f"{name} -> {sid}" for name, sid in participant_map.items())

    prompt = PROPOSITION_EXTRACTION_PROMPT.format(
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
        logger.error("proposition_extraction_failed", error=str(e))
        return []

    raw_props = data.get("propositions", [])
    logger.info("propositions_extracted", count=len(raw_props))

    creates: list[PropositionCreate] = []
    for p in raw_props:
        prop_type = p.get("proposition_type", "observation")
        observed_at = _parse_datetime(p.get("observed_at"), fallback=timestamp)
        evidence_type = p.get("evidence_type", "utterance")
        semantic_key = _normalize_semantic_key(p.get("semantic_key"))
        prior = get_prior(prop_type, evidence_type)

        pc = PropositionCreate(
            canonical_text=p["canonical_text"],
            proposition_type=prop_type,
            semantic_key=semantic_key,
            subject_id=p.get("subject_id") or None,
            valid_from=_parse_datetime(p.get("valid_from")),
            valid_until=_parse_datetime(p.get("valid_until")),
            first_observed_at=observed_at,
            tags=p.get("tags", []),
            importance=float(p.get("importance", 0.5)),
            prior=prior,
            evidence_type=evidence_type,
            speaker_id=p.get("speaker_id") or None,
            source_type="conversation",
            quoted_text=p.get("quoted_text"),
            observed_at=observed_at,
        )
        creates.append(pc)

    if creates:
        texts = [pc.canonical_text for pc in creates]
        try:
            embeddings = await embedding_service.embed_batch(texts)
            for pc, emb in zip(creates, embeddings):
                pc.embedding = emb
        except Exception as e:
            logger.error("embedding_failed", error=str(e))

    return creates
