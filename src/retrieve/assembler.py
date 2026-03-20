import structlog

from src.config import settings
from src.models.api import AssembledContext
from src.models.fact import ScoredFact
from src.models.profile import Profile
from src.retrieve.prompts import ASSEMBLE_PROMPT
from src.services import llm as llm_service

logger = structlog.get_logger(__name__)


async def assemble_context(
    query: str,
    candidates: list[ScoredFact],
    profile: Profile | None = None,
) -> AssembledContext:
    facts_text = "\n".join(
        f"[{f.id}] ({f.occurred_at.isoformat()}) {f.content}" for f in candidates
    )
    profile_summary = (
        profile.profile_data.summary if profile and profile.profile_data.summary else "No profile available"
    )

    prompt = ASSEMBLE_PROMPT.format(
        query=query,
        facts=facts_text,
        profile_summary=profile_summary,
    )

    try:
        data = await llm_service.complete_json(
            system_prompt=prompt,
            model=settings.ASSEMBLE_MODEL,
            temperature=0.1,
        )
        return AssembledContext(
            context=data.get("context", ""),
            selected_fact_ids=data.get("selected_fact_ids", []),
            confidence=float(data.get("confidence", 1.0)),
            information_gaps=data.get("information_gaps", []),
        )
    except Exception as e:
        logger.warning("assemble_context_failed", error=str(e))
        fallback_context = " ".join(f.content for f in candidates[:5])
        return AssembledContext(
            context=fallback_context,
            selected_fact_ids=[str(f.id) for f in candidates],
            confidence=0.0,
            information_gaps=["Context assembly failed; raw facts returned."],
        )
