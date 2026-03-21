from datetime import datetime, timedelta, timezone

import structlog

from src.config import settings
from src.extract.prompts import PROFILE_SYNTHESIS_PROMPT
from src.models.proposition import Proposition
from src.models.profile import Profile, ProfileData
from src.services import llm as llm_service
from src.store import cache, proposition_store, profile_store

logger = structlog.get_logger(__name__)


class ProfileUpdateTrigger:
    def __init__(self) -> None:
        self.prop_count_threshold = settings.PROFILE_FACT_THRESHOLD
        self.time_threshold = timedelta(hours=settings.PROFILE_TIME_THRESHOLD_HOURS)
        self.force_on_declaration = settings.PROFILE_FORCE_ON_DECLARATION

    async def should_update(
        self,
        user_id: str,
        new_propositions: list[Proposition],
        tenant_id: str = "default",
    ) -> bool:
        if self.force_on_declaration:
            if any(p.proposition_type == "declaration" for p in new_propositions):
                return True

        profile = await profile_store.get(user_id, tenant_id)
        if profile is None:
            return True

        props_since = await proposition_store.count_since(user_id, profile.updated_at, tenant_id)
        if props_since >= self.prop_count_threshold:
            return True

        updated_at = profile.updated_at
        if updated_at.tzinfo is None:
            updated_at = updated_at.replace(tzinfo=timezone.utc)
        if datetime.now(tz=timezone.utc) - updated_at > self.time_threshold:
            return True

        return False


async def synthesize_profile(
    user_id: str,
    new_propositions: list[Proposition],
    tenant_id: str = "default",
    scope: str = "global",
    group_id: str | None = None,
) -> Profile:
    existing = await profile_store.get(user_id, tenant_id, scope, group_id)
    existing_profile_str = existing.profile_data.model_dump_json() if existing else "{}"

    new_props_text = "\n".join(
        f"[{p.id}] ({p.proposition_type}) {p.canonical_text}" for p in new_propositions
    )

    prompt = PROFILE_SYNTHESIS_PROMPT.format(
        existing_profile=existing_profile_str,
        new_propositions=new_props_text,
    )

    try:
        data = await llm_service.complete_json(
            system_prompt=prompt,
            model=settings.EXTRACT_MODEL,
        )
        profile_data = ProfileData(
            skills=data.get("skills", []),
            personality=data.get("personality", []),
            preferences=data.get("preferences", []),
            goals=data.get("goals", []),
            relations=data.get("relations", []),
            summary=data.get("summary", ""),
        )
    except Exception as e:
        logger.error("profile_synthesis_failed", error=str(e))
        profile_data = existing.profile_data if existing else ProfileData()

    last_prop_id = new_propositions[-1].id if new_propositions else None
    updated = await profile_store.upsert(
        user_id=user_id,
        profile_data=profile_data,
        tenant_id=tenant_id,
        scope=scope,
        group_id=group_id,
        last_proposition_id=last_prop_id,
        fact_count=len(new_propositions),
    )
    await cache.invalidate_profile_cache(tenant_id, user_id, scope, group_id)
    return updated
