from collections import defaultdict

import structlog

from src.extract.contradiction import detect_contradictions, store_facts_with_contradictions
from src.extract.fact_extractor import extract_facts
from src.extract.profile_synthesizer import ProfileUpdateTrigger, synthesize_profile
from src.models.fact import Fact
from src.models.message import MessageBuffer

logger = structlog.get_logger(__name__)

_profile_trigger = ProfileUpdateTrigger()


async def process_buffered_messages(
    buffers: list[MessageBuffer],
    tenant_id: str,
    group_id: str,
    user_id: str = "unknown",
) -> list[Fact]:
    """Extract facts, resolve contradictions, persist, and update profiles.

    This is the single application-level use case for the memorize flow.
    Both the sync HTTP path and the async worker path call this function so
    system state is always consistent regardless of entry point.
    """
    new_facts = await extract_facts(messages=buffers, user_id=user_id)

    contradictions = []
    try:
        contradictions = await detect_contradictions(new_facts, user_id, tenant_id)
    except Exception as e:
        logger.warning("contradiction_detection_skipped", error=str(e))

    saved = await store_facts_with_contradictions(
        new_facts=new_facts,
        contradictions=contradictions,
        user_id=user_id,
        tenant_id=tenant_id,
        group_id=group_id,
    )

    facts_by_user: dict[str, list[Fact]] = defaultdict(list)
    for f in saved:
        facts_by_user[f.user_id].append(f)

    for uid, user_facts in facts_by_user.items():
        try:
            if await _profile_trigger.should_update(uid, user_facts, tenant_id):
                await synthesize_profile(uid, user_facts, tenant_id)
                logger.info("profile_updated", user_id=uid)
        except Exception as e:
            logger.warning("profile_update_skipped", user_id=uid, error=str(e))

    return saved
