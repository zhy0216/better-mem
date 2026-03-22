from collections import defaultdict
from uuid import UUID

import structlog

from src.extract.belief_updater import update_beliefs, store_propositions_with_belief_update
from src.extract.proposition_extractor import extract_propositions
from src.extract.profile_synthesizer import ProfileUpdateTrigger, synthesize_profile
from src.models.proposition import Proposition
from src.models.message import MessageBuffer
from src.store import buffer_store

logger = structlog.get_logger(__name__)

_profile_trigger = ProfileUpdateTrigger()


async def process_and_track(
    buffers: list[MessageBuffer],
    tenant_id: str,
    group_id: str,
    user_id: str = "unknown",
) -> list[Proposition]:
    """Full memorize lifecycle: mark_processing → extract → mark_consumed.

    On failure the buffers are reset to 'pending' and the exception re-raised.
    """
    ids = [b.id for b in buffers]
    await buffer_store.mark_processing(ids)
    try:
        saved = await process_buffered_messages(
            buffers=buffers,
            tenant_id=tenant_id,
            group_id=group_id,
            user_id=user_id,
        )
        await buffer_store.mark_consumed(ids)
        return saved
    except Exception:
        await buffer_store.mark_pending(ids)
        raise


async def process_buffered_messages(
    buffers: list[MessageBuffer],
    tenant_id: str,
    group_id: str,
    user_id: str = "unknown",
) -> list[Proposition]:
    """Extract propositions, update beliefs, persist, and update profiles.

    This is the single application-level use case for the memorize flow.
    Both the sync HTTP path and the async worker path call this function so
    system state is always consistent regardless of entry point.
    """
    new_propositions = await extract_propositions(messages=buffers, user_id=user_id)

    genuinely_new = new_propositions
    try:
        genuinely_new = await update_beliefs(new_propositions, user_id, tenant_id)
    except Exception as e:
        logger.warning("belief_update_skipped", error=str(e))

    saved = await store_propositions_with_belief_update(
        new_propositions=new_propositions,
        genuinely_new=genuinely_new,
        user_id=user_id,
        tenant_id=tenant_id,
        group_id=group_id,
    )

    props_by_user: dict[str, list[Proposition]] = defaultdict(list)
    for p in saved:
        props_by_user[p.user_id].append(p)

    for uid, user_props in props_by_user.items():
        try:
            if await _profile_trigger.should_update(uid, user_props, tenant_id):
                await synthesize_profile(uid, user_props, tenant_id)
                logger.info("profile_updated", user_id=uid)
        except Exception as e:
            logger.warning("profile_update_skipped", user_id=uid, error=str(e))

    return saved
