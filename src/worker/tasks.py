from collections import defaultdict

import structlog

from src.extract.contradiction import detect_contradictions, store_facts_with_contradictions
from src.extract.fact_extractor import extract_facts
from src.extract.profile_synthesizer import ProfileUpdateTrigger, synthesize_profile
from src.extract.session_detector import SessionDetector
from src.store import buffer_store
from src.store.cache import release_extract_lock

_STUCK_PROCESSING_MINUTES = 10

logger = structlog.get_logger(__name__)

_session_detector = SessionDetector()
_profile_trigger = ProfileUpdateTrigger()


async def process_group(ctx: dict, group_id: str, tenant_id: str = "default") -> dict:
    """Extract facts from buffered messages for a group."""
    logger.info("process_group_start", group_id=group_id, tenant_id=tenant_id)

    pending = await buffer_store.get_pending(group_id, tenant_id)
    if not pending:
        return {"status": "no_pending_messages"}

    user_ids = [str(m.content["speaker_id"]) for m in pending if m.content.get("speaker_id")]
    primary_user_id: str = user_ids[0] if user_ids else pending[0].user_id

    await buffer_store.mark_processing([m.id for m in pending])

    try:
        new_facts = await extract_facts(messages=pending, user_id=primary_user_id)

        contradictions = []
        try:
            contradictions = await detect_contradictions(new_facts, primary_user_id, tenant_id)
        except Exception as e:
            logger.warning("contradiction_detection_skipped", group_id=group_id, error=str(e))

        saved = await store_facts_with_contradictions(
            new_facts=new_facts,
            contradictions=contradictions,
            user_id=primary_user_id,
            tenant_id=tenant_id,
            group_id=group_id,
        )

        await buffer_store.mark_consumed([m.id for m in pending])

        facts_by_user: dict[str, list] = defaultdict(list)
        for f in saved:
            facts_by_user[f.user_id].append(f)

        for uid, user_facts in facts_by_user.items():
            try:
                if await _profile_trigger.should_update(uid, user_facts, tenant_id):
                    await synthesize_profile(uid, user_facts, tenant_id)
                    logger.info("profile_updated", user_id=uid)
            except Exception as e:
                logger.warning("profile_update_skipped", user_id=uid, error=str(e))

        logger.info("process_group_done", group_id=group_id, facts_saved=len(saved))
        return {"status": "ok", "facts_saved": len(saved)}

    except Exception as e:
        logger.error("process_group_failed", group_id=group_id, error=str(e))
        raise
    finally:
        await release_extract_lock(tenant_id, group_id)


async def scan_groups(ctx: dict) -> dict:
    """Periodic task: scan all groups with pending messages and trigger extraction."""
    from src.store.buffer_store import get_active_group_ids, recover_stuck_processing
    from src.store.cache import acquire_extract_lock

    await recover_stuck_processing(stuck_minutes=_STUCK_PROCESSING_MINUTES)

    groups = await get_active_group_ids()
    triggered = 0

    for tenant_id, group_id in groups:
        try:
            should = await _session_detector.should_extract(group_id, tenant_id)
            if not should:
                continue

            locked = await acquire_extract_lock(tenant_id, group_id)
            if not locked:
                logger.info("extract_lock_busy", group_id=group_id)
                continue

            await ctx["redis"].enqueue_job("process_group", group_id, tenant_id)
            triggered += 1
        except Exception as e:
            logger.error("scan_group_error", group_id=group_id, error=str(e))

    return {"triggered": triggered, "scanned": len(groups)}


async def decay_sweep(ctx: dict) -> dict:
    """Periodic task: expire facts that are past their valid_until date."""
    from src.store.database import get_pool

    pool = get_pool()
    async with pool.acquire() as conn:
        result = await conn.execute("""
            UPDATE facts
            SET status = 'expired', updated_at = now()
            WHERE status = 'active'
              AND valid_until IS NOT NULL
              AND valid_until < now()
        """)
    updated = int(result.split()[-1]) if result else 0
    logger.info("decay_sweep_done", expired_count=updated)
    return {"expired": updated}
