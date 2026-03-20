from datetime import datetime, timedelta, timezone

import structlog

from src.config import settings
from src.services.tokenizer import count_tokens
from src.store import buffer_store

logger = structlog.get_logger(__name__)


class SessionDetector:
    def __init__(self) -> None:
        self.time_gap = timedelta(minutes=settings.SESSION_TIME_GAP_MINUTES)
        self.max_messages = settings.SESSION_MAX_MESSAGES
        self.max_tokens = settings.SESSION_MAX_TOKENS
        self.min_messages = settings.SESSION_MIN_MESSAGES

    async def should_extract(self, group_id: str, tenant_id: str = "default") -> bool:
        pending = await buffer_store.get_pending(group_id, tenant_id)

        if len(pending) < self.min_messages:
            return False

        last_msg_time = pending[-1].created_at
        if last_msg_time.tzinfo is None:
            last_msg_time = last_msg_time.replace(tzinfo=timezone.utc)
        now = datetime.now(tz=timezone.utc)

        if now - last_msg_time > self.time_gap:
            logger.info("session_trigger_time_gap", group_id=group_id)
            return True

        if len(pending) >= self.max_messages:
            logger.info("session_trigger_max_messages", group_id=group_id)
            return True

        total_tokens = sum(
            count_tokens(str(m.content.get("content", ""))) for m in pending
        )
        if total_tokens >= self.max_tokens:
            logger.info("session_trigger_max_tokens", group_id=group_id, tokens=total_tokens)
            return True

        return False
