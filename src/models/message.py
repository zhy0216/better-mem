from datetime import datetime
from uuid import UUID

from pydantic import BaseModel


class Message(BaseModel):
    role: str
    speaker_id: str | None = None
    speaker_name: str | None = None
    content: str
    timestamp: datetime | None = None


class MessageBuffer(BaseModel):
    id: UUID
    tenant_id: str
    group_id: str
    user_id: str
    content: dict
    status: str
    batch_id: UUID | None
    created_at: datetime

    class Config:
        from_attributes = True
