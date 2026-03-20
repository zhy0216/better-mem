from datetime import datetime
from uuid import UUID

from pydantic import BaseModel


class ProfileData(BaseModel):
    skills: list[dict] = []
    personality: list[dict] = []
    preferences: list[dict] = []
    goals: list[dict] = []
    relations: list[dict] = []
    summary: str = ""


class Profile(BaseModel):
    id: UUID
    tenant_id: str
    user_id: str
    scope: str
    group_id: str | None
    profile_data: ProfileData
    version: int
    fact_count: int
    last_fact_id: UUID | None
    created_at: datetime
    updated_at: datetime

    class Config:
        from_attributes = True
