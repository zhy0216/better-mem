from datetime import datetime
from uuid import UUID

from pydantic import BaseModel, Field


class SearchFilters(BaseModel):
    group_id: str | None = None
    fact_types: list[str] | None = None
    time_range: dict | None = None
    tags: list[str] | None = None
    status: list[str] = Field(default_factory=lambda: ["active"])


class AssembledContext(BaseModel):
    context: str
    selected_fact_ids: list[str]
    confidence: float = 1.0
    information_gaps: list[str] = Field(default_factory=list)


class FactCreate(BaseModel):
    content: str
    fact_type: str = "observation"
    occurred_at: datetime | None = None
    valid_from: datetime | None = None
    valid_until: datetime | None = None
    importance: float = 0.5
    decay_rate: float = 0.01
    source_type: str = "conversation"
    source_id: str | None = None
    source_meta: dict | None = None
    tags: list[str] = Field(default_factory=list)
    metadata: dict = Field(default_factory=dict)
    embedding: list[float] | None = None
    speaker_id: str | None = None


class FactUpdate(BaseModel):
    tags: list[str] | None = None
    metadata: dict | None = None
    importance: float | None = None


class Fact(BaseModel):
    id: UUID
    tenant_id: str
    user_id: str
    group_id: str | None
    content: str
    fact_type: str
    occurred_at: datetime
    valid_from: datetime | None
    valid_until: datetime | None
    superseded_by: UUID | None
    supersedes: UUID | None
    status: str
    importance: float
    access_count: int
    last_accessed: datetime | None
    decay_rate: float
    source_type: str
    source_id: str | None
    source_meta: dict | None
    tags: list[str]
    metadata: dict
    created_at: datetime
    updated_at: datetime

    class Config:
        from_attributes = True


class ScoredFact(BaseModel):
    id: UUID
    content: str
    fact_type: str
    occurred_at: datetime
    importance: float
    decay_rate: float = 0.01
    access_count: int = 0
    metadata: dict
    tags: list[str]
    score: float
    source: str


class ContradictionPair(BaseModel):
    old_fact_id: UUID
    new_fact: FactCreate
    new_fact_index: int
    relation: str
