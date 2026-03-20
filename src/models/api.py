from datetime import datetime
from uuid import UUID

from pydantic import BaseModel, Field

from src.models.message import Message
from src.models.fact import ScoredFact


class MemorizeRequest(BaseModel):
    tenant_id: str = "default"
    group_id: str
    messages: list[Message]
    source_type: str = "conversation"
    extract_mode: str = "async"


class MemorizeAsyncResponse(BaseModel):
    status: str = "accepted"
    batch_id: UUID
    message_count: int


class MemorizeSyncResponse(BaseModel):
    status: str = "completed"
    facts: list[dict]


class SearchFilters(BaseModel):
    group_id: str | None = None
    fact_types: list[str] | None = None
    time_range: dict | None = None
    tags: list[str] | None = None
    status: list[str] = Field(default_factory=lambda: ["active"])


class RecallRequest(BaseModel):
    tenant_id: str = "default"
    user_id: str
    query: str
    top_k: int = 20
    filters: SearchFilters = Field(default_factory=SearchFilters)
    include_profile: bool = True
    assemble: bool = True


class ProfileSnippet(BaseModel):
    summary: str
    relevant_traits: list[str] = Field(default_factory=list)


class RecallAssembledResponse(BaseModel):
    context: str
    facts: list[dict]
    profile_snippet: ProfileSnippet | None = None
    total_candidates: int
    search_time_ms: float


class RecallRawResponse(BaseModel):
    facts: list[dict]
    total_candidates: int
    search_time_ms: float


class AssembledContext(BaseModel):
    context: str
    selected_fact_ids: list[str]
    confidence: float = 1.0
    information_gaps: list[str] = Field(default_factory=list)


class FactListRequest(BaseModel):
    user_id: str
    tenant_id: str = "default"
    fact_type: str | None = None
    status: str | None = "active"
    limit: int = 50
    offset: int = 0
