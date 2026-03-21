from datetime import datetime
from uuid import UUID

from pydantic import BaseModel, Field

from src.models.message import Message
from src.models.proposition import AssembledContext, SearchFilters


class MemorizeRequest(BaseModel):
    tenant_id: str = "default"
    group_id: str
    user_id: str = "unknown"
    messages: list[Message]
    source_type: str = "conversation"
    extract_mode: str = "async"


class MemorizeAsyncResponse(BaseModel):
    status: str = "accepted"
    batch_id: UUID
    message_count: int


class MemorizeSyncResponse(BaseModel):
    status: str = "completed"
    propositions: list[dict]


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
    propositions: list[dict]
    profile_snippet: ProfileSnippet | None = None
    total_candidates: int
    search_time_ms: float


class RecallRawResponse(BaseModel):
    propositions: list[dict]
    total_candidates: int
    search_time_ms: float


class PropositionListRequest(BaseModel):
    user_id: str
    tenant_id: str = "default"
    proposition_type: str | None = None
    status: str | None = "active"
    limit: int = 50
    offset: int = 0
