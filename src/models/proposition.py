from datetime import datetime
from enum import StrEnum
from typing import Literal
from uuid import UUID

from pydantic import BaseModel, Field


class PropositionType(StrEnum):
    OBSERVATION = "observation"
    DECLARATION = "declaration"
    PLAN = "plan"
    PREFERENCE = "preference"
    RELATION = "relation"


class BeliefStatus(StrEnum):
    ACTIVE = "active"
    UNCERTAIN = "uncertain"
    STALE = "stale"
    DEPRECATED = "deprecated"


class EvidenceType(StrEnum):
    UTTERANCE = "utterance"
    OBSERVATION = "observation"
    IMPORT = "import"
    INFERENCE = "inference"


class EvidenceDirection(StrEnum):
    SUPPORT = "support"
    CONTRADICT = "contradict"
    NEUTRAL = "neutral"


# ---------------------------------------------------------------------------
# Search / filter helpers
# ---------------------------------------------------------------------------

class SearchFilters(BaseModel):
    group_id: str | None = None
    proposition_types: list[PropositionType] | None = None
    time_range: dict | None = None
    tags: list[str] | None = None
    status: list[str] = Field(default_factory=lambda: ["active"])
    min_confidence: float | None = None


class AssembledContext(BaseModel):
    context: str
    selected_proposition_ids: list[str]
    confidence: float = 1.0
    information_gaps: list[str] = Field(default_factory=list)


# ---------------------------------------------------------------------------
# Create DTOs
# ---------------------------------------------------------------------------

class PropositionCreate(BaseModel):
    canonical_text: str
    proposition_type: PropositionType = PropositionType.OBSERVATION
    semantic_key: str | None = None
    subject_id: str | None = None

    valid_from: datetime | None = None
    valid_until: datetime | None = None
    first_observed_at: datetime | None = None

    tags: list[str] = Field(default_factory=list)
    metadata: dict = Field(default_factory=dict)
    embedding: list[float] | None = None

    # Evidence seed (first evidence for this proposition)
    evidence_type: EvidenceType = EvidenceType.UTTERANCE
    speaker_id: str | None = None
    source_type: str = "conversation"
    source_id: str | None = None
    source_meta: dict | None = None
    quoted_text: str | None = None
    observed_at: datetime | None = None

    # Belief hints from LLM
    importance: float = 0.5
    prior: float | None = None


class PropositionUpdate(BaseModel):
    canonical_text: str | None = None
    tags: list[str] | None = None
    metadata: dict | None = None


# ---------------------------------------------------------------------------
# Domain models (DB row representations)
# ---------------------------------------------------------------------------

class Proposition(BaseModel):
    id: UUID
    tenant_id: str
    user_id: str
    group_id: str | None
    subject_id: str | None

    canonical_text: str
    proposition_type: PropositionType
    semantic_key: str | None

    valid_from: datetime | None
    valid_until: datetime | None
    first_observed_at: datetime | None
    last_observed_at: datetime | None

    tags: list[str]
    metadata: dict

    created_at: datetime
    updated_at: datetime

    class Config:
        from_attributes = True


class Belief(BaseModel):
    id: UUID
    proposition_id: UUID

    confidence: float
    prior: float
    source_reliability: float

    utility_importance: float
    freshness_decay: float

    support_count: int = 0
    contradiction_count: int = 0

    access_count: int = 0
    last_accessed: datetime | None = None

    status: BeliefStatus

    created_at: datetime
    updated_at: datetime

    class Config:
        from_attributes = True


class Evidence(BaseModel):
    id: UUID
    proposition_id: UUID

    evidence_type: EvidenceType
    direction: EvidenceDirection

    source_type: str
    source_id: str | None
    source_meta: dict | None

    speaker_id: str | None
    quoted_text: str | None
    observed_at: datetime | None

    weight: float
    metadata: dict

    created_at: datetime

    class Config:
        from_attributes = True


# ---------------------------------------------------------------------------
# Composite / scored models for retrieval
# ---------------------------------------------------------------------------

class ScoredProposition(BaseModel):
    id: UUID
    canonical_text: str
    proposition_type: PropositionType
    semantic_key: str | None = None

    # Belief fields (joined)
    confidence: float = 0.5
    utility_importance: float = 0.5
    freshness_decay: float = 0.01
    access_count: int = 0
    belief_status: str = "active"

    first_observed_at: datetime | None = None
    last_observed_at: datetime | None = None

    metadata: dict = Field(default_factory=dict)
    tags: list[str] = Field(default_factory=list)

    score: float = 0.0
    source: str = "vector"


class BeliefUpdateCandidate(BaseModel):
    proposition_id: UUID
    canonical_text: str
    semantic_key: str | None
    confidence: float
    direction: EvidenceDirection = EvidenceDirection.SUPPORT


# ---------------------------------------------------------------------------
# Prior / weight / decay reference tables
# ---------------------------------------------------------------------------

PRIOR_TABLE: dict[str, dict[str, float]] = {
    "declaration": {"declaration": 0.85, "default": 0.60},
    "preference":  {"declaration": 0.80, "default": 0.55},
    "relation":    {"declaration": 0.75, "default": 0.50},
    "observation": {"declaration": 0.65, "default": 0.50},
    "plan":        {"declaration": 0.60, "default": 0.45},
}

EVIDENCE_WEIGHT_TABLE: dict[str, float] = {
    "utterance_self":   1.0,
    "utterance_other":  0.6,
    "observation":      0.7,
    "import":           0.8,
    "inference":        0.4,
}

DECAY_RATES: dict[str, float] = {
    "plan":        0.05,
    "observation": 0.02,
    "declaration": 0.005,
    "preference":  0.005,
    "relation":    0.003,
}


def get_prior(proposition_type: str, evidence_type: str) -> float:
    row = PRIOR_TABLE.get(proposition_type, PRIOR_TABLE["observation"])
    if evidence_type in ("utterance",):
        return row.get("declaration", row["default"])
    return row["default"]


def get_evidence_weight(evidence_type: str, is_self: bool = True) -> float:
    if evidence_type == "utterance":
        return EVIDENCE_WEIGHT_TABLE["utterance_self" if is_self else "utterance_other"]
    return EVIDENCE_WEIGHT_TABLE.get(evidence_type, 0.5)


def get_decay_rate(proposition_type: str) -> float:
    return DECAY_RATES.get(proposition_type, 0.01)
