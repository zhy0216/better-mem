import math
from datetime import datetime, timedelta, timezone
from uuid import uuid4

import pytest

from src.models.proposition import ScoredProposition
from src.retrieve.ranker import compute_retrieval_score, reciprocal_rank_fusion


def make_prop(score: float = 0.9, source: str = "vector", pid=None) -> ScoredProposition:
    return ScoredProposition(
        id=pid or uuid4(),
        canonical_text="some proposition",
        proposition_type="observation",
        confidence=0.7,
        utility_importance=0.5,
        freshness_decay=0.01,
        access_count=0,
        belief_status="active",
        first_observed_at=datetime(2024, 3, 14, tzinfo=timezone.utc),
        last_observed_at=datetime(2024, 3, 14, tzinfo=timezone.utc),
        metadata={},
        tags=[],
        score=score,
        source=source,
    )


def test_rrf_distinct_propositions():
    vec = [make_prop(0.9, "vector"), make_prop(0.85, "vector")]
    kw = [make_prop(0.8, "keyword"), make_prop(0.75, "keyword")]
    merged = reciprocal_rank_fusion(vec, kw)
    assert len(merged) == 4


def test_rrf_deduplicates_shared_id():
    shared_id = uuid4()
    vec = [make_prop(0.9, "vector", pid=shared_id)]
    kw = [make_prop(0.8, "keyword", pid=shared_id)]
    merged = reciprocal_rank_fusion(vec, kw)
    assert len(merged) == 1
    assert merged[0].id == shared_id


def test_rrf_higher_ranked_in_both_gets_top_score():
    id_a = uuid4()
    id_b = uuid4()
    vec = [make_prop(pid=id_a), make_prop(pid=id_b)]
    kw = [make_prop(pid=id_a), make_prop(pid=id_b)]
    merged = reciprocal_rank_fusion(vec, kw)
    assert merged[0].id == id_a


def test_compute_retrieval_score_decays_over_time():
    now = datetime.now(tz=timezone.utc)
    recent = datetime(2024, 3, 14, tzinfo=timezone.utc)
    old = recent - timedelta(days=365)
    score_recent = compute_retrieval_score(
        semantic_relevance=0.8, confidence=0.7, utility_importance=0.5,
        freshness_decay=0.01, last_observed_at=recent, access_count=0, now=now,
    )
    score_old = compute_retrieval_score(
        semantic_relevance=0.8, confidence=0.7, utility_importance=0.5,
        freshness_decay=0.01, last_observed_at=old, access_count=0, now=now,
    )
    assert score_recent > score_old


def test_compute_retrieval_score_access_boost():
    now = datetime.now(tz=timezone.utc)
    ts = datetime(2024, 3, 14, tzinfo=timezone.utc)
    score_no_access = compute_retrieval_score(
        semantic_relevance=0.8, confidence=0.5, utility_importance=0.5,
        freshness_decay=0.01, last_observed_at=ts, access_count=0, now=now,
    )
    score_with_access = compute_retrieval_score(
        semantic_relevance=0.8, confidence=0.5, utility_importance=0.5,
        freshness_decay=0.01, last_observed_at=ts, access_count=10, now=now,
    )
    assert score_with_access > score_no_access
