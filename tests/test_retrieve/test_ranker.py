import math
from datetime import datetime, timedelta, timezone
from uuid import uuid4

import pytest

from src.models.fact import ScoredFact
from src.retrieve.ranker import compute_relevance_score, reciprocal_rank_fusion


def make_fact(score: float = 0.9, source: str = "vector", fid=None) -> ScoredFact:
    return ScoredFact(
        id=fid or uuid4(),
        content="some fact",
        fact_type="observation",
        occurred_at=datetime(2024, 3, 14, tzinfo=timezone.utc),
        importance=0.5,
        metadata={},
        tags=[],
        score=score,
        source=source,
    )


def test_rrf_distinct_facts():
    vec = [make_fact(0.9, "vector"), make_fact(0.85, "vector")]
    kw = [make_fact(0.8, "keyword"), make_fact(0.75, "keyword")]
    merged = reciprocal_rank_fusion(vec, kw)
    assert len(merged) == 4


def test_rrf_deduplicates_shared_id():
    shared_id = uuid4()
    vec = [make_fact(0.9, "vector", fid=shared_id)]
    kw = [make_fact(0.8, "keyword", fid=shared_id)]
    merged = reciprocal_rank_fusion(vec, kw)
    assert len(merged) == 1
    assert merged[0].id == shared_id


def test_rrf_higher_ranked_in_both_gets_top_score():
    id_a = uuid4()
    id_b = uuid4()
    vec = [make_fact(fid=id_a), make_fact(fid=id_b)]
    kw = [make_fact(fid=id_a), make_fact(fid=id_b)]
    merged = reciprocal_rank_fusion(vec, kw)
    assert merged[0].id == id_a


def test_compute_relevance_score_decays_over_time():
    now = datetime.now(tz=timezone.utc)
    recent = datetime(2024, 3, 14, tzinfo=timezone.utc)
    old = recent - timedelta(days=365)
    score_recent = compute_relevance_score(0.7, 0.01, recent, 0, now=now)
    score_old = compute_relevance_score(0.7, 0.01, old, 0, now=now)
    assert score_recent > score_old


def test_compute_relevance_score_access_boost():
    now = datetime.now(tz=timezone.utc)
    ts = datetime(2024, 3, 14, tzinfo=timezone.utc)
    score_no_access = compute_relevance_score(0.5, 0.01, ts, 0, now=now)
    score_with_access = compute_relevance_score(0.5, 0.01, ts, 10, now=now)
    assert score_with_access > score_no_access
