"""Search-quality guards on the bundled fixture collection (CPU, deterministic).

These assert *correctness* properties (filters are honoured, judged documents exist,
critical identifier lookups work) and coarse quality floors that protect against
regressions. Floors are deliberately below measured values; see the report for
the measured numbers.
"""

from __future__ import annotations

import pytest

from searchforge.config.loader import LoadedConfig
from searchforge.domain.filters import parse_filters
from searchforge.domain.judgment import QuerySeverity
from searchforge.evaluation.datasets import load_qrels, load_queries
from searchforge.evaluation.runner import evaluate
from searchforge.indexes.handle import IndexHandle
from searchforge.service.search_service import SearchService

pytestmark = pytest.mark.quality
Fixture = tuple[IndexHandle, SearchService]


def _data(loaded: LoadedConfig):  # type: ignore[no-untyped-def]
    ev = loaded.config.evaluation
    assert ev is not None
    return load_queries(loaded.resolve(ev.queries)), load_qrels(loaded.resolve(ev.qrels))


def test_collection_integrity(fixture_loaded: LoadedConfig, fixture_service: Fixture) -> None:
    handle, _ = fixture_service
    queries, qrels = _data(fixture_loaded)
    indexed = set(handle.chunk_store.document_hashes())
    assert handle.manifest.document_count == 48 and len(queries) == 56
    types = {q.query_type for q in queries}
    assert types == {"keyword", "semantic", "identifier", "ambiguous", "filter"}
    assert {q.split.value for q in queries} == {"train", "validation", "test"}
    for q in queries:
        assert qrels.relevant(q.query_id), q.query_id
        assert set(qrels.for_query(q.query_id)) <= indexed, q.query_id


def test_filters_are_always_honoured(fixture_loaded: LoadedConfig, fixture_service: Fixture) -> None:
    handle, service = fixture_service
    queries, _ = _data(fixture_loaded)
    for q in (q for q in queries if q.filters):
        expr = parse_filters(q.filters, handle.filter_schema)
        for profile in ("lexical", "dense", "hybrid", "adaptive"):
            resp = service.search(q.text, filters=q.filters, profile=profile, top_k=20)
            for r in resp.results:
                chunk = handle.chunk_store.get_many([r.chunk_id])[r.chunk_id]
                assert expr.matches(chunk.metadata.attributes), (q.query_id, profile, r.document_id)
    rollback = service.search("rollback", filters={"published": {"gte": "2024-01-01"}}, top_k=20)
    assert "KB-OPS-002-OLD" not in {r.document_id for r in rollback.results}
    german = service.search("secrets", filters={"language": "de"}, top_k=5)
    assert {r.document_id for r in german.results} == {"KB-SEC-004-DE"}


def test_critical_identifier_queries_rank_first(fixture_loaded: LoadedConfig, fixture_service: Fixture) -> None:
    _, service = fixture_service
    queries, qrels = _data(fixture_loaded)
    critical = [q for q in queries if q.severity is QuerySeverity.CRITICAL]
    assert critical
    for profile in ("lexical", "adaptive"):
        run = evaluate(service, critical, qrels, profile=profile, k_values=[1, 10], warmup=False)
        assert run.aggregate()["mrr"] == pytest.approx(1.0), profile


@pytest.mark.parametrize(
    ("profile", "ndcg_floor"),
    [("lexical", 0.80), ("dense", 0.75), ("hybrid", 0.80), ("hybrid_rerank", 0.80), ("adaptive", 0.80)],
)
def test_quality_floors(
    fixture_loaded: LoadedConfig, fixture_service: Fixture, profile: str, ndcg_floor: float
) -> None:
    _, service = fixture_service
    queries, qrels = _data(fixture_loaded)
    run = evaluate(service, queries, qrels, profile=profile, k_values=[10, 20], warmup=False)
    agg = run.aggregate()
    assert agg["ndcg@10"] >= ndcg_floor
    assert agg["recall@20"] >= 0.8
    assert not run.excluded and all(o.error is None for o in run.outcomes)


def test_adaptive_routes_identifiers_to_lexical(fixture_loaded: LoadedConfig, fixture_service: Fixture) -> None:
    _, service = fixture_service
    queries, qrels = _data(fixture_loaded)
    run = evaluate(
        service, [q for q in queries if q.query_type == "identifier"], qrels, profile="adaptive", warmup=False
    )
    planned = {o.planned_profile for o in run.scored}
    assert planned == {"lexical"}
