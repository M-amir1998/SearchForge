"""Query analysis/planning/rewriting, reranking, diversity, cache, context, breaker, privacy, config."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from searchforge.caching.lru import BoundedLRUCache
from searchforge.caching.query_cache import query_cache_key
from searchforge.config.loader import load_config, resolve_path
from searchforge.config.models import (
    CapturePolicy,
    PlannerConfig,
    RerankerConfig,
    RerankerName,
    RewriteConfig,
    SearchConfig,
)
from searchforge.config.profiles import resolve_profiles
from searchforge.context.builder import CitationContextBuilder
from searchforge.domain.filters import FilterSchema, parse_filters
from searchforge.domain.query import QueryClass, SearchQuery
from searchforge.domain.result import RankingSignals, ScoredChunk, SearchResult
from searchforge.errors import CircuitOpenError, ConfigurationError, InvalidPlan, InvalidQuery
from searchforge.observability.privacy import capture_query, redact
from searchforge.query.analysis import QueryAnalyzer, is_identifier
from searchforge.query.planning import LLMQueryPlanner, StaticQueryPlanner, validate_plan
from searchforge.query.rewriting import QueryRewriter
from searchforge.ranking.diversity import cap_per_document, group_results, mmr
from searchforge.reranking import build_reranker
from searchforge.reranking.rerankers import CrossEncoderReranker, LexicalFeatureReranker, NoOpReranker
from searchforge.service.circuit_breaker import BreakerState, CircuitBreaker

pytestmark = pytest.mark.unit
ANALYZER = QueryAnalyzer(["category", "year"])


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("ERR-4012", QueryClass.IDENTIFIER),
        ("RetryPolicy.max_attempts", QueryClass.IDENTIFIER),
        ("next_page_token", QueryClass.IDENTIFIER),
        ("getUserById timeout", QueryClass.IDENTIFIER),
        ("rollback procedure", QueryClass.KEYWORD),
        ("rate-limit", QueryClass.KEYWORD),
        ("how do I undo a release", QueryClass.NATURAL_LANGUAGE),
        ("group many generation requests together to keep the accelerator busy", QueryClass.COMPLEX_SEMANTIC),
        ("category:security keys", QueryClass.FILTER_HEAVY),
    ],
)
def test_query_classification(text: str, expected: QueryClass) -> None:
    assert ANALYZER.analyze(text).query_class is expected


def test_query_analysis_features() -> None:
    a = ANALYZER.analyze('category:"security" year:2025 rotate "api keys" v2 1.4.2')
    assert a.extracted_filters == {"category": "security", "year": "2025"}
    assert "category" not in a.residual_text and a.quoted_phrases == ("api keys",)
    assert "1.4.2" in a.identifiers and a.as_dict()["extracted_filter_fields"] == ["category", "year"]
    assert ANALYZER.analyze("unknown:field stays").extracted_filters == {}
    assert ANALYZER.analyze("x", explicit_filter_count=2).query_class is QueryClass.FILTER_HEAVY
    assert not is_identifier("1234") and not is_identifier("well-known") and is_identifier("X-Request-Id")


def test_search_query_validation() -> None:
    with pytest.raises(InvalidQuery):
        SearchQuery(query_id="q", text="   ")
    with pytest.raises(InvalidQuery):
        SearchQuery(query_id="q", text="x", requested_k=0)


def test_rewriting_is_recorded_never_silent() -> None:
    r = QueryRewriter(
        RewriteConfig(enabled=True, multi_query=True, acronyms={"k8s": "kubernetes"}, synonyms={"undo": ["roll back"]})
    )
    rec = r.rewrite("  **undo**   k8s `deploy` ")
    assert rec.original == "  **undo**   k8s `deploy` "
    assert rec.rewritten == "undo k8s deploy kubernetes"
    assert rec.strategies == ("whitespace", "strip_formatting", "acronym_expansion", "synonyms")
    assert rec.variants == ("roll back k8s deploy kubernetes",) and rec.changed
    off = QueryRewriter(RewriteConfig(enabled=False)).rewrite("undo k8s")
    assert off.rewritten == "undo k8s" and not off.changed


def test_static_planner_rules_and_validation() -> None:
    profiles = resolve_profiles(SearchConfig())
    planner = StaticQueryPlanner(profiles, PlannerConfig())
    assert planner.plan(ANALYZER.analyze("ERR-4012"), "adaptive").profile == "lexical"
    assert planner.plan(ANALYZER.analyze("how do I undo a release"), "adaptive").profile == "hybrid"
    long_q = ANALYZER.analyze("group many generation requests together to keep the accelerator busy")
    assert planner.plan(long_q, "adaptive").profile == "hybrid_rerank"
    fixed = planner.plan(ANALYZER.analyze("ERR-4012"), "dense")
    assert fixed.profile == "dense" and fixed.reason == "fixed profile"
    with pytest.raises(InvalidPlan):
        planner.plan(ANALYZER.analyze("x"), "nope")
    with pytest.raises(InvalidPlan, match="does not provide"):
        validate_plan(fixed, available_retrievers={"lexical"}, top_k=10)
    with pytest.raises(InvalidPlan, match="candidate_k"):
        validate_plan(planner.plan(ANALYZER.analyze("x"), "lexical"), available_retrievers={"lexical"}, top_k=500)
    with pytest.raises(InvalidPlan, match="cycle"):
        StaticQueryPlanner(profiles, PlannerConfig(rules={QueryClass.KEYWORD: "adaptive"}))


def test_llm_planner_untrusted_output_falls_back() -> None:
    profiles = resolve_profiles(SearchConfig())
    static = StaticQueryPlanner(profiles, PlannerConfig())
    analysis = ANALYZER.analyze("how do I undo a release")
    good = LLMQueryPlanner(lambda prompt: json.dumps({"profile": "lexical", "reason": "short"}), static)
    assert good.plan(analysis, "adaptive").profile == "lexical"
    for bad in ("not json", json.dumps({"profile": "adaptive"}), json.dumps({"profile": "drop table"})):
        plan = LLMQueryPlanner(lambda prompt, b=bad: b, static).plan(analysis, "adaptive")
        assert plan.profile == "hybrid" and plan.fallback and plan.fallback.startswith("llm_invalid")
    assert good.plan(analysis, "dense").profile == "dense"  # fixed profiles bypass the LLM


def _sc(cid: str, doc: str, text: str, score: float, title: str = "") -> ScoredChunk:
    return ScoredChunk(
        chunk_id=cid,
        document_id=doc,
        text=text,
        score=score,
        retrieval_sources=["lexical"],
        metadata={"title": title, "section": None, "start_char": 0, "end_char": len(text)},
        attributes={},
        signals=RankingSignals(),
    )


CANDS = [
    _sc("c1", "d1", "general overview of deployments and releases", 0.9),
    _sc("c2", "d2", "ERR-4012 rate limit exceeded returns HTTP 429", 0.8, "Errors"),
    _sc("c3", "d3", "rate limits use a token bucket", 0.7),
]


def test_rerankers_preserve_candidate_set_and_order_by_score() -> None:
    for rr in (NoOpReranker(), LexicalFeatureReranker()):
        out = rr.rerank("ERR-4012 rate limit", CANDS)
        assert {c.chunk_id for c in out} == {c.chunk_id for c in CANDS} and len(out) == len(CANDS)
    out = LexicalFeatureReranker().rerank("ERR-4012 rate limit", CANDS)
    assert out[0].chunk_id == "c2" and out[0].signals.reranker_rank == 1
    assert [c.score for c in out] == sorted((c.score for c in out), reverse=True)
    with pytest.raises(ValueError, match="unknown"):
        LexicalFeatureReranker({"magic": 1.0})
    assert build_reranker(RerankerConfig(type=RerankerName.NONE)).name == "none"


def test_cross_encoder_with_fake_model() -> None:
    class Fake:
        def predict(self, pairs: list[tuple[str, str]], **kw: Any) -> list[float]:
            return [float("token" in t) for _, t in pairs]

    out = CrossEncoderReranker("fake", model_obj=Fake()).rerank("q", CANDS)
    assert out[0].chunk_id == "c3" and out[0].signals.reranker_score == 1.0


def test_diversity_cap_mmr_grouping() -> None:
    same_doc = [_sc(f"c{i}", "d1" if i < 3 else "d2", "t", 1 - i / 10) for i in range(5)]
    capped = cap_per_document(same_doc, 2)
    assert [c.document_id for c in capped[:3]] == ["d1", "d1", "d2"]
    assert capped[-1].signals.diversity_note and "demoted" in capped[-1].signals.diversity_note
    vecs = {"a": np.array([1, 0], np.float32), "b": np.array([1, 0], np.float32), "c": np.array([0, 1], np.float32)}
    pool = [_sc("a", "d", "", 1.0), _sc("b", "d", "", 0.95), _sc("c", "e", "", 0.5)]
    assert [c.chunk_id for c in mmr(pool, vecs, 1.0, 3)] == ["a", "b", "c"]  # pure relevance
    assert [c.chunk_id for c in mmr(pool, vecs, 0.3, 2)] == ["a", "c", "b"]  # novelty wins; rest kept
    res = [
        SearchResult("x1", "d1", 1, 1, (), {"source_id": "kb"}),
        SearchResult("x2", "d1", 1, 2, (), {"source_id": "kb"}),
    ]
    assert list(group_results(res)) == ["d1"] and list(group_results(res, "source")) == ["kb"]


def test_bounded_lru_cache() -> None:
    now = [0.0]
    cache: BoundedLRUCache[str] = BoundedLRUCache(
        max_entries=2, max_bytes=100, ttl_s=10, sizeof=len, clock=lambda: now[0]
    )
    cache.put("a", "x" * 10, cost_ms=5)
    cache.put("b", "y" * 10)
    assert cache.get("a") == "x" * 10 and cache.stats.latency_saved_ms == 5
    cache.put("c", "z" * 10)  # evicts LRU "b"
    assert cache.get("b") is None and cache.stats.evictions == 1 and len(cache) == 2
    assert not cache.put("huge", "h" * 101)
    cache.put("d", "w" * 90)  # byte limit forces eviction
    assert cache.stats.bytes <= 100
    now[0] = 100.0
    assert cache.get("d") is None and cache.stats.expirations == 1
    assert 0 <= cache.stats.hit_rate <= 1


def test_query_cache_key_is_version_aware() -> None:
    schema = FilterSchema.from_mapping({"category": "string"})
    base = {
        "text": "Rollback ",
        "filters": parse_filters({"category": "ops"}, schema),
        "index_id": "idx@v1",
        "planner_version": "p1",
        "profile": "hybrid",
        "top_k": 10,
        "explain": False,
    }
    k1 = query_cache_key(**base)  # type: ignore[arg-type]
    assert k1 == query_cache_key(**{**base, "text": "  Rollback  "})  # type: ignore[arg-type]
    # case is significant: identifier detection (and therefore planning) is case-sensitive
    assert k1 != query_cache_key(**{**base, "text": "rollback"})  # type: ignore[arg-type]
    for field, value in (
        ("index_id", "idx@v2"),
        ("planner_version", "p2"),
        ("top_k", 5),
        ("profile", "lexical"),
        ("filters", parse_filters({"category": "sec"}, schema)),
    ):
        assert query_cache_key(**{**base, field: value}) != k1  # type: ignore[arg-type]


def _res(cid: str, doc: str, rank: int, text: str, start: int = 0) -> SearchResult:
    return SearchResult(
        cid,
        doc,
        1.0,
        rank,
        ("lexical",),
        {
            "title": "T",
            "section": "S",
            "start_char": start,
            "end_char": start + len(text),
            "source_id": "kb",
            "uri": "u",
        },
        text=text,
    )


def test_context_builder_budget_dedupe_and_citations() -> None:
    results = [
        _res("a#1", "a", 1, "alpha " * 20),
        _res("a#2", "a", 2, "alpha " * 20, start=10),  # overlaps a#1 by > 50 %
        _res("b#1", "b", 3, "alpha " * 20, start=500),  # near-duplicate text of a#1
        _res("c#1", "c", 4, "gamma delta epsilon zeta"),
        _res("d#1", "d", 5, "x" * 5000),  # exceeds budget
        _res("e#1", "e", 6, "short tail fits"),
    ]
    pkg = CitationContextBuilder(max_chars=300, max_per_document=2).build(results)
    assert [i.chunk_id for i in pkg.items] == ["a#1", "c#1", "e#1"]
    reasons = {d["chunk_id"]: d["reason"] for d in pkg.dropped}
    assert reasons == {"a#2": "overlapping_chunk", "b#1": "near_duplicate", "d#1": "budget"}
    assert pkg.total_chars <= 300 and pkg.items[0].citation_id in pkg.render()
    assert set(pkg.citations) == {i.citation_id for i in pkg.items}
    assert CitationContextBuilder(max_chars=None, max_tokens=5).build(results).estimated_tokens <= 5
    with pytest.raises(ValueError, match="max_chars"):
        CitationContextBuilder(max_chars=None, max_tokens=None)


def test_circuit_breaker_transitions() -> None:
    now = [0.0]
    br = CircuitBreaker("x", failure_threshold=2, reset_timeout_s=10, clock=lambda: now[0])
    for _ in range(2):
        with pytest.raises(RuntimeError), br.guard():
            raise RuntimeError
    assert br.state is BreakerState.OPEN
    with pytest.raises(CircuitOpenError), br.guard():
        pass
    now[0] = 11
    assert br.state is BreakerState.HALF_OPEN
    with br.guard():
        pass
    assert br.state is BreakerState.CLOSED


def test_privacy_capture_policies() -> None:
    q = "contact jane.doe@example.com about ticket 1234567 at https://x.example/a token sk-abcdefghijklmnop"
    red = redact(q)
    assert "jane" not in red and "1234567" not in red and "https" not in red and "sk-abc" not in red
    assert capture_query(q, CapturePolicy.NONE) is None
    assert (capture_query(q, CapturePolicy.HASH_ONLY) or "").startswith("sha256:")
    assert capture_query(q, CapturePolicy.FULL) == q


def test_config_include_validation_and_pkg_paths(tmp_path: Path) -> None:
    (tmp_path / "base.yaml").write_text(
        "corpus: {id: c, sources: [{id: s, path: data}]}\nindex: {name: i}\n", encoding="utf-8"
    )
    (tmp_path / "top.yaml").write_text("include: base.yaml\nchunking: {strategy: sentence}\n", encoding="utf-8")
    loaded = load_config(tmp_path / "top.yaml")
    assert loaded.config.chunking.strategy.value == "sentence" and loaded.resolve("data") == (tmp_path / "data")
    (tmp_path / "bad.yaml").write_text("include: base.yaml\nindex: {name: i, unknown_key: 1}\n", encoding="utf-8")
    with pytest.raises(ConfigurationError, match="unknown_key"):
        load_config(tmp_path / "bad.yaml")
    with pytest.raises(ConfigurationError):
        resolve_path(tmp_path, "pkg:../../etc")
    assert resolve_path(tmp_path, "pkg:fixture/fixture.yaml").exists()
