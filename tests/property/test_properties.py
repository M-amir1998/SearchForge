"""Hypothesis property tests: invariants that must hold for arbitrary inputs."""

from __future__ import annotations

import itertools
import sqlite3
import string
from typing import Any

import pytest
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from searchforge.caching.lru import BoundedLRUCache
from searchforge.chunking.chunkers import build_chunker
from searchforge.config.models import ChunkingConfig, ChunkStrategy
from searchforge.domain.document import Document, DocumentMetadata, DocumentSource, SourceFormat
from searchforge.domain.filters import FilterSchema, parse_filters
from searchforge.domain.ids import make_chunk_id, make_document_id
from searchforge.domain.result import Candidate
from searchforge.errors import InvalidFilter
from searchforge.evaluation.metrics import evaluate_ranking
from searchforge.filtering.translate import to_sqlite
from searchforge.fusion.strategies import NormalizedScoreFusion, ReciprocalRankFusion
from searchforge.indexes.lexical.sqlite_fts import build_match_expression

pytestmark = pytest.mark.property
SETTINGS = settings(max_examples=150, deadline=None, suppress_health_check=[HealthCheck.too_slow])

ids = st.text(alphabet=string.ascii_lowercase, min_size=1, max_size=3)
ranked = st.lists(ids, min_size=0, max_size=30, unique=True)


def _cands(items: list[str]) -> list[Candidate]:
    return [
        Candidate(chunk_id=c, document_id=c, score=float(len(items) - i), rank=i + 1, source="x")
        for i, c in enumerate(items)
    ]


@SETTINGS
@given(st.lists(ranked, min_size=1, max_size=4), st.integers(min_value=1, max_value=100))
def test_rrf_invariants(lists: list[list[str]], k: int) -> None:
    named = {f"r{i}": _cands(lst) for i, lst in enumerate(lists)}
    fused = ReciprocalRankFusion(k=k).fuse(named, limit=1000)
    ids_out = [c.chunk_id for c in fused]
    assert len(ids_out) == len(set(ids_out)) == len(set().union(*map(set, lists)))  # unique, complete
    assert [c.rank for c in fused] == list(range(1, len(fused) + 1))
    assert all(0 < c.score <= len(lists) / (k + 1) + 1e-12 for c in fused)
    assert all(a.score >= b.score for a, b in itertools.pairwise(fused))
    # order of the input lists does not matter
    rev = ReciprocalRankFusion(k=k).fuse(dict(reversed(list(named.items()))), limit=1000)
    assert [c.chunk_id for c in rev] == ids_out


@SETTINGS
@given(ranked, st.integers(min_value=1, max_value=60))
def test_rrf_single_list_preserves_order(lst: list[str], k: int) -> None:
    assert [c.chunk_id for c in ReciprocalRankFusion(k=k).fuse({"x": _cands(lst)}, 1000)] == lst


@SETTINGS
@given(st.lists(ranked, min_size=1, max_size=3))
def test_normalized_fusion_unique_and_bounded(lists: list[list[str]]) -> None:
    fused = NormalizedScoreFusion().fuse({f"r{i}": _cands(lst) for i, lst in enumerate(lists)}, 1000)
    assert len({c.chunk_id for c in fused}) == len(fused)
    assert all(-1e-9 <= c.score <= len(lists) + 1e-9 for c in fused)


judgments = st.dictionaries(ids, st.integers(min_value=0, max_value=3), max_size=10)


@SETTINGS
@given(ranked, judgments, st.lists(st.integers(min_value=1, max_value=40), min_size=1, max_size=4))
def test_metric_bounds(ranking: list[str], judged: dict[str, int], ks: list[int]) -> None:
    metrics = evaluate_ranking(ranking, {k: float(v) for k, v in judged.items()}, sorted(set(ks)))
    assert all(0.0 <= v <= 1.0 + 1e-9 for v in metrics.values())


@SETTINGS
@given(judgments)
def test_ideal_ranking_has_ndcg_one(judged: dict[str, int]) -> None:
    rel = {k: float(v) for k, v in judged.items()}
    if not any(v > 0 for v in rel.values()):
        return
    ideal = sorted(rel, key=lambda d: -rel[d])
    assert evaluate_ranking(ideal, rel, [10])["ndcg@10"] == pytest.approx(1.0)


SCHEMA = FilterSchema.from_mapping({"category": "string", "year": "integer", "published": "date"})
scalars = st.one_of(st.text(max_size=20), st.integers(), st.floats(allow_nan=False), st.booleans(), st.none())
raw_filters = st.dictionaries(
    st.one_of(st.sampled_from(["category", "year", "published", "source"]), st.text(max_size=12)),
    st.one_of(
        scalars,
        st.lists(scalars, max_size=4),
        st.dictionaries(
            st.sampled_from(["eq", "ne", "in", "nin", "gt", "gte", "lt", "lte", "bogus"]),
            st.one_of(scalars, st.lists(scalars, max_size=3)),
            max_size=3,
        ),
    ),
    max_size=4,
)


@SETTINGS
@given(raw_filters)
def test_filter_parsing_is_total_and_sql_is_always_parameterized(raw: dict[str, Any]) -> None:
    """Arbitrary input either fails with InvalidFilter or yields SQL whose values are all bound."""
    try:
        expr = parse_filters(raw, SCHEMA)
    except InvalidFilter:
        return
    where, params = to_sqlite(expr, "attributes")
    assert where.count("?") == len(params)
    conn = sqlite3.connect(":memory:")
    conn.execute("CREATE TABLE t (attributes TEXT)")
    conn.execute('INSERT INTO t VALUES (\'{"category": "x", "year": 1}\')')
    conn.execute(f"SELECT COUNT(*) FROM t WHERE {where}", params).fetchone()


@SETTINGS
@given(st.text(max_size=300), st.lists(st.text(max_size=40), max_size=3))
def test_fts_match_expression_never_breaks_sqlite(text: str, phrases: list[str]) -> None:
    expr = build_match_expression(text, phrases)
    conn = sqlite3.connect(":memory:")
    conn.execute("CREATE VIRTUAL TABLE f USING fts5(body, tokenize='porter unicode61 remove_diacritics 2')")
    conn.execute("INSERT INTO f VALUES ('some indexed text')")
    if expr is not None:
        conn.execute("SELECT COUNT(*) FROM f WHERE f MATCH ?", (expr,)).fetchone()


words = st.text(alphabet=string.ascii_letters + "  .!?\n#", min_size=0, max_size=1500)


@SETTINGS
@given(
    words,
    st.sampled_from(list(ChunkStrategy)),
    st.integers(min_value=60, max_value=600),
    st.integers(min_value=0, max_value=59),
    st.integers(min_value=0, max_value=50),
)
def test_chunk_offset_invariants(
    text: str, strategy: ChunkStrategy, max_chars: int, overlap: int, min_chars: int
) -> None:
    cfg = ChunkingConfig(
        strategy=strategy,
        max_chars=max_chars,
        overlap_chars=overlap,
        min_chars=min_chars,
        max_tokens=max(9, max_chars // 6),
        overlap_tokens=min(overlap // 6, max(8, max_chars // 6) - 1),
    )
    doc = Document("D", text, DocumentSource("s", "k", SourceFormat.TEXT), DocumentMetadata(), "h")
    chunks = list(build_chunker(cfg).chunk(doc))
    for c in chunks:
        assert 0 <= c.metadata.start_char < c.metadata.end_char <= len(text)
        assert text[c.metadata.start_char : c.metadata.end_char] == c.text
    starts = [c.metadata.start_char for c in chunks]
    assert starts == sorted(starts)
    assert len({c.chunk_id for c in chunks}) == len(chunks)
    if text.strip():
        assert chunks
    assert [c.chunk_id for c in build_chunker(cfg).chunk(doc)] == [c.chunk_id for c in chunks]


@SETTINGS
@given(
    st.lists(st.tuples(st.text(max_size=4), st.text(max_size=50)), max_size=60),
    st.integers(min_value=1, max_value=10),
    st.integers(min_value=20, max_value=400),
)
def test_cache_never_exceeds_capacity(ops: list[tuple[str, str]], max_entries: int, max_bytes: int) -> None:
    cache: BoundedLRUCache[str] = BoundedLRUCache(
        max_entries=max_entries, max_bytes=max_bytes, ttl_s=60, sizeof=lambda v: len(v) + 1
    )
    for key, value in ops:
        cache.put(key, value)
        assert len(cache) <= max_entries
        assert cache.stats.bytes <= max_bytes
        assert cache.stats.entries == len(cache)


@SETTINGS
@given(st.text(max_size=40), st.text(max_size=40), st.text(max_size=10))
def test_stable_ids_are_deterministic(source_key: str, record_key: str, explicit: str) -> None:
    a = make_document_id(explicit_id=explicit or None, source_key=source_key, record_key=record_key)
    b = make_document_id(explicit_id=explicit or None, source_key=source_key, record_key=record_key)
    assert a == b and len(a) <= 200
    assert make_chunk_id(a, "fp", 0, 1, "x") == make_chunk_id(b, "fp", 0, 1, "x")
