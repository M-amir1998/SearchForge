"""Typed filters, backend translation, stable ids."""

from __future__ import annotations

import sqlite3

import pytest

from searchforge.domain.filters import FieldType, FilterOp, FilterSchema, merge_filters, parse_filters
from searchforge.domain.ids import make_chunk_id, make_citation_id, make_content_hash, make_document_id, query_hash
from searchforge.errors import InvalidFilter
from searchforge.filtering.translate import canonicalize_attributes, to_opensearch, to_qdrant_dict, to_sqlite

pytestmark = pytest.mark.unit

SCHEMA = FilterSchema.from_mapping(
    {"category": "string", "published": "date", "year": "integer", "score": "float", "public": "boolean"}
)


def test_parse_forms() -> None:
    expr = parse_filters(
        {
            "category": "security",
            "year": {"gte": 2020, "lt": 2025},
            "source": ["kb", "rn"],
            "published": {"gte": "2024-01-01T10:00:00Z"},
        },
        SCHEMA,
    )
    ops = {(c.field, c.op) for c in expr.conditions}
    assert (
        ("category", FilterOp.EQ) in ops
        and ("year", FilterOp.GTE) in ops
        and ("year", FilterOp.LT) in ops
        and ("source", FilterOp.IN) in ops
    )
    date_cond = next(c for c in expr.conditions if c.field == "published")
    assert date_cond.value == "2024-01-01"
    assert date_cond.field_type is FieldType.DATE


@pytest.mark.parametrize(
    "raw",
    [
        {"unknown": "x"},
        {"Category": "x"},
        {"category; DROP TABLE": "x"},
        {"year": {"between": [1, 2]}},
        {"year": "twenty"},
        {"public": {"gt": True}},
        {"category": {"in": "notalist"}},
        {"category": {"in": []}},
        {"category": {}},
        {"year": 2.5},
    ],
)
def test_invalid_filters_rejected(raw: dict[str, object]) -> None:
    with pytest.raises(InvalidFilter):
        parse_filters(raw, SCHEMA)


def test_matches_semantics_including_missing_fields() -> None:
    expr = parse_filters({"category": "security", "year": {"gte": 2024}}, SCHEMA)
    assert expr.matches({"category": "security", "year": 2025})
    assert not expr.matches({"category": "security", "year": 2023})
    assert not expr.matches({"year": 2025})
    ne = parse_filters({"category": {"ne": "hr"}, "source": {"nin": ["x"]}}, SCHEMA)
    assert ne.matches({})  # missing passes negations
    assert not ne.matches({"category": "hr"})
    assert ne.failing({"category": "hr"}) == ["category ne 'hr'"]
    assert parse_filters(None, SCHEMA).matches({"anything": 1})


def test_canonical_is_order_independent_and_merge_dedupes() -> None:
    a = parse_filters({"category": "a", "year": {"gte": 1}}, SCHEMA)
    b = parse_filters({"year": {"gte": 1}, "category": "a"}, SCHEMA)
    assert a.canonical() == b.canonical()
    assert len(merge_filters(a, b).conditions) == 2


def _sqlite_rows(expr_raw: dict[str, object]) -> list[str]:
    conn = sqlite3.connect(":memory:")
    conn.execute("CREATE TABLE t (id TEXT, attributes TEXT)")
    rows = [
        ("1", '{"category": "security", "year": 2025, "public": true, "published": "2025-02-01"}'),
        ("2", '{"category": "ops", "year": 2022, "public": false}'),
        ("3", '{"year": 2024}'),
    ]
    conn.executemany("INSERT INTO t VALUES (?, ?)", rows)
    where, params = to_sqlite(parse_filters(expr_raw, SCHEMA), "attributes")
    return [r[0] for r in conn.execute(f"SELECT id FROM t WHERE {where} ORDER BY id", params)]


def test_sqlite_translation_is_parameterized_and_correct() -> None:
    assert _sqlite_rows({"category": "security"}) == ["1"]
    assert _sqlite_rows({"year": {"gte": 2024}}) == ["1", "3"]
    assert _sqlite_rows({"category": {"ne": "ops"}}) == ["1", "3"]
    assert _sqlite_rows({"category": {"nin": ["ops"]}}) == ["1", "3"]
    assert _sqlite_rows({"category": ["ops", "security"]}) == ["1", "2"]
    assert _sqlite_rows({"public": True}) == ["1"]
    assert _sqlite_rows({"published": {"gte": "2025-01-01"}}) == ["1"]
    # an injection attempt is just a literal value
    assert _sqlite_rows({"category": "x' OR '1'='1"}) == []
    where, params = to_sqlite(parse_filters({"category": "zz-value"}, SCHEMA), "attributes")
    assert "zz-value" not in where
    assert params == ["$.category", "zz-value"]


def test_structured_backend_translations() -> None:
    expr = parse_filters(
        {"category": "security", "year": {"gte": 2024}, "published": {"lt": "2025-01-01"}, "source": {"nin": ["x"]}},
        SCHEMA,
    )
    q = to_qdrant_dict(expr)
    assert {"key": "attributes.category", "match": {"value": "security"}} in q["must"]
    assert {"key": "attributes.year", "range": {"gte": 2024}} in q["must"]
    assert {"key": "attributes.published", "datetime_range": {"lt": "2025-01-01"}} in q["must"]
    assert q["must_not"] == [{"key": "attributes.source", "match": {"any": ["x"]}}]
    os_ = to_opensearch(expr)
    assert {"term": {"attributes.category": "security"}} in os_
    assert {"range": {"attributes.year": {"gte": 2024}}} in os_


def test_canonicalize_attributes() -> None:
    out = canonicalize_attributes(
        {"published": "2025-03-05T10:00:00", "year": "2025", "other": "x", "public": "not-bool"}, SCHEMA
    )
    assert out == {"published": "2025-03-05", "year": 2025, "other": "x"}


def test_stable_ids() -> None:
    assert make_document_id(explicit_id="KB-1", source_key="a.md") == "KB-1"
    assert make_document_id(explicit_id=None, source_key="dir/a.md", record_key="#L3") == make_document_id(
        explicit_id=None, source_key="dir/a.md", record_key="#L3"
    )
    assert make_document_id(explicit_id=None, source_key="a.md") != make_document_id(
        explicit_id=None, source_key="b.md"
    )
    assert make_document_id(explicit_id="has space/../x", source_key="a").startswith("doc-")
    c1 = make_chunk_id("KB-1", "fp1", 0, 10, "hello")
    assert c1 == make_chunk_id("KB-1", "fp1", 0, 10, "hello")
    assert c1 != make_chunk_id("KB-1", "fp2", 0, 10, "hello")
    assert c1.startswith("KB-1#")
    assert make_citation_id(c1) == make_citation_id(c1) and make_citation_id(c1).startswith("S")
    assert make_content_hash("t", {"a": 1}) != make_content_hash("t", {"a": 2})
    assert query_hash("Hello  World") == query_hash("hello world")


def test_integer_filters_outside_int64_are_rejected() -> None:
    with pytest.raises(InvalidFilter):
        parse_filters({"year": 2**63}, SCHEMA)
    assert parse_filters({"year": 2**63 - 1}, SCHEMA).conditions[0].value == 2**63 - 1
