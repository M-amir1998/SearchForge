"""Translate canonical filters into backend-native *structured* queries.

No translator concatenates user-provided values into query strings. SQLite gets
bound parameters (field names are regex-validated *and* passed as a bound JSON
path); Qdrant/OpenSearch get structured dicts/models.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from searchforge.domain.document import AttributeValue
from searchforge.domain.filters import (
    Condition,
    FieldType,
    FilterExpression,
    FilterOp,
    FilterSchema,
    ScalarValue,
    coerce_value,
)
from searchforge.errors import InvalidFilter

_SQL_OPS = {FilterOp.GT: ">", FilterOp.GTE: ">=", FilterOp.LT: "<", FilterOp.LTE: "<="}


def _sql_value(value: ScalarValue) -> ScalarValue:
    # json_extract returns 1/0 for JSON booleans.
    return int(value) if isinstance(value, bool) else value


def to_sqlite(expr: FilterExpression, column: str) -> tuple[str, list[Any]]:
    """Return ``(sql_fragment, params)``; ``column`` is a trusted, code-defined identifier."""
    if not expr:
        return "1=1", []
    clauses: list[str] = []
    params: list[Any] = []
    for cond in expr.conditions:
        path = f"$.{cond.field}"
        ref = f"json_extract({column}, ?)"
        if cond.op is FilterOp.EQ:
            clauses.append(f"{ref} = ?")
            params += [path, _sql_value(_scalar(cond))]
        elif cond.op is FilterOp.NE:
            clauses.append(f"({ref} IS NULL OR {ref} != ?)")
            params += [path, path, _sql_value(_scalar(cond))]
        elif cond.op in (FilterOp.IN, FilterOp.NOT_IN):
            values = _values(cond)
            marks = ",".join("?" for _ in values)
            if cond.op is FilterOp.IN:
                clauses.append(f"{ref} IN ({marks})")
                params += [path, *(_sql_value(v) for v in values)]
            else:
                clauses.append(f"({ref} IS NULL OR {ref} NOT IN ({marks}))")
                params += [path, path, *(_sql_value(v) for v in values)]
        else:
            clauses.append(f"{ref} {_SQL_OPS[cond.op]} ?")
            params += [path, _sql_value(_scalar(cond))]
    return " AND ".join(clauses), params


def _scalar(cond: Condition) -> ScalarValue:
    if isinstance(cond.value, tuple):
        raise InvalidFilter(f"operator {cond.op.value} expects a scalar")
    return cond.value


def _values(cond: Condition) -> tuple[ScalarValue, ...]:
    if not isinstance(cond.value, tuple):
        raise InvalidFilter(f"operator {cond.op.value} expects a list")
    return cond.value


def to_opensearch(expr: FilterExpression, attributes_prefix: str = "attributes") -> list[dict[str, Any]]:
    """OpenSearch bool-``filter`` clauses (structured Query DSL, no query strings)."""
    must: list[dict[str, Any]] = []
    for cond in expr.conditions:
        field = f"{attributes_prefix}.{cond.field}"
        if cond.op is FilterOp.EQ:
            must.append({"term": {field: _scalar(cond)}})
        elif cond.op is FilterOp.NE:
            must.append({"bool": {"must_not": [{"term": {field: _scalar(cond)}}]}})
        elif cond.op is FilterOp.IN:
            must.append({"terms": {field: list(_values(cond))}})
        elif cond.op is FilterOp.NOT_IN:
            must.append({"bool": {"must_not": [{"terms": {field: list(_values(cond))}}]}})
        else:
            must.append({"range": {field: {cond.op.value: _scalar(cond)}}})
    return must


def to_qdrant_dict(expr: FilterExpression, payload_prefix: str = "attributes") -> dict[str, Any]:
    """Qdrant filter as a plain dict mirroring ``models.Filter`` (adapter converts it).

    Range conditions on DATE fields use ``datetime_range``; numeric fields use ``range``.
    """
    must: list[dict[str, Any]] = []
    must_not: list[dict[str, Any]] = []
    for cond in expr.conditions:
        key = f"{payload_prefix}.{cond.field}"
        if cond.op is FilterOp.EQ:
            must.append({"key": key, "match": {"value": _scalar(cond)}})
        elif cond.op is FilterOp.NE:
            must_not.append({"key": key, "match": {"value": _scalar(cond)}})
        elif cond.op is FilterOp.IN:
            must.append({"key": key, "match": {"any": list(_values(cond))}})
        elif cond.op is FilterOp.NOT_IN:
            must_not.append({"key": key, "match": {"any": list(_values(cond))}})
        else:
            kind = "datetime_range" if cond.field_type is FieldType.DATE else "range"
            must.append({"key": key, kind: {cond.op.value: _scalar(cond)}})
    out: dict[str, Any] = {}
    if must:
        out["must"] = must
    if must_not:
        out["must_not"] = must_not
    return out


def canonicalize_attributes(
    attributes: Mapping[str, AttributeValue], schema: FilterSchema
) -> dict[str, AttributeValue]:
    """Coerce schema fields to canonical types at index time (dates → ISO date)."""
    out: dict[str, AttributeValue] = {}
    for key, value in attributes.items():
        ftype = schema.fields.get(key)
        if ftype is None:
            out[key] = value
            continue
        try:
            out[key] = coerce_value(value, ftype, key)
        except InvalidFilter:
            continue  # untyped/garbage values are not filterable rather than wrong
    return out
