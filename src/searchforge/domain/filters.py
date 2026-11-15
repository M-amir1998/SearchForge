"""Canonical, typed metadata filters.

Filters are parsed from untrusted input (API/CLI/query syntax) into a validated
:class:`FilterExpression`. Backends never see raw user strings: each adapter
*translates* the canonical expression into a structured/parameterized backend
query (SQLite bound parameters, Qdrant ``Filter`` models, OpenSearch DSL dicts).

Semantics: all conditions are AND-ed. A document lacking the field fails every
condition except ``ne`` / ``nin`` (which pass, mirroring "is not X").
"""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import date, datetime
from enum import StrEnum
from typing import Any

from searchforge.domain.document import AttributeValue
from searchforge.errors import InvalidFilter

_FIELD_NAME = re.compile(r"^[a-z][a-z0-9_]{0,63}$")
MAX_SET_SIZE = 256
SYSTEM_FIELDS = ("document_id", "source")


class FieldType(StrEnum):
    STRING = "string"
    INTEGER = "integer"
    FLOAT = "float"
    BOOLEAN = "boolean"
    DATE = "date"


class FilterOp(StrEnum):
    EQ = "eq"
    NE = "ne"
    IN = "in"
    NOT_IN = "nin"
    GT = "gt"
    GTE = "gte"
    LT = "lt"
    LTE = "lte"


RANGE_OPS = frozenset({FilterOp.GT, FilterOp.GTE, FilterOp.LT, FilterOp.LTE})
SET_OPS = frozenset({FilterOp.IN, FilterOp.NOT_IN})
_ORDERED_TYPES = frozenset({FieldType.INTEGER, FieldType.FLOAT, FieldType.DATE, FieldType.STRING})

ScalarValue = str | int | float | bool


@dataclass(frozen=True, slots=True)
class Condition:
    """One validated condition. ``value`` is a scalar, or a tuple for set ops."""

    field: str
    op: FilterOp
    value: ScalarValue | tuple[ScalarValue, ...]
    field_type: FieldType

    def matches(self, attributes: Mapping[str, AttributeValue]) -> bool:
        present = self.field in attributes
        if not present:
            return self.op in (FilterOp.NE, FilterOp.NOT_IN)
        actual = attributes[self.field]
        try:
            actual_c = coerce_value(actual, self.field_type, self.field)
        except InvalidFilter:
            return False
        if self.op is FilterOp.EQ:
            return actual_c == self.value
        if self.op is FilterOp.NE:
            return actual_c != self.value
        if self.op is FilterOp.IN:
            assert isinstance(self.value, tuple)
            return actual_c in self.value
        if self.op is FilterOp.NOT_IN:
            assert isinstance(self.value, tuple)
            return actual_c not in self.value
        assert not isinstance(self.value, tuple)
        return _compare(actual_c, self.op, self.value)

    def describe(self) -> str:
        return f"{self.field} {self.op.value} {self.value!r}"


def _compare(left: ScalarValue, op: FilterOp, right: ScalarValue) -> bool:
    if isinstance(left, bool) or isinstance(right, bool):
        return False
    if isinstance(left, str) != isinstance(right, str):
        return False
    lv: Any = left
    rv: Any = right
    if op is FilterOp.GT:
        return bool(lv > rv)
    if op is FilterOp.GTE:
        return bool(lv >= rv)
    if op is FilterOp.LT:
        return bool(lv < rv)
    return bool(lv <= rv)


@dataclass(frozen=True, slots=True)
class FilterExpression:
    """Conjunction of conditions. The empty expression matches everything."""

    conditions: tuple[Condition, ...] = ()

    def __bool__(self) -> bool:
        return bool(self.conditions)

    def matches(self, attributes: Mapping[str, AttributeValue]) -> bool:
        return all(c.matches(attributes) for c in self.conditions)

    def failing(self, attributes: Mapping[str, AttributeValue]) -> list[str]:
        """Human-readable reasons a record fails this filter (explainability)."""
        return [c.describe() for c in self.conditions if not c.matches(attributes)]

    def canonical(self) -> list[tuple[str, str, Any]]:
        """Order-independent representation used in cache keys."""
        items = [
            (c.field, c.op.value, list(c.value) if isinstance(c.value, tuple) else c.value) for c in self.conditions
        ]
        return sorted(items, key=lambda t: (t[0], t[1], repr(t[2])))

    def fields(self) -> set[str]:
        return {c.field for c in self.conditions}


@dataclass(frozen=True, slots=True)
class FilterSchema:
    """The set of filterable fields and their types for an index."""

    fields: Mapping[str, FieldType]

    @classmethod
    def from_mapping(cls, raw: Mapping[str, str | FieldType]) -> FilterSchema:
        out: dict[str, FieldType] = dict.fromkeys(SYSTEM_FIELDS, FieldType.STRING)
        for name, ftype in raw.items():
            validate_field_name(name)
            out[name] = FieldType(ftype)
        return cls(fields=out)

    def field_type(self, name: str) -> FieldType:
        validate_field_name(name)
        if name not in self.fields:
            allowed = ", ".join(sorted(self.fields))
            raise InvalidFilter(f"unknown filter field {name!r}; filterable fields: {allowed}")
        return self.fields[name]


def validate_field_name(name: str) -> None:
    if not isinstance(name, str) or not _FIELD_NAME.match(name):
        raise InvalidFilter(f"invalid filter field name {name!r}")


def coerce_value(value: Any, ftype: FieldType, field: str) -> ScalarValue:
    """Coerce an incoming value to the field's canonical python type."""
    try:
        if ftype is FieldType.STRING:
            if isinstance(value, bool) or not isinstance(value, str | int | float):
                raise TypeError
            return str(value)
        if ftype is FieldType.BOOLEAN:
            if isinstance(value, bool):
                return value
            if isinstance(value, str) and value.lower() in ("true", "false"):
                return value.lower() == "true"
            raise TypeError
        if ftype is FieldType.INTEGER:
            if isinstance(value, bool):
                raise TypeError
            if isinstance(value, float) and not value.is_integer():
                raise TypeError
            return int(value)
        if ftype is FieldType.FLOAT:
            if isinstance(value, bool):
                raise TypeError
            return float(value)
        # DATE: canonical ISO-8601 date string (lexicographically ordered).
        if isinstance(value, datetime):
            return value.date().isoformat()
        if isinstance(value, date):
            return value.isoformat()
        if isinstance(value, str):
            text = value.strip()
            if len(text) >= 10:
                return date.fromisoformat(text[:10]).isoformat()
        raise TypeError
    except (TypeError, ValueError) as exc:
        raise InvalidFilter(f"value {value!r} is not a valid {ftype.value} for field {field!r}") from exc


def _parse_op(name: str) -> FilterOp:
    try:
        return FilterOp(name)
    except ValueError as exc:
        allowed = ", ".join(op.value for op in FilterOp)
        raise InvalidFilter(f"unknown filter operator {name!r}; allowed: {allowed}") from exc


def make_condition(field: str, op: FilterOp, value: Any, schema: FilterSchema) -> Condition:
    ftype = schema.field_type(field)
    if op in RANGE_OPS and ftype not in _ORDERED_TYPES:
        raise InvalidFilter(f"operator {op.value!r} is not supported for {ftype.value} field {field!r}")
    if op in SET_OPS:
        if isinstance(value, str | bytes) or not isinstance(value, Sequence):
            raise InvalidFilter(f"operator {op.value!r} requires a list of values")
        if not value:
            raise InvalidFilter(f"operator {op.value!r} requires a non-empty list")
        if len(value) > MAX_SET_SIZE:
            raise InvalidFilter(f"set filters are limited to {MAX_SET_SIZE} values")
        coerced = tuple(dict.fromkeys(coerce_value(v, ftype, field) for v in value))
        return Condition(field=field, op=op, value=coerced, field_type=ftype)
    return Condition(field=field, op=op, value=coerce_value(value, ftype, field), field_type=ftype)


def parse_filters(raw: Mapping[str, Any] | None, schema: FilterSchema) -> FilterExpression:
    """Parse the public filter syntax.

    ``{"category": "security"}``                     equality
    ``{"category": ["a", "b"]}``                      set membership
    ``{"year": {"gte": 2020, "lt": 2024}}``          range
    ``{"published": {"gte": "2024-01-01"}}``         date range
    ``{"language": {"nin": ["de"]}}``                 negated membership
    """
    if raw is None:
        return FilterExpression()
    if not isinstance(raw, Mapping):
        raise InvalidFilter("filters must be a JSON object")
    conditions: list[Condition] = []
    for field, spec in raw.items():
        if isinstance(spec, Mapping):
            if not spec:
                raise InvalidFilter(f"empty operator object for field {field!r}")
            for op_name, value in spec.items():
                conditions.append(make_condition(field, _parse_op(str(op_name)), value, schema))
        elif isinstance(spec, list | tuple):
            conditions.append(make_condition(field, FilterOp.IN, spec, schema))
        else:
            conditions.append(make_condition(field, FilterOp.EQ, spec, schema))
    return FilterExpression(conditions=tuple(conditions))


def merge_filters(*expressions: FilterExpression) -> FilterExpression:
    merged: list[Condition] = []
    for expr in expressions:
        merged.extend(c for c in expr.conditions if c not in merged)
    return FilterExpression(conditions=tuple(merged))
