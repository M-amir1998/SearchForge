"""Query models: the request, its analysis and the execution plan."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any

from searchforge.domain.filters import FilterExpression
from searchforge.errors import InvalidQuery

MAX_QUERY_CHARS = 2000
MAX_TOP_K = 200


class QueryClass(StrEnum):
    """Broad intent buckets used by the planner (rules-based in v0.1)."""

    IDENTIFIER = "identifier"
    KEYWORD = "keyword"
    NATURAL_LANGUAGE = "natural_language"
    COMPLEX_SEMANTIC = "complex_semantic"
    FILTER_HEAVY = "filter_heavy"


@dataclass(frozen=True, slots=True)
class SearchQuery:
    """A validated search request."""

    query_id: str
    text: str
    filters: FilterExpression = field(default_factory=FilterExpression)
    requested_k: int = 10
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.text or not self.text.strip():
            raise InvalidQuery("query text must not be empty")
        if len(self.text) > MAX_QUERY_CHARS:
            raise InvalidQuery(f"query text exceeds {MAX_QUERY_CHARS} characters")
        if not 1 <= self.requested_k <= MAX_TOP_K:
            raise InvalidQuery(f"requested_k must be within [1, {MAX_TOP_K}]")


@dataclass(frozen=True, slots=True)
class QueryAnalysis:
    """CPU-cheap, deterministic query features."""

    tokens: tuple[str, ...]
    token_count: int
    char_length: int
    quoted_phrases: tuple[str, ...]
    identifiers: tuple[str, ...]
    numbers: tuple[str, ...]
    has_question_word: bool
    stopword_ratio: float
    extracted_filters: dict[str, str]
    residual_text: str
    query_class: QueryClass

    def as_dict(self) -> dict[str, Any]:
        return {
            "token_count": self.token_count,
            "char_length": self.char_length,
            "quoted_phrases": len(self.quoted_phrases),
            "identifiers": len(self.identifiers),
            "numbers": len(self.numbers),
            "has_question_word": self.has_question_word,
            "stopword_ratio": round(self.stopword_ratio, 3),
            "extracted_filter_fields": sorted(self.extracted_filters),
            "query_class": self.query_class.value,
        }


@dataclass(frozen=True, slots=True)
class RewriteRecord:
    """Query rewriting is never silent: original, rewritten and strategies are kept."""

    original: str
    rewritten: str
    strategies: tuple[str, ...]
    variants: tuple[str, ...] = ()

    @property
    def changed(self) -> bool:
        return self.rewritten != self.original or bool(self.variants)
