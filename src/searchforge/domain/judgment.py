"""Relevance judgments and evaluation query sets."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any


class QuerySeverity(StrEnum):
    NORMAL = "normal"
    IMPORTANT = "important"
    CRITICAL = "critical"


class QuerySplit(StrEnum):
    TRAIN = "train"
    VALIDATION = "validation"
    TEST = "test"


class QueryProvenance(StrEnum):
    MANUAL = "manual"
    SYNTHETIC = "synthetic"
    SANITIZED_PRODUCTION = "sanitized_production"
    FAILURE_DERIVED = "failure_derived"
    PUBLIC_BENCHMARK = "public_benchmark"


@dataclass(frozen=True, slots=True)
class RelevanceJudgment:
    """Graded relevance of a document for a query (0 = not relevant ... 3 = perfect)."""

    query_id: str
    document_id: str
    relevance: int | float
    source: str = "manual"


@dataclass(frozen=True, slots=True)
class EvalQuery:
    """A query in an evaluation set."""

    query_id: str
    text: str
    query_type: str = "unspecified"
    split: QuerySplit = QuerySplit.TEST
    severity: QuerySeverity = QuerySeverity.NORMAL
    filters: dict[str, Any] = field(default_factory=dict)
    provenance: QueryProvenance = QueryProvenance.MANUAL
    notes: str = ""


class Qrels:
    """In-memory qrels: ``query_id -> document_id -> relevance``."""

    def __init__(self, judgments: Iterable[RelevanceJudgment] = ()) -> None:
        self._data: dict[str, dict[str, float]] = defaultdict(dict)
        for j in judgments:
            self.add(j)

    def add(self, judgment: RelevanceJudgment) -> None:
        self._data[judgment.query_id][judgment.document_id] = float(judgment.relevance)

    def for_query(self, query_id: str) -> Mapping[str, float]:
        return self._data.get(query_id, {})

    def relevant(self, query_id: str, threshold: float = 1.0) -> set[str]:
        return {d for d, r in self.for_query(query_id).items() if r >= threshold}

    def query_ids(self) -> list[str]:
        return sorted(self._data)

    def judgments(self) -> list[RelevanceJudgment]:
        return [
            RelevanceJudgment(query_id=q, document_id=d, relevance=r)
            for q in sorted(self._data)
            for d, r in sorted(self._data[q].items())
        ]

    def __len__(self) -> int:
        return sum(len(v) for v in self._data.values())

    def __contains__(self, query_id: object) -> bool:
        return query_id in self._data
