"""Incident-to-search-eval: turn an observed retrieval failure into a permanent regression case.

    retrieval failure → sanitize query → identify expected document(s)
      → add relevance judgment(s) → new regression query (provenance=failure_derived)

Mirrors reliability engineering: every incident leaves a test behind. Query text is
redacted (emails, URLs, numbers, tokens...) before it is persisted.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

from searchforge.domain.ids import query_hash
from searchforge.domain.judgment import (
    EvalQuery,
    Qrels,
    QueryProvenance,
    QuerySeverity,
    QuerySplit,
    RelevanceJudgment,
)
from searchforge.errors import EvaluationFailure
from searchforge.evaluation.datasets import load_qrels, load_queries, save_qrels, save_queries
from searchforge.observability.privacy import redact


@dataclass(frozen=True, slots=True)
class IncidentCase:
    raw_query: str
    expected_document_ids: Sequence[str]
    severity: QuerySeverity = QuerySeverity.IMPORTANT
    relevance: int = 3
    incident_ref: str = ""
    filters: dict[str, object] | None = None


def incident_to_eval(
    case: IncidentCase, known_document_ids: set[str] | None = None
) -> tuple[EvalQuery, list[RelevanceJudgment]]:
    if not case.expected_document_ids:
        raise EvaluationFailure("an incident case needs at least one expected document")
    if known_document_ids is not None:
        unknown = [d for d in case.expected_document_ids if d not in known_document_ids]
        if unknown:
            raise EvaluationFailure(f"expected documents are not in the index: {unknown}")
    text = redact(" ".join(case.raw_query.split()))
    qid = "q-inc-" + query_hash(text)[:10]
    query = EvalQuery(
        query_id=qid,
        text=text,
        query_type="incident",
        split=QuerySplit.TEST,
        severity=case.severity,
        filters=dict(case.filters or {}),
        provenance=QueryProvenance.FAILURE_DERIVED,
        notes=f"incident:{case.incident_ref}".rstrip(":"),
    )
    judgments = [RelevanceJudgment(qid, d, case.relevance, source="incident") for d in case.expected_document_ids]
    return query, judgments


def append_incident(
    case: IncidentCase, queries_path: Path, qrels_path: Path, known_document_ids: set[str] | None = None
) -> EvalQuery:
    query, judgments = incident_to_eval(case, known_document_ids)
    queries = load_queries(queries_path) if queries_path.exists() else []
    qrels = load_qrels(qrels_path) if qrels_path.exists() else Qrels()
    if any(q.query_id == query.query_id for q in queries):
        raise EvaluationFailure(f"incident query already recorded as {query.query_id}")
    queries.append(query)
    for j in judgments:
        qrels.add(j)
    save_queries(queries, queries_path)
    save_qrels(qrels, qrels_path)
    return query
