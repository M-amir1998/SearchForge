"""Query-level failure analysis with an explicit failure taxonomy.

For every relevant document that is missing from the final top-k, we record
**evidence** (observed facts: is it indexed, does it pass the filters, its rank in
each retriever's list, in the fused list and in the final list, duplicate pressure)
separately from a **hypothesis** (the most likely taxonomy category). Hypotheses
are rule-based and explicitly labelled as such — they are a triage aid, not a
root-cause proof.

Taxonomy: missed_lexical_match, semantic_miss, bad_fusion, bad_rerank,
filter_error, duplicate_results, chunking_failure, query_rewrite_failure,
index_staleness.
"""

from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any

from searchforge.domain.filters import FilterSchema, parse_filters
from searchforge.domain.judgment import Qrels
from searchforge.errors import InvalidFilter
from searchforge.evaluation.runner import QueryOutcome, RunResult
from searchforge.indexes.chunk_store import SqliteChunkStore
from searchforge.parsing.stopwords import STOPWORDS

_WORD = re.compile(r"\w+", re.UNICODE)


class FailureCategory(StrEnum):
    MISSED_LEXICAL_MATCH = "missed_lexical_match"
    SEMANTIC_MISS = "semantic_miss"
    BAD_FUSION = "bad_fusion"
    BAD_RERANK = "bad_rerank"
    FILTER_ERROR = "filter_error"
    DUPLICATE_RESULTS = "duplicate_results"
    CHUNKING_FAILURE = "chunking_failure"
    QUERY_REWRITE_FAILURE = "query_rewrite_failure"
    INDEX_STALENESS = "index_staleness"


@dataclass(slots=True)
class FailureRecord:
    query_id: str
    query_type: str
    document_id: str
    relevance: float
    category: FailureCategory
    evidence: dict[str, Any]
    hypothesis: str
    basis: str = "rule-based hypothesis"
    severity: str = "normal"

    def as_row(self) -> dict[str, Any]:
        return {
            "query_id": self.query_id,
            "query_type": self.query_type,
            "document_id": self.document_id,
            "relevance": self.relevance,
            "category": self.category.value,
            "hypothesis": self.hypothesis,
            "basis": self.basis,
            "severity": self.severity,
            **{f"evidence_{k}": (str(v) if isinstance(v, list | dict) else v) for k, v in self.evidence.items()},
        }


@dataclass(slots=True)
class FailureReport:
    records: list[FailureRecord] = field(default_factory=list)

    def counts(self) -> dict[str, int]:
        return dict(Counter(r.category.value for r in self.records).most_common())


def _rank(docs: list[str], doc: str) -> int | None:
    try:
        return docs.index(doc) + 1
    except ValueError:
        return None


def _terms(text: str) -> set[str]:
    return {t for t in _WORD.findall(text.lower()) if t not in STOPWORDS and len(t) > 1}


def analyze_failures(
    run: RunResult,
    qrels: Qrels,
    store: SqliteChunkStore,
    schema: FilterSchema,
    *,
    k: int = 10,
    threshold: float = 1.0,
) -> FailureReport:
    report = FailureReport()
    for outcome in run.scored:
        report.records.extend(_analyze_query(outcome, qrels, store, schema, k, threshold))
    return report


def _analyze_query(
    o: QueryOutcome, qrels: Qrels, store: SqliteChunkStore, schema: FilterSchema, k: int, threshold: float
) -> list[FailureRecord]:
    judgments = qrels.for_query(o.query.query_id)
    top = o.doc_ranking[:k]
    top_chunks = [c[1] for c in o.chunk_ranking[:k]]
    dup_pressure = len(top_chunks) - len(set(top_chunks))
    try:
        filters = parse_filters(o.query.filters, schema)
    except InvalidFilter:
        filters = None
    q_terms = _terms(o.query.text)
    out: list[FailureRecord] = []
    for doc_id, rel in sorted(judgments.items(), key=lambda kv: -kv[1]):
        if rel < threshold or doc_id in top:
            continue
        info = store.document_info(doc_id)
        evidence: dict[str, Any] = {
            "indexed": info is not None,
            "final_rank": _rank(o.doc_ranking, doc_id),
            "fused_rank": _rank(o.stage_docs.get("fused", []), doc_id),
            "lexical_rank": _rank(o.stage_docs.get("lexical", []), doc_id),
            "dense_rank": _rank(o.stage_docs.get("dense", []), doc_id),
            "planned_profile": o.planned_profile,
            "reranked": o.reranked,
            "duplicate_chunks_in_top_k": dup_pressure,
            "rewrite_strategies": o.rewrite_strategies,
        }
        category, hypothesis = _classify(o, doc_id, evidence, store, filters, q_terms, k, info)
        out.append(
            FailureRecord(
                query_id=o.query.query_id,
                query_type=o.query.query_type,
                document_id=doc_id,
                relevance=rel,
                category=category,
                evidence=evidence,
                hypothesis=hypothesis,
                severity=o.query.severity.value,
            )
        )
    return out


def _classify(
    o: QueryOutcome,
    doc_id: str,
    ev: dict[str, Any],
    store: SqliteChunkStore,
    filters: Any,
    q_terms: set[str],
    k: int,
    info: dict[str, object] | None,
) -> tuple[FailureCategory, str]:
    if info is None:
        return FailureCategory.INDEX_STALENESS, "relevant document is not in the served index version"
    chunks = store.chunks_for_document(doc_id)
    if filters and chunks and not filters.matches(chunks[0].metadata.attributes):
        return FailureCategory.FILTER_ERROR, "query filters exclude a document judged relevant"
    fused, lex, dense = ev["fused_rank"], ev["lexical_rank"], ev["dense_rank"]
    if fused is not None and fused <= k and ev["reranked"]:
        return FailureCategory.BAD_RERANK, f"in fused top-{k} (rank {fused}) but pushed out by the reranker"
    retrieved = (lex is not None and lex <= k) or (dense is not None and dense <= k)
    if retrieved and (fused is None or fused > k):
        return FailureCategory.BAD_FUSION, "a retriever ranked it in its top-k but fusion demoted it"
    if ev["duplicate_chunks_in_top_k"] >= max(2, k // 3):
        return FailureCategory.DUPLICATE_RESULTS, "several chunks of the same documents occupy the top-k"
    if lex is None and q_terms:
        per_chunk = [_terms(c.text) | _terms(c.metadata.section or "") for c in chunks]
        whole = set().union(*per_chunk) if per_chunk else set()
        coverage = len(q_terms & whole) / len(q_terms)
        best_chunk = max((len(q_terms & pc) / len(q_terms) for pc in per_chunk), default=0.0)
        if coverage >= 0.75 and best_chunk < 0.5 and len(chunks) > 1:
            return FailureCategory.CHUNKING_FAILURE, "query terms are spread across chunks; no single chunk matches"
    rewrote = bool({"acronym_expansion", "synonyms"} & set(o.rewrite_strategies))
    if rewrote and lex is None and dense is None:
        return (
            FailureCategory.QUERY_REWRITE_FAILURE,
            "rewritten query surfaced nothing relevant (compare without rewrite)",
        )
    used_dense = "dense" in o.stage_docs
    if lex is None and dense is None:
        if used_dense:
            return FailureCategory.SEMANTIC_MISS, "neither lexical nor dense retrieval surfaced the document"
        return (
            FailureCategory.MISSED_LEXICAL_MATCH,
            "lexical-only plan; document not among the lexical candidates",
        )
    if lex is not None and (dense is None or (lex <= dense)):
        return FailureCategory.MISSED_LEXICAL_MATCH, f"weak lexical match (lexical rank {lex}) and no semantic rescue"
    return FailureCategory.SEMANTIC_MISS, f"weak semantic match (dense rank {dense})"
