"""Domain models. No backend, web-framework or ML-library objects live here."""

from __future__ import annotations

from searchforge.domain.chunk import Chunk, ChunkMetadata
from searchforge.domain.document import (
    AttributeValue,
    Document,
    DocumentMetadata,
    DocumentSource,
    SourceFormat,
)
from searchforge.domain.filters import (
    Condition,
    FieldType,
    FilterExpression,
    FilterOp,
    FilterSchema,
    parse_filters,
)
from searchforge.domain.index import EmbeddingSpec, IndexManifest
from searchforge.domain.judgment import (
    EvalQuery,
    Qrels,
    QueryProvenance,
    QuerySeverity,
    QuerySplit,
    RelevanceJudgment,
)
from searchforge.domain.query import QueryAnalysis, QueryClass, RewriteRecord, SearchQuery
from searchforge.domain.result import (
    Candidate,
    RankingSignals,
    ScoredChunk,
    SearchResponse,
    SearchResult,
    SearchTrace,
)

__all__ = [
    "AttributeValue",
    "Candidate",
    "Chunk",
    "ChunkMetadata",
    "Condition",
    "Document",
    "DocumentMetadata",
    "DocumentSource",
    "EmbeddingSpec",
    "EvalQuery",
    "FieldType",
    "FilterExpression",
    "FilterOp",
    "FilterSchema",
    "IndexManifest",
    "Qrels",
    "QueryAnalysis",
    "QueryClass",
    "QueryProvenance",
    "QuerySeverity",
    "QuerySplit",
    "RankingSignals",
    "RelevanceJudgment",
    "RewriteRecord",
    "ScoredChunk",
    "SearchQuery",
    "SearchResponse",
    "SearchResult",
    "SearchTrace",
    "SourceFormat",
    "parse_filters",
]
