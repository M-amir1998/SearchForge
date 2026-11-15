"""Adapter for external RAG systems: SearchForge produces *evidence*, not generations.

``SearchForgeRetriever`` is framework-agnostic: it returns plain dataclasses /
dicts with stable citation ids that downstream generators must cite. No LLM
library is imported here.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, Protocol, runtime_checkable

from searchforge.context.builder import CitationContextBuilder, ContextPackage
from searchforge.domain.ids import make_citation_id
from searchforge.service.search_service import SearchService


@dataclass(frozen=True, slots=True)
class Evidence:
    citation_id: str
    chunk_id: str
    document_id: str
    text: str
    score: float
    rank: int
    title: str | None
    uri: str | None


@runtime_checkable
class EvidenceRetriever(Protocol):
    def retrieve(self, query: str, filters: Mapping[str, Any] | None = None) -> list[Evidence]: ...

    def context(self, query: str, filters: Mapping[str, Any] | None = None) -> ContextPackage: ...


class SearchForgeRetriever:
    def __init__(
        self,
        service: SearchService,
        *,
        profile: str = "hybrid_rerank",
        top_k: int = 8,
        context_builder: CitationContextBuilder | None = None,
    ) -> None:
        self.service = service
        self.profile = profile
        self.top_k = top_k
        self.context_builder = context_builder or CitationContextBuilder(max_chars=6000, max_per_document=2)

    def retrieve(self, query: str, filters: Mapping[str, Any] | None = None) -> list[Evidence]:
        resp = self.service.search(query, top_k=self.top_k, filters=filters, profile=self.profile)
        return [
            Evidence(
                citation_id=make_citation_id(r.chunk_id),
                chunk_id=r.chunk_id,
                document_id=r.document_id,
                text=r.text or "",
                score=r.score,
                rank=r.rank,
                title=r.metadata.get("title"),
                uri=r.metadata.get("uri"),
            )
            for r in resp.results
        ]

    def context(self, query: str, filters: Mapping[str, Any] | None = None) -> ContextPackage:
        resp = self.service.search(query, top_k=self.top_k, filters=filters, profile=self.profile)
        return self.context_builder.build(resp.results)
