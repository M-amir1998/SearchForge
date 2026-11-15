"""Retrievers: thin, engine-independent adapters from requests to ranked candidates."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from searchforge.domain.result import Candidate
from searchforge.errors import EmbeddingFailure, RetrievalFailure
from searchforge.ports import EmbeddingProvider, FusionStrategy, LexicalIndex, RetrievalRequest, Retriever, VectorIndex


@dataclass(slots=True)
class LexicalRetriever:
    """BM25-style retrieval over any :class:`LexicalIndex` backend."""

    index: LexicalIndex
    name: str = "lexical"

    def retrieve(self, request: RetrievalRequest) -> list[Candidate]:
        try:
            return self.index.search(request.text, request.filters, request.k, request.phrases)
        except RetrievalFailure:
            raise
        except Exception as exc:
            raise RetrievalFailure(f"lexical retrieval failed: {type(exc).__name__}: {exc}") from exc


@dataclass(slots=True)
class DenseRetriever:
    """Embeds the query with an :class:`EmbeddingProvider` and searches a :class:`VectorIndex`."""

    embedder: EmbeddingProvider
    index: VectorIndex
    name: str = "dense"

    def retrieve(self, request: RetrievalRequest) -> list[Candidate]:
        try:
            vector = self.embedder.embed_query(request.text)
            return self.index.search(vector, request.filters, request.k)
        except (RetrievalFailure, EmbeddingFailure):
            raise
        except Exception as exc:
            raise RetrievalFailure(f"dense retrieval failed: {type(exc).__name__}: {exc}") from exc


@dataclass(slots=True)
class HybridRetriever:
    """Runs several retrievers and fuses their candidate lists.

    This is the simple, synchronous composition (handy for library/RAG use). The
    :class:`~searchforge.service.search_service.SearchService` performs the same
    steps concurrently with per-retriever timeouts and degraded-mode handling.
    """

    retrievers: Sequence[Retriever]
    fusion: FusionStrategy
    candidate_k: int = 50
    name: str = "hybrid"

    def retrieve(self, request: RetrievalRequest) -> list[Candidate]:
        sub = RetrievalRequest(text=request.text, k=self.candidate_k, filters=request.filters, phrases=request.phrases)
        lists = {r.name: r.retrieve(sub) for r in self.retrievers}
        return self.fusion.fuse(lists, limit=request.k)
