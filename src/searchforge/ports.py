"""Ports: the engine-independent interfaces of SearchForge.

Application services depend only on these protocols. Concrete backends
(SQLite FTS5, NumPy, Qdrant, OpenSearch, Sentence Transformers, HTTP embedding
services) are adapters that implement them. No backend object crosses this line.
"""

from __future__ import annotations

from collections.abc import Iterable, Iterator, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Protocol, runtime_checkable

import numpy as np
from numpy.typing import NDArray

from searchforge.domain.chunk import Chunk
from searchforge.domain.document import AttributeValue, Document
from searchforge.domain.filters import FilterExpression
from searchforge.domain.index import EmbeddingSpec
from searchforge.domain.result import Candidate, ScoredChunk

FloatMatrix = NDArray[np.float32]
FloatVector = NDArray[np.float32]


@dataclass(frozen=True, slots=True)
class RetrievalRequest:
    """What a retriever needs; already rewritten, validated and filter-parsed."""

    text: str
    k: int
    filters: FilterExpression = field(default_factory=FilterExpression)
    phrases: tuple[str, ...] = ()


@runtime_checkable
class Chunker(Protocol):
    """Splits a normalized document into chunks with exact character offsets."""

    name: str

    def fingerprint(self) -> str:
        """Stable hash of the chunker type and its configuration."""
        ...

    def chunk(self, document: Document) -> Iterator[Chunk]: ...


@runtime_checkable
class EmbeddingProvider(Protocol):
    """Turns text into L2-normalized float32 vectors. Separate from storage."""

    @property
    def spec(self) -> EmbeddingSpec: ...

    def embed_documents(self, texts: Sequence[str]) -> FloatMatrix: ...

    def embed_query(self, text: str) -> FloatVector: ...


@runtime_checkable
class ChunkStore(Protocol):
    """Source of truth for chunk text/metadata (used for hydration and citations)."""

    def get_many(self, chunk_ids: Sequence[str]) -> dict[str, Chunk]: ...

    def iter_chunks(self, batch_size: int = 512) -> Iterator[list[Chunk]]: ...

    def document_hashes(self) -> dict[str, str]: ...

    def count_chunks(self) -> int: ...

    def count_documents(self) -> int: ...


@runtime_checkable
class LexicalIndex(Protocol):
    """A term-matching index with BM25 (or equivalent) ranking and filtering."""

    backend: str

    def search(self, text: str, filters: FilterExpression, k: int, phrases: Sequence[str] = ()) -> list[Candidate]: ...

    def add(self, chunks: Sequence[Chunk]) -> None: ...

    def delete_documents(self, document_ids: Sequence[str]) -> None: ...

    def count(self) -> int: ...


@runtime_checkable
class VectorIndex(Protocol):
    """Nearest-neighbour search over pre-computed vectors with metadata filtering."""

    backend: str

    @property
    def dimension(self) -> int: ...

    def search(self, vector: FloatVector, filters: FilterExpression, k: int) -> list[Candidate]: ...

    def add(
        self,
        chunk_ids: Sequence[str],
        document_ids: Sequence[str],
        attributes: Sequence[Mapping[str, AttributeValue]],
        vectors: FloatMatrix,
    ) -> None: ...

    def delete_documents(self, document_ids: Sequence[str]) -> None: ...

    def get_vectors(self, chunk_ids: Sequence[str]) -> dict[str, FloatVector]: ...

    def count(self) -> int: ...


@runtime_checkable
class Retriever(Protocol):
    """Produces a ranked candidate list for one retrieval request."""

    name: str

    def retrieve(self, request: RetrievalRequest) -> list[Candidate]: ...


@runtime_checkable
class FusionStrategy(Protocol):
    """Combines several ranked lists (one per retriever / query variant)."""

    name: str

    def fuse(self, ranked_lists: Mapping[str, Sequence[Candidate]], limit: int) -> list[Candidate]: ...


@runtime_checkable
class Reranker(Protocol):
    """Re-scores the top-N hydrated candidates. Returns them sorted by new score."""

    name: str

    def rerank(self, query: str, candidates: Sequence[ScoredChunk]) -> list[ScoredChunk]: ...


@runtime_checkable
class DocumentSourceReader(Protocol):
    """Streams documents from one configured source."""

    def iter_documents(self) -> Iterable[Document]: ...
