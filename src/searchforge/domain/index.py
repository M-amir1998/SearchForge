"""Index manifest: the immutable description of one completed index version."""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class EmbeddingSpec(BaseModel):
    """Identity of the embedding model used to build a dense index."""

    model_config = ConfigDict(frozen=True)

    provider: str
    model: str
    revision: str
    dimension: int
    preprocessing_version: str
    normalized: bool = True


class FileChecksum(BaseModel):
    model_config = ConfigDict(frozen=True)

    path: str
    sha256: str
    bytes: int


class IndexManifest(BaseModel):
    """Written last when a build completes; a version without a manifest is incomplete.

    Changing the corpus, the chunking strategy, the embedding model/revision or the
    lexical configuration changes ``build_fingerprint`` and therefore the version.
    """

    model_config = ConfigDict(frozen=True)

    schema_version: int = 1
    index_id: str
    index_name: str
    version: str
    parent_version: str | None = None
    status: str = "complete"
    corpus_id: str
    corpus_version: str
    document_count: int
    chunk_count: int
    index_types: list[str]
    lexical_backend: str | None = None
    lexical_config: dict[str, Any] = Field(default_factory=dict)
    dense_config: dict[str, Any] = Field(default_factory=dict)
    vector_backend: str | None = None
    embedding: EmbeddingSpec | None = None
    chunking_config: dict[str, Any]
    chunker_fingerprint: str
    filter_schema: dict[str, str]
    build_fingerprint: str
    git_sha: str | None = None
    searchforge_version: str
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    build_seconds: float | None = None
    peak_rss_mb: float | None = None
    files: list[FileChecksum] = Field(default_factory=list)
    update_summary: dict[str, int] = Field(default_factory=dict)

    @property
    def has_dense(self) -> bool:
        return "dense" in self.index_types

    @property
    def has_lexical(self) -> bool:
        return "lexical" in self.index_types

    def size_bytes(self) -> int:
        return sum(f.bytes for f in self.files)

    def size_by_component(self) -> dict[str, int]:
        out: dict[str, int] = {}
        for f in self.files:
            component = f.path.split("/", 1)[0].split(".", 1)[0]
            out[component] = out.get(component, 0) + f.bytes
        return out
