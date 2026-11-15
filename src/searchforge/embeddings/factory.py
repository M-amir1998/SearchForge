"""Construct embedding providers from config, and restore them for a built index."""

from __future__ import annotations

from pathlib import Path

from searchforge.config.models import DenseConfig, EmbeddingProviderName
from searchforge.domain.index import EmbeddingSpec
from searchforge.embeddings.hashing import HashEmbeddingProvider
from searchforge.embeddings.tfidf import TfidfEmbeddingProvider
from searchforge.errors import ConfigurationError, IndexVersionMismatch
from searchforge.ports import EmbeddingProvider

EMBEDDER_DIR = "dense/embedder"


def create_embedder(dense: DenseConfig) -> EmbeddingProvider | TfidfEmbeddingProvider:
    """A fresh provider. TF-IDF is returned *unfitted* (the builder fits it)."""
    if dense.provider is EmbeddingProviderName.HASH:
        return HashEmbeddingProvider(dimension=dense.dimension)
    if dense.provider is EmbeddingProviderName.TFIDF:
        return TfidfEmbeddingProvider(dense.tfidf)
    if dense.provider is EmbeddingProviderName.SENTENCE_TRANSFORMER:
        from searchforge.embeddings.sentence_transformer import SentenceTransformerEmbeddingProvider

        return SentenceTransformerEmbeddingProvider(
            dense.model, dense.revision, batch_size=dense.batch_size, allow_download=dense.allow_download
        )
    if dense.provider is EmbeddingProviderName.REMOTE:
        from searchforge.embeddings.remote import RemoteEmbeddingProvider

        if dense.remote is None:
            raise ConfigurationError("dense.remote is required for the remote provider")
        return RemoteEmbeddingProvider(dense.remote)
    raise ConfigurationError(f"unknown embedding provider {dense.provider}")  # pragma: no cover


def restore_embedder(spec: EmbeddingSpec, version_dir: Path, dense: DenseConfig) -> EmbeddingProvider:
    """Rebuild the exact provider an index was built with, verifying its identity."""
    provider: EmbeddingProvider
    if spec.provider == "tfidf":
        provider = TfidfEmbeddingProvider.load(version_dir / EMBEDDER_DIR)
    elif spec.provider == "hash":
        provider = HashEmbeddingProvider(dimension=spec.dimension)
    elif spec.provider == "sentence_transformer":
        from searchforge.embeddings.sentence_transformer import SentenceTransformerEmbeddingProvider

        provider = SentenceTransformerEmbeddingProvider(
            spec.model,
            None if spec.revision == "unpinned" else spec.revision,
            batch_size=dense.batch_size,
            allow_download=dense.allow_download,
        )
    elif spec.provider == "remote":
        from searchforge.embeddings.remote import RemoteEmbeddingProvider

        if dense.remote is None or dense.remote.model != spec.model:
            raise IndexVersionMismatch("index was built with a remote model that is not configured")
        provider = RemoteEmbeddingProvider(dense.remote)
    else:
        raise IndexVersionMismatch(f"unknown embedding provider in manifest: {spec.provider}")
    got = provider.spec
    pinned = spec.provider in ("tfidf", "hash") or (
        spec.provider == "sentence_transformer" and spec.revision != "unpinned"
    )
    if got.dimension != spec.dimension or (pinned and got.revision != spec.revision):
        raise IndexVersionMismatch(
            f"embedding provider mismatch: index={spec.model}@{spec.revision}/{spec.dimension}, "
            f"loaded={got.model}@{got.revision}/{got.dimension}"
        )
    return provider
