"""Open a published index version for (read-only) search."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from searchforge.config.models import DenseConfig, SearchForgeConfig
from searchforge.domain.filters import FilterSchema
from searchforge.domain.index import IndexManifest
from searchforge.embeddings.factory import restore_embedder
from searchforge.errors import ConfigurationError, IndexNotFound
from searchforge.indexes.chunk_store import SqliteChunkStore
from searchforge.indexes.dense.numpy_index import NumpyVectorIndex
from searchforge.indexes.lexical.sqlite_fts import SqliteFtsLexicalIndex
from searchforge.indexes.lifecycle import IndexLayout
from searchforge.ports import EmbeddingProvider, LexicalIndex, VectorIndex


@dataclass(slots=True)
class IndexHandle:
    manifest: IndexManifest
    path: Path
    chunk_store: SqliteChunkStore
    lexical: LexicalIndex | None
    vectors: VectorIndex | None
    embedder: EmbeddingProvider | None
    filter_schema: FilterSchema

    @property
    def index_id(self) -> str:
        return self.manifest.index_id

    def close(self) -> None:
        self.chunk_store.close()
        if isinstance(self.lexical, SqliteFtsLexicalIndex):
            self.lexical.close()


def _open_vectors(manifest: IndexManifest, vdir: Path, dense: DenseConfig) -> VectorIndex:
    if manifest.vector_backend == "qdrant":
        from searchforge.indexes.external.qdrant import QdrantVectorIndex, make_client

        if dense.qdrant is None:
            raise ConfigurationError("index uses Qdrant but dense.qdrant is not configured")
        meta = json.loads((vdir / "dense" / "qdrant.json").read_text(encoding="utf-8"))
        assert manifest.embedding is not None
        return QdrantVectorIndex(make_client(dense.qdrant), meta["collection"], manifest.embedding.dimension)
    return NumpyVectorIndex.load(vdir / "dense", mmap=True)


def open_index(workspace: Path, ref: str, config: SearchForgeConfig) -> IndexHandle:
    layout = IndexLayout(workspace)
    manifest, vdir = layout.resolve(ref)
    if not (vdir / SqliteChunkStore.FILENAME).exists():
        raise IndexNotFound(f"index {manifest.index_id} has no chunk store")
    store = SqliteChunkStore(vdir / SqliteChunkStore.FILENAME, read_only=True)
    lexical: LexicalIndex | None = None
    if manifest.has_lexical:
        lc = manifest.lexical_config
        lexical = SqliteFtsLexicalIndex(
            vdir / SqliteFtsLexicalIndex.FILENAME,
            read_only=True,
            text_weight=float(lc.get("text_weight", 1.0)),
            title_weight=float(lc.get("title_weight", 2.0)),
        )
    vectors: VectorIndex | None = None
    embedder: EmbeddingProvider | None = None
    if manifest.has_dense and manifest.embedding is not None:
        vectors = _open_vectors(manifest, vdir, config.index.dense)
        embedder = restore_embedder(manifest.embedding, vdir, config.index.dense)
    schema = FilterSchema.from_mapping(manifest.filter_schema)
    return IndexHandle(
        manifest=manifest,
        path=vdir,
        chunk_store=store,
        lexical=lexical,
        vectors=vectors,
        embedder=embedder,
        filter_schema=schema,
    )
