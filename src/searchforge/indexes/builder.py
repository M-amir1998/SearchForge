"""Index builds: full (versioned, reusable by fingerprint) and incremental (content-hash diff).

Full build
    1. safety checks (data profile, disk budget),
    2. pass 1 streams the corpus to compute ``corpus_version`` (doc ids + content hashes);
       if an identical build fingerprint exists the existing version is reused,
    3. pass 2 streams documents → chunker → chunk store + FTS5 (batched commits),
    4. dense: fit the provider if needed (TF-IDF), then embed chunks in batches
       through the embedding cache into the vector index,
    5. checksums + manifest, atomic publish, read-only files, CURRENT pointer.

Incremental update
    Copies the parent version, then applies only *new*, *updated* (content hash
    changed) and *deleted* documents. Unchanged documents are not re-chunked or
    re-embedded. Requires an identical chunker/lexical/embedding configuration;
    otherwise :class:`IndexVersionMismatch` asks for a full rebuild.
"""

from __future__ import annotations

import dataclasses
import shutil
import time
from collections.abc import Iterator, Sequence
from dataclasses import dataclass, field
from pathlib import Path

from searchforge import __version__
from searchforge.benchmarking.memory import PeakRSSMonitor
from searchforge.chunking.chunkers import build_chunker
from searchforge.config.loader import LoadedConfig
from searchforge.domain.chunk import Chunk
from searchforge.domain.document import Document
from searchforge.domain.filters import FilterSchema
from searchforge.domain.ids import canonical_json, sha256_hex
from searchforge.domain.index import IndexManifest
from searchforge.embeddings.cache import CachedEmbeddingProvider, EmbeddingCache
from searchforge.embeddings.factory import EMBEDDER_DIR, create_embedder, restore_embedder
from searchforge.embeddings.tfidf import TfidfEmbeddingProvider
from searchforge.errors import ConfigurationError, IndexVersionMismatch
from searchforge.filtering.translate import canonicalize_attributes
from searchforge.indexes.chunk_store import SqliteChunkStore
from searchforge.indexes.dense.numpy_index import NumpyVectorIndex
from searchforge.indexes.lexical.sqlite_fts import SqliteFtsLexicalIndex
from searchforge.indexes.lifecycle import (
    IndexLayout,
    checksum_files,
    git_sha,
    make_writable,
    new_version_id,
    remove_tree,
)
from searchforge.ingestion.readers import iter_corpus
from searchforge.ingestion.safety import IngestionReport
from searchforge.ports import EmbeddingProvider, VectorIndex
from searchforge.storage.budget import check_disk_budget, check_profile_allows_local, enforce_chunk_limit

DENSE_DIR = "dense"


def embedding_text(chunk: Chunk) -> str:
    """Text sent to embedding models: title/section context + chunk body."""
    head = " — ".join(p for p in (chunk.metadata.title, chunk.metadata.section) if p)
    return f"{head}\n{chunk.text}" if head else chunk.text


@dataclass(slots=True)
class BuildResult:
    manifest: IndexManifest
    path: Path
    reused: bool = False
    report: IngestionReport = field(default_factory=IngestionReport)


class IndexBuilder:
    def __init__(self, loaded: LoadedConfig) -> None:
        self.loaded = loaded
        self.cfg = loaded.config
        self.layout = IndexLayout(loaded.workspace)
        self.schema = FilterSchema.from_mapping(self.cfg.corpus.filterable_fields)
        self.chunker = build_chunker(self.cfg.chunking)
        self.name = self.cfg.index.name

    # ------------------------------------------------------------------ helpers
    def _canonical(self, doc: Document) -> Document:
        attrs = canonicalize_attributes(doc.metadata.attributes, self.schema)
        return dataclasses.replace(doc, metadata=dataclasses.replace(doc.metadata, attributes=attrs))

    def corpus_version(self) -> tuple[str, int]:
        pairs = sorted((d.document_id, d.content_hash) for d in iter_corpus(self.loaded))
        return sha256_hex(canonical_json(pairs))[:16], len(pairs)

    def _config_fingerprint_parts(self) -> dict[str, object]:
        idx = self.cfg.index
        return {
            "chunker": self.chunker.fingerprint(),
            "lexical": idx.lexical.model_dump(mode="json") if idx.lexical.enabled else None,
            "dense": idx.dense.model_dump(mode="json", exclude={"cache_enabled", "batch_size"})
            if idx.dense.enabled
            else None,
            "filters": {k: v.value for k, v in self.schema.fields.items()},
        }

    def _fingerprint(self, corpus_version: str, mode: str) -> str:
        return sha256_hex(canonical_json({"corpus": corpus_version, "mode": mode, **self._config_fingerprint_parts()}))

    def _embedding_cache(self) -> EmbeddingCache | None:
        if not self.cfg.index.dense.cache_enabled:
            return None
        return EmbeddingCache(self.loaded.workspace / "cache" / "embeddings")

    def _wrap(self, provider: EmbeddingProvider, cache: EmbeddingCache | None) -> EmbeddingProvider:
        return CachedEmbeddingProvider(provider, cache) if cache is not None else provider

    def _open_writers(self, vdir: Path) -> tuple[SqliteChunkStore, SqliteFtsLexicalIndex | None]:
        lex_cfg = self.cfg.index.lexical
        store = SqliteChunkStore(vdir / SqliteChunkStore.FILENAME, read_only=False)
        lexical = None
        if lex_cfg.enabled:
            if lex_cfg.backend != "sqlite_fts":
                raise ConfigurationError(
                    "the local builder writes sqlite_fts; OpenSearch lexical indexes are built with "
                    "the opensearch adapter (see docs/architecture/indexing.md)"
                )
            lexical = SqliteFtsLexicalIndex(
                vdir / SqliteFtsLexicalIndex.FILENAME,
                read_only=False,
                tokenizer=lex_cfg.tokenizer,
                text_weight=lex_cfg.text_weight,
                title_weight=lex_cfg.title_weight,
            )
        return store, lexical

    def _ingest(self, docs: Iterator[Document], store: SqliteChunkStore, lexical: SqliteFtsLexicalIndex | None) -> int:
        n_chunks = 0
        batch_docs = 0
        for doc in docs:
            chunks = list(self.chunker.chunk(doc))
            store.add_document(doc, chunks)
            if lexical is not None:
                lexical.add(chunks)
            n_chunks += len(chunks)
            batch_docs += 1
            if batch_docs >= self.cfg.index.write_batch_size:
                store.commit()
                if lexical is not None:
                    lexical.commit()
                batch_docs = 0
        store.commit()
        if lexical is not None:
            lexical.commit()
        return n_chunks

    def _embed_into(self, provider: EmbeddingProvider, vindex: VectorIndex, batches: Iterator[list[Chunk]]) -> None:
        for batch in batches:
            vectors = provider.embed_documents([embedding_text(c) for c in batch])
            vindex.add(
                [c.chunk_id for c in batch],
                [c.document_id for c in batch],
                [c.metadata.attributes for c in batch],
                vectors,
            )

    def _vector_index_for_write(self, dimension: int, vdir: Path, version: str) -> VectorIndex:
        dense = self.cfg.index.dense
        if dense.backend == "qdrant":
            from searchforge.indexes.external.qdrant import QdrantVectorIndex, make_client

            if dense.qdrant is None:
                raise ConfigurationError("dense.backend=qdrant requires dense.qdrant settings")
            collection = f"{dense.qdrant.collection_prefix}_{self.name}_{version}".replace(".", "_")
            (vdir / DENSE_DIR).mkdir(parents=True, exist_ok=True)
            (vdir / DENSE_DIR / "qdrant.json").write_text(canonical_json({"collection": collection}), encoding="utf-8")
            return QdrantVectorIndex(make_client(dense.qdrant), collection, dimension, create=True)
        return NumpyVectorIndex(dimension)

    def _manifest(
        self,
        *,
        version: str,
        vdir: Path,
        corpus_version: str,
        fingerprint: str,
        store: SqliteChunkStore,
        provider: EmbeddingProvider | None,
        started: float,
        peak_mb: float | None,
        parent: str | None = None,
        summary: dict[str, int] | None = None,
    ) -> IndexManifest:
        idx = self.cfg.index
        types = (["lexical"] if idx.lexical.enabled else []) + (["dense"] if idx.dense.enabled else [])
        return IndexManifest(
            index_id=f"{self.name}@{version}",
            index_name=self.name,
            version=version,
            parent_version=parent,
            corpus_id=self.cfg.corpus.id,
            corpus_version=corpus_version,
            document_count=store.count_documents(),
            chunk_count=store.count_chunks(),
            index_types=types,
            lexical_backend=idx.lexical.backend if idx.lexical.enabled else None,
            lexical_config=idx.lexical.model_dump(mode="json") if idx.lexical.enabled else {},
            vector_backend=idx.dense.backend if idx.dense.enabled else None,
            embedding=provider.spec if provider is not None else None,
            chunking_config=self.cfg.chunking.model_dump(mode="json"),
            chunker_fingerprint=self.chunker.fingerprint(),
            filter_schema={k: v.value for k, v in self.schema.fields.items()},
            build_fingerprint=fingerprint,
            git_sha=git_sha(Path.cwd()),
            searchforge_version=__version__,
            build_seconds=round(time.perf_counter() - started, 3),
            peak_rss_mb=peak_mb,
            files=checksum_files(vdir),
            update_summary=summary or {},
        )

    # ------------------------------------------------------------------ full build
    def build(self, *, force: bool = False, set_current: bool = True) -> BuildResult:
        check_profile_allows_local(self.loaded)
        check_disk_budget(self.loaded)
        corpus_version, _ = self.corpus_version()
        fingerprint = self._fingerprint(corpus_version, "full")
        existing = None if force else self.layout.find_by_fingerprint(self.name, fingerprint)
        if existing is not None:
            if set_current:
                self.layout.set_current(self.name, existing.version)
            return BuildResult(existing, self.layout.version_dir(self.name, existing.version), reused=True)

        version = new_version_id(fingerprint)
        vdir = self.layout.ensure_new_version_dir(self.name, version)
        report = IngestionReport()
        started = time.perf_counter()
        cache = self._embedding_cache()
        try:
            with PeakRSSMonitor() as mon:
                store, lexical = self._open_writers(vdir)
                docs = (self._canonical(d) for d in iter_corpus(self.loaded, report))
                n_chunks = self._ingest(docs, store, lexical)
                enforce_chunk_limit(self.cfg.index.data_profile, n_chunks)
                if lexical is not None:
                    lexical.optimize()
                    lexical.close()
                provider: EmbeddingProvider | None = None
                if self.cfg.index.dense.enabled:
                    provider = self._build_dense(store, vdir, version, cache)
                manifest = self._manifest(
                    version=version,
                    vdir=vdir,
                    corpus_version=corpus_version,
                    fingerprint=fingerprint,
                    store=store,
                    provider=provider,
                    started=started,
                    peak_mb=None,
                )
                store.close()
            manifest = manifest.model_copy(update={"peak_rss_mb": mon.peak_mb})
            path = self.layout.publish(self.name, version, manifest, set_current=set_current)
        except BaseException:
            remove_tree(vdir)
            raise
        finally:
            if cache is not None:
                cache.close()
        return BuildResult(manifest, path, report=report)

    def _build_dense(
        self, store: SqliteChunkStore, vdir: Path, version: str, cache: EmbeddingCache | None
    ) -> EmbeddingProvider:
        dense = self.cfg.index.dense
        raw = create_embedder(dense)
        if isinstance(raw, TfidfEmbeddingProvider):
            raw.fit(embedding_text(c) for batch in store.iter_chunks(1024) for c in batch)
            raw.save(vdir / EMBEDDER_DIR)
        provider = self._wrap(raw, cache)
        vindex = self._vector_index_for_write(provider.spec.dimension, vdir, version)
        self._embed_into(provider, vindex, store.iter_chunks(dense.batch_size))
        if isinstance(vindex, NumpyVectorIndex):
            vindex.save(vdir / DENSE_DIR)
        return raw

    # ------------------------------------------------------------------ incremental
    def update(
        self,
        *,
        parent_ref: str | None = None,
        delete_missing: bool = True,
        delete_ids: Sequence[str] = (),
        set_current: bool = True,
    ) -> BuildResult:
        """Derive a new version from ``parent_ref`` (default: CURRENT) with only changed docs."""
        check_profile_allows_local(self.loaded)
        parent, parent_dir = self.layout.resolve(parent_ref or self.name)
        self._check_compatible(parent)
        if parent.vector_backend == "qdrant":
            raise IndexVersionMismatch(
                "incremental updates are implemented for the local NumPy backend; rebuild Qdrant-backed "
                "indexes (each version is a new collection)"
            )

        probe_version = new_version_id(parent.build_fingerprint)
        vdir = self.layout.ensure_new_version_dir(self.name, probe_version)
        report = IngestionReport()
        started = time.perf_counter()
        cache = self._embedding_cache()
        try:
            for item in parent_dir.iterdir():
                if item.name == "manifest.json":
                    continue
                if item.is_dir():
                    shutil.copytree(item, vdir / item.name)
                else:
                    shutil.copy2(item, vdir / item.name)
            make_writable(vdir)
            with PeakRSSMonitor() as mon:
                store, lexical = self._open_writers(vdir)
                parent_hashes = store.document_hashes()
                seen: set[str] = set()
                touched: list[str] = []
                stale: list[str] = []
                summary = {"added": 0, "updated": 0, "deleted": 0, "unchanged": 0}
                explicit_deletes = set(delete_ids)

                def changed_docs() -> Iterator[Document]:
                    for doc in iter_corpus(self.loaded, report):
                        if doc.document_id in explicit_deletes:
                            continue
                        seen.add(doc.document_id)
                        old = parent_hashes.get(doc.document_id)
                        if old == doc.content_hash:
                            summary["unchanged"] += 1
                            continue
                        if old is not None:
                            summary["updated"] += 1
                            stale.append(doc.document_id)
                            store.delete_documents([doc.document_id])
                            if lexical is not None:
                                lexical.delete_documents([doc.document_id])
                        else:
                            summary["added"] += 1
                        touched.append(doc.document_id)
                        yield self._canonical(doc)

                self._ingest(changed_docs(), store, lexical)
                removed = set(explicit_deletes & parent_hashes.keys())
                if delete_missing:
                    removed |= parent_hashes.keys() - seen
                if removed:
                    store.delete_documents(sorted(removed))
                    if lexical is not None:
                        lexical.delete_documents(sorted(removed))
                    store.commit()
                summary["deleted"] = len(removed)
                stale.extend(sorted(removed))
                enforce_chunk_limit(self.cfg.index.data_profile, store.count_chunks())
                new_pairs = sorted(store.document_hashes().items())
                corpus_version = sha256_hex(canonical_json(new_pairs))[:16]

                if not touched and not removed:
                    store.close()
                    if lexical is not None:
                        lexical.close()
                    remove_tree(vdir)
                    return BuildResult(parent, parent_dir, reused=True, report=report)

                if lexical is not None:
                    lexical.optimize()
                    lexical.close()
                provider: EmbeddingProvider | None = None
                if parent.embedding is not None:
                    provider = restore_embedder(parent.embedding, vdir, self.cfg.index.dense)
                    vindex = NumpyVectorIndex.load(vdir / DENSE_DIR, mmap=False)
                    vindex.delete_documents(stale)
                    self._embed_into(
                        self._wrap(provider, cache),
                        vindex,
                        store.iter_chunks_for_documents(touched, self.cfg.index.dense.batch_size),
                    )
                    (vdir / DENSE_DIR / NumpyVectorIndex.VECTORS).unlink()
                    (vdir / DENSE_DIR / NumpyVectorIndex.ROWS).unlink()
                    vindex.save(vdir / DENSE_DIR)
                fingerprint = sha256_hex(
                    canonical_json(
                        {
                            "corpus": corpus_version,
                            "parent": parent.build_fingerprint,
                            "embedding": provider.spec.revision if provider else None,
                            **self._config_fingerprint_parts(),
                        }
                    )
                )
                manifest = self._manifest(
                    version=probe_version,
                    vdir=vdir,
                    corpus_version=corpus_version,
                    fingerprint=fingerprint,
                    store=store,
                    provider=provider,
                    started=started,
                    peak_mb=None,
                    parent=parent.version,
                    summary=summary,
                )
                store.close()
            manifest = manifest.model_copy(update={"peak_rss_mb": mon.peak_mb})
            path = self.layout.publish(self.name, probe_version, manifest, set_current=set_current)
        except BaseException:
            remove_tree(vdir)
            raise
        finally:
            if cache is not None:
                cache.close()
        return BuildResult(manifest, path, report=report)

    def _check_compatible(self, parent: IndexManifest) -> None:
        reasons = []
        if parent.chunker_fingerprint != self.chunker.fingerprint():
            reasons.append("chunking configuration changed")
        idx = self.cfg.index
        if idx.lexical.enabled != parent.has_lexical or (
            idx.lexical.enabled and parent.lexical_config != idx.lexical.model_dump(mode="json")
        ):
            reasons.append("lexical configuration changed")
        if idx.dense.enabled != parent.has_dense:
            reasons.append("dense enablement changed")
        elif parent.embedding is not None and parent.embedding.provider != idx.dense.provider.value:
            reasons.append("embedding provider changed")
        if parent.filter_schema != {k: v.value for k, v in self.schema.fields.items()}:
            reasons.append("filter schema changed")
        if reasons:
            raise IndexVersionMismatch("incremental update impossible (" + "; ".join(reasons) + "); run a full build")
