# ADR-001: Engine-independent retrieval through ports and adapters

## Status

Accepted

## Context

SearchForge compares lexical, dense and hybrid retrieval strategies and has to keep working when the
storage engine changes (a local SQLite/NumPy index on a laptop, Qdrant or OpenSearch elsewhere). If
application code calls engine clients directly, every experiment becomes engine-specific, results
are hard to compare, and swapping a backend means touching search, evaluation and API code at once.

## Decision

- [`src/searchforge/ports.py`](../../src/searchforge/ports.py) defines the engine-independent
  interfaces as `typing.Protocol` classes: `Chunker`, `EmbeddingProvider`, `ChunkStore`,
  `LexicalIndex`, `VectorIndex`, `Retriever`, `FusionStrategy`, `Reranker` and
  `DocumentSourceReader`, plus the `RetrievalRequest` value object.
- Application services (`service/search_service.py`, `retrieval/retrievers.py`, evaluation and the
  API) depend only on these protocols and on domain types (`Candidate`, `ScoredChunk`,
  `FilterExpression`). No backend object crosses the port boundary.
- Concrete engines are adapters:
  - `indexes/lexical/sqlite_fts.py` - `LexicalIndex` on SQLite FTS5 (ADR-002).
  - `indexes/dense/numpy_index.py` - exact `VectorIndex` on NumPy (memory-mapped `vectors.npy`).
  - `indexes/external/qdrant.py` - `VectorIndex` on Qdrant (Query API, `qdrant` extra).
  - `indexes/external/opensearch.py` - `LexicalIndex` (BM25 `multi_match`) and `VectorIndex`
    (k-NN) on OpenSearch (`opensearch` extra).
- Metadata filters are parsed once into a canonical, typed `FilterExpression`
  (`domain/filters.py`) and translated per backend in `filtering/translate.py`
  (`to_sqlite`, `to_qdrant_dict`, `to_opensearch`).
- Optional engine libraries are imported lazily inside the adapter modules, so the core package
  installs and runs without them.

## Consequences

Positive:

- Retrieval strategies, fusion, reranking and evaluation are written once and measured identically
  on every backend.
- Backends can be tested in isolation: Qdrant runs in-memory in CI; OpenSearch query construction
  is tested with a fake client, and live tests run only when `OPENSEARCH_URL` is set.
- Fusion stays client-side, so its parameters do not depend on engine defaults (ADR-004).

Negative:

- Engine-native features (server-side hybrid queries, Qdrant prefetch, OpenSearch search
  pipelines) are not used by default. Some latency or recall optimisations are left unused.
- The protocols are a lowest common denominator. For example, `OpenSearchLexicalIndex.search`
  accepts `phrases` but ignores them, and `SqliteFtsLexicalIndex` uses them.
- Wiring is incomplete in this version. The index builder (`indexes/builder.py`) writes only
  `sqlite_fts` lexical indexes and rejects `lexical.backend: opensearch`. `DenseConfig.backend`
  accepts only `numpy` and `qdrant`. `indexes/handle.py` always opens the SQLite chunk store and
  SQLite FTS5. The OpenSearch adapters are usable only programmatically.

## Alternatives considered

- **Use one engine directly (for example OpenSearch everywhere).** This is simpler, but it needs a
  JVM service with several GB of RAM for every test and demo. That conflicts with ADR-009, and it
  ties the evaluation results to one engine.
- **Adopt a framework retriever abstraction (LangChain, LlamaIndex).** This rejected option adds a
  large dependency surface. Those frameworks also hide filter translation and scoring details,
  which SearchForge needs to measure and explain.
- **Use abstract base classes instead of protocols.** Rejected: structural typing lets thin
  wrappers and test fakes satisfy a port without inheriting from it, and `mypy --strict` still
  checks conformance.
