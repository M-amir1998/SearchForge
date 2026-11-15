# ADR-003: Separate the embedding provider from the vector index

## Status

Accepted

## Context

Many vector databases and RAG frameworks combine "turn text into vectors" with "store and search
vectors". SearchForge needs to vary these independently. It compares embedding models (TF-IDF/LSA,
Sentence Transformers, a remote service) on the same storage, and it moves the same vectors
between storage engines (NumPy, Qdrant, OpenSearch k-NN). It also needs to cache embeddings and
to prove which model produced an index.

## Decision

- [`ports.py`](../../src/searchforge/ports.py) defines two independent protocols:
  - `EmbeddingProvider`: `spec`, `embed_documents(texts)` and `embed_query(text)`. It returns
    L2-normalised `float32` vectors.
  - `VectorIndex`: `add(chunk_ids, document_ids, attributes, vectors)`,
    `search(vector, filters, k)`, `get_vectors`, `delete_documents` and `count`. It only accepts
    pre-computed vectors and never sees text.
- Every provider exposes an `EmbeddingSpec` (`domain/index.py`): `provider`, `model`, `revision`,
  `dimension`, `preprocessing_version` and `normalized`.
- Providers in `src/searchforge/embeddings/`:
  - `tfidf` - the default; scikit-learn fitting and vocabulary counting, NumPy/SciPy weighting.
  - `hash` - deterministic, non-semantic; for tests and CI only.
  - `sentence_transformer` - optional `neural` extra; CPU only; no download unless allowed.
  - `remote` - an OpenAI-compatible `/embeddings` HTTP endpoint.
- `DenseRetriever` (`retrieval/retrievers.py`) composes one provider and one index at query time.
- The `EmbeddingSpec` is written into the index manifest. When an index is opened,
  `embeddings/factory.py::restore_embedder` rebuilds the provider from the manifest. It raises
  `IndexVersionMismatch` on a dimension mismatch, and for `tfidf`/`hash` also on a revision
  mismatch.
- A persistent embedding cache (`embeddings/cache.py`, SQLite in the workspace) wraps any
  provider. Its key is a SHA-256 over the text hash, provider, model, revision, preprocessing
  version and dimension.

## Consequences

Positive:

- Embedding models and storage engines can be combined freely and compared on equal terms.
- The model identity used to build an index is recorded and checked at load time. This prevents
  querying an index with vectors from a different model or dimension.
- Cached vectors are never reused across model revisions or preprocessing changes.
- A fitted TF-IDF model is stored inside the index version (`dense/embedder/`, JSON plus `.npy`).
  That keeps the index self-contained.

Negative:

- Revision checks are only as strong as the provider's revision string.
  - Sentence Transformers models without a pinned `revision` are recorded as `unpinned`.
  - Remote models are recorded as `remote-unversioned`.
  - For these providers `restore_embedder` checks only the dimension (and, for remote, the
    configured model name).
- Every call site has to pair a provider with an index explicitly. Engine-side vectorisation
  (for example OpenSearch ingest pipelines) is not used.
- Query embedding happens in the SearchForge process (or the remote service) on every uncached
  query, so it adds latency.

## Alternatives considered

- **Engine-integrated vectorisation (vector DB or search-engine inference plugins).** Rejected:
  model choice would depend on the engine, and comparisons across engines would be confounded.
- **Store vectors next to chunk text in the chunk store.** Rejected: the chunk store is the source
  of truth for text and citations. Putting vectors there would couple it to one model and one
  dimension.
- **No embedding cache.** This is simpler, but repeated benchmark runs and incremental updates
  would re-embed unchanged text on CPU.
