# Dense retrieval

Dense retrieval embeds chunks and queries into the same vector space and ranks chunks by vector
similarity. In SearchForge it is the `DenseRetriever` in
[retrievers.py](../../src/searchforge/retrieval/retrievers.py): embed the query with an
`EmbeddingProvider`, then search a `VectorIndex`. Providers live in
[embeddings/](../../src/searchforge/embeddings/); construction and restore logic is in
[factory.py](../../src/searchforge/embeddings/factory.py).

All providers return float32, **L2-normalized** vectors. The text embedded for a chunk is its
title and section path followed by the chunk body (`embedding_text` in
[builder.py](../../src/searchforge/indexes/builder.py)); the query is embedded as-is.

## Embedding providers

Selected with `index.dense.provider` ([models.py](../../src/searchforge/config/models.py)).
The default is `tfidf`.

### `hash`: feature hashing (tests only, not semantic)

[hashing.py](../../src/searchforge/embeddings/hashing.py) hashes lower-cased word unigrams and
bigrams into `dimension` buckets (default 256) with a signed BLAKE2b hash and `1 + ln(tf)` weights.
Similarity is purely lexical overlap with hash collisions; there is no semantic knowledge. It exists
so that dense code paths (vector index, fusion, MMR, caching) run deterministically in tests and CI
without a model. **Never report hash-provider results as semantic search quality.**

### `tfidf`: TF-IDF with optional LSA (the CPU baseline)

[tfidf.py](../../src/searchforge/embeddings/tfidf.py):

- **Fitting.** At index build time a scikit-learn `TfidfVectorizer` is fitted on the embedding text
  of the indexed chunks (at most `fit_max_chunks`, default 200,000, taken in store order). Defaults:
  word n-grams up to `ngram_max = 2`, `sublinear_tf = true`, `min_df = 1`,
  `max_features = 100,000`, scikit-learn's English stopword list.
- **LSA.** If `svd_components` is set (default 256; the fixture uses 128), a `TruncatedSVD` is fitted
  on the TF-IDF matrix. The effective number of components is
  `min(svd_components, n_features - 1, n_chunks - 1)`; below 2, SVD is skipped. With SVD, vectors
  capture term co-occurrence (some vocabulary-mismatch tolerance); without it, vectors are dense
  TF-IDF and essentially lexical. Either way this is a classical baseline, not a neural model.
- **Transform.** Re-implemented with NumPy/SciPy: counts, `1 + ln(tf)`, idf weighting, L2
  normalization, projection onto the SVD components, L2 normalization again.
- **Pickle-free persistence.** The vocabulary and config are stored as JSON; `idf` and SVD
  components as `.npy` loaded with `allow_pickle=False`, so loading an index cannot execute code.
- **Identity.** The provider revision is `fit-<hash>` of the fitted state; restoring an index
  verifies the dimension and revision.
- **Frozen vocabulary.** Incremental updates (`searchforge index update`) restore the parent's
  fitted provider instead of refitting. Terms that first appear in new documents are invisible to
  this provider until a full rebuild. Lexical retrieval is unaffected.

Because the model is fitted on the corpus being searched, its quality depends on corpus size: on
a few dozen documents the co-occurrence statistics are thin.

### `sentence_transformer`: optional neural embeddings

[sentence_transformer.py](../../src/searchforge/embeddings/sentence_transformer.py), installed with
the `neural` extra. Requires `dense.model`.

- Runs on CPU only (`device="cpu"`), embeddings normalized.
- **Offline by default.** Unless `allow_download: true`, only locally cached checkpoints load
  (`local_files_only`); a missing model fails fast instead of downloading.
- **Pin `revision`** to a commit hash for reproducible indexes. An unpinned model is recorded as
  `unpinned` in the manifest.

### `remote`: OpenAI-compatible endpoint

[remote.py](../../src/searchforge/embeddings/remote.py) POSTs `{"model", "input"}` to a configured
`/embeddings` URL.

- **Corpus transmission is opt-in.** `embed_documents` raises before any network I/O unless
  `dense.remote.allow_corpus_transmission: true`. This gate covers document text only: **query text
  is always sent** at search time, and if `dimension` is not configured a short probe string is sent
  to discover it.
- The API key is read from the environment variable named in `api_key_env`; it is never written to
  config or manifests.
- Requests go through a circuit breaker; failures surface as embedding errors and the search is
  served degraded (lexical-only) when other retrievers succeed.
- The remote model has no version identity (`remote-unversioned`). If the provider changes the model
  behind the same name, existing vectors silently become inconsistent; rebuild after any change.

## Exact inner-product search

[numpy_index.py](../../src/searchforge/indexes/dense/numpy_index.py) is the default `VectorIndex`:

- Vectors are L2-normalized float32, so the inner product equals cosine similarity.
- Vectors are stored in `vectors.npy` and memory-mapped for search.
- Scoring is blockwise (`block_rows = 65,536`): each block computes `matrix @ q`, keeps its top-k
  with `argpartition`, and the merged candidates are sorted by score descending, then row ascending
  (deterministic ties).
- Filters are **pre-filters**: a boolean mask over row attributes (memoized per distinct filter)
  restricts scoring, so a filtered query still returns up to k matching rows.
- Deletes are tombstones; saving writes a compacted copy.

Cost is `O(N * d)` per query and `4 * d` bytes per vector.

### Why exact search is the evaluation baseline

Approximate nearest-neighbour (ANN) search can miss true nearest neighbours. If an evaluation uses
ANN, a drop in nDCG could come from the embedding model or from the ANN index, and the two cannot be
separated. Exact search removes that confounder: dense metrics then measure the embedding model
only. It is also deterministic, and at the data profiles SearchForge targets on a laptop the
brute-force cost is small.

## ANN at scale (Qdrant)

For collections where `O(N * d)` per query is too slow, set `dense.backend: qdrant`
([qdrant.py](../../src/searchforge/indexes/external/qdrant.py), `qdrant` extra). The adapter
creates one collection per index version with cosine distance, pushes filters down to Qdrant, and
keeps fusion client-side so RRF parameters are identical across backends. It sets no HNSW or
exact-search parameters, so Qdrant's defaults apply. Incremental updates are not supported for this
backend; rebuild instead.

General ANN trade-offs (HNSW and similar graph indexes):

- **Recall vs latency.** Larger search beams and graph degree raise recall and cost.
- **Memory and build time.** The graph adds memory beyond the raw vectors and takes time to build.
- **Filtered search.** Restrictive filters interact with graph traversal; recall under filters can
  differ from unfiltered recall.

To attribute a quality change correctly, build the same embedder into both a `numpy` and a `qdrant`
index and compare: differences between them are ANN effects, not model effects.

## Related

- [bm25.md](bm25.md), [hybrid-search.md](hybrid-search.md), [ranking-fusion.md](ranking-fusion.md)
