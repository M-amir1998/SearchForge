# Indexing

From source files to a published, immutable index version. The orchestration lives in
[indexes/builder.py](../../src/searchforge/indexes/builder.py); this page follows the data.

## Ingestion

[ingestion/readers.py](../../src/searchforge/ingestion/readers.py) turns each configured source into a
stream of `Document` objects; [ingestion/safety.py](../../src/searchforge/ingestion/safety.py) enforces
the limits. Formats come from `source.format` or the file suffix:

| Format | Suffixes | Read pattern | Record key |
|---|---|---|---|
| text | `.txt`, `.text` | whole file; title = first non-empty line if at most 120 chars, else file stem | empty |
| markdown | `.md`, `.markdown` | whole file; YAML front matter (`yaml.safe_load`) becomes fields; title = front matter `title`, else first H1, else stem | empty |
| html | `.html`, `.htm` | whole file; scripts, styles, iframes etc. dropped (never executed); headings rendered as Markdown ATX headings | empty |
| json | `.json` | whole file; an object or a list of objects | `#<i>` |
| jsonl | `.jsonl`, `.ndjson` | line by line | `#L<line>` |
| csv | `.csv` | whole file through `csv.DictReader` | `#R<i>` |
| parquet | `.parquet` | record batches of `PARQUET_BATCH_ROWS` (1024) | `#R<i>` |

Safety rules (`IngestionLimits`, defaults in [config/models.py](../../src/searchforge/config/models.py)):

| Rule | Behaviour |
|---|---|
| Path containment | every file must resolve under its source root (`ensure_within`); a glob containing `..` raises `UnsafePathError` |
| Symlinks | skipped unless `follow_symlinks: true` |
| `max_file_bytes` (50 MiB) | checked on `stat` for every format, including JSONL and Parquet, and again while reading whole-file formats |
| `max_record_chars` (2,000,000) | longer records are skipped with a warning |
| `max_documents` | optional cap on emitted documents |
| `on_decode_error` | `replace` (default), `skip` or `fail`; UTF-8 with optional BOM |
| Duplicate ids | first occurrence wins; later ones are counted and warned |

The `IngestionReport` keeps counters and up to 200 warnings, never document text. Text, Markdown,
HTML, JSON and CSV files are read whole (bounded by `max_file_bytes`); only JSONL and Parquet stream
below file level. Documents themselves are always yielded one at a time.

Attributes: if `metadata_fields` is set, only those keys are kept; otherwise every non-reserved key.
Only scalar values with identifier-like keys survive (keys are lowercased). The reader adds `source`
(the source id), `language` (when present) and `document_id`.

## Normalization

[parsing/normalize.py](../../src/searchforge/parsing/normalize.py) is deliberately conservative: Unicode
NFC by default (NFKC is opt-in because it can change identifiers), CRLF/CR to LF, removal of control and
zero-width characters (newline and tab kept), collapsing of horizontal whitespace, removal of trailing
spaces and a cap of `max_blank_lines` (default 2). Casing, punctuation and Markdown structure are
preserved. Queries use `normalize_query_text` (NFC, controls removed, single-spaced).

## Chunking

[chunking/chunkers.py](../../src/searchforge/chunking/chunkers.py) works on character spans of the
normalized text, so `document.text[start_char:end_char] == chunk.text` always holds.

| Strategy | Parameters (defaults) | Behaviour |
|---|---|---|
| `section` (default) | `max_chars` 1200, `overlap_chars` 150, `min_chars` 80 | split at ATX headings outside code fences; sections longer than `max_chars` are sentence-packed; chunk records the heading path, e.g. `Deployment > Rollback` |
| `sentence` | same | greedy packing of whole sentences with sentence-aligned overlap |
| `fixed_token` | `max_tokens` 200, `overlap_tokens` 30, `min_chars` | windows of whitespace tokens (an approximation of model tokens) |

Spans shorter than `min_chars` are merged into a neighbour instead of being dropped. Over-long sentences
are split at whitespace near `max_chars`. The chunker fingerprint is a hash of the strategy name and its
parameters.

## Stable identifiers

[domain/ids.py](../../src/searchforge/domain/ids.py):

| Id | Depends on | Consequence |
|---|---|---|
| `document_id` | explicit id that is a safe identifier: used verbatim; explicit id that is not: `doc-` + hash of that id; no explicit id: `doc-` + hash of `source_key` and record key | stable across chunking/index changes; an edited document keeps its id |
| `content_hash` | normalized text + title, uri, attributes, created/updated timestamps | detects updates (see [index updates](../operations/index-updates.md)) |
| `chunk_id` | `document_id`, chunker fingerprint, span and text | `<document_id>#<hash>`; changes when chunking changes |
| citation id | `chunk_id` | `S` + upper-case hash, used by the context builder |

`source_key` is POSIX and relative to the source root; for a single-file source it is the file name.
The source id is not part of the derived id, so two sources that yield the same relative key and
record key without explicit ids produce the same `document_id`; the second is dropped as a duplicate.

## Chunk store and lexical index

[indexes/chunk_store.py](../../src/searchforge/indexes/chunk_store.py) (`chunks.sqlite`) is the source of
truth for chunk text, offsets, title, section, uri, timestamps and JSON attributes. It is used for
hydration, citations and failure analysis. Before chunking, attributes declared in
`corpus.filterable_fields` are coerced to their canonical type (dates become ISO `YYYY-MM-DD`); values
that cannot be coerced are dropped rather than stored wrongly
([filtering/translate.py](../../src/searchforge/filtering/translate.py) `canonicalize_attributes`).

[indexes/lexical/sqlite_fts.py](../../src/searchforge/indexes/lexical/sqlite_fts.py) (`lexical.sqlite`)
holds an FTS5 table with `body` (chunk text) and `title` (title and section) columns plus a side table
with chunk id, document id and attributes JSON.

- **Safe MATCH construction.** User text is never interpolated. Quoted phrases with at least two tokens
  become phrase literals; in addition every `\w+` token of the text (lowercased, stopwords removed
  unless the query is only stopwords) becomes a quoted literal with embedded quotes doubled. Duplicate
  literals are dropped and the list is joined with `OR`; token literals stop being added once
  `MAX_MATCH_TERMS` (64) literals exist. FTS5 operators in user input are therefore inert. The expression and all
  filter values are bound parameters.
- **Scoring.** FTS5 `bm25()` is lower-is-better; the index negates it so higher is better, like every
  other retriever. k1 and b are fixed by SQLite.
- **Column weights.** `lexical.text_weight` (1.0) and `lexical.title_weight` (2.0) are passed to
  `bm25()` as bound parameters.
- **Tokenizer.** `porter unicode61 remove_diacritics 2` by default; `porter unicode61` and `unicode61`
  are the alternatives. The FTS index is optimized once after the build.

## Dense index and embeddings

[indexes/dense/numpy_index.py](../../src/searchforge/indexes/dense/numpy_index.py) is an exact
inner-product index over float32, L2-normalized vectors (inner product equals cosine):

- `dense/vectors.npy` plus `dense/rows.jsonl` (chunk id, document id, attributes per row);
- loaded with `mmap_mode="r"` for search, so pages are read on demand;
- scoring in blocks of `block_rows` (65,536) with a per-block top-k, then a deterministic merge
  (score descending, row ascending);
- filters are pre-applied as a boolean mask over row attributes; up to 64 masks are memoized per
  distinct canonical filter;
- deletes are in-memory tombstones; `save` writes a compacted copy.

Embedding providers ([embeddings/](../../src/searchforge/embeddings/)) are separate from storage:

| Provider | Notes |
|---|---|
| `tfidf` (default) | scikit-learn TF-IDF (+ truncated SVD, `svd_components` 256) fitted on up to `fit_max_chunks` chunks; state persisted pickle-free under `dense/embedder/` |
| `hash` | feature hashing; non-semantic, for tests and CI only |
| `sentence_transformer` | optional `[neural]` extra; CPU; no download unless `allow_download`; revision recorded as `unpinned` when not set |
| `remote` | OpenAI-compatible `/embeddings`; corpus transmission requires explicit opt-in; circuit breaker |

The text embedded per chunk is title and section (when present) followed by the chunk body
(`embedding_text`). Document embeddings pass through the persistent embedding cache
([caching.md](caching.md)). When an index is opened, `restore_embedder` rebuilds the provider from the
manifest's `EmbeddingSpec` and rejects a dimension mismatch (and, for `tfidf` / `hash`, a revision
mismatch) with `IndexVersionMismatch`.

## External adapters

| Adapter | Wired into builder / `open_index` | Notes |
|---|---|---|
| [Qdrant](../../src/searchforge/indexes/external/qdrant.py) `VectorIndex` | yes, `dense.backend: qdrant` | one collection per version, `<collection_prefix>_<index>_<version>` (dots replaced); UUIDv5 point ids, original ids in payload; cosine; Query API (`query_points`); filters as structured `Filter` models |
| [OpenSearch](../../src/searchforge/indexes/external/opensearch.py) `LexicalIndex` and `VectorIndex` | no | BM25 `multi_match` over `body` and `title^2`; `knn_vector` HNSW (faiss, cosine); structured Query DSL only |

Qdrant can use `url` (server), `path` (embedded on disk) or `location` (default `:memory:`).
`:memory:` clients are shared within one process, so an index built in memory can be searched by the
same process (tests, demos) but disappears when it exits; use `url` or `path` for anything durable. The SQLite chunk store and FTS5
index are still local for Qdrant-backed versions.

The local builder only writes `sqlite_fts`; `lexical.backend: opensearch` raises `ConfigurationError`.
The OpenSearch classes implement the ports and are used by constructing them directly and injecting
retrievers, e.g. `SearchService(handle, config, retrievers={"lexical": LexicalRetriever(os_index)})`.
Populating the OpenSearch index (`create=True`, `add`) is the caller's responsibility.

## Builder and version layout

`IndexBuilder.build()` ([indexes/builder.py](../../src/searchforge/indexes/builder.py)):

1. `check_profile_allows_local` and `check_disk_budget` ([storage](../operations/storage.md)).
2. Pass 1 streams the corpus to compute `corpus_version` (hash of sorted document ids and content
   hashes). The build fingerprint hashes `corpus_version`, the mode, the chunker fingerprint, the
   lexical config, the dense config (excluding `cache_enabled` and `batch_size`) and the filter
   schema. If a version with the same fingerprint exists it is reused (unless `--force`).
3. Pass 2 streams documents through the chunker into the chunk store and FTS5, committing every
   `write_batch_size` (512) documents. The data-profile chunk limit is checked after ingestion.
4. Dense: fit TF-IDF if needed, then embed chunks in `dense.batch_size` batches into the vector index.
5. Checksums, manifest, atomic publish, read-only files, `CURRENT` pointer. Any exception removes the
   `.partial` directory.

The manifest ([domain/index.py](../../src/searchforge/domain/index.py)) records index id
(`name@version`), parent version, corpus id and version, counts, index types, lexical config, vector
backend, `EmbeddingSpec`, chunking config and fingerprint, filter schema, build fingerprint, git SHA,
SearchForge version, build seconds, sampled peak RSS, per-file SHA-256 checksums and the update summary.
Layout, validation and rollback: [index lifecycle](../operations/index-lifecycle.md).
