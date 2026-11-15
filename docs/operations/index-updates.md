# Incremental index updates

`searchforge index update` derives a new immutable version from the `CURRENT` version by processing
only documents that were added, changed or deleted. Code: `IndexBuilder.update` and
`_check_compatible` in [indexes/builder.py](../../src/searchforge/indexes/builder.py).

```
searchforge index update CONFIG                         # apply corpus changes, delete missing docs
searchforge index update CONFIG --keep-missing          # never delete docs absent from the corpus
searchforge index update CONFIG --delete KB-0042 --delete KB-0043
```

The parent is always `CURRENT` from the CLI; the Python API accepts `parent_ref="name@version"`. On
success the CLI prints the new index id, its parent and the update summary; with no changes it prints
`no changes; CURRENT stays <index id>`. Errors exit with code 2.

## Content-hash diff

Every document from the corpus is compared with the parent's `document_id -> content_hash` table
(`content_hash` covers normalized text, title, uri, attributes and timestamps; see
[indexing](../architecture/indexing.md#stable-identifiers)).

| Status | Condition | Action |
|---|---|---|
| `added` | id not in parent | chunk, add to chunk store and FTS5, embed |
| `updated` | id in parent, hash differs | delete old chunks, re-chunk, re-embed |
| `unchanged` | id in parent, same hash | nothing (not re-chunked, not re-embedded) |
| `deleted` | id passed with `--delete` and present in parent, or (default) id in parent but absent from the corpus | delete chunks and vectors |

Ids passed with `--delete` are skipped during ingestion even when they are still in the corpus, so
they end up deleted. The counts are stored as `update_summary` in the new manifest, together with
`parent_version`.

If nothing was added, updated or deleted, the partial version is discarded, the parent is returned
unchanged and `CURRENT` is not touched.

## Copy-on-write from the parent

The parent version directory is never modified. The update:

1. checks the data profile (`check_profile_allows_local`) and the compatibility rules below;
2. creates `<new-version>.partial/` and copies every parent file except `manifest.json` into it, then
   makes the copies writable;
3. applies the diff to the copied `chunks.sqlite` and `lexical.sqlite`, enforces the data-profile chunk
   limit on the resulting total and recomputes `corpus_version`;
4. for dense indexes, restores the parent's embedder from the parent's `EmbeddingSpec`, loads the copied
   vectors into memory (no memory map), tombstones updated and deleted documents, embeds the chunks of
   added and updated documents (through the embedding cache) and writes a compacted `vectors.npy`;
5. writes checksums and the manifest, publishes atomically and moves `CURRENT`
   ([index lifecycle](index-lifecycle.md)).

The whole parent is copied, so an update temporarily needs roughly the parent's size plus the new
data on disk. Like a full build, `update` runs the profile and disk-budget checks first
(`StorageBudgetExceeded` if the reserve would be violated). Any failure removes the partial directory.

The new version's fingerprint hashes the new `corpus_version`, the parent's build fingerprint, the
embedding revision and the configuration parts. It never matches a full-build fingerprint, so a later
`index build` of the same corpus creates a fresh full version instead of reusing the updated one.

## Frozen TF-IDF vocabulary

With the default `tfidf` provider, the fitted vocabulary, IDF weights and SVD components are part of
the parent version and are reused unchanged. Consequences:

- terms that first appear in added or updated documents are invisible to the dense TF-IDF vectors
  (they still reach the lexical FTS5 index, whose BM25 statistics are maintained by SQLite);
- IDF weights reflect the corpus at the last full build, so the dense representation drifts as the
  corpus changes.

Run a full `searchforge index build CONFIG --force` periodically, or whenever many new terms are
expected, to refit the provider. Sentence Transformers, remote and hash providers have no fitted state.

## Compatibility checks that force a full rebuild

`_check_compatible` raises `IndexVersionMismatch: incremental update impossible (...); run a full build`
when any of these differ between the parent manifest and the current configuration:

| Check | Compared |
|---|---|
| chunking configuration changed | chunker fingerprint (strategy and its parameters) |
| lexical configuration changed | lexical enabled flag and the full lexical config |
| dense enablement changed | dense enabled flag |
| embedding provider changed | provider name |
| dense configuration changed | the manifest's recorded `dense_config` (model, revision, dimension, TF-IDF parameters, vector backend; cache and batch settings excluded) |
| filter schema changed | `corpus.filterable_fields` plus system fields |

Any change to settings that affect vectors therefore requires a full build; an update always keeps
the parent's embedder (for TF-IDF: the frozen vocabulary). Manifests written before `dense_config`
was recorded skip this comparison. For `remote`, `restore_embedder` requires a
configured `dense.remote` whose model equals the manifest's model and a matching dimension; otherwise
the update fails with `IndexVersionMismatch`.

## Qdrant-backed indexes

In v0.1 incremental updates are implemented for the local NumPy backend only. If the parent's
`vector_backend` is `qdrant`, `update` raises `IndexVersionMismatch` and asks for a rebuild, because
every version owns a separate Qdrant collection. Run `searchforge index build CONFIG` instead.

## Deletes and missing documents

- Deleting missing documents is the default. A document that disappears from the corpus because a glob
  no longer matches, a file was moved or `max_documents` truncated the stream is deleted in the new
  version. A configured source path that does not exist at all aborts the update with `CorpusNotFound`.
- Use `--keep-missing` when the corpus snapshot is partial (for example when only new files are staged).
- Explicit deletes use document ids, not file paths; unknown ids are ignored.
- Deleted documents remain in the parent version, so rolling back restores them
  ([index lifecycle](index-lifecycle.md#rollback)).
