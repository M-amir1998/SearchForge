# ADR-007: Immutable, versioned indexes with manifests

## Status

Accepted

## Context

Evaluation results are only reproducible if we know exactly which corpus, chunking, lexical
configuration and embedding model produced the index that was measured. Mutating an index in
place makes earlier results unattributable. A crash during a build can also leave a half-written
index that is then served. Query caches must not return results from an older index after a
rebuild.

## Decision

Index lifecycle is implemented in
[`indexes/lifecycle.py`](../../src/searchforge/indexes/lifecycle.py),
[`indexes/builder.py`](../../src/searchforge/indexes/builder.py) and
[`domain/index.py`](../../src/searchforge/domain/index.py).

- **Layout.** Each build writes `<workspace>/indexes/<name>/<version>.partial/`. On success the
  manifest is written last, the directory is atomically renamed to `<version>/`, its files are made
  read-only, and the `CURRENT` pointer file is atomically replaced. A `.partial` directory is never
  served. A failed build deletes it.
- **Manifest.** `IndexManifest` is a frozen Pydantic model. It records identity and lineage
  (`name@version`, `parent_version`), the corpus (`corpus_version`, a hash of document ids and
  content hashes; counts), the configuration (index types, backends, `EmbeddingSpec` from ADR-003,
  chunking, filter schema), provenance (`build_fingerprint`, git SHA, version, build time, peak
  RSS) and a SHA-256 checksum and size for every file.
- **Immutability.** A version directory with a manifest is never modified (`IndexImmutableError`).
  `searchforge index update` copies the parent into a new `.partial` version and applies only
  added, changed and deleted documents. It needs an identical chunker, lexical, embedding-provider
  and filter configuration; otherwise `IndexVersionMismatch` asks for a full rebuild.
- **Reuse.** The build fingerprint covers corpus version, chunker, lexical and dense configuration
  and filter schema. If a version with the same fingerprint exists, it is reused instead of
  rebuilt, unless `--force` is given.
- **Validation.** `searchforge index validate <name[@version]>` recomputes every checksum and
  reports missing, changed and unexpected files. It then checks document and chunk counts against
  the manifest.
- **Serving.** Search opens SQLite files with `mode=ro` and memory-maps `vectors.npy` read-only.
  The query-result cache key includes the index id, so a new `CURRENT` version cannot serve stale
  cached results.

## Consequences

Positive:

- Every metric in a run can be traced to an exact index version and its configuration.
- Rollback means re-pointing `CURRENT` to an older complete version.
- Corrupted or partially copied index directories are detected by `index validate`.

Negative:

- Every change creates a new directory, and incremental updates copy the parent first. Disk use
  grows until old versions are removed with `searchforge storage clean`. Storage budgets are
  checked before builds.
- Checksums are not signatures. The manifest is unsigned, so `index validate` detects accidental
  corruption, not tampering by someone who can also rewrite `manifest.json`.
- Validation is not run automatically when an index is opened (`indexes/handle.py::open_index`).
  It has to be invoked explicitly, as the Jenkins pipeline does.
- For Qdrant-backed indexes, the manifest covers only local files (`dense/qdrant.json` names the
  collection), not the remote collection contents. Incremental updates are not supported.
- `mode=ro` and read-only file permissions protect against accidental writes. They are not an
  access-control boundary.

## Alternatives considered

- **Mutable index with in-place upserts.** Rejected: results become unattributable, and a crash can
  leave mixed states.
- **Engine-managed aliases only (OpenSearch aliases, Qdrant collection aliases).** These are useful
  for remote deployments. They do not cover the local SQLite/NumPy indexes or record build
  provenance.
- **Signed manifests.** Deferred: it needs key management. It is worth adding before indexes are
  distributed between parties that do not trust each other.
