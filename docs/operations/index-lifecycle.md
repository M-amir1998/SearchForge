# Index lifecycle

Every build or update produces a new, immutable index version. Serving reads whatever `CURRENT` points
to at startup. Code: [indexes/lifecycle.py](../../src/searchforge/indexes/lifecycle.py),
[indexes/builder.py](../../src/searchforge/indexes/builder.py),
[domain/index.py](../../src/searchforge/domain/index.py).

## On-disk layout

```
<workspace>/indexes/<name>/
    CURRENT                     text file holding the serving version id
    <version>/                  immutable once manifest.json exists
        manifest.json
        chunks.sqlite           chunk store (text, offsets, citation metadata)
        lexical.sqlite          FTS5 index + filter attributes
        dense/vectors.npy       float32, L2-normalized (NumPy backend)
        dense/rows.jsonl
        dense/embedder/...      fitted TF-IDF state, pickle-free
        dense/qdrant.json       collection name (Qdrant backend)
    <version>.partial/          build in progress; never served
```

The workspace is `workspace:` from the config (default `.searchforge`), resolved against the current
working directory, not the config file's directory. Version ids are
`v<UTC timestamp with microseconds>-<first 8 hex chars of a fingerprint>`.

## Build, publish and immutability

1. `ensure_new_version_dir` refuses to reuse an existing final version directory
   (`IndexImmutableError`), removes a leftover `.partial` with the same id and creates a fresh one.
2. All files are written into `<version>.partial/`. Any exception removes the partial directory.
3. `publish` writes `manifest.json` last (refusing to overwrite one), renames the partial directory to
   its final name with `os.replace` (atomic on one filesystem) and marks every file read-only.
4. `CURRENT` is replaced atomically: the new id is written to `CURRENT.tmp`, then `os.replace`d.

A directory without `manifest.json` is incomplete: it is not listed, cannot be resolved and cannot be
made `CURRENT`. Search opens SQLite files with `mode=ro` and NumPy vectors as a read-only memory map.
For Qdrant-backed versions, each version writes its own collection
`<collection_prefix>_<name>_<version>`; that collection is outside the version directory and is not
removed when the directory is deleted.

## Build fingerprint and reuse

A full build first streams the corpus to compute `corpus_version` (hash of sorted document ids and
content hashes) and then the build fingerprint:

```
build_fingerprint = sha256(corpus_version, mode, chunker fingerprint,
                           lexical config, dense config without cache_enabled/batch_size,
                           filter schema)
```

If any existing version of the index has the same fingerprint, the build is skipped, that version is
made `CURRENT` (even if it is older) and the CLI prints `reused existing`. `searchforge index build
CONFIG --force` always creates a new version.

## What forces a new version

| Change | New version on `index build`? | Why |
|---|---|---|
| Document text, title, uri, attributes or timestamps | yes | content hash, hence `corpus_version` |
| Normalization or ingestion settings that change emitted documents | yes | same |
| `chunking.*` | yes | chunker fingerprint |
| `index.lexical.*` (tokenizer, weights, backend, enabled) | yes | lexical config |
| `index.dense.*` (provider, model, revision, dimension, backend, TF-IDF, remote, Qdrant settings) | yes | dense config |
| `corpus.filterable_fields` | yes | filter schema |
| `index.dense.cache_enabled`, `index.dense.batch_size` | no | excluded from the fingerprint |
| `index.data_profile`, `index.write_batch_size`, `storage.*` | no | not part of the fingerprint |
| `search.*` (profiles, fusion, reranking, timeouts, cache, planner) | no | applied at query time |
| SearchForge code version or git SHA | no | recorded in the manifest only; use `--force` after upgrading code that changes index contents |

Incremental updates always derive a new version from a parent ([index updates](index-updates.md)).

## Inspecting and validating

```
searchforge index list -c CONFIG              # all versions, * marks CURRENT
searchforge index inspect fixture -c CONFIG   # manifest of CURRENT (or name@version)
searchforge index validate fixture@<version> -c CONFIG
```

`--workspace DIR` can replace `-c CONFIG`; without either, `./.searchforge` is used. `validate`
recomputes SHA-256 checksums and reports missing files, checksum mismatches and unexpected files; if
the files are intact it compares chunk and document counts in `chunks.sqlite` with the manifest. It
exits with code 1 on any problem. It does not inspect Qdrant collections.

## Rollback

Rollback means pointing `CURRENT` at an older complete version. There is no CLI command for it; use
`IndexLayout.set_current`, which refuses versions without a manifest:

```python
from pathlib import Path
from searchforge.indexes.lifecycle import IndexLayout

IndexLayout(Path(".searchforge").resolve()).set_current("fixture", "<older-version-id>")
```

Alternatives that do not touch `CURRENT`: pin a version explicitly with
`searchforge serve CONFIG --index fixture@<version>` or `searchforge search ... --index fixture@<version>`.

Notes:

- A running server keeps the index handle it opened at startup; restart it after changing `CURRENT`.
- Old versions are the rollback targets. `searchforge storage clean --include-old-versions --yes`
  deletes every non-`CURRENT` version ([storage](storage.md)).
- Re-running `index build` with a configuration whose fingerprint matches an older version moves
  `CURRENT` back to that version.
- The query cache key includes the index id, so a rolled-back server cannot serve entries computed
  against another version ([caching](../architecture/caching.md)).
