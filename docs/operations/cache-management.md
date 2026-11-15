# Cache management

Operational guide for the two caches described in [caching](../architecture/caching.md): the in-memory
query-result cache of each search process and the on-disk embedding cache of each workspace.

## Query-result cache

The query cache lives inside one `SearchService` instance (one `searchforge serve` process, one CLI
invocation). There is no CLI for it: it is sized by configuration, emptied by a restart and observed
through Prometheus.

```yaml
search:
  cache:
    enabled: true        # false disables lookups and stores
    max_entries: 2048    # hard cap on entries (1 to 10,000,000)
    max_bytes: 67108864  # cap on estimated bytes (64 MiB; minimum 1024)
    ttl_s: 600           # seconds an entry may be served after insertion
```

Sizing (`CacheConfig` in [config/models.py](../../src/searchforge/config/models.py), `response_size`
in [caching/query_cache.py](../../src/searchforge/caching/query_cache.py)):

- Estimated entry size is a fixed per-response overhead plus, per result, a fixed overhead, twice the
  text length and the metadata JSON length (plus explanation JSON for `explain` requests). With `top_k`
  results of about `C` characters each, an entry is roughly `top_k x 2C` bytes. Choose `max_bytes`
  from that estimate and `max_entries` from the number of distinct queries you expect to repeat within
  `ttl_s`.
- The estimate ignores the stored trace and candidate lists, so real memory use is higher than
  `max_bytes`; leave headroom when sizing the process.
- `ttl_s` bounds how long a cached answer can outlive a runtime profile replacement
  (`register_profile`). Index changes never serve stale entries because the index id is in the key, and
  a server only changes index by restarting.
- Explain and non-explain requests, and different `top_k` values, are cached separately.

Observe it with:

```
sum(rate(searchforge_cache_events_total{cache="query",event="hit"}[5m]))
  / sum(rate(searchforge_cache_events_total{cache="query"}[5m]))
searchforge_query_cache_entries
searchforge_query_cache_bytes
```

Degraded responses are never cached, so a low hit ratio during an incident is expected. Evaluation
and benchmark latency runs bypass the cache.

## Embedding cache

Location: `<workspace>/cache/embeddings/embeddings.sqlite` (default workspace `.searchforge`, resolved
against the current directory). It is written by `searchforge index build` and `index update` when
`index.dense.cache_enabled` is true (default) and holds document vectors only.

```
searchforge cache stats -c CONFIG       # entries and vector bytes
searchforge cache clear -c CONFIG       # delete all entries, then VACUUM the file
searchforge cache stats --workspace DIR
```

- `stats` reports the number of cached vectors and the sum of vector blob sizes (not the SQLite file
  size, which includes page overhead). Hit and miss counters are not persisted, so they are not shown.
- `clear` removes every row and runs `VACUUM` to return the space; it prints the number removed.
- Both print `no embedding cache` when the file does not exist.
- The cache has no size limit or TTL. `searchforge storage status` reports its size; clear it when it
  grows beyond what you want to keep.
- To stop caching, set `index.dense.cache_enabled: false`. This does not change the build fingerprint,
  so no index rebuild follows.

### When to clear it

Keys include provider, model, revision, preprocessing version, dimension and a hash of the exact text,
so most changes invalidate entries automatically. Clear the cache manually when:

| Situation | Reason |
|---|---|
| A `remote` embedding service changed its model under the same name | revision is the constant `remote-unversioned` |
| An unpinned `sentence_transformer` model was updated in the local model cache | revision is the constant `unpinned` |
| Disk pressure | entries are recomputable |

Avoid running `cache clear` or `storage clean` while a build or update is writing to the cache.

## What is safe to delete

| Artifact | Safe? | Effect of deleting |
|---|---|---|
| Query-result cache | yes | restart the process; subsequent queries recompute |
| `<workspace>/cache/embeddings/` | yes, when no build is running | the next build or update re-embeds documents; for `remote` providers this sends corpus text to the endpoint again (only if `allow_corpus_transmission` is enabled) |
| `<workspace>/cache/` (whole directory) | yes, when no build is running | same; `searchforge storage clean` removes it ([storage](storage.md)) |
| `<version>/dense/embedder/` | no | this is fitted TF-IDF state required to embed queries, not a cache; deleting it breaks the version (and `index validate` reports the missing files) |
| `<workspace>/indexes/<name>/CURRENT` | no | the index can no longer be resolved by name |
