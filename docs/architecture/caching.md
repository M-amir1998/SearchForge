# Caching

SearchForge has two independent caches:

| Cache | Scope | Stores | Code |
|---|---|---|---|
| Query-result cache | in memory, one per `SearchService` process | complete `SearchResponse` objects | [caching/query_cache.py](../../src/searchforge/caching/query_cache.py), [caching/lru.py](../../src/searchforge/caching/lru.py) |
| Embedding cache | on disk, shared per workspace | document embedding vectors | [embeddings/cache.py](../../src/searchforge/embeddings/cache.py) |

Operational commands are in [cache management](../operations/cache-management.md).

## Query-result cache

### Key composition

`query_cache_key` hashes (SHA-256 over canonical JSON) everything that is meant to change the answer:

| Key part | Value |
|---|---|
| `v` | key schema version `qc-v1` |
| `q` | search text after inline-filter extraction, whitespace-collapsed (case preserved) |
| `f` | canonical filters: sorted `(field, op, value)` triples, so filter order does not matter |
| `i` | index id `name@version` |
| `p` | planner version (`static-v1` by default) |
| `pr` | requested profile name |
| `k` | `top_k` |
| `e` | explain flag |

**Version-aware invalidation.** A new index version has a new index id, so entries computed against
the previous version can never be addressed again; they age out through LRU and TTL. Changing the
planner version has the same effect for planner changes.

What the key does not capture:

- **Profile contents.** Only the profile name is in the key. Profiles come from configuration loaded at
  startup; `SearchService.register_profile` can replace a profile at runtime under the same name, and
  doing so clears the query cache so results computed with the old definition are never served.
- **Letter case.** The text is whitespace-normalized but **not** lowercased (key version `qc-v2`),
  because identifier detection (`camelCase`, `PascalCase`) and cased embedding models are
  case-sensitive and may plan or rank differently.

### When the cache is bypassed

- `search.cache.enabled: false`;
- `use_cache=False` (the evaluation runner always passes it);
- `ranking_depth` is set (evaluation rankings are never cached);
- the response is degraded (it is computed but not stored).

**Why degraded responses are not cached.** A degraded response reflects a transient condition: a
retriever timeout, a remote embedding outage behind an open circuit breaker, a reranker fallback or an
index inconsistency. Caching it would serve the partial ranking for up to `ttl_s` after the backend
recovered, and those requests would be counted as `cache_hit` instead of `degraded`.

On a hit the stored response is returned with the new `query_id`, `trace.cache_hit = true`, a single
`cache_lookup` timing and the measured lookup latency. Planner decision and candidate lists are those
of the original computation.

### Bounded LRU with TTL and bytes

`BoundedLRUCache` is thread-safe (one lock) and never holds more than `max_entries` entries or more
than `max_bytes` estimated bytes. `CacheConfig` defaults
([config/models.py](../../src/searchforge/config/models.py)):

| Setting | Default | Notes |
|---|---|---|
| `enabled` | true | |
| `max_entries` | 2048 | range 1 to 10,000,000 |
| `max_bytes` | 64 MiB | minimum 1024 |
| `ttl_s` | 600 | per entry, from insertion |

Behaviour:

- `put` evicts least-recently-used entries until both limits hold; a single response larger than
  `max_bytes` is not cached.
- `get` refreshes recency. TTL is checked lazily on `get`: an expired entry is removed and counted as an
  expiration and a miss. Expired entries that are never read keep occupying space until evicted.
- Size is estimated by `response_size`: a fixed overhead per response and per result, twice the result
  text length, the metadata JSON and the explanation JSON. The trace and the candidate lists kept on the
  response are not counted, so real memory use is higher than `max_bytes`.

Statistics (`CacheStats`): hits, misses, evictions, expirations, entries, bytes, hit and miss rates, and
`latency_saved_ms` (sum of the original latency of every served hit).

### Metrics

| Metric | Updated |
|---|---|
| `searchforge_cache_events_total{cache="query", event="hit"\|"miss"}` | on every lookup |
| `searchforge_query_cache_entries` | after every store |
| `searchforge_query_cache_bytes` | after every store |

The benchmark runner reports cold versus warm latency and hit counts for the baseline profile when
`measure_cache` is enabled ([evaluation.md](evaluation.md)).

## Embedding cache

`EmbeddingCache` is a SQLite table `emb(key, dim, vec, created)` in
`<workspace>/cache/embeddings/embeddings.sqlite`, holding float32 vectors as blobs.
`CachedEmbeddingProvider` wraps the provider during full builds and incremental updates when
`index.dense.cache_enabled` is true (default). Only document embeddings are cached; query embeddings are
computed per request.

### Key composition

```
key = sha256( sha256(text) | provider | model | revision | preprocessing_version | dimension )
```

`text` is the exact string sent to the model (title and section, then the chunk body), so any change to
normalization or chunking that changes that string produces a new key. `preprocessing_version` is
`EMBED_PREPROCESSING_VERSION` (`embed-pre-v1`); `NORMALIZATION_VERSION` is not part of the key.

| Provider | `revision` in the key | Staleness risk |
|---|---|---|
| `tfidf` | `fit-<hash of vocabulary, idf and SVD state>` | none: a refit with different state yields new keys |
| `hash` | `v1-s<seed>-b<bigrams>` | none |
| `sentence_transformer` | configured `revision`, or `unpinned` | an unpinned model whose local weights change keeps hitting old vectors; pin `revision` |
| `remote` | `remote-unversioned` | a server-side model change under the same name keeps hitting old vectors; clear the cache |

The cache has no size limit, TTL or eviction; it grows until cleared. Hit and miss counters are kept in
memory by the process that uses the cache; the persistent file only knows entries and bytes.
`cache_enabled` is excluded from the build fingerprint, so toggling it never forces a new index version.
