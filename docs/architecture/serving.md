# Serving

`SearchService` ([service/search_service.py](../../src/searchforge/service/search_service.py)) executes
one query end to end. The CLI (`searchforge search`) and the FastAPI app
([api/app.py](../../src/searchforge/api/app.py)) are thin wrappers around it.

## Pipeline

| Step | What happens | Errors / flags |
|---|---|---|
| 1. Deadline | `deadline = start + timeouts.overall_ms` | |
| 2. Normalize | `normalize_query_text`; `query_id` is the caller's id or `q-` + 12 hex chars of the query hash | |
| 3. Filters | explicit filters parsed against the index filter schema | `InvalidFilter` |
| 4. Analyze | features, query class, inline filters ([query-planning.md](query-planning.md)); filters merged; `SearchQuery` checks non-empty text, at most 2000 chars, `1 <= top_k <= 200` | `InvalidQuery` |
| 5. Plan | planner, `_plan` adaptation, `validate_plan` | `InvalidPlan` |
| 6. Cache lookup | version-aware key ([caching.md](caching.md)); a hit returns immediately | `cache_hit` |
| 7. Rewrite | `QueryRewriter` per profile; variants only with `multi_query` | recorded in trace |
| 8. Retrieve | every (variant, retriever) pair submitted to a thread pool of `search.max_workers` (4) | timeouts / errors degrade |
| 9. Fuse | fusion if more than one list ([retrieval.md](retrieval.md)) | |
| 10. Hydrate | chunk text and metadata from the chunk store | missing chunks degrade |
| 11. Rerank | top `reranker.top_n` in a worker with a timeout | timeout / error degrade |
| 12. Diversity | MMR, then per-document cap | |
| 13. Results | first `top_k` (or `ranking_depth`) chunks, explanations if requested | |
| 14. Finish | trace, metrics; response cached only if not degraded | |

## Timeouts

`search.timeouts` ([config/models.py](../../src/searchforge/config/models.py)):

| Setting | Default | Meaning |
|---|---|---|
| `overall_ms` | 3000 | deadline that bounds retriever waits and the rerank wait |
| `retriever_ms` | 1500 | maximum wait per retriever list |
| `reranker_ms` | 1500 | maximum wait for the reranker |
| `reranker_fallback` | true | on rerank timeout or error, keep fused order and flag degraded |
| `allow_degraded` | true | return partial retriever results instead of failing |

Semantics as implemented:

- Retriever futures run in parallel but are awaited one after another in submission order, each with
  `min(retriever_ms, time left until deadline)`.
- Python threads cannot be killed. A timed-out retrieval or rerank keeps running in its worker; its
  result is discarded. The pool is bounded by `max_workers`.
- If the deadline has already passed before reranking, rerank is skipped with reason
  `deadline_before_rerank`. Otherwise the wait is `min(reranker_ms, time left)`.
- Fusion, hydration and diversity are not bounded by the deadline.

## Degraded mode

A degraded response has `degraded: true` and explicit `degradation_reasons`:

| Reason | Cause |
|---|---|
| `<list>_timeout` | a retriever list (`lexical`, `dense`, `dense#v1`, ...) exceeded its wait |
| `<list>_error:<Exception>` | the retriever raised `RetrievalFailure` or `EmbeddingFailure` |
| `index_inconsistency:<n>_chunks_missing` | fused chunk ids absent from the chunk store |
| `reranker_timeout`, `reranker_error` | rerank fell back to fused order |
| `deadline_before_rerank` | no time left for reranking |

If every list fails the request fails: `SearchTimeout` when any failure was a timeout, else
`RetrievalFailure`. With `allow_degraded: false`, any retriever failure fails the request. With
`reranker_fallback: false`, a rerank timeout or error fails it. A reranker that changes the candidate
set raises `RerankerFailure` regardless of `reranker_fallback`. A reranker that cannot be constructed
(for example a missing cross-encoder model) degrades the response with reason `reranker_unavailable`
when `reranker_fallback` is on, and fails the request otherwise.

## Circuit breaker

[service/circuit_breaker.py](../../src/searchforge/service/circuit_breaker.py) implements
CLOSED -> OPEN -> HALF_OPEN. It opens after `failure_threshold` (5) consecutive failures; after
`reset_timeout_s` (30) one trial call is allowed; success closes it, failure re-opens it. Concurrent
calls during the trial are rejected with `CircuitOpenError`. In v0.1 it guards only
`RemoteEmbeddingProvider`, which converts `CircuitOpenError` into `EmbeddingFailure`. For a dense
retriever that becomes a `dense_error:EmbeddingFailure` degradation while lexical retrieval continues.

## HTTP API

`searchforge serve CONFIG [--host 127.0.0.1] [--port 8080] [--index name@version]` starts Uvicorn with
access logging disabled. `serve-fixture` builds or reuses the bundled fixture index and binds
`0.0.0.0` (container entrypoint). The service keeps the index handle it opened at startup; a change of
`CURRENT` takes effect only after a restart.

| Endpoint | Purpose |
|---|---|
| `GET /health` | liveness, version |
| `GET /ready` | 200 if the chunk store has chunks and at least one retriever exists, else 503; lists retrievers and profiles |
| `POST /search` | search; body `{"query", "top_k", "filters", "profile"}` |
| `POST /search/explain` | same, plus per-result `explanation` and the full `trace` |
| `GET /indexes` | versions of the served index name, which one is `CURRENT`, which one is served |
| `GET /indexes/{index_id}` | manifest view for `version` or `name@version` of the served index name; the version must match `v<timestamp>-<8 hex>` before any filesystem access, otherwise 404 |
| `GET /metrics` | Prometheus text exposition |

Request limits: `query` 1 to 2000 chars, `top_k` 1 to 100 (default 10; the domain allows up to 200),
`profile` at most 64 chars, unknown body fields rejected. Responses contain `query_id`, `results`
(with text), `latency_ms`, `profile` (requested), `planned_profile`, `index_version`, `degraded` and
`degradation_reasons`; no configuration, paths or secrets.

| Exception | HTTP status |
|---|---|
| request body validation (FastAPI) | 422 |
| `InvalidFilter`, `InvalidQuery`, `InvalidPlan` | 400 |
| `IndexNotFound` | 404 |
| `SearchTimeout` | 504 |
| `RetrievalFailure`, `RerankerFailure` | 503 |
| any other `SearchForgeError` (for example `EmbeddingFailure`) | 500 |

Error bodies are `{"detail": {"error": <class name>, "message": <first 500 chars>}}`.

## Prometheus metrics

[observability/metrics.py](../../src/searchforge/observability/metrics.py), one registry per
`SearchService`. Counters are exposed with the `_total` suffix.

| Metric | Type | Labels |
|---|---|---|
| `searchforge_search_requests_total` | counter | `profile`, `status` (`ok`, `degraded`, `error`, `cache_hit`) |
| `searchforge_search_latency_seconds` | histogram | `profile` |
| `searchforge_stage_latency_seconds` | histogram | `stage` |
| `searchforge_cache_events_total` | counter | `cache` (`query`), `event` (`hit`, `miss`) |
| `searchforge_errors_total` | counter | `stage`, `kind` |
| `searchforge_degraded_total` | counter | `reason` |
| `searchforge_result_count` | histogram | - |
| `searchforge_query_cache_entries`, `searchforge_query_cache_bytes` | gauge | - |

Label policy: values come from small closed sets. `profile` is the requested profile name, or
`invalid` if unknown; `stage` outside the known set becomes `other`; `kind` is an exception class name
truncated to 40 chars; `reason` is the degradation reason up to the first `:` with any `#vN` variant
suffix removed. Query text, document ids, chunk ids and user ids are never labels. Stage latencies
are observed for `analysis`, `lexical`, `dense`, `fusion`, `hydrate`, `rerank` and `diversity`. Latency buckets range from 1 ms to 5 s.

```
rate(searchforge_search_requests_total[1m])
histogram_quantile(0.95, sum by (le) (rate(searchforge_search_latency_seconds_bucket[5m])))
```

## OpenTelemetry spans

[observability/tracing.py](../../src/searchforge/observability/tracing.py) creates spans through the
global tracer when `opentelemetry` is importable (`[observability]` extra); otherwise `span()` is a
no-op. SearchForge does not configure an SDK or exporter; the host process does. Spans:
`searchforge.query_analysis` (attribute `query_hash`), `searchforge.lexical_retrieval` and
`searchforge.dense_retrieval` (`k`), `searchforge.fusion` (`strategy`), `searchforge.reranking`
(`reranker`, `candidates`). Only scalar attributes are set; raw query text never is. Retrieval spans
are opened in pool threads, which do not inherit the caller's span context.

## Query capture policy and traces

`search.capture_policy` ([observability/privacy.py](../../src/searchforge/observability/privacy.py))
controls the `query` field of each `SearchTrace`:

| Policy | Recorded |
|---|---|
| `none` | nothing |
| `hash_only` (default) | `sha256:` + 16 hex chars of an unsalted hash of the lowercased, whitespace-normalized query |
| `redacted` | text with emails, URLs, IPv4 addresses, UUIDs, key-like tokens, long tokens and long numbers masked |
| `full` | verbatim text (opt-in) |

The hash is unsalted, so it correlates identical queries and short queries can be guessed by
enumeration. Independent of the policy, the default `query_id` and `rewrite.original_hash` are also
derived from query hashes.

`SearchTrace` ([domain/result.py](../../src/searchforge/domain/result.py)) holds `query_id`, the captured
query, policy, requested profile, `planner_decision`, retrievers, candidate counts per list (plus
`fused` and `reranked`), fusion and reranker names, per-stage latency (`analysis`, each retriever,
`fusion`, `hydrate`, `rerank`, `diversity`, `total`), result chunk ids, degradation flags, `cache_hit`
and the rewrite record. The service does not persist traces; they are returned by
`POST /search/explain` and the CLI (`--explain`, `--json`).
