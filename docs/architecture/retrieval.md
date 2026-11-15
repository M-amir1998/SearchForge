# Retrieval, fusion and ranking

How a validated plan turns into a ranked list: retrievers, backend filtering, fusion, hydration,
reranking, diversity and the signals recorded for explanations. The orchestration (parallelism,
timeouts, degraded mode) is in [serving.md](serving.md).

## Retrieval request and retrievers

`RetrievalRequest` ([ports.py](../../src/searchforge/ports.py)) carries what a retriever needs, already
rewritten, validated and filter-parsed: `text`, `k`, `filters` (a canonical `FilterExpression`) and
`phrases` (quoted phrases from query analysis).

| Retriever | File | Behaviour |
|---|---|---|
| `LexicalRetriever` | [retrieval/retrievers.py](../../src/searchforge/retrieval/retrievers.py) | calls `LexicalIndex.search(text, filters, k, phrases)`; unexpected exceptions become `RetrievalFailure` |
| `DenseRetriever` | same | embeds the query (`embed_query`, not cached) and calls `VectorIndex.search(vector, filters, k)`; `EmbeddingFailure` passes through |
| `HybridRetriever` | same | synchronous composition for library use: every retriever at `candidate_k`, then fusion to `k`; no timeouts or degraded mode |

`SearchService` builds one `lexical` and one `dense` retriever from the opened index (or uses
retrievers injected through its constructor). Each plan retriever is called with
`k = plan.candidate_k` (profile default 50). With multi-query rewriting, every variant is retrieved
separately and its list is named `<retriever>#v<n>`.

## Filters are applied inside the backends

Filters are parsed from untrusted input into a typed `FilterExpression`
([domain/filters.py](../../src/searchforge/domain/filters.py)) and applied as pre-filters inside each
backend, so `k` counts only matching chunks.

```json
{"category": "security", "team": ["core", "infra"], "year": {"gte": 2020, "lt": 2024},
 "language": {"nin": ["de"]}, "published": {"gte": "2024-01-01"}}
```

- Operators: `eq`, `ne`, `in`, `nin`, `gt`, `gte`, `lt`, `lte`. A scalar means `eq`, a list means `in`.
- Field names must match `^[a-z][a-z0-9_]{0,63}$` and be declared in `corpus.filterable_fields`;
  `document_id` and `source` are always filterable strings. Unknown fields raise `InvalidFilter` with
  the list of allowed fields.
- Range operators require an ordered type (integer, float, date, string); set operators accept at most
  `MAX_SET_SIZE` (256) values. Values are coerced to the field type; dates compare as ISO strings.
- All conditions are AND-ed. A chunk without the field fails every condition except `ne` and `nin`.

| Backend | Translation ([filtering/translate.py](../../src/searchforge/filtering/translate.py)) |
|---|---|
| SQLite FTS5 | `json_extract(attributes, ?)` clauses; JSON path and values are bound parameters |
| NumPy | boolean mask from `FilterExpression.matches` over row attributes (memoized) |
| Qdrant | `must` / `must_not` with `MatchValue`, `MatchAny`, `Range`, `DatetimeRange` on `attributes.<field>` |
| OpenSearch | `bool.filter` clauses (`term`, `terms`, `range`, `must_not`); k-NN uses the efficient `filter` option |

## Fusion

[fusion/strategies.py](../../src/searchforge/fusion/strategies.py). Fusion runs only when there is more
than one list; a single list is truncated to `candidate_k` and re-ranked from 1.

| Strategy | Score | Configuration |
|---|---|---|
| `rrf` (default) | sum over lists of `w / (k + rank)` | `fusion.k` (60), optional `weights` (default 1.0) |
| `weighted_rrf` | same formula, weights mandatory | config validation rejects `weighted_rrf` without `fusion.weights` |
| `normalized` | weighted sum of per-list normalized scores | `normalization: minmax` (default) or `zscore` |

Details that affect results:

- Ranks are 1-based; a list contributes at most once per chunk.
- RRF uses ranks only, so BM25 and cosine scores never need calibration. Normalized fusion never sums
  raw scores: min-max maps each list to [0, 1] (absent items get 0); z-score assigns absent items the
  list's minimum z.
- Weights are looked up by full list name, then by the base retriever name, so `dense#v1` inherits the
  `dense` weight.
- Ties are broken by best contributing rank, then chunk id, so fusion is deterministic.

## Hydration

Fused candidates are hydrated from the chunk store (text, title, section, uri, offsets, attributes).
Candidates whose chunk is missing are dropped and the response is flagged
`index_inconsistency:<n>_chunks_missing`.

## Reranking

[reranking/rerankers.py](../../src/searchforge/reranking/rerankers.py). Contract: the reranker receives
the top `reranker.top_n` (default 50) hydrated candidates in first-stage order and returns the same set,
sorted by its new score, with `reranker_score` and `reranker_rank` set. `SearchService` checks the set
and raises `RerankerFailure` if it changed.

| Reranker | Behaviour |
|---|---|
| `none` | identity; no rerank stage is run |
| `lexical_features` | weighted sum of CPU features (defaults): `prior` 1.0, `coverage` 1.0, `title_coverage` 0.4, `phrase` 0.5, `identifier` 1.0, `proximity` 0.4 |
| `cross_encoder` | Sentence Transformers `CrossEncoder` on CPU (`[neural]` extra), candidate text truncated to 2000 chars, no download unless allowed |

Lexical features: `prior` is `1 / (1 + position)` in first-stage order; `coverage` and `title_coverage`
are the share of (lightly stemmed, non-stopword) query terms found in the body or in title + section;
`phrase` is 1 when the query has at least two content terms and its normalized word sequence occurs
in the text; `identifier` is the share of query
identifiers found verbatim (case-insensitive); `proximity` is the number of matched terms divided by the
smallest window containing all of them. Override weights with `reranker.feature_weights`; unknown
feature names are rejected.

The reranker receives the search text after inline-filter extraction and before rewriting. After
reranking, the head carries reranker scores while the tail keeps fusion scores, so result scores are not
on a single scale across that boundary.

## Diversity

[ranking/diversity.py](../../src/searchforge/ranking/diversity.py), applied after reranking:

- **MMR** (`diversity.mmr_lambda`, only when the index has vectors): over the first `3 x top_k`
  candidates, greedily select `top_k` items maximizing
  `lambda * rel - (1 - lambda) * max_sim_to_selected`, where `rel` is the min-max normalized score
  within the pool and similarity is the dot product of stored (normalized) vectors. The rest keep
  their order.
- **Per-document cap** (`diversity.max_per_document`): chunks beyond the cap are demoted to the tail
  (not removed) with a `diversity_note`.

`group_results` groups final results by `document_id` or source for display.

## Explainability signals

With `explain` set (CLI `--explain`, API `POST /search/explain`), each result carries
`{"signals": ..., "plan_profile": ..., "query_class": ...}`. `RankingSignals`
([domain/result.py](../../src/searchforge/domain/result.py)) records actual ranking inputs, not model
interpretability:

| Field | Meaning |
|---|---|
| `lexical_rank`, `lexical_score` | position and negated BM25 score in the lexical list (first query variant) |
| `dense_rank`, `dense_score` | position and cosine score in the dense list |
| `per_retriever` | rank and score in every list, including `#vN` variant lists |
| `fusion_score`, `fusion_rank` | after fusion (or the single list) |
| `reranker_score`, `reranker_rank` | after reranking, when it ran |
| `filter_reasons` | `passed: <field> <op> <value>` for every active filter condition |
| `diversity_note` | why a chunk was demoted by the per-document cap |

`filter_reasons` are derived from the active filter, since backends pre-filter; they are not a
re-evaluation per chunk. `FilterExpression.failing` provides per-record reasons when needed.

## Built-in profiles

[config/profiles.py](../../src/searchforge/config/profiles.py); user profiles with the same name override
them.

| Profile | Retrievers | Fusion | Reranker / diversity |
|---|---|---|---|
| `lexical` | lexical | - | - |
| `dense` | dense | - | - |
| `hybrid` | lexical, dense | RRF k=60 | - |
| `weighted_hybrid` | lexical, dense | weighted RRF, lexical 1.0, dense 0.5 | - |
| `hybrid_normalized` | lexical, dense | min-max, 0.5 / 0.5 | - |
| `hybrid_rerank` | lexical, dense | RRF | `lexical_features` top 50, max 3 per document |
| `hybrid_cross_encoder` | lexical, dense | RRF | `cross_encoder` top 50 |
| `adaptive` | planned per query | planned | planned (see [query-planning.md](query-planning.md)) |
