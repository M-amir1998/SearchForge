# ADR-004: Reciprocal Rank Fusion as the default hybrid fusion

## Status

Accepted

## Context

Hybrid search combines a lexical ranked list (BM25: unbounded, query-dependent scores) with a
dense ranked list (cosine similarity in [-1, 1]). Adding raw scores is meaningless because the
scales are not comparable. Score normalisation requires choices (min-max or z-score, how to treat
missing items) that interact with the corpus and the query. SearchForge also runs the same
profiles on different engines and needs fused rankings that do not change when the engine does.

## Decision

- The default fusion for hybrid profiles is Reciprocal Rank Fusion, implemented client-side in
  [`fusion/strategies.py`](../../src/searchforge/fusion/strategies.py):
  `score(d) = sum_i w_i / (k + rank_i(d))`, where `rank` is 1-based and `w_i` defaults to 1.0.
- `k` defaults to 60 (`FusionConfig.k`, minimum 1). The built-in `hybrid`, `hybrid_rerank` and
  `adaptive` profiles use it.
- Each list contributes at most once per chunk. Ties are broken by best contributing rank and then
  by chunk id, so fusion is deterministic.
- Two alternatives are available per profile:
  - `weighted_rrf` - explicit per-retriever weights; tune them on train/validation splits only
    (`evaluation/tuning.py`).
  - `normalized` - per-list min-max or z-score normalisation, then a weighted sum. Raw scores are
    never summed without normalisation.
- Fusion always runs in SearchForge, after each retriever returns its list. The Qdrant and
  OpenSearch adapters return plain ranked candidates and do not use server-side fusion. Multi-query
  rewrite variants are fused the same way, as lists named `retriever#vN` that inherit the weight of
  their base retriever.

## Consequences

Positive:

- RRF uses ranks only, so it needs no score calibration between BM25 and cosine, and adding a
  retriever needs no normalisation design.
- `k`, weights and tie-breaking are identical across backends. Engine defaults differ: Qdrant's
  documentation states that its server-side RRF uses `k = 2` when `k` is not set, with zero-based
  ranks, and OpenSearch exposes the constant as `rank_constant` in a search pipeline (see
  [docs/references/current-search-stack.md](../references/current-search-stack.md)). With
  client-side fusion, a profile means the same thing everywhere.
- It avoids depending on Elasticsearch's RRF retriever. The reference notes record an Elastic
  staff forum statement that the RRF and linear retrievers require an Enterprise license. The
  official subscription matrix could not be read in a tier-resolved form, so this is reported
  there as forum-sourced.

Negative:

- RRF discards score magnitude. A very confident match in one list counts the same as a marginal
  one at the same rank.
- Client-side fusion needs each retriever to return `candidate_k` results. That means more data
  transfer than engine-side fusion and two round-trips for remote backends.
- `k = 60` is a literature default, not a value tuned for any SearchForge corpus. Whether another
  `k` or a weighted variant is better is an empirical question for `searchforge compare` and
  `searchforge tune`.

## Alternatives considered

- **Normalised score fusion as the default.** It is available as `normalized`, but not the default:
  it is sensitive to outliers and to the normalisation method, and missing items need an arbitrary
  fill value.
- **Engine-native hybrid (Qdrant prefetch with RRF/DBSF, OpenSearch hybrid query with a
  normalisation pipeline, Elasticsearch RRF retriever).** Rejected as the default because
  parameters and defaults differ per engine, and because of the Elasticsearch licensing caveat
  above.
- **Learned fusion or learning-to-rank.** Deferred: it needs training labels at a scale the
  fixture and public subsets do not provide, and it risks overfitting the evaluation set.
