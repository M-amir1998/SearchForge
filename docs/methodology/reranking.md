# Reranking

Rerankers are second-stage scorers: more expensive per item than a first-stage retriever, so they
only see a short candidate list. Implementations are in
[rerankers.py](../../src/searchforge/reranking/rerankers.py); orchestration (depth, timeout,
fallback) is in `SearchService._rerank` in
[search_service.py](../../src/searchforge/service/search_service.py).

## Two-stage retrieval

```
retrievers (candidate_k each) -> fusion (candidate_k) -> hydrate
  -> rerank head top_n -> append unchanged tail -> diversity -> top_k
```

- `candidate_k` (profile, default 50) bounds what any later stage can see.
- `reranker.top_n` (default 50) is how many fused candidates are rescored; it is raised to `top_k` if
  smaller. Candidates beyond `top_n` keep their fused order and are appended after the reranked head.
- Diversity (`max_per_document`, MMR) runs **after** reranking.

The reranker can only reorder; recall at depth `top_n` is fixed by the first stage. A relevant
document outside the fused top `top_n` stays where it is.

## Contract

Input: the top-N hydrated candidates in first-stage order. Output: the **same set** of candidates,
sorted by the new score descending, with `signals.reranker_score` and `signals.reranker_rank` set.
Rerankers never add or drop candidates. The service checks set equality after every rerank and
raises `RerankerFailure` on a violation; this check is outside the timeout/error fallback path, so a
contract violation fails the request instead of being served degraded.

`NoOpReranker` returns its input unchanged and is the baseline for measuring a reranker's marginal
value.

## LexicalFeatureReranker (`lexical_features`)

A CPU-cheap, interpretable reranker. Query terms are lower-cased `\w+` tokens with stopwords removed
(the list in [analysis.py](../../src/searchforge/query/analysis.py)) and a light suffix stripper
applied, de-duplicated. For candidate at 0-based first-stage position `p`:

| Feature | Definition | Default weight |
|---|---|---|
| `prior` | `1 / (1 + p)`: 1.0, 0.5, 0.333, ... (the first-stage *score* is not used) | 1.0 |
| `coverage` | fraction of query terms present in the chunk body | 1.0 |
| `title_coverage` | fraction of query terms present in title + section | 0.4 |
| `phrase` | 1 if the query has >= 2 terms and its full normalized token sequence (stopwords included) occurs contiguously in the body, else 0 | 0.5 |
| `identifier` | fraction of query identifiers (error codes, `snake_case`, dotted paths, versions, ...) found case-insensitively in the body text | 1.0 |
| `proximity` | for >= 2 matched terms: `m / w`, where `m` is the number of matched terms and `w` the width in tokens of the smallest body window containing all of them; else 0 | 0.4 |

Score = sum of `weight * feature`. Ties keep first-stage order. Weights can be overridden per profile
with `reranker.feature_weights` (keys must be the feature names above; unknown keys are rejected).

The `prior` term keeps the reranker anchored to the first stage: a candidate needs several feature
wins to overtake one ranked far above it. The features were designed alongside the fixture
collection, so validate them on independent data before relying on them.

## CrossEncoderReranker (`cross_encoder`)

Optional (`neural` extra). Loads a Sentence Transformers `CrossEncoder`
(default `cross-encoder/ms-marco-MiniLM-L6-v2`) on CPU. As with embeddings, it loads only locally
cached models unless `allow_download: true`, and `revision` should be pinned. Each (query, chunk)
pair is scored jointly, with chunk text truncated to 2,000 characters, in batches of
`batch_size` (default 32). It runs only over the top `top_n` candidates.

## Cost

- Reranking cost grows linearly with `top_n`. For the lexical reranker each candidate is tokenized
  and scanned (cost proportional to chunk length); for a cross-encoder each candidate is a model
  forward pass and usually dominates request latency on CPU.
- Deeper candidate pools give the reranker more chances to promote a relevant document, but also
  more distractors, and the `prior` contribution of deep positions is small.
- The reranking time is recorded as the `rerank` stage timing in the trace, together with the number
  of reranked candidates.

### Candidate-depth experiment

The fixture benchmark ([benchmark.yaml](../../src/searchforge/datasets/fixture/benchmark.yaml))
runs `hybrid_rerank` with `candidate_k = top_n` set to 20, 50 and 100 (systems D20, D50, D100) and
reports quality and latency for each. Caveats when reading it:

- Evaluated rankings are at most `candidate_k` chunks long, so at depth 20 the document ranking has
  at most 20 entries (fewer after collapsing chunks to documents), which limits Recall@20 and MAP.
- `hybrid_rerank` also caps chunks per document (`max_per_document: 3`), so comparing `hybrid`
  against `hybrid_rerank` changes two things at once. To isolate the reranker, compare against a
  custom profile that differs only in `reranker`.

## Timeouts and fallback

The rerank call runs in the service's thread pool with timeout
`min(search.timeouts.reranker_ms, time left before overall_ms)` (default 1500 ms).

- If the overall deadline has already passed, reranking is skipped (`deadline_before_rerank`).
- On timeout or `RerankerFailure`, with `reranker_fallback: true` (default), the fused order is
  returned and the response is flagged degraded (`reranker_timeout` / `reranker_error`). With
  `reranker_fallback: false` the request fails.
- Python threads cannot be killed: timed-out work keeps running in its worker and the result is
  discarded. Degraded responses are not cached.
