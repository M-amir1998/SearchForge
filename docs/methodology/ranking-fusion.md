# Ranking fusion

Fusion combines the ranked candidate lists of several retrievers into one list. Implementations are
in [strategies.py](../../src/searchforge/fusion/strategies.py); configuration is the profile's
`fusion` block (`FusionConfig` in [models.py](../../src/searchforge/config/models.py)).

Common rules for all strategies:

- Items are keyed by **chunk id**; `Candidate.rank` is **1-based**.
- Within one list an item contributes at most once (first occurrence wins).
- Output order is fused score descending, then best contributing rank ascending, then chunk id,
  so fusion is deterministic. The output is truncated to the profile's `candidate_k`.
- Fusion only runs when at least two lists survive retrieval; a single list passes through.
- Weights are looked up by list name. Query-variant lists from multi-query rewriting are named
  `retriever#vN`; they use an exact-name weight if configured, otherwise the base retriever's weight.
  A list with no configured weight gets 1.0. Weights must be non-negative and are not normalized.

## Reciprocal Rank Fusion (RRF)

```
score(d) = sum over lists i containing d of  w_i / (k + rank_i(d))
```

with `rank_i(d)` 1-based, `w_i = 1.0` unless configured, and `k = 60` by default (must be >= 1). RRF
(Cormack, Clarke and Buettcher, 2009) uses ranks only, so it needs no calibration between BM25
and cosine scores. A document absent from a list gets no contribution from it.

`k` controls how steeply contributions fall with rank. The ratio between the contribution at rank 1
and rank 10 is `(k + 10) / (k + 1)`: about 1.15 for `k = 60`, 5.5 for `k = 1`. Large `k` makes RRF
behave like a vote count across lists; small `k` lets top positions dominate.

**Backend note.** Qdrant's server-side RRF uses zero-based ranks and, per its documentation, a
default `k` of 2, not 60 (see [current-search-stack.md](../references/current-search-stack.md)).
SearchForge always fuses client-side, so its RRF parameters are the same for every vector backend.
If you reproduce results with server-side fusion, set `k` explicitly and account for the rank base.

### Worked example

Lexical list: `[A, B, C, D]`; dense list: `[C, A, E]`; `k = 60`, equal weights.

| Chunk | Lexical rank | Dense rank | Score |
|---|---|---|---|
| A | 1 | 2 | 1/61 + 1/62 = 0.016393 + 0.016129 = 0.032522 |
| C | 3 | 1 | 1/63 + 1/61 = 0.015873 + 0.016393 = 0.032266 |
| B | 2 | - | 1/62 = 0.016129 |
| E | - | 3 | 1/63 = 0.015873 |
| D | 4 | - | 1/64 = 0.015625 |

Fused order: `A, C, B, E, D`. Documents found by both retrievers rise to the top; among
single-list documents, rank decides regardless of which retriever found them.

## Weighted RRF

`strategy: weighted_rrf` is RRF with explicit per-retriever weights, an explicit hypothesis such as
"lexical evidence counts double". The built-in `weighted_hybrid` profile uses
`{lexical: 1.0, dense: 0.5}` as an a priori choice. If a `weighted_rrf` profile is configured with an
empty `weights` map, `build_fusion` substitutes `{lexical: 1.0, dense: 1.0}`, which is plain RRF.

Same example with `{lexical: 1.0, dense: 0.5}`:

| Chunk | Score |
|---|---|
| A | 1/61 + 0.5/62 = 0.024458 |
| C | 1/63 + 0.5/61 = 0.024070 |
| B | 1/62 = 0.016129 |
| D | 1/64 = 0.015625 |
| E | 0.5/63 = 0.007937 |

Order: `A, C, B, D, E`. The dense-only document E now falls below the lexical-only D.

## Normalized score fusion

`strategy: normalized` normalizes each list's raw scores separately, then takes a weighted sum
(the built-in `hybrid_normalized` profile uses weights 0.5/0.5). `normalization` selects:

- **min-max** (default): `(s - min) / (max - min)` within the list. If all scores in a list are
  equal, every item gets 1.0. An item **missing** from a list gets 0.0 for that list, the same value
  as that list's worst retrieved item.
- **z-score**: `(s - mean) / std` with population standard deviation. If `std = 0`, every item gets
  0.0 and missing items get 0.0. Otherwise a missing item gets the list's **minimum z-score**, so
  absence is treated like the worst observed score rather than a neutral 0.

Normalization is per query and per list, over the `candidate_k` items actually retrieved. The values
are therefore relative ("best of this list"), not calibrated probabilities.

Example (min-max, weights 0.5/0.5). Lexical scores `A 12.0, B 9.0, C 3.0`; dense scores
`C 0.82, A 0.80, E 0.41`:

| Chunk | Lexical norm | Dense norm | Fused |
|---|---|---|---|
| A | 1.0 | 0.9512 | 0.9756 |
| C | 0.0 | 1.0 | 0.5000 |
| B | 0.6667 | missing (0.0) | 0.3333 |
| E | missing (0.0) | 0.0 | 0.0000 |

## Why raw BM25 and cosine must never be added

Adding raw scores in the example above gives `A 12.8, B 9.0, C 3.82, E 0.41`: the dense retriever's
top result C drops below B, which dense did not retrieve at all, because a cosine contributes at most
1 while BM25 contributes about 12. In general:

- BM25 is unbounded and its scale depends on the query: number of terms, their IDF, column weights
  and collection statistics. The same document scores differently for a longer query.
- Cosine similarity lies in `[-1, 1]` and its typical range depends on the embedding model.

A raw sum therefore lets whichever retriever has the larger scale on that query dominate, and the
effective weighting changes from query to query. SearchForge only combines scores after per-list
normalization, or uses ranks (RRF).

## Tuning weights without touching the test split

[tuning.py](../../src/searchforge/evaluation/tuning.py) (`searchforge tune`) selects fusion weights
with strict split discipline:

1. A grid of `lexical` x `dense` weights (defaults `{0.25, 0.5, 1.0, 1.5, 2.0}` x `{0.25, 0.5, 1.0}`)
   is evaluated on the **train** split, ranked by `ndcg@10`.
2. The best 3 settings are re-evaluated on **validation**; the winner is chosen there.
3. The winner is evaluated **once** on **test**, next to the untuned base profile.

All three splits must be non-empty. `k` is inherited from the base profile; the strategy can be
`weighted_rrf` (default) or `normalized`. Weights chosen by looking at test results are not valid
evidence; see [fair-comparisons.md](fair-comparisons.md).
