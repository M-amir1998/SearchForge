# Search profiles

A search profile names a complete retrieval pipeline: which retrievers run, how deep, how lists are
fused, whether and how results are reranked, diversified and rewritten. Built-in profiles are defined
in [profiles.py](../../src/searchforge/config/profiles.py); the schema is `SearchProfile` in
[models.py](../../src/searchforge/config/models.py). Select one with `--profile` on `search` and
`evaluate`, name two as the positional arguments of `compare`, or set `search.default_profile`
(default `hybrid`).

## Defaults

Any field a profile does not set takes the `SearchProfile` default:

| Field | Default |
|---|---|
| `retrievers` | `[lexical]` |
| `candidate_k` | 50 (1-1000; raised to `top_k` at plan time if smaller) |
| `fusion` | `strategy: rrf`, `k: 60`, no weights, `normalization: minmax` |
| `reranker` | `type: none` (`top_n` 50 when enabled; raised to `top_k` if smaller) |
| `diversity` | no per-document cap, no MMR |
| `rewrite` | disabled |
| `adaptive` | `false` |

Fusion only runs when more than one retriever list is present ([ranking-fusion.md](../methodology/ranking-fusion.md)).

## Built-in profiles

| Profile | Retrievers | Fusion | Reranker | Diversity | Notes |
|---|---|---|---|---|---|
| `lexical` | lexical | - | none | - | BM25 over SQLite FTS5 (B0) |
| `dense` | dense | - | none | - | dense vectors, exact search by default (B1) |
| `hybrid` | lexical, dense | RRF, k=60 | none | - | B2; the default profile |
| `weighted_hybrid` | lexical, dense | weighted RRF, k=60, lexical 1.0 / dense 0.5 | none | - | B3; weights are a priori, tune on train/validation only |
| `hybrid_normalized` | lexical, dense | normalized, min-max, 0.5 / 0.5 | none | - | score fusion baseline |
| `hybrid_rerank` | lexical, dense | RRF, k=60 | `lexical_features`, top_n 50 | max 3 chunks per document | B4 |
| `hybrid_cross_encoder` | lexical, dense | RRF, k=60 | `cross_encoder` (`cross-encoder/ms-marco-MiniLM-L6-v2`), top_n 50 | - | needs the `neural` extra and a locally cached model |
| `adaptive` | routed per query | of target | of target | of target | B5; rules-based query planner |

The fixture configuration adds one more profile
([fixture.yaml](../../src/searchforge/datasets/fixture/fixture.yaml)):

| Profile | Retrievers | Fusion | Rewrite |
|---|---|---|---|
| `hybrid_rewrite` | lexical, dense | RRF, k=60 | enabled, `multi_query: true`; acronyms `k8s`, `oom`, `sso`, `pii`, `tls`; synonyms for `credentials`, `outdated`, `undo` |

With `rewrite.enabled`, acronym expansions are appended to the query. Synonym variants additionally
require `multi_query`: each variant (up to `max_variants`, default 3) is retrieved separately and
fused as an extra list ([rewriting.py](../../src/searchforge/query/rewriting.py)).

## The `adaptive` profile

`adaptive: true` makes the static planner ([planning.py](../../src/searchforge/query/planning.py))
classify each query ([analysis.py](../../src/searchforge/query/analysis.py)) and execute another
profile chosen by `search.planner.rules`:

| Query class | Rule (first match wins) | Default target |
|---|---|---|
| `identifier` | contains an identifier, at most 4 tokens, not a question | `lexical` |
| `filter_heavy` | at least one filter (inline or explicit), at most 4 residual tokens | `hybrid` |
| `keyword` | at most 3 tokens and not a question, or a quoted phrase with at most 5 tokens | `hybrid` |
| `complex_semantic` | at least 9 tokens | `hybrid_rerank` |
| `natural_language` | everything else | `hybrid` |

Classes without a rule use `planner.default_profile` (`hybrid`). Rule targets must exist and must not
be adaptive (checked at startup). If the index lacks a retriever the target needs, the adaptive plan
drops it and records a note; a fixed profile instead fails with `InvalidPlan`. The executed profile
and the reason are recorded in the trace (`planner_decision`).

## Custom profiles in YAML

Define profiles under `search.profiles`. A custom profile with the same name as a built-in
**replaces** it entirely (it is not merged field by field). Unknown keys are rejected.
[configs/search/profiles.yaml](../../configs/search/profiles.yaml) shows two examples:

```yaml
search:
  profiles:
    hybrid_mmr:
      description: Hybrid RRF with MMR diversification (lambda 0.7) and max 2 chunks per document
      retrievers: [lexical, dense]
      diversity: {mmr_lambda: 0.7, max_per_document: 2}
    lexical_rerank:
      description: BM25 top-50 + lexical-feature reranker
      retrievers: [lexical]
      reranker: {type: lexical_features, top_n: 50}
```

To use them with another config, compose with `include:` (paths are relative to the including file;
`pkg:` paths resolve inside the installed package; included files are deep-merged in order, then the
including file on top):

```yaml
# configs/evaluations/my-eval.yaml
include:
  - pkg:fixture/fixture.yaml
  - ../search/profiles.yaml
search:
  profiles:
    hybrid_deep:
      retrievers: [lexical, dense]
      candidate_k: 100
      fusion: {strategy: weighted_rrf, k: 60, weights: {lexical: 1.0, dense: 0.75}}
      reranker:
        type: lexical_features
        top_n: 100
        feature_weights: {identifier: 1.5, prior: 0.8}
```

Available keys:

- `retrievers`: `lexical`, `dense` (unique; at least one unless `adaptive`).
- `candidate_k`: per-retriever and fused candidate depth.
- `fusion`: `strategy` (`rrf`, `weighted_rrf`, `normalized`), `k`, `weights` (non-negative, keyed by
  retriever), `normalization` (`minmax`, `zscore`).
- `reranker`: `type` (`none`, `lexical_features`, `cross_encoder`), `top_n`, `model`, `revision`,
  `batch_size`, `allow_download`, `feature_weights` (lexical features only;
  see [reranking.md](../methodology/reranking.md)).
- `diversity`: `max_per_document` (extra chunks are demoted to the tail, not removed), `mmr_lambda`
  (0-1; MMR over the top `3 * top_k` using dense vectors, skipped if the index has no vectors).
- `rewrite`: `enabled`, `acronyms`, `synonyms`, `multi_query`, `max_variants`.
- `adaptive`: route through the planner.
- `description`: free text; also offered to the optional LLM planner as the profile's description.

Benchmarks can also derive variants without a new profile, through `profile_overrides` on a system
(deep-merged into the base profile), as the candidate-depth systems in
[benchmark.yaml](../../src/searchforge/datasets/fixture/benchmark.yaml) do. Evaluate any new profile
with `searchforge compare` against the profile it is meant to replace
([regression-gates.md](../evaluation/regression-gates.md)).
