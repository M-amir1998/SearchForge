# ADR-005: Multi-stage retrieval and reranking

## Status

Accepted

## Context

High-recall first-stage retrieval (BM25, dense, hybrid) is cheap per document but coarse. More
precise scorers such as cross-encoders are too expensive to run over a whole corpus, especially on
a CPU-only machine (ADR-009). The pipeline must also stay predictable under timeouts and backend
failures, and it must be explainable stage by stage for evaluation and failure analysis.

## Decision

`SearchService` ([`service/search_service.py`](../../src/searchforge/service/search_service.py))
runs a fixed sequence of stages.

1. **Analyse and plan.** Normalise the query and parse typed filters. Classify the query. Choose a
   profile (fixed or `adaptive`) and validate the plan: retrievers known and available,
   `1 <= candidate_k <= 1000`, `candidate_k >= top_k` and `reranker.top_n >= top_k`.
2. **Retrieve.** Retrievers (and optional multi-query variants) run in parallel on a bounded
   thread pool, each with its own timeout.
3. **Fuse** (ADR-004), only when more than one list is present. The result is truncated to
   `candidate_k`.
4. **Hydrate** candidates from the chunk store, which is the source of truth for text, offsets and
   citations. Missing chunks are reported as an `index_inconsistency` degradation.
5. **Rerank** only the top `reranker.top_n` (default 50, maximum 1000). The rest keep their fused
   order. `reranking/rerankers.py` provides `none`, `lexical_features` (CPU features: term and
   title coverage, exact phrase, identifier match, proximity, first-stage rank prior) and
   `cross_encoder` (optional `neural` extra; CPU; no download unless allowed).
6. **Diversify** (`ranking/diversity.py`): a per-document cap (extra chunks are demoted, not
   dropped) and optional MMR over dense vectors.
7. Return the top `k`, with an optional per-result explanation and a trace of every stage.

Reranker contract: the output is the same candidate set as the input, re-sorted. The service
checks this and raises `RerankerFailure` if the set changes.

## Consequences

Positive:

- Expensive scoring is bounded by `top_n`, independent of corpus size, which keeps CPU latency
  predictable.
- Each stage is timed and traced: per-retriever candidate counts, fusion name, reranker name,
  degradation reasons. Evaluation can attribute a miss to a stage (`evaluation/failures.py`).
- Failures degrade visibly. A failed or timed-out retriever is dropped and the response is flagged
  `degraded` (unless `allow_degraded: false`). A reranker timeout or error falls back to the fused
  order (unless `reranker_fallback: false`). Degraded responses are not cached.

Negative:

- A relevant chunk outside the first-stage `candidate_k` can never be recovered by the reranker.
  First-stage recall caps final quality.
- Python threads cannot be killed. Timed-out retriever or reranker work keeps running in its
  worker thread and its result is discarded, so CPU is still spent under overload.
- A contract violation (the reranker changed the candidate set) raises immediately and is not
  handled by `reranker_fallback`.
- Profiles have several knobs (`candidate_k`, `top_n`, diversity), and each one has to be justified
  by measurement (`searchforge compare`, `searchforge benchmark`).

## Alternatives considered

- **Single-stage retrieval only.** This is simpler and is still available (`lexical`, `dense`,
  `hybrid` profiles), but it cannot use more precise scorers within a CPU budget.
- **Rerank the whole candidate list or the whole corpus.** Rejected: cost grows with candidate
  depth or corpus size, with no latency bound.
- **Engine-side rescoring (OpenSearch rescore, Qdrant multi-stage queries).** Rejected for the same
  portability reasons as ADR-001 and ADR-004.
- **An LLM as reranker.** Out of scope (ADR-006). The `Reranker` port allows such an adapter
  outside the core package.
