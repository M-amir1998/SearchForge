# Hybrid search

Hybrid search runs a lexical retriever ([bm25.md](bm25.md)) and a dense retriever
([dense-retrieval.md](dense-retrieval.md)) for the same query and fuses their candidate lists
([ranking-fusion.md](ranking-fusion.md)). This page explains why that can help, when it does not,
and how SearchForge decides whether the extra cost is justified.

## Lexical and dense retrieval fail differently

| Situation | Lexical (BM25) | Dense |
|---|---|---|
| Exact identifiers (`ERR-4012`, `max_attempts`, version strings) | Strong: rare tokens get high IDF | Weak: identifiers carry little meaning for a model; TF-IDF/LSA and neural models blur them |
| Vocabulary mismatch (`undo a deploy` vs `rollback`) | Fails: no shared terms, no match | Can succeed if the model relates the terms |
| Long natural-language questions | Noisy: many weakly informative terms | Usually better at matching intent |
| Negation, exact phrasing | Quoted phrases add score for the exact wording (they are OR-ed, not required) | Embeddings largely ignore it |
| Out-of-vocabulary terms after incremental updates | Indexed immediately | TF-IDF vocabulary is frozen until rebuild |

Because the error patterns are partly independent, a document missed by one retriever is often
found by the other. Fusion exploits that complementarity; it cannot fix cases both retrievers miss.

## Candidate sets

In [search_service.py](../../src/searchforge/service/search_service.py):

1. Each retriever in the plan is submitted to a bounded thread pool with the same request: query
   text, filters, quoted phrases and `k = candidate_k` (profile default 50, raised to `top_k` if
   smaller).
2. Each retriever has its own timeout (`search.timeouts.retriever_ms`, default 1500 ms) inside the
   overall deadline (`overall_ms`, default 3000 ms). A retriever that times out or fails is
   dropped and the response is flagged `degraded` with the reason; if all fail, the search fails.
   With `allow_degraded: false`, any retriever failure fails the search.
3. If more than one list survives, the lists are fused and truncated to `candidate_k`. If only one
   list survives, it is used directly (ranks renumbered) and no fusion runs.
4. Fused chunk ids are hydrated from the chunk store, then optionally reranked, diversified and
   cut to `top_k`.

With multi-query rewriting enabled (profile `rewrite.multi_query`), each query variant is retrieved
separately and enters fusion as an extra list named `lexical#v1`, `dense#v1`, and so on.

The candidate set bounds everything downstream: a document absent from every retriever's top
`candidate_k` cannot be recovered by fusion or reranking.

## When hybrid helps and when it hurts

Helps:

- Mixed query traffic where some queries need exact matching and others need vocabulary tolerance.
- Recall-oriented metrics (Recall@20) when the union of candidate sets covers more relevant
  documents than either set alone.

Hurts or does not pay off:

- When one retriever is much weaker on a query type. Plain RRF gives both lists an equal vote, so
  noise from the weak list can displace a correct result ranked first by the strong one. Identifier
  lookups are the typical case; the `adaptive` profile routes them to lexical-only for this reason
  ([profiles.md](../retrieval/profiles.md)).
- When both retrievers return mostly the same documents: latency increases with little new recall.
- When the dense model is a corpus-fitted baseline on a very small corpus: its signal may be too
  weak to add value.

These are hypotheses to measure, not rules.

## Is the extra latency worth the relevance gain?

Hybrid search costs at least the slower retriever (for dense, including query embedding) plus
fusion and hydration of a larger candidate set. Retrievers run
concurrently; how much they overlap in practice depends on how much of each call releases the
Python GIL. SearchForge treats "is it worth it" as an explicit measurement:

- **Per-stage timings.** Every response carries a trace with timings for analysis, each retriever,
  fusion, hydration, reranking and diversity, plus candidate counts per list.
- **Paired evaluation.** `searchforge compare lexical hybrid` evaluates both profiles on the same
  index, queries and split, then reports per-query improved/regressed/unchanged counts, mean metric
  deltas and p50/p95 latency deltas ([regression-gates.md](../evaluation/regression-gates.md)).
- **Per-query-type breakdown.** Run summaries aggregate metrics by `query_type`, which shows where
  hybrid helps (for example semantic queries) and where it costs (for example identifiers).
- **Experiment matrix.** `searchforge benchmark` evaluates B0 (lexical), B1 (dense), B2 (hybrid RRF)
  and further systems in one run and plots quality against latency
  ([runner.py](../../src/searchforge/benchmarking/runner.py)).
- **Stage attribution.** The evaluation runner records document rankings per retriever and after
  fusion, so failure analysis can tell "no retriever found it" from "a retriever found it but fusion
  demoted it" ([failure-taxonomy.md](../evaluation/failure-taxonomy.md)).
- **Gates.** A latency budget can be enforced with `p95_ms_max` next to the quality gates.

The decision is made from the measured trade-off for the target query mix. Measurement rules are in
[fair-comparisons.md](fair-comparisons.md).
