# Adaptive Query Planning — Local Demonstration

> **LOCAL DEMONSTRATION RESULTS** on the synthetic fixture collection (all splits, CPU only).
> Not a benchmark; the planner rules were written for this kind of collection.

Question: can query-aware routing keep search quality comparable to *always* running
hybrid retrieval + reranking while skipping dense retrieval / reranking where lexical search suffices?

| System | nDCG@10 | Recall@20 | MRR | p50 | p95 | QPS |
|---|---|---|---|---|---|---|
| always-hybrid-rerank | 0.882 | 0.952 | 0.902 | 5.26 ms | 7.47 ms | 153.1 |
| adaptive | 0.884 | 0.952 | 0.902 | 2.10 ms | 6.32 ms | 311.5 |
| always-lexical | 0.850 | 0.891 | 0.891 | 0.57 ms | 0.93 ms | 1365.1 |

## By query class (as classified by the planner)

| Query class | n | Routed to | nDCG@10 fixed | nDCG@10 adaptive | mean ms fixed | mean ms adaptive |
|---|---|---|---|---|---|---|
| complex_semantic | 8 | hybrid_rerank | 0.635 | 0.635 | 6.37 | 6.17 |
| filter_heavy | 10 | hybrid | 0.995 | 0.995 | 2.69 | 2.28 |
| identifier | 13 | lexical | 0.958 | 0.993 | 5.26 | 0.57 |
| keyword | 16 | hybrid | 0.931 | 0.933 | 5.54 | 2.18 |
| natural_language | 9 | hybrid | 0.777 | 0.741 | 5.97 | 2.33 |

Adaptive routing skipped the reranker for 48/56 queries and dense retrieval for 13/56 queries. Versus always-hybrid-rerank: ΔnDCG@10 = +0.003, Δp95 = -1.15 ms, 6 improved / 4 regressed / 46 unchanged queries.

CPU cost proxy: the share of queries that avoided dense retrieval and/or reranking (above). Wall-clock differences on a ~50-document collection are small in absolute terms; the relative
cost matters once a neural encoder or cross-encoder replaces the CPU baselines.
