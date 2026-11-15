# Offline evaluation

Evaluation runs a judged query set through a real `SearchService` and scores document-level
rankings. Code: [evaluation/](../../src/searchforge/evaluation/),
[benchmarking/runner.py](../../src/searchforge/benchmarking/runner.py),
[reporting/](../../src/searchforge/reporting/), [hard_negatives/](../../src/searchforge/hard_negatives/).

## Inputs

`evaluation` section of the config ([config/models.py](../../src/searchforge/config/models.py)):
`queries` and `qrels` paths, `k_values` (default 1, 3, 5, 10, 20), `profiles` (default `lexical`,
`dense`, `hybrid`), optional `splits`, `latency_repeats` (3), `relevance_threshold` (1.0) and `gates`.
Each `EvalQuery` has a `query_type`, `split` (train / validation / test), `severity` (normal /
important / critical), optional `filters` and a `provenance`. Qrels are graded per document
(0 to 3 by convention). Loaders for JSONL, TREC TSV, Parquet and BEIR live in
[evaluation/datasets.py](../../src/searchforge/evaluation/datasets.py).

## Runner rules

[evaluation/runner.py](../../src/searchforge/evaluation/runner.py) `evaluate()`:

| Rule | Implementation |
|---|---|
| Judged queries only | queries with no judgment at or above the threshold are excluded and listed; if none remain, `EvaluationFailure` |
| Cache bypass | every call uses `use_cache=False`, so latency is real work |
| Warm-up | the first (up to 5) judged queries run once, unmeasured, errors suppressed |
| Repeats | each query runs `repeats` times; latency is wall clock around `service.search`; the ranking comes from the first repeat |
| Ranking depth | `ranking_depth=100` returns up to that many chunks without changing the plan; the list is still limited by the plan's `candidate_k` (default 50) |
| Chunk to document collapse | chunk rankings are reduced to document rankings by first occurrence; a document's rank is its best chunk's rank |
| Errors | a `SearchForgeError` stops that query's repeats; if no repeat succeeded the outcome is marked as an error, excluded from aggregates and counted under `errors` |

Per-query metrics ([evaluation/metrics.py](../../src/searchforge/evaluation/metrics.py)), for every `k`
in `k_values`: `ndcg@k` (linear gain on graded judgments, IDCG from all judged grades), `recall@k`,
`precision@k`, `hit@k`, plus `mrr` and `map`. Binary metrics use `relevance_threshold`. Latency
percentiles (p50, p95, p99, mean) are linear-interpolated over all measured samples. Runs also
aggregate by `query_type`, count degraded queries and count planned profiles (useful for `adaptive`).

```
searchforge evaluate configs/evaluations/fixture.yaml -p hybrid -p hybrid_rerank --split test
```

## Regression comparison and gates

[evaluation/regression.py](../../src/searchforge/evaluation/regression.py) `compare_runs` pairs queries
scored in both runs. Each query is `improved`, `regressed` or `unchanged` by its primary-metric delta
(default `ndcg@10`) against `epsilon` (`gates.regression_epsilon`, default 1e-9). The summary contains
counts, `regression_rate`, aggregate metrics and mean deltas over the common queries, p50 / p95 latency
deltas over the common queries and the ids of regressed `critical` queries. A query that errored keeps
all-zero metrics in the comparison, so an erroring candidate counts as a regression (and trips the
critical-query gate). Only queries absent from one run's query set are listed in `missing_queries`.

| Gate (`evaluation.gates`) | Default | Passes when |
|---|---|---|
| `ndcg_at_10_min_delta` | -0.005 | mean nDCG@10 delta (candidate minus baseline) is at least the threshold |
| `recall_at_20_min` | off | candidate Recall@20 is at least the threshold |
| `p95_ms_max` | off | candidate p95 latency is at most the threshold |
| `max_query_regression_rate` | 0.03 | share of regressed queries (primary metric) is at most the threshold |
| `block_on_critical_regression` | true | no `critical` query regressed |

A gate whose metric was not computed (for example `recall@20` without 20 in `k_values`) fails.

```
searchforge compare hybrid hybrid_rerank -c configs/evaluations/fixture.yaml --out comparison.json
searchforge compare hybrid hybrid --candidate-config candidate.yaml   # same profile, other index
```

`compare` exits with code 1 when a gate fails (disable with `--no-gates`) and 2 on other errors.

## Failure analysis

[evaluation/failures.py](../../src/searchforge/evaluation/failures.py) inspects every relevant document
missing from the final top-k (k = 10) of each scored query. **Evidence** (observed facts) is kept
separate from a rule-based **hypothesis**:

- evidence: whether the document is indexed; its rank in the final, fused, lexical and dense document
  lists; planned profile; whether reranking ran; duplicate chunks in the top-k; rewrite strategies;
- hypothesis (first matching rule):

| Order | Category | Rule |
|---|---|---|
| 1 | `index_staleness` | document not in the served index version |
| 2 | `filter_error` | the query filters exclude the document's first chunk |
| 3 | `bad_rerank` | fused rank within k, but reranking ran and it is not in the final top-k |
| 4 | `bad_fusion` | a retriever ranked it within k, fusion did not |
| 5 | `duplicate_results` | at least `max(2, k // 3)` duplicate-document chunks in the top-k |
| 6 | `chunking_failure` | absent from lexical; the document has several chunks that together cover at least 75% of query terms, but no single chunk covers half |
| 7 | `query_rewrite_failure` | acronym or synonym rewriting ran and neither retriever found it |
| 8 | `semantic_miss` / `missed_lexical_match` | neither retriever found it (semantic if dense ran) |
| 9 | `missed_lexical_match` / `semantic_miss` | found but ranked weakly; lexical if its lexical rank is the better one |

Records carry `basis: rule-based hypothesis` and are a triage aid, not a root-cause proof.

## Hard negatives

[hard_negatives/mining.py](../../src/searchforge/hard_negatives/mining.py): per query, documents within
the first `max_rank` (10) **document** ranks (chunks collapsed, as in evaluation) that are not judged
relevant, up to `per_query` (5). `judged` separates documents judged below the threshold from unjudged
(probable) negatives; `rank` is the document rank. Output is JSONL, or Parquet for a `.parquet` path:
`searchforge hard-negatives build CONFIG --profile hybrid --out results/hard_negatives.jsonl`.

## Incident to eval

[evaluation/incidents.py](../../src/searchforge/evaluation/incidents.py) turns an observed failure into
a permanent regression case: the query is whitespace-normalized and redacted, the id is
`q-inc-<hash>`, `query_type` is `incident`, split `test`, provenance `failure_derived`, severity
`important` by default, and every expected document gets relevance 3. Expected ids must exist in the
index; a duplicate query id is rejected.

```
searchforge incident add "token refresh fails with ERR-4012" -e KB-0042 -c CONFIG --ref INC-17
```

## Benchmark runner and reports

`searchforge benchmark SPEC [--out results/benchmark] [--workspace DIR]` runs an experiment matrix. A
spec names a `base_config`, `systems` (profile, optional `profile_overrides`, optional
`config_overrides`, `group`), an optional `baseline` (default: first system), `splits` (default `test`),
`repeats` (3), `k_values`, `primary_metric` and `throughput` (`threads` 4, `queries` 200).

- One index is built and opened per distinct `index.name` (`IndexBuilder.build()`, which also moves
  `CURRENT`). Systems whose overrides keep the same `index.name` reuse the first index built under that
  name, so a `config_overrides` that changes indexing must also rename the index.
- Each system's effective profile is registered on that service under the system name, evaluated
  (sampled peak RSS), measured for throughput (cache bypassed, concurrent client threads) and analysed
  for failures. Every other system is compared with the baseline.
- With `measure_cache` (default), cold and warm passes of the baseline profile measure the query cache.

[reporting/report.py](../../src/searchforge/reporting/report.py) writes only measured values. Report
sections include setup and environment, main results, latency / throughput / memory, quality by query
type, adaptive routing, index size by component, query cache, regressions vs baseline, failure analysis
and limitations. The
header text describes the bundled fixture collection regardless of the corpus used.

## Results directory layout

[evaluation/store.py](../../src/searchforge/evaluation/store.py) writes one directory per run:

```
results/<run-id>/              run-id = <UTC timestamp>-<name>-<6 hex>
    config.yaml  manifest.json  metrics.json  performance.json
    per_query.parquet  regressions.parquet  failures.parquet  plots/
results/benchmark/             searchforge benchmark
    report.md  metrics.json  plots/quality_latency.svg  hard_negatives.jsonl  runs/<run-id>/...
```

Query text in `per_query.parquet` is verbatim for `manual`, `synthetic` and `public_benchmark`
provenance and redacted for `sanitized_production` and `failure_derived`. `searchforge report DIR` prints
`report.md`, or renders a minimal one from `metrics.json`.
