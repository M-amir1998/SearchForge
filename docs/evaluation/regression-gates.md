# Regression analysis and quality gates

Per-query comparison of a baseline and a candidate, plus release gates, implemented in
[regression.py](../../src/searchforge/evaluation/regression.py) and exposed as
`searchforge compare` ([main.py](../../src/searchforge/cli/main.py)).

## `compare_runs` semantics

Input: two evaluation runs on the same query selection.

1. Queries present in **both runs' query sets** are compared. A query that raised an error keeps
   all-zero metrics, so an erroring candidate query counts as a regression. Queries absent from one
   run are listed in `missing_queries`.
2. For each common query, `delta = candidate - baseline` on the primary metric (`ndcg@10` by
   default).
3. Status, with `epsilon = evaluation.gates.regression_epsilon` (default `1e-9`):
   - `improved` if `delta > epsilon`
   - `regressed` if `delta < -epsilon`
   - `unchanged` otherwise
4. Each `QueryDelta` also carries the deltas of every other metric, the baseline and candidate top-10
   document ids, the relevant document ids, the query type and the severity (taken from the
   baseline run's query).

With the default epsilon any measurable drop counts as a regression; raise it to ignore
floating-point-level or negligible changes.

## Summary fields

| Field | Meaning |
|---|---|
| `queries` | number of compared (common) queries |
| `improved`, `regressed`, `unchanged` | status counts |
| `regression_rate` | `regressed / queries` |
| `baseline_metrics`, `candidate_metrics` | mean metrics over the common queries |
| `mean_deltas` | `candidate_metrics - baseline_metrics` per metric |
| `latency` | each run's p50/p95/p99/mean, plus `p50_delta_ms` and `p95_delta_ms` (over each run's scored queries) |
| `critical_regressions` | ids of regressed queries with severity `critical` |

## Gates

`GateConfig` lives under `evaluation.gates` ([models.py](../../src/searchforge/config/models.py)).
Setting a gate to `null` disables it. The report passes only if every enabled check passes.

| Gate | Default | Check |
|---|---|---|
| `ndcg_at_10_min_delta` | `-0.005` | mean nDCG@10 delta `>=` threshold |
| `recall_at_20_min` | off | candidate mean Recall@20 `>=` threshold (absolute) |
| `p95_ms_max` | off | candidate p95 latency `<=` threshold (ms) |
| `max_query_regression_rate` | `0.03` | `regression_rate <=` threshold |
| `block_on_critical_regression` | `true` | no critical query regressed |
| `regression_epsilon` | `1e-9` | tolerance used for per-query status |

The nDCG@10 and Recall@20 gates fail if the metric was not computed, so keep 10 and 20 in
`evaluation.k_values`. The regression-rate gate uses the comparison's primary metric (nDCG@10 for
`searchforge compare`).

Example:

```yaml
evaluation:
  queries: evaluation/queries.jsonl
  qrels: evaluation/qrels.jsonl
  k_values: [1, 3, 5, 10, 20]
  gates:
    ndcg_at_10_min_delta: -0.005      # mean nDCG@10 may drop by at most 0.005
    recall_at_20_min: null            # no absolute recall floor
    p95_ms_max: 250                   # latency budget for the candidate (in-process ms)
    max_query_regression_rate: 0.10   # at most 10% of queries may regress
    block_on_critical_regression: true
    regression_epsilon: 0.001         # ignore per-query changes smaller than 0.001
```

The fixture config uses `ndcg_at_10_min_delta: -0.005`, `max_query_regression_rate: 0.10` and
`block_on_critical_regression: true`.

## Critical queries

A query with `severity: critical` in the query set
([relevance-judgments.md](relevance-judgments.md)) blocks the release if its primary metric regresses
by more than `epsilon`, regardless of the mean. Use it for a small set of queries whose failure is
unacceptable (security procedures, top-traffic lookups, past incidents). Keep the set small and the
judgments stable; with the default epsilon a single rank swap among relevant documents can trip it.

## `searchforge compare`

```
searchforge compare BASELINE CANDIDATE [--config CONFIG] [--candidate-config CONFIG2]
                    [--split test] [--no-gates] [--out comparison.json]
```

- `BASELINE` and `CANDIDATE` are **profile names**, evaluated live (not stored runs). `--config`
  defaults to `configs/evaluations/fixture.yaml`. `--candidate-config` evaluates the candidate on
  another config/index, for example a new chunking or embedding setup.
- Queries come from `evaluation.queries`, restricted by `--split` (or `evaluation.splits`; all
  queries if neither is set). Latency repeats and the relevance threshold come from the config;
  evaluation bypasses the query cache.
- Output: improved/regressed/unchanged counts, mean deltas of nDCG@10, Recall@20, MRR and MAP,
  p50/p95 latency deltas, up to 10 regressed queries with expected documents and old/new top-3, and
  each gate as PASS/FAIL.
- `--out` writes `{"summary": ..., "gates": ...}` as JSON before the exit status is decided.

Exit codes:

| Code | Meaning |
|---|---|
| 0 | comparison ran and all gates passed, or `--no-gates` was given |
| 1 | a gate failed (`QualityGateFailure`) |
| 2 | a SearchForge error during evaluation, e.g. an unknown profile or no judged queries |

Errors raised before evaluation starts (loading the config, opening the index) and an invalid
`--split` value are not mapped by the command and surface as an unhandled exception, which also
exits non-zero (typically 1). In CI, treat the presence of the `--out` JSON with
`gates.passed: false` as the definitive gate failure.

## CI usage

The Jenkins pipeline ([Jenkinsfile](../../Jenkinsfile)) builds the fixture index, then gates the
adaptive planner against BM25 on the held-out split and archives the JSON:

```
uv run searchforge compare lexical adaptive \
  --config configs/evaluations/fixture.yaml --split test --out reports/regression-gate.json
```

Guidelines: compare on the `test` split only, keep the baseline profile fixed between releases, run
on a dedicated machine if `p95_ms_max` is enabled (in-process latency is machine-dependent), and
archive the JSON so regressions can be inspected after the build.
