# Fair comparisons

Rules SearchForge follows (and that users should follow) when comparing retrieval systems. They are
implemented in [runner.py](../../src/searchforge/evaluation/runner.py),
[tuning.py](../../src/searchforge/evaluation/tuning.py),
[benchmarking/runner.py](../../src/searchforge/benchmarking/runner.py) and
[report.py](../../src/searchforge/reporting/report.py).

## Same corpus, same index, same queries

- Every system in a comparison is evaluated on the same query selection and the same qrels.
- `searchforge compare BASELINE CANDIDATE` evaluates both **profiles** against one index unless
  `--candidate-config` deliberately points the candidate at another config/index.
- `searchforge benchmark` builds one index per distinct `index.name`; systems that only differ in
  search profile share that index. Experiments that need a different index (chunking) get their own
  index name explicitly, built from the same corpus.
- Metrics are document-level ([retrieval-metrics.md](retrieval-metrics.md)), so systems with
  different chunkers are scored against identical judgments.
- Per-query paired analysis (`compare_runs`) uses queries present in both runs (errored queries
  count with zero metrics); latency deltas use the same common queries; queries missing from either
  side are listed as `missing_queries`.

## Train / validation / test discipline

Each query carries a `split` ([relevance-judgments.md](../evaluation/relevance-judgments.md)). The
fixture query set has 23 train, 13 validation and 20 test queries.

- **Tune on train, select on validation, report on test once.** `searchforge tune` enforces this for
  fusion weights; it refuses to run if any split is empty.
- **Never tune on test.** Any choice made after looking at test results (weights, thresholds, rules,
  feature weights) turns the test split into training data. The built-in `weighted_hybrid` weights
  are an a priori choice, not tuned.
- The fixture core matrix and the CI gate report the **test** split only. The adaptive study
  (`adaptive.yaml`) uses all splits and says so in its spec.
- `searchforge evaluate` uses `evaluation.splits` from the config, or all queries if it is unset
  (the fixture config leaves it unset); pass `--split test` for held-out numbers.
- The rules-based planner and the lexical-feature reranker were designed alongside the fixture
  collection. Treat fixture results for them as optimistic until confirmed on independent data
  (for example the prepared BEIR SciFact study, `configs/benchmarks/scifact.yaml`).

## Latency measurement

- **Query cache bypassed.** Evaluation calls `search(..., use_cache=False)`, and responses requested
  with a ranking depth are never cached, so every measured call does real work. Cache benefit is
  measured separately by the benchmark (cold pass, then warm pass).
- **Warm-up.** Before measuring, the first `min(5, n)` judged queries run once each (errors
  ignored) to load memory maps, caches and lazily imported code.
- **Repetition.** Each query runs `latency_repeats` times (default 3; `--repeats` overrides);
  percentiles are taken over all samples with linear interpolation. Metrics come from the first
  repetition.
- **Throughput** is measured separately with N client threads (default 4) over a fixed number of
  queries, cache bypassed.

### In-process caveats

Latency is `time.perf_counter()` around `SearchService.search` in the same Python process: no
network, HTTP, serialization or client overhead. It includes query analysis, planning, retrieval,
fusion, hydration, reranking and diversity. Numbers depend on the machine, CPU frequency scaling,
garbage collection and other load, and are only comparable within one run on one machine. On tiny
collections stage costs are dominated by fixed overheads and do not extrapolate to large corpora.

### Approximate peak RSS

`PeakRSSMonitor` ([memory.py](../../src/searchforge/benchmarking/memory.py)) samples the whole
process's resident set size every 20 ms. It can miss short spikes, and it includes the Python
runtime and every index opened so far. The benchmark keeps earlier systems' indexes open, so a
system's peak RSS also depends on its position in the system list. Treat it as a coarse
process-level indicator, not a per-component measurement.

## Small collections are noisy

The bundled fixture has 48 documents and 56 queries; the test split has 20. A single query changing
its nDCG@10 by 1.0 moves a 20-query mean by 0.05. Differences of a few hundredths are therefore within
noise. SearchForge performs **no significance testing** at fixture size. Prefer per-query analysis
(improved / regressed / unchanged, per query type) over a single mean, and confirm conclusions on a
larger, independent collection.

## Reporting

- **Report only measured numbers.** Every figure in a report is produced by the run that wrote it;
  nothing is hard-coded or copied from elsewhere. Run artifacts (config, index manifest, metrics,
  per-query tables, failures) are written to `results/<run-id>/`.
- Record the environment (Python version, platform, CPUs, RAM, GPU presence); the benchmark report
  does this automatically.
- **Label fixture results.** Results on the bundled synthetic collection are headed
  **LOCAL DEMONSTRATION RESULTS**: they demonstrate the methodology, are not industry benchmarks, and
  do not predict behaviour on other corpora.
- State the limitations that apply (collection size, corpus-fitted dense baseline, in-process
  latency, components designed on the same data).
