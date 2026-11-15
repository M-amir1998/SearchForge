# Retrieval metrics

Definitions used by SearchForge. They match
[metrics.py](../../src/searchforge/evaluation/metrics.py) exactly; the evaluation loop is in
[runner.py](../../src/searchforge/evaluation/runner.py).

## Notation

- `d_1, d_2, ..., d_n`: the **document-level** ranking for one query (see the collapse rule below).
- `rel(d)`: graded judgment of document `d` for the query (scale 0-3); unjudged documents have
  `rel(d) = 0`.
- `t`: the relevance threshold for binary metrics, `evaluation.relevance_threshold` (default 1.0,
  must be > 0).
- `R = { d : rel(d) >= t }`: the set of relevant documents (judged documents, whether or not they
  were retrieved).
- `top_k = { d_1, ..., d_k }`.

## Binary metrics

```
Precision@k = |R intersect top_k| / k
Recall@k    = |R intersect top_k| / |R|                (0 if R is empty)
HitRate@k   = 1 if R intersect top_k is non-empty, else 0
RR          = 1 / min{ i : d_i in R }                  (0 if no relevant document is ranked)
AP          = (1 / |R|) * sum over i with d_i in R of Precision@i     (0 if R is empty)
```

- Precision@k divides by `k` even when fewer than `k` documents were returned.
- AP sums over the whole evaluated ranking; relevant documents that were never retrieved contribute
  0 but still count in `|R|`.
- MRR and MAP are the means of RR and AP over queries.

## Graded metrics: DCG and nDCG

```
DCG@k  = sum_{i=1..k} gain(g_i) / log2(i + 1),    g_i = max(0, rel(d_i))
gain(g) = g                  (linear; default)
gain(g) = 2^g - 1            (exponential variant)
nDCG@k = DCG@k / IDCG@k      (0 if IDCG@k = 0)
```

`IDCG@k` is the DCG@k of **all judged grades of the query** (clamped at 0) sorted in descending
order, regardless of whether those documents were retrieved, are in the index, or meet the
threshold. nDCG does **not** use the threshold: a grade-1 judgment contributes gain even when
`t = 2`.

The linear gain matches trec_eval's default `ndcg` gain. The evaluation runner (`evaluate_ranking`)
always reports linear-gain nDCG; the exponential variant is available only by calling
`ndcg_at_k(..., exponential=True)` directly. Exponential gain emphasizes grade-3 documents much more
than grade-1, so the two variants can rank systems differently.

## Chunk-to-document collapse

Retrieval returns chunks; judgments are per document. Before scoring, the chunk ranking is
collapsed to a document ranking by **first occurrence** (`collapse_to_documents`): each document
appears once, at the position of its best-ranked chunk.

Why:

- Judgments refer to documents, so scoring must be at document level.
- Different chunking strategies produce different chunk ids but the same document ids. Collapsing
  makes systems with different chunkers comparable on the same qrels.
- Without it, several chunks of one relevant document would inflate Precision@k and DCG.

Consequence: `k` positions in a metric are `k` distinct documents, which may need more than `k`
chunks. The runner therefore asks for up to `ranking_depth = 100` chunks without changing the plan.
That depth is still bounded by the profile's `candidate_k` (50 by default), so the document ranking
can be shorter than 20 entries; Recall@20 and AP are computed over what is available.

## Query inclusion and aggregation

- A query is **excluded** when it has no judgment with `rel >= t`. Excluded query ids are reported
  in the run summary (`excluded`), not scored as 0.
- Aggregate metrics are arithmetic means over the scored queries. Queries whose search raised an
  error are recorded with an `error` field and are not part of the aggregates; the summary reports
  their count separately (`errors`).
- `k_values` (default `[1, 3, 5, 10, 20]`) are de-duplicated and sorted; every `k` must be >= 1.
- Metrics are computed from the first measured repetition of each query; repetitions only add
  latency samples.

Metric keys: `ndcg@k`, `recall@k`, `precision@k`, `hit@k` for each `k`, plus `mrr` and `map`.

## Latency percentiles

`percentile(values, q)` uses linear interpolation between order statistics (NumPy's `linear`
method):

```
x = values sorted ascending, n = len(x)
pos = (n - 1) * q / 100,  lo = floor(pos),  hi = ceil(pos)
P_q = x[lo] + (x[hi] - x[lo]) * (pos - lo)
```

An empty list gives 0; a single value is returned as-is. Example: `[10, 20, 30, 40]` gives
`P50 = 25` and `P95 = 30 + 0.85 * 10 = 38.5`. Run-level p50/p95/p99 are computed over all measured
samples (queries x repeats); a query's own latency is the p50 of its samples.

## Worked example

Ranking `[A, B, C, D, E]`, judgments `{A: 3, C: 2, F: 1}`, `t = 1`, `k = 3`.
`R = {A, C, F}` (F is relevant but not retrieved).

| Metric | Computation | Value |
|---|---|---|
| Precision@3 | A, C in top 3: 2 / 3 | 0.6667 |
| Recall@3 | 2 of 3 relevant: 2 / 3 | 0.6667 |
| HitRate@3 | A in top 3 | 1 |
| RR | first relevant at rank 1 | 1 |
| AP | (P@1 + P@3) / \|R\| = (1 + 2/3) / 3 | 0.5556 |
| DCG@3 | 3/log2(2) + 0/log2(3) + 2/log2(4) = 3 + 0 + 1 | 4 |
| IDCG@3 | grades sorted (3, 2, 1): 3/1 + 2/log2(3) + 1/log2(4) = 3 + 1.2619 + 0.5 | 4.7619 |
| nDCG@3 | (3 + 1) / (3 + 2/log2(3) + 0.5) | 0.8400 |
| nDCG@3, exponential | (7 + 3/2) / (7 + 3/log2(3) + 1/2) | 0.9049 |

These values are what `evaluate_ranking(["A","B","C","D","E"], {"A": 3, "C": 2, "F": 1}, [3])`
returns.
