# Relevance judgments and query sets

Offline evaluation needs a query set and relevance judgments (qrels). Models are in
[judgment.py](../../src/searchforge/domain/judgment.py); file I/O is in
[datasets.py](../../src/searchforge/evaluation/datasets.py).

## Judgments and qrels

A `RelevanceJudgment` is `(query_id, document_id, relevance, source)`. Judgments are per
**document**, never per chunk, so they stay valid when chunking changes. `Qrels` maps
`query_id -> document_id -> relevance`; a later judgment for the same pair replaces the earlier one.

### Graded scale

| Grade | Meaning |
|---|---|
| 0 | not relevant |
| 1 | partially relevant |
| 2 | relevant |
| 3 | perfect answer |

Binary metrics treat `relevance >= evaluation.relevance_threshold` (default 1) as relevant; nDCG
uses the grades directly ([retrieval-metrics.md](../methodology/retrieval-metrics.md)). Unjudged
documents count as 0. Review import enforces integer grades 0-3; the file loaders accept any number.

### File formats

| Format | Layout |
|---|---|
| `.jsonl` | one object per line: `query_id`, `document_id`, `relevance`, optional `source` |
| `.tsv` / `.txt` | TREC `qid 0 docid rel`, or BEIR `query-id corpus-id score`; lines starting with `query-id` or `qid` are skipped as headers; fields are whitespace-separated |
| `.parquet` | columns `query_id`, `document_id`, `relevance`, optional `source` |

`searchforge qrels convert SRC DST` converts between them; TSV output is always 4-column TREC with
integral grades written as integers. The in-memory `Qrels` does not keep each judgment's `source`, so
files written by SearchForge record every judgment as `manual`.

## Query sets

Query sets are JSONL only. Fields:

| Field | Required | Values / default |
|---|---|---|
| `query_id` | yes | unique within the set |
| `text` | yes | query text |
| `query_type` | no | free label, default `unspecified`; metrics are also aggregated per type |
| `split` | no | `train`, `validation`, `test` (default `test`) |
| `severity` | no | `normal` (default), `important`, `critical`; critical regressions can block a release ([regression-gates.md](regression-gates.md)) |
| `filters` | no | filter object applied when the query is run |
| `provenance` | no | `manual` (default), `synthetic`, `sanitized_production`, `failure_derived`, `public_benchmark` |
| `notes` | no | free text |

Provenance also sets the text policy for persisted results: `manual`, `synthetic` and
`public_benchmark` query text is stored verbatim in `per_query.parquet`; anything else is redacted.

## The fixture collection

[datasets/fixture](../../src/searchforge/datasets/fixture/fixture.yaml) is a synthetic knowledge base
of a fictional company, shipped with the package:

- **48 documents**: 39 Markdown KB articles, 4 release-note records (JSONL) and 5 error-code rows
  (CSV), with filterable metadata (`category`, `team`, `language`, `published`, `year`, `version`,
  `severity`). It deliberately includes a legacy version of an article and a German translation.
- **56 queries**, all `synthetic`, in 5 query types: `semantic` 14, `keyword` 12, `identifier` 12,
  `filter` 10, `ambiguous` 8. Splits: 23 train, 13 validation, 20 test. Severity: 4 critical,
  3 important, 49 normal.
- **116 judgments** (grades 1-3); every query has at least one relevant document.

It is large enough to exercise every code path and small enough to inspect by hand. It is not large
enough for statistically meaningful comparisons ([fair-comparisons.md](../methodology/fair-comparisons.md)).

## Blinded human review

[review.py](../../src/searchforge/evaluation/review.py):

1. `searchforge review export CONFIG --profile lexical --profile dense --profile hybrid --depth 10`
   evaluates each profile, pools the top `depth` documents per query across runs, drops pairs that
   already have a judgment, de-duplicates, and shuffles deterministically (seed 7).
2. The CSV (`review_id, query_id, query_text, document_id, title, excerpt, relevance`) contains no
   run names, ranks or scores, so reviewers cannot tell which system retrieved what. The excerpt is
   the first 600 characters of the document's chunks.
3. Reviewers fill `relevance` with 0-3. Blank rows are skipped.
4. `searchforge review import FILE --out evaluation/reviewed_qrels.jsonl` validates the grades and
   writes qrels.

Pooling fills the unjudged gaps that make new systems look worse than they are. Because the export
runs through the evaluator, only queries that already have at least one relevant judgment are pooled.

## Click logs are not relevance labels

[clicks.py](../../src/searchforge/evaluation/clicks.py) (library API, no CLI command) reads anonymized
JSONL events `{query_key, document_id, position, clicked}`; a raw `query` field is rejected.

Clicks are biased evidence:

- **Position / exposure bias.** Users click what they see, and top positions are seen far more.
- **Presentation bias.** Attractive titles and snippets attract clicks regardless of usefulness.
- **Unexamined results.** A result without a click was often never looked at; it is not a negative.

The module estimates an examination propensity per position as `CTR(p) / CTR(1)`, clamped to
`[0.05, 1]`, and computes inverse-propensity-weighted click rates per (query key, document) with at
least 5 impressions: `weak_relevance = min(1, sum(1 / propensity) / impressions)`. This estimator is
crude (without randomized rankings it confounds relevance with position). Outputs are labelled
`click_weak` and are **never merged into qrels automatically**; use them to choose pairs for human
review.

## Query set curation

- **Manual**: written by people who know the domain; best for intent coverage.
- **Synthetic**: generated or authored for a fixture; useful for coverage and tests, not for claims
  about real users.
- **Sanitized production**: sampled from real traffic after redaction; representative but needs
  privacy review.
- **Failure-derived**: created from incidents (below); keeps known failures from returning.
- **Public benchmark**: e.g. BEIR via `load_beir` (only judged queries, `split: test`,
  `query_type: beir`).

Balance query types, assign splits before looking at any results, and mark business-critical
queries `critical`.

## Incident to search eval

[incidents.py](../../src/searchforge/evaluation/incidents.py) turns an observed failure into a
permanent regression case:

```
searchforge incident add "query text" -e KB-OPS-002 --severity important --ref INC-123
```

The query is whitespace-normalized and redacted (emails, URLs, IPs, UUIDs, key-like tokens, long
numbers); it gets id `q-inc-<hash>`, `query_type: incident`, `split: test`,
`provenance: failure_derived` and `notes: incident:<ref>`. Each expected document must exist in the
index and receives grade 3 (`source: incident`). Re-recording the same query is rejected. Cases are
appended to `evaluation/incident_queries.jsonl` and `evaluation/incident_qrels.jsonl` by default; point
`evaluation.queries` / `qrels` at them (or merge them) to include them in evaluation.

## Hard-negative mining

[mining.py](../../src/searchforge/hard_negatives/mining.py) (`searchforge hard-negatives build`)
walks each query's final ranking, takes the first chunk of each document up to `max_rank` document
ranks (default 10), skips documents judged relevant (`>= threshold`), and records up to `per_query`
(default 5) rows with document rank, score and the retrievers that surfaced the chunk.

`judged` separates the two kinds of negatives:

- `judged = true`: the document has a judgment below the threshold (with threshold 1, an explicit 0):
  a confirmed negative.
- `judged = false`: the document was never judged: a **probable** negative that may in fact be
  relevant. Review these before using them for training.
