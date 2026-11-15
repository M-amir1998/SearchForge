# Failure taxonomy

Query-level failure analysis is implemented in
[failures.py](../../src/searchforge/evaluation/failures.py). For every relevant document that is
missing from a query's final top-k, it records **evidence** (observed facts) and a **hypothesis**
(the most likely category, chosen by fixed rules). Hypotheses are a triage aid, not proof of a root
cause; every record carries `basis = "rule-based hypothesis"`.

## What counts as a failure

For each scored query (queries whose search raised an error are skipped) and each judged document
with `relevance >= threshold` that is not among the first `k` documents of the final document ranking
(`k = 10` by default), one `FailureRecord` is produced. Relevant documents are processed in order of
decreasing grade.

## Evidence

| Field | Meaning |
|---|---|
| `indexed` | the document exists in the served index version |
| `final_rank` | 1-based rank in the final document ranking, or empty |
| `fused_rank` | rank in the fused candidate list (collapsed to documents), or empty |
| `lexical_rank`, `dense_rank` | rank in that retriever's candidate list (collapsed to documents), or empty |
| `planned_profile` | profile actually executed (for `adaptive`, the routed target) |
| `reranked` | a reranker was configured in the executed plan (also true if it then timed out or failed) |
| `duplicate_chunks_in_top_k` | number of top-k chunk positions occupied by an already-seen document |
| `rewrite_strategies` | rewrite strategies applied to the query |

An empty rank means "not within that list's `candidate_k` candidates" (or the retriever did not run).
It does **not** mean the document shares no terms with the query. Ranks of multi-query variant lists
are not part of the evidence.

## Categories and decision rules

`_classify` applies these rules in order; the first match wins.

| # | Category | Rule |
|---|---|---|
| 1 | `index_staleness` | the document is not in the served index (`indexed = false`) |
| 2 | `filter_error` | the query has explicit `filters`, the document has chunks, and the filters do not match the **first chunk's** attributes |
| 3 | `bad_rerank` | `fused_rank <= k` and `reranked` |
| 4 | `bad_fusion` | `lexical_rank <= k` or `dense_rank <= k`, and `fused_rank` is empty or `> k` |
| 5 | `duplicate_results` | `duplicate_chunks_in_top_k >= max(2, k // 3)` (3 for `k = 10`) |
| 6 | `chunking_failure` | `lexical_rank` is empty, the query has content terms, the document has more than one chunk, the union of its chunks covers >= 75% of the query terms, but the best single chunk covers < 50% |
| 7 | `query_rewrite_failure` | rewrite strategies include `acronym_expansion` or `synonyms`, and both `lexical_rank` and `dense_rank` are empty |
| 8 | `semantic_miss` | both ranks empty and a dense candidate list was returned |
| 8 | `missed_lexical_match` | both ranks empty and no dense list was returned (lexical-only plan, or the dense retriever timed out or failed) |
| 9 | `missed_lexical_match` | `lexical_rank` present and (`dense_rank` empty or `lexical_rank <= dense_rank`): weak lexical match, no semantic rescue |
| 10 | `semantic_miss` | otherwise: `dense_rank` present and better than `lexical_rank` (weak semantic match) |

Details of the rules:

- Query terms for rule 6 are lower-cased `\w+` tokens without stopwords and without one-character
  tokens; chunk terms come from chunk text plus section heading. No stemming is applied.
- Rule 2 checks only explicit query filters (inline `field:value` filters extracted from the text
  are not re-checked) and only the first chunk's attributes, which are document-level metadata.
- Rule 3 fires whenever a reranker was configured and the document was in the fused top-k. If
  diversity (per-document cap or MMR) demoted the document instead, it is still attributed to the
  reranker. Without a reranker, such a demotion falls through to later rules.
- `index_staleness` covers any absence from the index: a stale version, a document not yet ingested,
  or a judgment that refers to a wrong document id.

## Evidence versus hypothesis

Keep them apart when triaging. Evidence fields are facts about this run and can be trusted as such.
The category and its `hypothesis` text are an inference from those facts. For example, `bad_fusion`
says "a retriever ranked it in its top-k but fusion demoted it"; it does not establish that a
different fusion setting would fix it. Confirm a hypothesis by changing the suspected stage and
re-running `searchforge compare`.

## Using `failures.parquet`

`searchforge evaluate` writes `results/<run-id>/failures.parquet`; `searchforge benchmark` writes
one per system under `runs/<run-id>/`. Columns:

- `query_id`, `query_type`, `document_id`, `relevance`, `category`, `hypothesis`, `basis`,
  `severity`
- one `evidence_<field>` column per evidence field (lists are stored as strings)

A run without failures writes a table with a single `empty` column. Example (with pandas installed):

```python
import pandas as pd

f = pd.read_parquet("results/<run-id>/failures.parquet")
if "category" in f:
    print(f.groupby(["query_type", "category"]).size())
    print(f[f.severity == "critical"][["query_id", "document_id", "category", "hypothesis"]])
    # documents a retriever found but fusion lost
    print(f[f.category == "bad_fusion"][["query_id", "evidence_lexical_rank", "evidence_dense_rank",
                                         "evidence_fused_rank"]])
```

Typical uses: count categories per system to see which stage loses most relevant documents, look at
critical queries first, and turn recurring failures into regression cases
([relevance-judgments.md](relevance-judgments.md#incident-to-search-eval)).
