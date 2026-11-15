# Query analysis and planning

Before retrieval, every query is analyzed, optionally stripped of inline filters, routed to a
profile and validated. Code: [query/analysis.py](../../src/searchforge/query/analysis.py),
[query/rewriting.py](../../src/searchforge/query/rewriting.py),
[query/planning.py](../../src/searchforge/query/planning.py) and `SearchService._plan` in
[service/search_service.py](../../src/searchforge/service/search_service.py).

## QueryAnalyzer

Deterministic, CPU-cheap features computed on the normalized query (after inline-filter removal):

| Feature | Definition |
|---|---|
| `tokens` | regex `[\w.\-/()]+`, surrounding punctuation stripped, lowercased |
| `quoted_phrases` | text inside double quotes (1 to 200 chars) |
| `identifiers` | tokens matching an identifier pattern (below) |
| `numbers` | tokens that are integers or decimals |
| `has_question_word` | first token is a question word (`how`, `why`, `what`, `can`, `is`, ...) or the text ends with `?` |
| `stopword_ratio` | share of tokens in the stopword list |
| `extracted_filters`, `residual_text` | inline `field:value` filters and the remaining text |

Identifier patterns: codes with `-` or `_` that also contain a digit, an underscore or a capital
(`ERR-4012`, `max_attempts`, `X-Request-Id`; plain `rate-limit` does not count); `camelCase`;
`PascalCase`; dotted paths with optional `()`; letters followed by digits (`http2`, `sha256`); dotted
versions (`1.4.2`). Pure numbers and one-character tokens are never identifiers.

### Classification rules

First match wins. `n` is the token count; filters count inline filters plus explicit filter conditions
(a range with `gte` and `lt` counts as two).

| Order | Class | Rule (default thresholds) |
|---|---|---|
| 1 | `identifier` | at least one identifier, `n <= 4`, and no question word |
| 2 | `filter_heavy` | at least one filter and `n <= 4` |
| 3 | `keyword` | `n <= 3` and no question word, or a quoted phrase and `n <= 5` |
| 4 | `complex_semantic` | `n >= 9` |
| 5 | `natural_language` | everything else |

Thresholds come from `AnalyzerThresholds` (`identifier_max_tokens` 4, `filter_residual_max_tokens` 4,
`keyword_max_tokens` 3, `complex_min_tokens` 9). They are constructor arguments of `QueryAnalyzer`;
`SearchService` uses the defaults and they are not exposed in the YAML configuration.

## Inline field:value filters

When `search.planner.extract_inline_filters` is true (default), tokens of the form `field:value` or
`field:"quoted value"` are extracted if `field` is filterable in the opened index (`document_id` is
excluded; `source` is allowed). Other `word:word` tokens stay in the text.

```
rollback runbook team:infra year:2024   ->  text "rollback runbook", filters team = infra, year = 2024
```

Extracted values are parsed with the index filter schema into equality conditions and merged with
explicit filters (duplicates removed). If any extracted value fails type coercion, all inline filters
of that query are discarded and the original text is searched unchanged; the analysis still reports the
extracted field names in the trace.

## QueryRewriter: recorded, never silent

Rewriting is configured per profile (`rewrite`, disabled by default). Strategies, each recorded only if
it changed something:

| Strategy | Effect |
|---|---|
| `whitespace` | collapse and trim whitespace (always checked) |
| `strip_formatting` | remove Markdown emphasis, backticks, quote and heading markers |
| `acronym_expansion` | append configured expansions not already present (`k8s` -> `k8s kubernetes`) |
| `synonyms` | build up to `max_variants` (3) variants by substituting configured synonyms; used only with `multi_query: true` |

The result is a `RewriteRecord` (original, rewritten, strategies, variants). The trace stores
`rewrite.original_hash`, the strategy list, the variant count and a `changed` flag. Variants are
retrieved as separate lists and fused ([retrieval.md](retrieval.md)). Whether rewriting helps is an
empirical question; compare profiles with `searchforge compare` before enabling it.

## StaticQueryPlanner

The default planner. For a non-adaptive profile it returns that profile unchanged (reason
`fixed profile`). For an adaptive profile it maps the query class to a target profile through
`search.planner.rules`, falling back to `default_profile`. At construction, every rule target must
exist and must not itself be adaptive (no cycles); otherwise `InvalidPlan`.

`PlannerConfig` defaults ([config/models.py](../../src/searchforge/config/models.py)):

| Setting | Default |
|---|---|
| `version` | `static-v1` (part of the query-cache key) |
| `rules.identifier` | `lexical` |
| `rules.keyword` | `hybrid` |
| `rules.natural_language` | `hybrid` |
| `rules.complex_semantic` | `hybrid_rerank` |
| `rules.filter_heavy` | `hybrid` |
| `default_profile` | `hybrid` |
| `extract_inline_filters` | `true` |

A `QueryPlan` carries the profile name, retrievers, `candidate_k`, fusion, reranker, diversity and
rewrite settings, the query class, planner name, reason, an optional `fallback` and `notes`.

## LLMQueryPlanner

Optional and never constructed by the CLI or API; it is injected in code, e.g.
`SearchService(handle, config, planner=LLMQueryPlanner(complete, fallback=static_planner))`, where
`complete(prompt) -> str` is any text-completion callable.

- It is consulted only for adaptive profiles; fixed profiles bypass it.
- The prompt lists non-adaptive profiles with their descriptions, the analysis features and the
  residual query text. The query text therefore reaches whatever `complete` calls.
- The output is untrusted. It must be JSON with a `profile` naming an existing, non-adaptive profile
  (`reason` is truncated to 200 chars). Any exception while parsing or checking it yields the static
  plan with `fallback: llm_invalid:<ExceptionName>`.
- A syntactically valid choice is then subject to the same adaptation and validation as any plan
  (below), not to the static fallback.

## validate_plan

Every plan is validated before execution. `InvalidPlan` (HTTP 400) is raised when:

- the plan has no retrievers, or names a retriever other than `lexical` / `dense`;
- the opened index does not provide a planned retriever;
- `candidate_k` is outside 1 to `MAX_CANDIDATES` (1000), or smaller than `top_k`;
- a reranker is enabled and `reranker.top_n < top_k`.

## How SearchService adapts plans

`SearchService._plan` adjusts the planner output before validation:

1. If the requested profile is adaptive and the index lacks a planned retriever (for example a
   lexical-only index), that retriever is dropped and a note is added. For fixed profiles nothing is
   dropped and validation fails instead.
2. If `candidate_k < top_k`, `candidate_k` is raised to `top_k` (note `candidate_k raised to top_k`).
3. If a reranker is enabled and `top_n < top_k`, `top_n` is raised to `top_k` (no note).
4. `validate_plan` runs on the result.

The planner decision is recorded in the trace as `planner_decision`: the plan fields, the analysis
features (counts only for phrases, identifiers and numbers; no query text) and `planner_version`. The
API response reports both the requested `profile` and the `planned_profile`.
