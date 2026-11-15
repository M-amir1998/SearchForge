# ADR-002: SQLite FTS5 as the local lexical backend

## Status

Accepted

## Context

Lexical retrieval (BM25) is the baseline for every comparison in SearchForge, and it has to work
on a CPU-only laptop, in CI and inside a small container without extra services (ADR-009). The
options were a search server (OpenSearch, Elasticsearch), a pure-Python BM25 library, or an
embedded engine.

## Decision

Use SQLite FTS5 through the Python standard library `sqlite3` module
([`indexes/lexical/sqlite_fts.py`](../../src/searchforge/indexes/lexical/sqlite_fts.py)).

- **No extra dependencies.** FTS5 ships with the SQLite in standard CPython builds.
  `searchforge doctor` reports whether it is available and exits non-zero if it is missing.
- **Schema.** A `lex_chunks` table holds the chunk id, the document id and the filter attributes
  as a JSON string. An FTS5 virtual table `lex_fts(body, title)` holds the text; `title` contains
  the document title and the section heading. The tokenizer can be configured from a fixed
  allow-list. The default is `porter unicode61 remove_diacritics 2`.
- **Ranking.** Scores come from `bm25(lex_fts, text_weight, title_weight)`, with per-column
  weights from config (defaults 1.0 and 2.0). FTS5 returns lower-is-better values, so the adapter
  negates them: larger is better, as for every other retriever.
- **Safe query construction.** User text is never interpolated into SQL.
  `build_match_expression` extracts `\w+` tokens and drops a small stopword list (unless only
  stopwords remain). It wraps each token or quoted phrase as an FTS5 string literal (embedded `"`
  doubled) and joins them with `OR`, up to 64 terms. FTS5 operators in user input (`AND`, `NEAR`,
  `*`, `^`, column filters) are therefore inert. The MATCH expression, the column weights, the
  filter values and `LIMIT` are all bound parameters.
- **Filters.** Canonical filters become `json_extract(attributes, ?)` comparisons with bound JSON
  paths and values (`filtering/translate.py::to_sqlite`); field names are also validated.
- **Concurrency.** Searches open the database read-only (`mode=ro`) with one connection per
  thread. Builds use a single writer connection inside a `.partial` version directory (ADR-007).

## Consequences

Positive:

- Lexical search works offline and in CI with zero services, and indexes are single files that
  can be checksummed and copied.
- Query-syntax injection is structurally impossible: user text only ever reaches FTS5 as quoted
  string literals.

Negative:

- BM25 parameters are fixed at SQLite's built-in `k1 = 1.2` and `b = 0.75`. Only column weights
  can be tuned, so BM25 parameter sweeps need another backend.
- The query is always an OR of terms and phrases. There are no FTS5 boolean, prefix or `NEAR`
  semantics for users, and no per-query field boosts beyond the two column weights.
- SQLite has a single writer per database. Index builds are serialised per version directory and
  are not designed for concurrent ingestion. Published indexes are read-only, and changes go
  through new versions (incremental update copies the parent).
- Filtering on `json_extract` over a JSON text column is not indexed. It is acceptable for the
  `tiny`/`small`/`research` data profiles but does not scale like a native keyword index.
- Tokenisation (Porter stemming, `unicode61`) is English-centric. Other analysers need another
  backend.

## Alternatives considered

- **OpenSearch / Elasticsearch BM25.** These offer full analyser and BM25 control but need a JVM
  service with several GB of RAM. An OpenSearch adapter exists for deployments that run it
  (ADR-001). Elasticsearch has licensing caveats noted in
  [docs/references/current-search-stack.md](../references/current-search-stack.md).
- **`rank-bm25` / `bm25s` (pure Python / NumPy).** Rejected: they add a dependency and would need
  their own persistence, filtering and incremental-update story. They also hold the index in
  memory.
- **Tantivy or Lucene bindings.** These are better engines, but they add native dependencies and
  build complexity for little gain at the target corpus sizes.
