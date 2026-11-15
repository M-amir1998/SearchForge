# ADR-010: Raw queries are not persisted by default

## Status

Accepted

## Context

Search queries often contain personal or confidential data: names, email addresses, ticket
numbers, internal URLs, sometimes pasted credentials. Traces, logs, metrics and evaluation
artefacts are kept longer and shared more widely than the search index itself. SearchForge still
needs to correlate requests, debug ranking decisions and turn failures into regression cases.

## Decision

Query text is treated as sensitive by default. The controls live in
[`src/searchforge/observability/`](../../src/searchforge/observability/) and are applied by the
search service and the evaluation tools.

- **Capture policy.** `search.capture_policy` decides what the per-request trace stores in its
  `query` field: `none`; `hash_only` (default: `sha256:` plus a 16-hex-character SHA-256 prefix of
  the lower-cased, whitespace-normalised query, `domain/ids.py::query_hash`); `redacted` (emails,
  URLs, IPv4 addresses, UUIDs, key-like and long tokens, long numbers masked by
  `observability/privacy.py::redact`); or `full` (verbatim; opt-in, for local debugging on
  synthetic data).
- **Derived identifiers.** Query ids (`q-...`), rewrite records in traces, and OpenTelemetry span
  attributes carry the hash, never the text. The analysis summary in a trace holds counts and
  feature flags, not tokens. The in-memory query-result cache key is a SHA-256 digest.
- **No access logs.** `searchforge serve` and `serve-fixture` run Uvicorn with
  `access_log=False`. Queries are sent in POST bodies, not URLs.
- **Prometheus label policy** (`observability/metrics.py`). Labels come only from closed sets:
  configured profile names (unknown ones become `invalid`), code-defined stage names (others
  become `other`), status enums, exception class names (at most 40 characters) and
  degradation-reason prefixes. Query text, document ids, chunk ids and user ids are never labels.
- **Click logs** (`evaluation/clicks.py`) are keyed by `query_key`, which is meant to be a hash. A
  row with a `query` field is rejected. The loader does not verify that `query_key` is a hash.
- **Evaluation artefacts.** Persisted per-query tables store text verbatim only for `synthetic`,
  `public_benchmark` and `manual` provenance (`evaluation/store.py`). `sanitized_production` and
  `failure_derived` queries are redacted. Incident cases are redacted before they are written to
  the query set (`evaluation/incidents.py`).

## Consequences

Positive:

- Shipping traces or metrics to a log or monitoring backend does not, by default, copy query text
  into it.
- Repeated queries can still be correlated by hash, which is enough for cache analysis and
  per-query regressions.
- Metric cardinality stays bounded, which also protects Prometheus from label explosions.

Negative:

- The hash is unsalted and truncated. A party holding traces can confirm a guessed query or
  dictionary-attack short, common queries. `hash_only` is pseudonymisation, not anonymisation. Use
  `none` where even that is unacceptable.
- Redaction is pattern-based. It misses names, free-text secrets and domain-specific identifiers,
  and it can over-mask. Review redacted output before sharing it.
- The capture policy covers SearchForge's own traces and artefacts only. With the `remote`
  embedding provider, query text is sent to that service on every dense query
  (`allow_corpus_transmission` gates corpus text, not queries). An injected `LLMQueryPlanner`
  receives the query's residual text.
- Debugging relevance on real queries needs a deliberate policy change and a short retention
  period.

## Alternatives considered

- **Log full queries and restrict access to the logs.** Rejected: logs are copied, exported and
  retained beyond their intended audience. The default should be safe.
- **Salted or keyed hashes (HMAC).** This is stronger against dictionary attacks but needs secret
  management and rotation. A possible later change; the policy enum allows adding it.
- **No query-derived data at all.** Rejected as the default: it prevents correlating requests and
  turning incidents into regression cases. It is available as `capture_policy: none`.
