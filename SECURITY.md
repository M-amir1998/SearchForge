# Security Policy

SearchForge ingests documents, builds local indexes, serves a search API and runs evaluations. This
document describes the threat model, the controls implemented in the code, and the known
limitations. It applies to version `0.1.0.dev0` (pre-alpha). Only the latest commit on `main`
receives security fixes.

## Reporting a vulnerability

Please report vulnerabilities privately through GitHub:

1. Open the repository's **Security** tab:
   <https://github.com/M-amir1998/SearchForge-/security>.
2. Choose **Report a vulnerability**. This opens a private security advisory draft visible only to
   the maintainers.
3. Include the following:
   - the affected component;
   - the version or commit;
   - reproduction steps or a proof of concept;
   - the impact you observed.

Do not open a public issue, pull request or discussion with vulnerability details. If private
reporting is unavailable, open a public issue that only asks for a private contact channel and
contains no technical details.

Reports are handled on a best-effort basis. Fixes are published through a GitHub security advisory,
and the reporter is credited unless they prefer otherwise.

## Threat model

### Assets

- Corpus content and derived artifacts: the chunk store, lexical and vector indexes, embedding
  caches.
- Query text, search traces, evaluation runs and click logs.
- Backend credentials: Qdrant, OpenSearch and remote embedding API keys.
- The integrity of index versions and evaluation results.

### Trust boundaries

| Input | Trust |
|---|---|
| Configuration files and CLI arguments | **Trusted** (operator-controlled). Paths, URLs and hosts come from here. |
| Documents in a corpus | **Untrusted.** They may be malformed or hostile. |
| Search API request bodies, filters, query text | **Untrusted.** |
| Downloaded datasets and archives | **Untrusted.** |
| Index directories received from someone else | **Untrusted** until validated. |
| Remote embedding service, Qdrant, OpenSearch | Trusted to operate correctly, but they receive data, and their responses are validated. |

## Controls

### Untrusted documents

Implemented in `ingestion/safety.py`, `ingestion/readers.py`, `parsing/` and `config/models.py`.

- **Size limits.** Files larger than `corpus.limits.max_file_bytes` (default 50 MiB) are skipped.
  The size is checked again while reading, in case a file grows. Records longer than
  `max_record_chars` (default 2,000,000) are skipped. The CSV field size limit is set to the same
  value, and `max_documents` optionally caps the corpus.
- **Encoding policy.** Input must be UTF-8 (a BOM is tolerated). `on_decode_error` is `replace`
  (default; replacements are counted and warned), `skip` or `fail`.
- **Normalisation.** C0/C1 control characters, zero-width characters and BOMs are removed, and
  line endings are normalised. NFC is the default; NFKC is opt-in.
- **Metadata.** Only scalar values are kept, and only under keys that are valid Python
  identifiers. Filterable fields are coerced to their declared types at index time; values that
  fail coercion are dropped.
- **Document ids.** Explicit ids are used verbatim only if they match
  `^[A-Za-z0-9][A-Za-z0-9._:\-/]{0,199}$`; otherwise they are hashed. Derived ids use
  source-relative keys, never absolute paths.

### Malicious HTML and text

- HTML is converted to text with the standard library `html.parser`. The contents of `script`,
  `style`, `noscript`, `template`, `svg`, `iframe` and `object` are dropped. Nothing is executed,
  rendered or fetched. Attributes, including URLs, are ignored.
- Markdown front matter is parsed with `yaml.safe_load` only. Non-mapping or invalid front matter
  is ignored. Configuration files also use `yaml.safe_load`, and unknown configuration keys are
  rejected.
- JSON and JSONL are parsed with `json.loads`. Invalid records are skipped and counted.
- **No pickle in index artifacts.** Manifests and `rows.jsonl` are JSON. Vectors are `.npy` files
  loaded with `allow_pickle=False`. The fitted TF-IDF state is JSON (vocabulary, parameters) plus
  `.npy` (idf, SVD components), also loaded with `allow_pickle=False`. A downloaded index therefore
  cannot run code through deserialisation.
- Search results return document text as data. Consumers that render results as HTML must escape
  them.

### Path traversal

- `ensure_within` resolves every candidate file and requires it to stay under the source root.
  Files outside it are skipped with a warning.
- Glob patterns containing `..` are rejected (`UnsafePathError`).
- Symlinks are skipped by default (`follow_symlinks: false`). Even when they are followed, the
  resolved target must stay inside the root.
- `pkg:` configuration paths are resolved inside the installed package's data directory. Paths
  that escape it raise `ConfigurationError`. Configuration includes are limited to a depth of 8.
- `searchforge storage clean` only removes project-owned artifacts inside the workspace, and it
  defaults to a dry run.

### Decompression bombs and downloads

- `evaluation/beir.py::safe_extract` applies these checks before extracting a ZIP archive:
  - rejects archives with more than 10,000 members;
  - rejects archives whose total declared uncompressed size exceeds 2 GiB;
  - rejects absolute paths, `..` components, symlink members, and members that resolve outside
    the destination.
- `searchforge datasets fetch-beir` runs only when invoked explicitly, never in tests or CI. It
  accepts only an allow-list of dataset names, uses a fixed URL template, and aborts once the
  download exceeds 200 MB.

### Query and search-log privacy

See [ADR-010](docs/adr/ADR-010-raw-queries-not-persisted-by-default.md).

- **Capture policy.** `search.capture_policy` defaults to `hash_only`, so traces carry a hash
  instead of the query text. The other options are `none`, `redacted` (emails, URLs, IPs, UUIDs,
  key-like tokens and long numbers masked) and `full` (opt-in).
- **Hash-only identifiers.** Query ids, rewrite records and OpenTelemetry span attributes carry
  hashes, never text. Uvicorn access logs are disabled.
- **Click logs.** Rows are keyed by `query_key`, which should be a hash. Rows with a raw `query`
  field are rejected. The loader does not check that `query_key` is actually a hash, so do not put
  query text there.
- **Evaluation artefacts.** Production-derived and incident-derived queries are redacted before
  they are persisted. Only synthetic, public-benchmark and manually written queries are stored
  verbatim.

### Prometheus label policy

Metric labels come only from closed sets:

- configured profile names (unknown names become `invalid`);
- code-defined stage names (unknown names become `other`);
- status enums;
- exception class names (at most 40 characters);
- degradation-reason prefixes.

Query text, document ids, chunk ids and user ids are never used as labels. This protects privacy
and keeps cardinality bounded (`observability/metrics.py`).

### SSRF

- These URLs and hosts come only from operator configuration, never from request input:
  - the remote embedding URL (`dense.remote.url`);
  - Qdrant `url`;
  - OpenSearch `hosts`.
- The search API has no endpoint that fetches URLs. Document URIs are stored as metadata and are
  never fetched.
- The only built-in download (BEIR) uses a fixed URL template and an allow-list of dataset names.

### Backend authentication

- **Secrets come from environment variables.** The configuration schema has no dedicated secret
  fields. Credentials are referenced only by the names of environment variables:
  - Qdrant: `api_key_env`;
  - OpenSearch: `username_env` and `password_env`;
  - remote embeddings: `api_key_env`.

  URL and host fields are free-form strings. Never embed credentials in them (for example a
  `user:password@` userinfo prefix in a host URL). The lexical configuration, including OpenSearch `hosts`, is copied
  into index manifests, which `searchforge index inspect` prints.
- OpenSearch TLS certificate verification is on by default (`verify_certs: true`).
- `.env` files, keys and credential files are git-ignored. `detect-secrets` and
  `detect-private-key` run in pre-commit, and `detect-secrets` also runs in Jenkins.

### Filter and query injection

- Filters are parsed from untrusted input into a typed `FilterExpression` (`domain/filters.py`):
  - unknown fields are rejected;
  - field names must match `^[a-z][a-z0-9_]{0,63}$`;
  - operators come from an allow-list and must suit the field type;
  - values are type-coerced;
  - set filters are limited to 256 values.
- Each backend receives a structured translation, never a concatenated string:
  - SQLite gets bound parameters for both the JSON path and the value;
  - Qdrant gets `Filter` models;
  - OpenSearch gets `term`, `terms` and `range` DSL clauses.
- The SQLite FTS5 `MATCH` expression is built from extracted word tokens, each quoted as an FTS5
  string literal, and passed as a bound parameter. FTS5 operators in user input have no effect.
- OpenSearch receives the query text only as the value of a `multi_match` query field, never in
  query-string syntax.
- API request bodies reject unknown fields. They allow 1-2,000 query characters, `top_k` from 1 to
  100, and profile names up to 64 characters. Plans are validated before execution. An optional
  LLM planner's output is treated as untrusted and must name an existing profile.

### Embedding service trust

- With `dense.provider: remote`, corpus text is sent to the service only when
  `dense.remote.allow_corpus_transmission: true` is set. Otherwise document embedding raises
  `EmbeddingFailure`.
- Responses are validated before use:
  - the number of vectors must match the request;
  - the dimension must match the configuration;
  - vectors are L2-normalised.
- Failures go through a circuit breaker. Error messages do not include the response body.
- Sentence Transformers models load only from the local cache unless `allow_download: true`. Pin
  `revision` to a commit hash and load models only from trusted publishers.

### Remote and shared index trust

- Every index version has a manifest with a SHA-256 checksum and size for each file.
  `searchforge index validate <name[@version]>` recomputes them and reports missing, modified and
  unexpected files. It then checks chunk and document counts.
- Run `index validate` before serving an index directory you did not build yourself.
- Published index files are read-only, and search opens SQLite files with `mode=ro`.

### Network exposure

- `searchforge serve` binds to `127.0.0.1` by default.
- Docker Compose publishes every service on `127.0.0.1` only.

### Containers

- The Docker image runs as the non-root user `searchforge` (uid/gid 10001) on `python:3.12-slim`.
  It sets `HF_HUB_OFFLINE=1` and `TRANSFORMERS_OFFLINE=1`.
- Docker Compose adds `no-new-privileges` and memory and CPU limits.

## Known limitations

These are not mitigated in the current version. Account for them in deployments.

- **API.**
  - The API has no authentication, authorisation, TLS or rate limiting. `/metrics` and `/indexes`
    are unauthenticated. Put it behind a reverse proxy that provides these before exposing it
    beyond localhost.
  - `searchforge serve-fixture`, the container entrypoint, binds to `0.0.0.0` by default.
  - Index version references in `GET /indexes/{index_id}` are not validated against a strict
    pattern before they are joined into a filesystem path.
- **Index integrity.**
  - Manifests are not signed, so checksums detect corruption, not tampering by someone who can
    also rewrite `manifest.json`.
  - Validation is not run automatically when an index is opened.
  - For Qdrant-backed indexes, the remote collection contents are not covered by checksums.
  - SQLite database files from untrusted sources are parsed by SQLite. Validate their checksums
    against a manifest from a trusted source.
- **Query privacy.**
  - The `hash_only` query hash is unsalted and truncated. It can be dictionary-attacked for
    short, common queries.
  - Redaction is pattern-based and incomplete.
  - With the remote embedding provider, query text is always sent to the service.
  - If `dimension` is not configured, a probe string is sent to discover it.
  - An injected LLM planner receives the query text.
- **Backend authentication.** If an environment variable named by Qdrant `api_key_env`, OpenSearch
  `username_env`/`password_env` or the remote embedder's `api_key_env` is configured but unset, client
  creation fails with a configuration error instead of connecting unauthenticated.
- **Ingestion.** Parquet files are bounded by `max_file_bytes` on disk only. Decompressed column
  data is not limited before records are materialised in batches of 1,024 rows.
- **Downloads.** `fetch-beir` follows redirects and does not verify a checksum of the downloaded
  archive.
- **Local-only Compose settings.** The `opensearch` profile disables the OpenSearch security
  plugin, and the `observability` profile enables anonymous Grafana viewer access. Both are for
  local development on `127.0.0.1` only.
