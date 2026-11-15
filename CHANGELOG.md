# Changelog

All notable changes to this project are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and this project
aims to follow [Semantic Versioning](https://semver.org/spec/v2.0.0.html) once it reaches 1.0.

## [Unreleased] - 0.1.0.dev0

First development version. Everything below is new.

### Added

#### Architecture

- Engine-independent ports (`ports.py`): `Chunker`, `EmbeddingProvider`, `ChunkStore`,
  `LexicalIndex`, `VectorIndex`, `Retriever`, `FusionStrategy`, `Reranker`, `DocumentSourceReader`.
- Typed configuration with Pydantic v2 (frozen models, unknown keys rejected), YAML `include:`
  composition, and `pkg:` paths for packaged data.
- An error hierarchy rooted at `SearchForgeError`, mapped to CLI exit codes and HTTP statuses.

#### Ingestion and chunking

- Streaming readers for text, Markdown (YAML front matter via `safe_load`), HTML (visible text
  only; scripts and styles dropped), JSON, JSONL, CSV and Parquet.
- Ingestion safety:
  - path containment;
  - `..` glob rejection;
  - symlinks skipped by default;
  - file and record size limits;
  - UTF-8 decoding policy (`replace` / `skip` / `fail`);
  - an ingestion report with counts and warnings.
- Conservative text normalisation (NFC by default, control and zero-width characters removed,
  structure preserved).
- Deterministic document, chunk and citation ids, and content hashes for change detection.
- Chunkers: `fixed_token`, `sentence` and `section` (Markdown/HTML heading aware), with exact
  character offsets (`document.text[start:end] == chunk.text`).

#### Indexing

- SQLite chunk store as the source of truth for text, offsets and citation metadata.
- SQLite FTS5 lexical index with BM25 ranking, configurable tokenizer and column weights, quoted
  and parameterised `MATCH` expressions, and typed metadata filters.
- Exact NumPy vector index (memory-mapped, blockwise scoring, filter pre-masking).
- Optional Qdrant vector adapter (`qdrant` extra; Query API; in-memory, embedded or server).
- Optional OpenSearch lexical (BM25) and k-NN adapters (`opensearch` extra). These can be used
  programmatically and are not yet selectable through index configuration.
- Canonical filter translation to SQLite (bound parameters), Qdrant (`Filter` models) and
  OpenSearch (Query DSL).
- Immutable, versioned index directories:
  - atomic publish from `.partial`;
  - read-only files;
  - a `CURRENT` pointer;
  - manifests with per-file SHA-256 checksums, the embedding spec, configuration fingerprints and
    build provenance.
- Index reuse by build fingerprint, and incremental updates (added, changed and deleted documents
  only) for local indexes.
- Data profiles (`tiny`, `small`, `research`, `large`) with chunk limits, and disk-budget checks
  before builds.

#### Embeddings

- TF-IDF with optional LSA (TruncatedSVD) as the CPU dense baseline. Its fitted state is stored
  pickle-free in the index version.
- Deterministic feature-hashing embeddings for tests and CI (non-semantic).
- Optional Sentence Transformers provider (`neural` extra; CPU; no download unless allowed).
- Optional remote provider for OpenAI-compatible `/embeddings` endpoints. Corpus transmission must
  be opted into explicitly; calls go through a circuit breaker.
- Persistent SQLite embedding cache keyed by text hash, model, revision, preprocessing version and
  dimension.

#### Search

- `SearchService` pipeline:
  - query normalisation;
  - typed filter parsing;
  - query analysis and classification;
  - planning and plan validation;
  - version-aware result cache;
  - deterministic rewriting (whitespace, formatting, acronyms, synonym variants);
  - parallel retrieval with per-retriever timeouts;
  - fusion;
  - hydration;
  - top-N reranking with timeout fallback;
  - diversity;
  - explanations and per-stage traces.
- Fusion strategies:
  - Reciprocal Rank Fusion (default `k = 60`);
  - weighted RRF;
  - min-max or z-score normalised score fusion.
- Rerankers: no-op, CPU lexical-feature reranker, and optional cross-encoder (`neural` extra).
- Diversity: per-document cap and Maximal Marginal Relevance.
- Built-in profiles:
  - `lexical`, `dense`;
  - `hybrid`, `weighted_hybrid`, `hybrid_normalized`;
  - `hybrid_rerank`, `hybrid_cross_encoder`;
  - `adaptive` (rules-based routing by query class).
- Optional LLM query planner via an injected callable, with validated output and static fallback.
- Degraded-mode handling: failed or timed-out retrievers and rerankers are reported, never hidden.

#### Evaluation and benchmarking

- Document-level IR metrics: P@k, Recall@k, HitRate@k, MRR, MAP, nDCG@k (linear or exponential
  gain).
- Offline evaluation runner with warm-up, repeated latency measurement and cache bypass.
- Per-query baseline-vs-candidate regression analysis and configurable quality gates.
- Fusion-weight tuning with strict train / validation / test separation.
- Failure analysis with an evidence/hypothesis split and a failure taxonomy.
- Hard-negative mining.
- Blinded human relevance review (export and import).
- Incident-to-regression-case workflow with redaction.
- Click-log weak labels with position-bias (inverse propensity) correction.
- Query and qrels I/O for JSONL, TREC TSV and Parquet, and BEIR loading.
- Explicit, opt-in BEIR download with archive-safety checks.
- Benchmark and experiment-matrix runner measuring quality, latency percentiles, throughput, peak
  RSS, build time and index size.
- Markdown, SVG and JSON reports generated only from measured runs.
- Citation-ready context packages with budgets and duplicate suppression.
- Citation-validity check and a JSON-Schema-validated evaluation export.

#### Interfaces

- Bundled synthetic fixture collection with queries and qrels, used by tests, CI and the demo.
- `searchforge` CLI. Commands:
  - `doctor`;
  - `corpus inspect` and `corpus prepare`;
  - `index build`, `index update`, `index list`, `index inspect` and `index validate`;
  - `search`;
  - `evaluate`, `compare` and `tune`;
  - `hard-negatives build`;
  - `benchmark` and `report`;
  - `serve` and `serve-fixture`;
  - `storage status` and `storage clean`;
  - `cache stats` and `cache clear`;
  - `incident add`;
  - `review export` and `review import`;
  - `qrels convert`;
  - `datasets fetch-beir`;
  - `demo` and `demo adaptive`.
- FastAPI service with these endpoints:
  - `/health` and `/ready`;
  - `/search` and `/search/explain`;
  - `/indexes` and `/indexes/{index_id}`;
  - Prometheus `/metrics`.
- Framework-agnostic RAG evidence retriever and agent tool definition. No LLM is served.

#### Privacy and observability

- Query capture policy (`none`, `hash_only` (default), `redacted`, `full`).
- Pattern-based redaction.
- Prometheus metrics with a closed-set label policy.
- Optional OpenTelemetry spans (`observability` extra).

#### Tooling and operations

- Tests:
  - unit, Hypothesis property, integration (including in-memory Qdrant and OpenSearch query
    construction), search-quality and CPU end-to-end tests;
  - live OpenSearch tests run only when `OPENSEARCH_URL` is set.
- Jenkins pipeline (`Jenkinsfile`). Stages:
  - lockfile, format, lint and strict typing;
  - tests with coverage;
  - fixture index build and validation, and a CLI search smoke test;
  - retrieval regression gate and demo;
  - secret scan and package build;
  - optional Docker build.
- Pre-commit hooks (ruff, detect-secrets, private-key and large-file checks) and
  `scripts/pre_push_check.sh`.
- Non-root, CPU-only Docker image.
- Docker Compose with the SearchForge API by default, plus opt-in `qdrant`, `observability`
  (Prometheus and Grafana dashboard) and `opensearch` profiles. All services bind to localhost
  and have resource limits.
- Documentation:
  - Architecture Decision Records ([docs/adr/](docs/adr/README.md));
  - Jenkins setup guide ([docs/ci/jenkins.md](docs/ci/jenkins.md));
  - [SECURITY.md](SECURITY.md) and [CONTRIBUTING.md](CONTRIBUTING.md).

[Unreleased]: https://github.com/M-amir1998/SearchForge-/commits/main
