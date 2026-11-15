# Project Status — SearchForge

_Last updated: 2026-11-15_

| Item | Status |
|---|---|
| Repository | https://github.com/M-amir1998/SearchForge- (public; the repository name has a trailing dash) |
| Version | `0.1.0.dev0` — not tagged; no release until the real-scale studies below are done |
| Git authors | Single author: the repository owner's configured identity. No AI co-author or generated-by trailers (enforced by `scripts/pre_push_check.sh`). |
| CI | Jenkins only ([Jenkinsfile](Jenkinsfile)); no `.github/workflows/`. |

## What is implemented

| Area | Implementation |
|---|---|
| Architecture | Ports & adapters ([ports.py](src/searchforge/ports.py)); domain free of backend/web/ML objects; typed Pydantic config with `include:` composition |
| Corpus pipeline | Streaming readers for text, Markdown (front matter, sections), JSON, JSONL, CSV, Parquet (batches), HTML; path-traversal, size, symlink and encoding safeguards; conservative NFC normalization; stable document ids (source identity), content hashes |
| Chunkers | `FixedTokenApproxChunker`, `SentenceChunker`, `SectionChunker` with configurable size/overlap/minimum and exact offsets |
| Index backends | SQLite FTS5 (BM25) lexical; NumPy exact dense (mmap, blockwise top-k, filter masks); **Qdrant** adapter (`query_points`, filters, in-memory/embedded/server — tested in-memory); **OpenSearch** lexical + k-NN adapters (Query DSL tested; live test needs `OPENSEARCH_URL`); pgvector documented only |
| Index lifecycle | Immutable versions, manifests with checksums, `.partial` builds + atomic publish, read-only files, CURRENT pointer, fingerprint reuse, content-hash incremental updates (add/update/delete), validation |
| Embedding providers | Hash (tests only), TF-IDF + LSA (pickle-free), Sentence Transformers (optional, offline-by-default), remote OpenAI-compatible (explicit corpus opt-in, circuit breaker); persistent version-aware embedding cache |
| Retrievers | Lexical, Dense, Hybrid |
| Fusion | RRF, weighted RRF, normalized (min-max / z-score) score fusion |
| Rerankers | NoOp, LexicalFeature (CPU), CrossEncoder (optional) with timeout fallback |
| Query planner | Deterministic analysis + rules classification; static planner; optional LLM planner with untrusted-output fallback; recorded rewriting (acronyms, synonyms, multi-query) |
| Filters | Typed eq/ne/in/nin/range/date-range; parameterized SQL, Qdrant and OpenSearch translation; inline `field:value` extraction |
| Diversity | Per-document cap, MMR, grouping |
| Caching | Bounded LRU query-result cache (entries/bytes/TTL, index-version-aware keys); embedding cache with stats/cleanup |
| Citation builder | Budgeted (chars/tokens), overlap and near-duplicate suppression, stable citation ids; RAG retriever and agent tool adapters |
| Evaluation | P@k, R@k, HitRate@k, MRR, MAP, nDCG@k; qrels JSONL/TSV/Parquet; BEIR loader; runner with cache bypass, warm-up, repeats |
| Retrieval regression | Per-query improved/regressed/unchanged, metric and latency deltas, critical queries, YAML quality gates (`compare` exits 1) |
| Failure analysis | Evidence + rule-based hypothesis across 9 taxonomy categories |
| Hard negatives | Mining with judged/unjudged distinction and source retrievers |
| Incident → eval | Sanitized failing query → judgments → regression case (`searchforge incident add`) |
| Human review / clicks | Blinded review export/import; click-log weak labels with position-bias correction (never auto-merged) |
| API | FastAPI `/health`, `/ready`, `/search`, `/search/explain`, `/indexes`, `/indexes/{id}`, `/metrics` |
| Observability | Prometheus metrics with low-cardinality labels; optional OpenTelemetry spans; HASH_ONLY query capture by default; Grafana dashboard |
| Storage safety | Disk-budget estimate + reserve, data profiles (tiny/small/research/large), `storage status`, `storage clean` (dry-run default, workspace-only) |
| CLI | doctor, corpus, index, search, evaluate, compare, tune, hard-negatives, benchmark, report, serve, serve-fixture, storage, cache, incident, review, qrels, datasets, demo, demo adaptive |
| Docker | Non-root CPU image; compose with opt-in Qdrant / observability / OpenSearch profiles and memory limits |

## Verification (this machine: Windows 11, Python 3.12.13, 12 logical CPUs, 15.6 GB RAM, no GPU)

| Check | Result |
|---|---|
| Tests | 113 collected: **112 passed, 0 failed, 1 skipped** (live OpenSearch test; needs `OPENSEARCH_URL`) |
| Test layers | unit, Hypothesis property, integration (incl. Qdrant in-memory, FastAPI), search-quality, CPU end-to-end (demo, regression-gate failure on a bad candidate, CLI) |
| Coverage | **87%** line+branch (`pytest --cov=searchforge`) |
| Ruff (lint + format) | clean |
| mypy `--strict` | clean (110 source files) |
| Package build | wheel + sdist built with `uv build` |
| Secret scan | detect-secrets over all git-tracked files: no findings |
| Git authors | sole author verified over the whole history; no co-author / generated-by trailers |
| Large files | none over 1 MB in history (largest: `uv.lock`) |
| GitHub Actions | none (`.github/workflows/` absent) |
| Jenkins | Every `sh` step of the `Jenkinsfile` was executed in Git Bash on this Windows machine (lockfile check with `UV_FROZEN` unset, sync, doctor, format, Ruff, mypy, all test stages, local index E2E smoke with its assertion, compare gate, demo, security scan, package build); the Docker stage takes its documented skip path because no daemon was running. **The pipeline itself has not run on a Jenkins server** (no Jenkins/Java here). |
| Docker | `Dockerfile` and `docker-compose.yml` written; `docker compose config` validates; **image build not verified** (Docker daemon not running here, ~1.6 GB RAM free during the session) |
| Local CPU demo | `searchforge demo` and `searchforge demo adaptive` complete offline; quality gates for B4 vs B0 pass |
| First public study | BEIR SciFact test (300 queries), CPU nDCG@10: BM25 0.687, TF-IDF+LSA dense 0.467, hybrid RRF 0.594, hybrid+rerank 0.622, adaptive 0.607 |

## Local CPU demo results

See [README → Results](README.md#results) and [results/local-demo/report.md](results/local-demo/report.md).
They are labelled **LOCAL DEMONSTRATION RESULTS** and are not benchmarks.

## External backend status

| Backend | Status |
|---|---|
| Qdrant | Adapter implemented against qdrant-client 1.20 `query_points`; exercised in-memory in tests and via `configs/indexes/fixture-qdrant.yaml`; server profile in docker-compose (not run here). Incremental updates of Qdrant-backed indexes require a rebuild. |
| OpenSearch | Lexical (BM25) and k-NN adapters implemented; query DSL unit-tested; live test skipped (no cluster). Not selectable through the local builder config in v0.1 (adapters are used programmatically). |
| pgvector | Documented in docs/references only; no adapter. |
| Sentence Transformers / cross-encoders | Implemented, tested with injected fake models; no model downloaded or measured. |

## Known limitations

- The fixture collection is small and synthetic; results there are demonstrations only.
- Dense retrieval in all committed results is TF-IDF + LSA. On SciFact it is much weaker than BM25 and drags hybrid fusion down; neural embeddings are the obvious next experiment.
- Exact NumPy search targets up to ~hundreds of thousands of chunks; larger corpora need an ANN backend.
- The `hash_only` query capture is an unsalted hash; short queries can be guessed. Use `none` for strict privacy.
- OpenTelemetry spans for retrievers run in worker threads and are not parented to the request span.
- The API has no authentication; it binds to 127.0.0.1 by default (`serve`).
- Latency is in-process and noisy at sub-10 ms; no significance testing is done.

## Exact next real-scale experiment

```bash
uv sync --extra dev --extra neural
uv run searchforge datasets fetch-beir scifact --out data/beir
# set dense.provider: sentence_transformer, dense.model: BAAI/bge-small-en-v1.5 (pin revision),
#     dense.allow_download: true (once), index.name: scifact-bge in a copy of configs/evaluations/scifact.yaml
uv run searchforge benchmark configs/benchmarks/scifact.yaml --out results/beir-scifact-bge
```

Systems: BM25 vs BGE-small dense vs hybrid RRF vs hybrid + `cross-encoder/ms-marco-MiniLM-L6-v2`
(top-50) vs adaptive, on the 300 SciFact test queries, reporting nDCG@10, Recall@20, MRR, p50/p95
and peak RSS. Then the same on NFCorpus, with fusion weights tuned on the BEIR dev split.
