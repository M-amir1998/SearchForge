# ADR-009: CPU-first, local architecture

## Status

Accepted

## Context

The reference development and CI machine has 16 GB of RAM and no GPU. Retrieval experiments should
run on such a machine without cloud services, paid APIs, model downloads during tests, or services
that need several GB of heap just to start. Many retrieval stacks assume a GPU for embeddings and a search cluster for indexing. Under those
assumptions the project could not be run, tested or reproduced locally.

## Decision

- **Default dense baseline: TF-IDF with optional LSA** (`embeddings/tfidf.py`). It is fitted with
  scikit-learn (`TfidfVectorizer`, then `TruncatedSVD`, 256 components by default). At query time a
  fixed-vocabulary `CountVectorizer` is followed by NumPy/SciPy weighting. The fitted state is
  stored in the index version as JSON plus `.npy`. This is a classical, CPU-cheap baseline and is
  not presented as neural semantic search.
- **Neural models are optional.** Sentence Transformers embeddings and the cross-encoder reranker
  need the `neural` extra. They run with `device="cpu"` and load only locally cached checkpoints
  unless `allow_download: true` is set. CI never installs the extra and sets `HF_HUB_OFFLINE=1`.
- **Deterministic hash embeddings** (`embeddings/hashing.py`) exist only so dense code paths can
  be tested without a model. They are labelled non-semantic.
- **Local engines.** Lexical search uses SQLite FTS5 (ADR-002). Dense search is exact brute-force
  inner product on a memory-mapped `vectors.npy`, scored blockwise with filter pre-masking
  (`indexes/dense/numpy_index.py`). Exact search keeps ANN recall loss from confounding evaluation.
- **Data profiles** (`index.data_profile`, `storage/budget.py`) put explicit limits on local runs:

  | Profile    | Max chunks | Notes                                                        |
  |------------|------------|--------------------------------------------------------------|
  | `tiny`     | 5,000      | default; fixture and tests                                   |
  | `small`    | 50,000     |                                                              |
  | `research` | 1,000,000  | public benchmark subsets                                     |
  | `large`    | no limit   | refused with the local NumPy index; requires `dense.backend: qdrant` |

- **Storage budgets.** Before a full build, disk use is estimated. The build is refused if it would
  leave less than `storage.reserve_free_gb` (default 5 GB) free, or exceed `storage.max_index_gb`
  (default 20 GB).
- **Measured resources and containers.** Peak RSS is sampled during builds and evaluations and
  recorded in manifests and reports. Docker Compose starts only the API (1 GB limit) by default;
  Qdrant, OpenSearch, Prometheus and Grafana are opt-in profiles with memory limits and are not
  used by the test suite or Jenkins.

## Consequences

Positive:

- The whole pipeline runs offline on a laptop and in CI: ingestion, indexing, hybrid search,
  reranking, evaluation and reporting. It needs no GPU, credentials or downloads.
- Results are reproducible: TF-IDF fitting uses a fixed `random_state`, and every artefact is
  versioned (ADR-007).
- Profiles and budgets fail fast instead of exhausting RAM or disk mid-build.

Negative:

- TF-IDF/LSA vectors capture term co-occurrence, not learned semantics. Dense and hybrid results
  from the default configuration understate what neural embeddings can do.
- Exact NumPy search is linear in corpus size. It is suitable up to the `research` profile, beyond
  which an ANN backend is needed.
- The TF-IDF vocabulary is frozen at fit time. Incremental updates reuse the parent's fitted state,
  so new terms are invisible to the dense retriever until a full rebuild.
- Data-profile limits are enforced after ingestion (on the actual chunk count), so an oversized
  corpus is detected only after it has been chunked.

## Alternatives considered

- **GPU-first with neural embeddings by default.** Rejected: it excludes the target hardware and
  makes CI depend on model downloads.
- **ANN locally (FAISS, HNSW libraries).** Deferred: it adds native dependencies, and approximate
  recall would confound relevance measurements at the target sizes.
- **Hosted embedding APIs by default.** Rejected: cost, API keys and sending corpus text to third
  parties. Supported only as an explicit opt-in (ADR-003, [SECURITY.md](../../SECURITY.md)).
