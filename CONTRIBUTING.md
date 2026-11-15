# Contributing to SearchForge

Thank you for helping. This guide covers the development setup, the checks every change has to
pass, the coding standards, and the project rules. Architecture decisions are recorded in
[docs/adr/](docs/adr/README.md). Security-relevant changes must also follow [SECURITY.md](SECURITY.md).

## Development setup

Requirements:

- [uv](https://docs.astral.sh/uv/). CI pins `0.11.21`.
- Python 3.12. The repository's `.python-version` pins it, and uv can install it for you. The
  package supports Python 3.11 or newer.
- `git`.
- SQLite with FTS5. This is standard in CPython builds; `searchforge doctor` checks it.

No GPU, model downloads, API keys or running services are needed for development or tests.

```bash
uv sync --extra dev --extra qdrant     # dev tools + in-memory Qdrant for integration tests
uv run pre-commit install              # ruff, detect-secrets, large-file and private-key hooks
uv run searchforge doctor              # environment report (CPU-only is the normal setup)
```

Optional extras, none of which the default test suite needs:

| Extra | Adds | Notes |
|---|---|---|
| `neural` | `sentence-transformers` | CPU only. Models load only from the local cache unless `allow_download: true`. |
| `opensearch` | `opensearch-py` | The OpenSearch adapters. Live tests need a running OpenSearch. |
| `observability` | OpenTelemetry API/SDK | Enables tracing spans; without it they are no-ops. |

`uv.lock` is committed. If you change dependencies in `pyproject.toml`, run `uv lock` and commit
both files.

## Running checks

Run the pre-push gate before every push:

```bash
scripts/pre_push_check.sh          # full run
scripts/pre_push_check.sh --fast   # skips e2e tests
```

It runs these checks in order:

1. ruff lint;
2. ruff format check;
3. `mypy` in strict mode;
4. pytest;
5. `uv build`;
6. `detect-secrets` over tracked files;
7. checks that every commit in the history is authored by your configured
   `git config user.name`/`user.email`;
8. checks that no commit message contains `Co-authored-by`, `Generated-by` or `AI-assisted`
   trailers;
9. checks that no blob in the history is larger than 1 MB;
10. checks that there are no `.github/workflows` files.

The author check runs over the whole history. It fails on any clone that contains commits by
someone other than the configured user, which today means only the repository owner can pass the
full script. Contributors can run the individual commands below until the check is scoped to new
commits.

The Jenkins pipeline runs more than this: lockfile check, index build and validation, CLI smoke
search, regression gate, demo and Docker build. See [docs/ci/jenkins.md](docs/ci/jenkins.md).

Individual commands:

```bash
uv run ruff check src tests
uv run ruff format --check src tests     # `uv run ruff format src tests` to fix
uv run mypy                              # strict mode, configured in pyproject.toml
uv run pytest                            # all tests (markers are strict)
uv run pytest -m unit                    # one category
uv run pytest -m "not e2e"               # skip the slower end-to-end flows
```

Test markers, declared in `pyproject.toml` (`--strict-markers` rejects undeclared ones):

| Marker | Meaning | Directory |
|---|---|---|
| `unit` | Fast, isolated tests | `tests/unit` |
| `property` | Hypothesis property tests (fusion, metrics, filter parsing, FTS5 expressions, chunk offsets, ...) | `tests/property` |
| `integration` | Multi-component tests on local backends, the API and adapters | `tests/integration` |
| `quality` | Search-quality assertions on the fixture collection | `tests/quality` |
| `e2e` | CPU end-to-end flows (demo, regression gate) | `tests/e2e` |
| `regression` | Retrieval regression and quality-gate tests | declared; not used yet |
| `qdrant` | Needs `qdrant-client`; runs in-memory, no server | `tests/integration` |
| `opensearch` | Needs a live OpenSearch at `OPENSEARCH_URL`; skipped otherwise | `tests/integration` |
| `neural` | Needs `sentence-transformers` and locally cached models | declared; not used yet |

Tests must run offline on CPU. `tests/conftest.py` sets `HF_HUB_OFFLINE=1`. Never add a test that
downloads a model or dataset, or that needs a running service, without a marker and a skip
condition.

## Coding standards

- **Typed.** All code passes `mypy --strict` (`warn_unreachable` on, Pydantic plugin). Use precise
  types. `Any` is acceptable only at third-party boundaries, such as optional backend clients.
- **Small, focused modules.** One responsibility per module, named after what it does
  (`fusion/strategies.py`, `ingestion/safety.py`). Do not create `utils.py`, `helpers.py` or
  `common.py` dumping grounds. Put a function in the package whose concept it belongs to.
- **Ports and adapters** ([ADR-001](docs/adr/ADR-001-engine-independent-retrieval.md)):
  - Application services depend only on the protocols in `src/searchforge/ports.py` and on
    `domain/` types.
  - Backend libraries (Qdrant, OpenSearch, Sentence Transformers) are imported lazily, inside
    their adapter modules only.
  - FastAPI objects stay in `api/`, and Typer and Rich objects stay in `cli/`.
- **Specific exceptions.** Raise the most specific class from
  [`src/searchforge/errors.py`](src/searchforge/errors.py). All of them derive from
  `SearchForgeError`, for example `InvalidFilter`, `IndexIntegrityError`, `RetrievalFailure` or
  `EmbeddingFailure`. Do not raise or catch bare `Exception`. The exception is an adapter
  boundary, which must translate backend errors into a SearchForge error such as
  `RetrievalFailure`. Add a new error class only when callers need to handle it differently.
- **Configuration.** New settings go into the Pydantic models in `config/models.py`. They are
  frozen, reject unknown keys, and are validated with bounds. Sizes, thresholds and limits are
  configuration, not constants buried in code.
- **Imports and style.** Absolute imports only (relative imports are banned by ruff), a 120-column
  line length and double quotes. ruff's `S` (bandit) rules are on. The `S608` per-file ignores
  exist only for SQL that interpolates `?` placeholder lists and code-defined identifiers.
- **Safe query construction.** Never build SQL, FTS5, Query DSL or filter syntax by concatenating
  user input. Use bound parameters and structured queries ([SECURITY.md](SECURITY.md)).
- **Determinism.** Seed randomness and break ties explicitly, for example by id, so runs are
  reproducible.

## Commit conventions

Use [Conventional Commits](https://www.conventionalcommits.org/):

```text
<type>(<optional scope>): <imperative summary>
```

Types to use:

- `feat`, `fix`, `perf`, `refactor`;
- `test`, `docs`;
- `build`, `ci`, `chore`.

Examples:

```text
feat(cli): add searchforge command-line interface
test: add unit coverage for metrics, fusion, filters, corpus, indexes and ranking
ci: add Jenkins search pipeline and pre-commit hooks
```

Keep commits focused. A behaviour change and its tests belong in the same commit.

## Project rules

- **Never commit data, indexes, caches or secrets.** `.gitignore` excludes the following:
  - workspaces: `.searchforge/`, `*.sqlite`, `*.npy`, `*.npz`, `*.faiss`;
  - model files: `*.safetensors`, `*.pt`, `*.onnx`;
  - vector database volumes;
  - `/data/` and downloaded datasets;
  - `.env*`, keys and credential files.

  pre-commit and Jenkins enforce this (detect-secrets, private-key detection, a large-file limit
  of 1 MB, and a tracked-artifact check). Datasets are downloaded explicitly at run time and keep
  their upstream licences.
- **No GitHub Actions workflows.** Jenkins is the CI
  ([ADR-008](docs/adr/ADR-008-jenkins-instead-of-github-actions.md)). Do not add
  `.github/workflows/`; both the pre-push script and the pipeline fail if it exists.
- **Results are only ever measured.** Never hand-write a metric, latency or memory figure into
  documentation, reports or code. Reports are generated from runs (`searchforge benchmark`,
  `searchforge report`, `searchforge demo`). Results on the synthetic fixture are labelled as
  demonstrations, not benchmarks. Tune on train and validation splits only, and report the test
  split once.
- **Privacy by default.** Do not log or persist raw query text. Do not add Prometheus labels with
  unbounded values
  ([ADR-010](docs/adr/ADR-010-raw-queries-not-persisted-by-default.md)).
- **CPU first.** Core features must work without a GPU, network services or model downloads
  ([ADR-009](docs/adr/ADR-009-cpu-first-local-architecture.md)). Neural and remote features stay
  optional extras.

## Adding a new backend adapter

1. **Implement the protocol.** Implement `LexicalIndex` or `VectorIndex` from
   [`src/searchforge/ports.py`](src/searchforge/ports.py) in a new module. Use
   `src/searchforge/indexes/external/` for service-backed engines and `indexes/lexical/` or
   `indexes/dense/` for embedded ones.
   - Set a `backend` name.
   - Return `Candidate` objects with 1-based `rank` and a larger-is-better `score`. Negate
     lower-is-better engine scores.
   - Do not apply engine-side fusion. Fusion stays client-side
     ([ADR-004](docs/adr/ADR-004-rrf-default-hybrid-fusion.md)).
2. **Translate canonical filters.** Add a translator to
   [`src/searchforge/filtering/translate.py`](src/searchforge/filtering/translate.py). It must
   produce structured or parameterised backend queries and never strings containing user values.
   Implement every `FilterOp` with the canonical semantics: conditions are AND-ed, and a missing
   field fails every operator except `ne`/`nin`.
3. **Handle errors at the boundary.** Wrap backend exceptions in `RetrievalFailure`, without
   leaking credentials or response bodies into messages.
4. **Add the dependency as an optional extra.** Add it to `[project.optional-dependencies]` and
   import it lazily in the adapter. Add a mypy `ignore_missing_imports` override if the library
   ships no type hints.
5. **Add configuration.** Add a `StrictModel` settings class in `config/models.py`.
   - Secrets are referenced only by environment-variable name (`*_env` fields), never as values.
   - Extend the `backend` `Literal` in `LexicalConfig` or `DenseConfig`.
6. **Wire it in.** Update these places:
   - `indexes/builder.py` (`_open_writers` / `_vector_index_for_write`);
   - `indexes/handle.py` (`open_index` / `_open_vectors`);
   - any local state needed to reopen the index (as `dense/qdrant.json` does for Qdrant), stored
     in the version directory so the manifest checksums it.

   Either support incremental updates or reject them with `IndexVersionMismatch`.
7. **Test it.**
   - Unit tests for the filter translation.
   - Integration tests against an in-memory mode or a fake client.
   - Live tests behind a new marker (declared in `pyproject.toml`) that skip when the service is
     not configured. The default test run and Jenkins must not need the service.
8. **Document it.** Update [SECURITY.md](SECURITY.md) (authentication, TLS, data sent to the
   service). For a significant decision, add an ADR.
