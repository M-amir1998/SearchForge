# Jenkins CI

SearchForge uses Jenkins as its only CI system ([ADR-008](../adr/ADR-008-jenkins-instead-of-github-actions.md)).
The pipeline is defined in the [`Jenkinsfile`](../../Jenkinsfile) at the repository root. GitHub
Actions is intentionally not used. The pipeline fails if a `.github/workflows` directory exists.

The pipeline is CPU only. It needs no GPU, CUDA, API keys, remote search service, paid embedding
API or model download, and it never installs the `neural` extra.

## Agent requirements

| Requirement | Details |
|---|---|
| Label | `linux` (`agent { label 'linux' }`) |
| Shell tools | `bash`, `git`, `curl`, `nproc`; `free` and `df` are optional (failures ignored) |
| uv | Used from `PATH` if present. If missing, the `Environment Validation` stage installs uv `0.11.21` into `${WORKSPACE}/.local/bin` from `https://astral.sh/uv/0.11.21/install.sh`. A uv already on `PATH` is used as is; its version is printed but not enforced. |
| Python | 3.12 (`UV_PYTHON=3.12`). If no 3.12 interpreter is found, uv downloads a managed one, which requires network access to the uv Python mirror. |
| Network | Outbound HTTPS to GitHub (checkout), astral.sh (uv bootstrap only) and PyPI (dependency install). The optional Docker build also pulls `python:3.12-slim` from Docker Hub and the uv image from `ghcr.io`. No model hubs. |
| Docker | Optional. CLI and a reachable daemon for the `Docker Build` stage. The stage is skipped with a message if `docker info` fails. |
| Memory | The reference agent has 16 GB RAM; see [Resource limits](#resource-limits). |

Jenkins plugins used by the Jenkinsfile:

- Pipeline, including Declarative Pipeline;
- Git;
- JUnit (`junit` step);
- Timestamper (`timestamps()` option).

For webhooks you also need the GitHub plugin, and for multibranch jobs the GitHub Branch Source
plugin.

## Parameters

| Parameter | Default | Effect |
|---|---|---|
| `RUN_DOCKER_BUILD` | `true` | Runs the `Docker Build` stage, which still skips itself when Docker is not available. |
| `RUN_E2E` | `true` | Runs the `Retrieval Regression Test` stage: e2e tests, the regression gate and the demo. The `Local Index E2E` stage always runs, whatever this value. |

Jenkins learns the parameters from the Jenkinsfile, so they appear in the UI ("Build with
Parameters") only after the first build of a job.

## Environment

| Variable | Value | Purpose |
|---|---|---|
| `UV_VERSION` | `0.11.21` | uv version installed by the bootstrap |
| `UV_CACHE_DIR` | `${WORKSPACE}/.cache/uv` | Per-workspace uv cache, reused across builds on the same agent |
| `UV_PYTHON` | `3.12` | Interpreter version for `uv sync` / `uv run` |
| `UV_FROZEN` | `1` | Install exactly from `uv.lock`, never re-resolve (see [Known issues](#known-issues)) |
| `HF_HUB_OFFLINE` | `1` | Hugging Face Hub never contacted |
| `TRANSFORMERS_OFFLINE` | `1` | Same for `transformers` |
| `PYTHONDONTWRITEBYTECODE` | `1` | No `.pyc` files in the workspace |
| `SF_WORKSPACE` | `${WORKSPACE}/.searchforge-ci` | Workspace for the demo run |
| `PATH` | `${WORKSPACE}/.local/bin:$PATH` | Makes a bootstrapped uv visible |

## Stages

| # | Stage | What it runs |
|---|---|---|
| 1 | Checkout | `checkout scm` (default checkout is skipped and done here), then prints the last commit. |
| 2 | Environment Validation | Bootstraps uv if missing. Prints the uv version, CPU count, memory and disk. Reports (does not use) a GPU if `nvidia-smi` exists. Fails if `.github/workflows` exists. |
| 3 | Lockfile Validation | `env -u UV_FROZEN uv lock --check` (with `UV_FROZEN=1` the check would only validate syntax, not freshness), then `uv sync --frozen --extra dev --extra qdrant`, then `uv run searchforge doctor` (fails if SQLite FTS5 is missing). |
| 4 | Format Check | `uv run ruff format --check src tests` |
| 5 | Ruff | `uv run ruff check src tests` |
| 6 | Strict mypy | `uv run mypy` (strict mode, configured in `pyproject.toml`) |
| 7 | Unit Tests | `pytest tests/unit` with coverage; JUnit to `reports/junit-unit.xml` |
| 8 | Property Tests | `pytest tests/property` (Hypothesis); coverage appended |
| 9 | Integration Tests | `pytest tests/integration`. Qdrant runs in-memory in the test process. Live OpenSearch tests skip unless `OPENSEARCH_URL` is set, and it is not set in CI. |
| 10 | Search Quality Tests | `pytest tests/quality`; writes `reports/coverage.xml` covering stages 7-10 |
| 11 | Local Index E2E | Builds the fixture index (`searchforge index build configs/evaluations/fixture.yaml --force`). Validates checksums and counts (`searchforge index validate fixture`). Runs `searchforge search "ERR-4012" --profile adaptive --top-k 3 --json` and asserts that the top result is document `ERR-4012`. |
| 12 | Retrieval Regression Test (`RUN_E2E`) | `pytest tests/e2e`. `searchforge compare lexical adaptive --split test --out reports/regression-gate.json` exits 1 if the configured quality gates fail; this is the step that fails the build on a retrieval regression. `searchforge demo --out reports/demo` runs the full CPU demo and prints its own gate result, but does not change the exit code. Its `metrics.json` is copied to `reports/benchmark-summary.json`. |
| 13 | Security Scan | `detect-secrets scan` over git-tracked files (excluding `uv.lock`); any finding fails the build. Also fails if tracked files match `*.npy`, `*.sqlite`, `*.faiss`, `*.safetensors`, `*.pt`, `*.onnx` or `.env`. |
| 14 | Package Build | `uv build -o dist` (wheel and sdist) |
| 15 | Docker Build (`RUN_DOCKER_BUILD`) | If Docker is usable: `docker build --pull -t searchforge:ci-<build>`, then `docker run --rm --memory 1g --cpus 2 searchforge:ci-<build> doctor`, then removes the image. |

Post actions:

- `always` publishes `reports/junit-*.xml` and archives the artifacts listed below.
- `Checkout` empties `reports/` and `dist/` first, so a previous build's reports can never be archived as this build's.
- `cleanup` deletes `.searchforge`, `.searchforge-ci` and `reports/demo` from the workspace.

## Setting up the job

Both job types run the Jenkinsfile from SCM. The pipeline uses `checkout scm`, so an inline
pipeline script will not work.

### Option A: Multibranch Pipeline (recommended)

1. **New Item -> Multibranch Pipeline.**
2. **Branch Sources -> GitHub.**
   - Repository HTTPS URL: `https://github.com/M-amir1998/SearchForge-`.
   - Credentials: none are needed to clone a public repository. A read-only GitHub credential
     raises the GitHub API rate limit used for branch scanning (see [Credentials](#credentials)).
3. **Build Configuration:** by Jenkinsfile, Script Path `Jenkinsfile`.
4. **Behaviours:** discover branches. Discover pull requests only if you have decided how to treat
   untrusted fork code, because PR builds execute that code on the agent.
5. Save. Jenkins scans the repository and runs the first build per branch, which also registers
   the parameters.

### Option B: Pipeline job

1. **New Item -> Pipeline.**
2. **Pipeline -> Definition:** "Pipeline script from SCM".
   - SCM: Git.
   - Repository URL: `https://github.com/M-amir1998/SearchForge-.git`.
   - Branch Specifier: `*/main`.
   - Script Path: `Jenkinsfile`.
3. **Build Triggers:** tick "GitHub hook trigger for GITScm polling". The Jenkinsfile has no
   `triggers {}` block, so the trigger must be configured on the job.
4. **Project url** (GitHub project property): `https://github.com/M-amir1998/SearchForge-/`.
5. Run **Build Now** once so Jenkins reads the parameters.

## GitHub webhook

1. Make sure the GitHub plugin is installed. It provides the `/github-webhook/` endpoint, which
   multibranch jobs also use.
2. In the GitHub repository, open **Settings -> Webhooks -> Add webhook** and set:
   - **Payload URL:** `https://<jenkins>/github-webhook/`. The trailing slash is required.
   - **Content type:** `application/json`.
   - **Secret:** recommended. Store the same value as a Jenkins "Secret text" credential and
     select it as the webhook shared secret in the GitHub plugin's section of the Jenkins system
     configuration. The exact location depends on the plugin version.
   - **Events:** "Just the push event". Add "Pull requests" only for multibranch jobs that build
     PRs.
   - **Active:** checked.
3. Check **Recent Deliveries** in GitHub after a push. Jenkins must be reachable from GitHub over
   HTTPS. If it is on a private network, use a relay or reverse proxy rather than exposing the
   whole controller.

The Jenkinsfile does not publish commit statuses to GitHub. Status reporting, if wanted, comes from
the GitHub Branch Source plugin (multibranch) or a separate plugin.

## Manual and parameterised builds

- **Build Now** (first build) or **Build with Parameters** (later builds). Untick `RUN_DOCKER_BUILD`
  on agents without Docker or to save time. Untick `RUN_E2E` for a faster run without the
  regression gate and demo.
- **Replay** can re-run a build with an edited Jenkinsfile for debugging. Commit the real change
  afterwards.

## Optional Docker tests

The `Docker Build` stage needs the `docker` CLI on the agent and a daemon the Jenkins user can
reach. Membership of the `docker` group is effectively root on the agent, so grant it only on a
dedicated CI host. The stage:

1. builds the image from the repository [`Dockerfile`](../../Dockerfile) with `--pull`. This is a
   multi-stage build on `python:3.12-slim`, running as non-root user `searchforge` (uid 10001);
2. runs `searchforge doctor` in a container limited to `--memory 1g --cpus 2`;
3. deletes the image `searchforge:ci-<BUILD_NUMBER>`.

If Docker is missing or the daemon is unreachable, the stage prints a message and succeeds. The
Docker Compose services (Qdrant, OpenSearch, Prometheus, Grafana) are never started by CI.

## Credentials

- **Checkout:** none needed for the public repository. If the repository becomes private, create a
  Jenkins credential (GitHub App, or username plus a fine-grained read-only token) in the Jenkins
  credentials store and select it in the job's SCM configuration.
- **Never put credentials in the Jenkinsfile,** in configuration files or in job parameters. If a
  future stage needs one, bind it from the credentials store with `withCredentials`.
- **No API keys are used in CI.** No embedding API, no model hub token, no Qdrant or OpenSearch
  credentials. Tests that would need them are skipped.

## Resource limits

- `disableConcurrentBuilds()`: one build of a job at a time.
- `timeout(time: 45, unit: 'MINUTES')`: the whole build is aborted after 45 minutes.
- On a 16 GB host, configure **one executor** on the agent, so that builds of different jobs or
  branches do not run test suites, index builds and a Docker build at the same time.
  `disableConcurrentBuilds()` applies per job (per branch in a multibranch job), not per agent.
- The uv cache (`.cache/uv`) and the virtual environment (`.venv`) stay in the workspace between
  builds to avoid re-downloading packages. Delete the workspace to reclaim disk.

## Artifact retention

`buildDiscarder(logRotator(numToKeepStr: '30', artifactNumToKeepStr: '10'))` keeps the last 30 build
records and the artifacts of the last 10 builds.

Archived, with fingerprints:

- `reports/junit-*.xml` - test results, also published with the `junit` step;
- `reports/coverage.xml` - Cobertura XML coverage report;
- `reports/benchmark-summary.json` - demo benchmark metrics on the synthetic fixture. It also
  records host details: Python version, platform, CPU, RAM;
- `reports/regression-gate.json` - comparison summary and gate checks;
- `dist/*.whl`, `dist/*.tar.gz` - the built packages.

Never archived: index versions, embedding caches, corpora, vector database data, `.env` files or
model files. The `cleanup` step deletes the CI index workspaces.

## Local equivalent: `scripts/pre_push_check.sh`

[`scripts/pre_push_check.sh`](../../scripts/pre_push_check.sh) (`--fast` skips e2e tests) is the
closest local equivalent. It is not identical to the pipeline.

| Check | Jenkins | Pre-push script |
|---|---|---|
| ruff lint, ruff format, strict mypy | yes | yes |
| pytest | per directory, with coverage and JUnit | `pytest -q` (or `-m "not e2e"` with `--fast`) |
| `uv build` | yes | yes (to a temporary directory) |
| detect-secrets over tracked files | yes | yes |
| No `.github/workflows` | yes | yes |
| `uv lock --check`, `searchforge doctor` | yes | no |
| Index build, validate and CLI search smoke test | yes | no |
| `searchforge compare` regression gate and demo | yes | no |
| Tracked index or model file check | yes | no (relies on `.gitignore` and the pre-commit large-file hook) |
| Docker build | optional | no |
| Commit author is the configured `git config user.name` / `user.email`, over the whole history | no | yes |
| No `Co-authored-by`, `Generated-by` or `AI-assisted` trailers in history | no | yes |
| No blobs larger than 1 MB in history | no | yes |

To reproduce the regression gate locally, after `uv sync --extra dev --extra qdrant`:

```bash
uv run searchforge index build configs/evaluations/fixture.yaml --force
uv run searchforge compare lexical adaptive --config configs/evaluations/fixture.yaml --split test
```

## Known issues

- **`UV_FROZEN=1` weakens `uv lock --check`.** With `UV_FROZEN=1` in the environment, uv
  (observed with 0.11.21) only checks that `uv.lock` is valid, not that it matches
  `pyproject.toml`, and prints a warning saying so. A stale lockfile therefore passes the
  `Lockfile Validation` stage. Running that one command with `UV_FROZEN` unset restores the
  intended check.
- **Stale archived reports.** The workspace is not wiped before checkout, and `reports/` is not
  cleaned at the start of a build. `archiveArtifacts` does not check file age. For example, a
  build with `RUN_E2E` unticked archives the previous build's `regression-gate.json` and
  `benchmark-summary.json` as if they were its own. An early failure can likewise re-archive an
  older `coverage.xml`. Use "Wipe out repository and force clone" or a clean step if this matters.
- **Docker image left behind on failure.** If `docker run ... doctor` fails, `set -e` stops the
  script before `docker image rm`, and `searchforge:ci-<n>` stays on the agent. Prune images
  periodically.
- **Pinned uv not enforced.** A uv already on the agent's `PATH` is used even if its version
  differs from `UV_VERSION`.
- **`.secrets.baseline` not used.** The Security Scan stage counts every detect-secrets finding.
  The baseline used by pre-commit is not applied, so an allow-listed finding still fails CI.
