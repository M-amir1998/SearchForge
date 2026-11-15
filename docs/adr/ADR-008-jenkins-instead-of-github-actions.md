# ADR-008: Jenkins instead of GitHub Actions

## Status

Accepted

## Context

The repository is hosted on GitHub, but no GitHub Actions minutes are budgeted for it. CI still
has to run the full CPU suite on every push:

- lint, strict typing and tests;
- an index build and validation;
- a retrieval-quality regression gate;
- package build and optional Docker build.

The CI host should be the same kind of machine the project targets: CPU only, 16 GB RAM
(ADR-009). It must not need cloud credentials or paid APIs.

## Decision

- The [`Jenkinsfile`](../../Jenkinsfile) at the repository root is the only CI definition. Builds
  run on a self-hosted Jenkins agent with the label `linux`, triggered by a GitHub webhook or
  started manually. Setup is described in [docs/ci/jenkins.md](../ci/jenkins.md).
- No `.github/workflows/` directory is kept. Both the Jenkins `Environment Validation` stage and
  `scripts/pre_push_check.sh` fail if `.github/workflows` exists (Jenkins) or contains YAML files
  (pre-push script).
- The pipeline is CPU only and offline with respect to models:
  - `HF_HUB_OFFLINE=1` and `TRANSFORMERS_OFFLINE=1` are set;
  - the `neural` extra is never installed;
  - Qdrant runs in-memory inside the test process;
  - no API keys or credentials are used.
- `disableConcurrentBuilds()` and a 45-minute timeout limit resource use on a 16 GB agent.
- Developers run `scripts/pre_push_check.sh` locally before pushing. It is the closest local
  equivalent of the pipeline, not an identical one.

## Consequences

Positive:

- CI costs no hosted minutes, and the hardware is under the project's control.
- Latency and memory behaviour in CI matches the target environment, because the agent is the same
  class of machine.
- One pipeline definition lives in the repository and is reviewed like code.

Negative:

- Someone has to operate the Jenkins controller and agent: upgrades, plugins, disk cleanup and
  security hardening. That is more work than a hosted service.
- Pull requests from forks are not built automatically unless the Jenkins job is configured for
  that (for example a multibranch job with a GitHub branch source). Builds from untrusted forks
  run code on the agent and need a deliberate policy.
- GitHub shows build status only if Jenkins is configured to report it. The Jenkinsfile does not
  publish commit statuses itself.
- The webhook needs the Jenkins URL to be reachable from GitHub, or a relay.
- Serialised builds queue under load.

## Alternatives considered

- **GitHub Actions.** Rejected: no Actions minutes are available. The project also wants CI on the
  same hardware class it targets.
- **Other hosted CI (GitLab CI, CircleCI, Buildkite hosted agents).** Rejected for the same cost
  reasons and because of an extra third-party account.
- **Local pre-push checks only.** Rejected as the sole gate: they depend on each developer and do
  not leave archived results (JUnit, coverage, regression-gate output).
