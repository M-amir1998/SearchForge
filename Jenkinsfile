// SearchForge CI — the official pipeline (GitHub Actions is intentionally not used).
//
// CPU only: no GPU, no CUDA, no external API keys, no remote search service, no paid
// embedding API, no model downloads (HF_HUB_OFFLINE=1). The neural extra is never installed.
// Builds are serialized (disableConcurrentBuilds) because the reference agent has 16 GB RAM.
//
// Agent requirements: Linux, bash, git, curl, Python 3.12 (or let uv download it), optional docker.
// See docs/ci/jenkins.md for setup, webhook, credentials and retention.

pipeline {
  agent { label 'linux' }

  options {
    disableConcurrentBuilds()
    timeout(time: 45, unit: 'MINUTES')
    buildDiscarder(logRotator(numToKeepStr: '30', artifactNumToKeepStr: '10'))
    timestamps()
    skipDefaultCheckout(true)
  }

  parameters {
    booleanParam(name: 'RUN_DOCKER_BUILD', defaultValue: true, description: 'Build the Docker image when docker is available')
    booleanParam(name: 'RUN_E2E', defaultValue: true, description: 'Run CPU end-to-end tests (demo, regression gate)')
  }

  environment {
    UV_VERSION = '0.11.21'
    UV_CACHE_DIR = "${WORKSPACE}/.cache/uv"
    UV_PYTHON = '3.12'
    UV_FROZEN = '1'
    HF_HUB_OFFLINE = '1'
    TRANSFORMERS_OFFLINE = '1'
    PYTHONDONTWRITEBYTECODE = '1'
    SF_WORKSPACE = "${WORKSPACE}/.searchforge-ci"
    PATH = "${WORKSPACE}/.local/bin:${PATH}"
  }

  stages {
    stage('Checkout') {
      steps {
        checkout scm
        sh 'git log -1 --format="%h %an %s"'
      }
    }

    stage('Environment Validation') {
      steps {
        sh '''#!/usr/bin/env bash
          set -euo pipefail
          if ! command -v uv >/dev/null 2>&1; then
            curl -LsSf "https://astral.sh/uv/${UV_VERSION}/install.sh" | env UV_INSTALL_DIR="${WORKSPACE}/.local/bin" sh
          fi
          uv --version
          echo "CPUs: $(nproc)"; free -g || true; df -h . || true
          if command -v nvidia-smi >/dev/null 2>&1; then echo "GPU present but NOT used by this pipeline"; fi
          test ! -d .github/workflows || { echo "active GitHub Actions workflows are not allowed"; exit 1; }
        '''
      }
    }

    stage('Lockfile Validation') {
      steps {
        sh 'uv lock --check'
        sh 'uv sync --frozen --extra dev --extra qdrant'
        sh 'uv run searchforge doctor'
      }
    }

    stage('Format Check') {
      steps { sh 'uv run ruff format --check src tests' }
    }

    stage('Ruff') {
      steps { sh 'uv run ruff check src tests' }
    }

    stage('Strict mypy') {
      steps { sh 'uv run mypy' }
    }

    stage('Unit Tests') {
      steps {
        sh 'mkdir -p reports'
        sh 'uv run pytest tests/unit -q --junitxml=reports/junit-unit.xml --cov=searchforge --cov-report= '
      }
    }

    stage('Property Tests') {
      steps {
        sh 'uv run pytest tests/property -q --junitxml=reports/junit-property.xml --cov=searchforge --cov-append --cov-report='
      }
    }

    stage('Integration Tests') {
      steps {
        // Qdrant runs in-memory inside the test process; live OpenSearch tests skip without OPENSEARCH_URL.
        sh 'uv run pytest tests/integration -q --junitxml=reports/junit-integration.xml --cov=searchforge --cov-append --cov-report='
      }
    }

    stage('Search Quality Tests') {
      steps {
        sh 'uv run pytest tests/quality -q --junitxml=reports/junit-quality.xml --cov=searchforge --cov-append --cov-report=xml:reports/coverage.xml'
      }
    }

    stage('Local Index E2E') {
      steps {
        sh '''#!/usr/bin/env bash
          set -euo pipefail
          cfg=configs/evaluations/fixture.yaml
          uv run searchforge index build "$cfg" --force
          uv run searchforge index validate fixture --config "$cfg"
          uv run searchforge search "ERR-4012" --config "$cfg" --profile adaptive --top-k 3 --json > reports/search-smoke.json
          uv run python -c "import json; r=json.load(open('reports/search-smoke.json'))['results']; assert r[0]['document_id']=='ERR-4012', r[0]"
        '''
      }
    }

    stage('Retrieval Regression Test') {
      when { expression { return params.RUN_E2E } }
      steps {
        sh 'uv run pytest tests/e2e -q --junitxml=reports/junit-e2e.xml'
        // Quality gate: the adaptive planner must not regress BM25 on the held-out test split.
        sh 'uv run searchforge compare lexical adaptive --config configs/evaluations/fixture.yaml --split test --out reports/regression-gate.json'
        sh 'uv run searchforge demo --out reports/demo --workspace "$SF_WORKSPACE/demo"'
        sh 'cp reports/demo/metrics.json reports/benchmark-summary.json'
      }
    }

    stage('Security Scan') {
      steps {
        sh '''#!/usr/bin/env bash
          set -euo pipefail
          n=$(uv run detect-secrets scan --exclude-files 'uv\\.lock' | uv run python -c "import json,sys; print(len(json.load(sys.stdin)['results']))")
          test "$n" = "0" || { echo "detect-secrets: $n file(s) with findings"; exit 1; }
          if git ls-files | grep -E '\\.(npy|sqlite|faiss|safetensors|pt|onnx)$|(^|/)\\.env$'; then
            echo "index artifacts, model files or .env must never be committed"; exit 1
          fi
        '''
      }
    }

    stage('Package Build') {
      steps {
        sh 'rm -rf dist && uv build -o dist'
        sh 'ls -l dist'
      }
    }

    stage('Docker Build') {
      when { expression { return params.RUN_DOCKER_BUILD } }
      steps {
        sh '''#!/usr/bin/env bash
          set -euo pipefail
          if ! command -v docker >/dev/null 2>&1 || ! docker info >/dev/null 2>&1; then
            echo "docker not available on this agent; skipping image build"; exit 0
          fi
          docker build --pull -t searchforge:ci-${BUILD_NUMBER} .
          docker run --rm --memory 1g --cpus 2 searchforge:ci-${BUILD_NUMBER} doctor
          docker image rm searchforge:ci-${BUILD_NUMBER}
        '''
      }
    }
  }

  post {
    always {
      junit allowEmptyResults: true, testResults: 'reports/junit-*.xml'
      // Archive ONLY small, non-sensitive artifacts. Never indexes, embedding caches, corpora or vector DB data.
      archiveArtifacts allowEmptyArchive: true, fingerprint: true,
        artifacts: 'reports/junit-*.xml, reports/coverage.xml, reports/benchmark-summary.json, reports/regression-gate.json, dist/*.whl, dist/*.tar.gz'
    }
    cleanup {
      sh 'rm -rf .searchforge .searchforge-ci reports/demo || true'
    }
  }
}
