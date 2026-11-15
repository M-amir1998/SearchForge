#!/usr/bin/env bash
# Pre-push gate: lint, format, strict typing, tests, build, secret scan, author and large-file checks.
# Usage: scripts/pre_push_check.sh [--fast]   (--fast skips slow e2e tests)
set -euo pipefail
cd "$(dirname "$0")/.."

step() { printf '\n==> %s\n' "$*"; }

step "ruff"
uv run ruff check src tests
step "format"
uv run ruff format --check src tests
step "mypy --strict"
uv run mypy
step "pytest"
if [[ "${1:-}" == "--fast" ]]; then
  uv run pytest -q -m "not e2e"
else
  uv run pytest -q
fi
step "build"
tmp_dist="$(mktemp -d)"
uv build -q -o "$tmp_dist"
ls "$tmp_dist"
rm -rf "$tmp_dist"
step "secret scan (detect-secrets over git-tracked files, i.e. what gets published)"
findings=$(uv run detect-secrets scan --exclude-files 'uv\.lock' \
  | uv run python -c "import json,sys; print(len(json.load(sys.stdin)['results']))")
[[ "$findings" == "0" ]] || { echo "secret scan found $findings file(s) with findings"; exit 1; }
echo "no findings"
step "commit authors / trailers"
expected="$(git config user.name) <$(git config user.email)>"
bad_authors=$(git log --format='%an <%ae>' | sort -u | grep -vxF "$expected" || true)
[[ -z "$bad_authors" ]] || { echo "unexpected authors: $bad_authors"; exit 1; }
if git log --format='%B' | grep -iqE 'co-authored-by|generated-by|ai-assisted'; then
  echo "forbidden attribution trailer found"; exit 1
fi
echo "sole author: $expected"
step "large files in history (>1 MB)"
large=$(git rev-list --objects --all | git cat-file --batch-check='%(objecttype) %(objectsize) %(rest)' \
  | awk '$1=="blob" && $2>1048576 {print $3" ("$2" bytes)"}')
[[ -z "$large" ]] || { echo "large blobs: $large"; exit 1; }
echo "none"
step "no active GitHub Actions workflows"
if ls .github/workflows/*.y*ml >/dev/null 2>&1; then echo "found .github/workflows — Jenkins is the CI"; exit 1; fi
echo "none"
step "working tree"
git status --short
echo "pre-push checks passed"
