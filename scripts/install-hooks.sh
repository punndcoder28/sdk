#!/usr/bin/env bash
# Install git hooks for this repo so they run for every developer (no agent required).
#
# Uses the `pre-commit` framework (Python equivalent of Husky for JS repos):
#   - pre-commit:  ruff lint/format + conflict marker scan
#   - commit-msg:  conventional commits + required body
#   - pre-push:    merge/conflict check vs parent/base branch
set -euo pipefail

ROOT=$(cd "$(dirname "$0")/.." && pwd)
cd "$ROOT"

if ! command -v pre-commit >/dev/null 2>&1 && ! command -v uv >/dev/null 2>&1; then
  echo "install-hooks: need 'pre-commit' or 'uv' on PATH" >&2
  echo "  uv sync --extra dev   # then re-run this script" >&2
  echo "  # or: pip install pre-commit ruff" >&2
  exit 1
fi

run_pc() {
  if command -v uv >/dev/null 2>&1 && [ -f "$ROOT/pyproject.toml" ]; then
    uv run pre-commit "$@"
  else
    pre-commit "$@"
  fi
}

echo "install-hooks: installing pre-commit + commit-msg + pre-push hooks…"
run_pc install --hook-type pre-commit
run_pc install --hook-type commit-msg
run_pc install --hook-type pre-push

echo ""
echo "install-hooks: done. Hooks will run on every commit/push in this clone."
echo ""
echo "  Manual full run:  uv run pre-commit run --all-files"
echo "  Stacked PR push:  GITHOOK_PARENT_BRANCH=<parent-branch> git push"
echo "  Example:          GITHOOK_PARENT_BRANCH=feat/foo-01-schema git push -u origin HEAD"
