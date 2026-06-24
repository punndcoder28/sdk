#!/usr/bin/env bash
# Verify current HEAD merges cleanly with the intended parent/base branch.
# Used by pre-push (and callable manually).
#
# Parent resolution order:
#   1. GITHOOK_PARENT_BRANCH / PRE_PUSH_PARENT_BRANCH env
#   2. origin/HEAD default branch (usually main)
#   3. main, then master
set -euo pipefail

fail() { echo "merge-check: $*" >&2; exit 1; }
warn() { echo "merge-check: warning: $*" >&2; }

default_branch() {
  if git symbolic-ref -q refs/remotes/origin/HEAD >/dev/null 2>&1; then
    git symbolic-ref refs/remotes/origin/HEAD | sed 's@^refs/remotes/origin/@@'
    return
  fi
  for b in main master; do
    if git rev-parse --verify "origin/$b" >/dev/null 2>&1; then
      echo "$b"
      return
    fi
  done
  echo "main"
}

resolve_tip() {
  local name=$1
  if git rev-parse --verify "origin/${name}" >/dev/null 2>&1; then
    echo "origin/${name}"
  elif git rev-parse --verify "${name}" >/dev/null 2>&1; then
    echo "${name}"
  else
    return 1
  fi
}

PARENT_BRANCH=${GITHOOK_PARENT_BRANCH:-${PRE_PUSH_PARENT_BRANCH:-}}
if [ -z "$PARENT_BRANCH" ]; then
  PARENT_BRANCH=$(default_branch)
fi

git fetch origin "$PARENT_BRANCH" --quiet 2>/dev/null || true

PARENT_TIP=$(resolve_tip "$PARENT_BRANCH") \
  || fail "cannot resolve parent branch '${PARENT_BRANCH}' (set GITHOOK_PARENT_BRANCH for stacked PRs, e.g. export GITHOOK_PARENT_BRANCH=feat/foo-01-schema)"

HEAD_TIP=$(git rev-parse HEAD)

# Same commit as parent => trivial pass
if [ "$HEAD_TIP" = "$(git rev-parse "$PARENT_TIP")" ]; then
  echo "merge-check: ok (HEAD == ${PARENT_TIP})"
  exit 0
fi

TMP_OUT=$(mktemp)
TMP_ERR=$(mktemp)
trap 'rm -f "$TMP_OUT" "$TMP_ERR"' EXIT

if git merge-tree --write-tree "$PARENT_TIP" "$HEAD_TIP" >"$TMP_OUT" 2>"$TMP_ERR"; then
  if grep -qE 'CONFLICT|changed in both' "$TMP_OUT" "$TMP_ERR" 2>/dev/null; then
    fail "merge conflicts with ${PARENT_TIP} (parent branch: ${PARENT_BRANCH}). Rebase/merge parent, resolve, then retry."
  fi
else
  # Older merge-tree invocation: three-way form
  if git merge-tree "$(git merge-base "$PARENT_TIP" "$HEAD_TIP")" "$PARENT_TIP" "$HEAD_TIP" 2>"$TMP_ERR" | grep -qE '^\+<<<<<<<|^<<<<<<<'; then
    fail "merge conflicts with ${PARENT_TIP} (parent branch: ${PARENT_BRANCH}). Rebase/merge parent, resolve, then retry."
  fi
  if grep -qE 'CONFLICT' "$TMP_ERR" 2>/dev/null; then
    fail "merge conflicts with ${PARENT_TIP} (parent branch: ${PARENT_BRANCH}). Rebase/merge parent, resolve, then retry."
  fi
fi

BEHIND=$(git rev-list --count "${HEAD_TIP}..${PARENT_TIP}" 2>/dev/null || echo 0)
if [ "${BEHIND:-0}" -gt 0 ]; then
  warn "branch is ${BEHIND} commit(s) behind ${PARENT_TIP}; consider rebase before opening/updating the PR"
fi

echo "merge-check: ok (parent=${PARENT_BRANCH} -> ${PARENT_TIP})"
exit 0
