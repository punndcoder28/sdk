#!/usr/bin/env bash
# Fail if staged files contain unresolved merge conflict markers.
set -euo pipefail

STAGED=$(git diff --cached --name-only --diff-filter=ACMR || true)
[ -z "$STAGED" ] && exit 0

echo "$STAGED" | while IFS= read -r f; do
  [ -z "$f" ] && continue
  [ -f "$f" ] || continue
  if git show ":$f" 2>/dev/null | grep -qE '^(<<<<<<< |>>>>>>> )'; then
    echo "conflict-markers: unresolved markers in staged file: $f" >&2
    exit 1
  fi
done

# while in pipeline may not propagate exit; re-scan in current shell
failed=0
while IFS= read -r f; do
  [ -z "$f" ] && continue
  [ -f "$f" ] || continue
  if git show ":$f" 2>/dev/null | grep -qE '^(<<<<<<< |>>>>>>> )'; then
    echo "conflict-markers: unresolved markers in staged file: $f" >&2
    failed=1
  fi
done <<< "$STAGED"
exit $failed
