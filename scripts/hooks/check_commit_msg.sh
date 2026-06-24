#!/usr/bin/env bash
# Enforce conventional commits + non-empty body (runs via pre-commit commit-msg stage).
set -euo pipefail

MSG_FILE=${1:-}
if [ -z "$MSG_FILE" ] || [ ! -f "$MSG_FILE" ]; then
  echo "commit-msg: missing commit message file" >&2
  exit 1
fi

# Read non-comment lines into a temp file (portable; no mapfile)
TMP=$(mktemp)
trap 'rm -f "$TMP"' EXIT
grep -v '^#' "$MSG_FILE" | sed -e 's/[[:space:]]*$//' >"$TMP" || true

SUBJECT=""
SAW_BLANK_AFTER_SUBJECT=false
BODY_OK=false

while IFS= read -r line || [ -n "$line" ]; do
  if [ -z "$SUBJECT" ]; then
    if [ -z "$line" ]; then
      continue
    fi
    SUBJECT=$line
    continue
  fi
  # second logical line must be blank
  if [ "$SAW_BLANK_AFTER_SUBJECT" = false ]; then
    if [ -n "$line" ]; then
      echo "commit-msg: second line must be blank (separate subject from body)" >&2
      exit 1
    fi
    SAW_BLANK_AFTER_SUBJECT=true
    continue
  fi
  # body lines
  if [ -z "$line" ]; then
    continue
  fi
  if echo "$line" | grep -qE '^[A-Za-z0-9-]+: '; then
    continue
  fi
  BODY_OK=true
done <"$TMP"

if [ -z "$SUBJECT" ]; then
  echo "commit-msg: empty subject" >&2
  exit 1
fi

if echo "$SUBJECT" | grep -qiE '^(fixup!|squash!|wip\b|WIP\b)'; then
  echo "commit-msg: WIP/fixup/squash commits are blocked" >&2
  exit 1
fi

if ! echo "$SUBJECT" | grep -qE '^(feat|fix|refactor|perf|test|docs|build|ci|chore|style)(\([a-zA-Z0-9._-]+\))?(!)?: .+'; then
  echo "commit-msg: subject must use conventional commits, e.g.:" >&2
  echo "  feat(providers): add native anthropic client" >&2
  echo "got: $SUBJECT" >&2
  exit 1
fi

if echo "$SUBJECT" | grep -q '\.$'; then
  echo "commit-msg: subject must not end with a period" >&2
  exit 1
fi

LEN=${#SUBJECT}
if [ "$LEN" -gt 100 ]; then
  echo "commit-msg: subject too long ($LEN > 100 chars)" >&2
  exit 1
fi
if [ "$LEN" -gt 72 ]; then
  echo "commit-msg: warning: subject is $LEN chars (prefer <= 72)" >&2
fi

if [ "$SAW_BLANK_AFTER_SUBJECT" != true ]; then
  echo "commit-msg: body required — add a blank line after the subject, then describe the change" >&2
  exit 1
fi

if [ "$BODY_OK" != true ]; then
  echo "commit-msg: body must include at least one descriptive line (not only trailers)" >&2
  exit 1
fi

exit 0
