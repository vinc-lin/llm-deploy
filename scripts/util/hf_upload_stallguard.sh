#!/usr/bin/env bash
# Upload large files to HF one at a time, killing and retrying any stream that
# STOPS ADVANCING. Safe for these repos: it drives hf_upload_file.py, i.e.
# single `upload_file` commits, which touch neither repo settings nor the hub
# README.
#
# THE FAILURE THIS EXISTS TO PREVENT
# ----------------------------------
# 2026-08-16: uploading eight ~926 MB bundles hung for ~4 HOURS on the first
# one, frozen at 725/926 MB with the progress bar redrawing the same line 365
# times. Zero bundles landed. Nothing errored.
#
# hf_upload_file.py's --retries only catches EXCEPTIONS. A stalled stream never
# raises -- it just stops moving -- so the retry logic never fires and the
# process sits there indefinitely. docs/REFERENCE.md section 9 says exactly
# this: "The progress-freeze detector (STALL_SECS=240) is the reliable signal",
# not the socket detector. The existing hf_upload_watchdog.sh has that detector
# but wraps `hf upload-large-folder`, which applies its own repo defaults and
# has silently flipped this repo's visibility and clobbered its README. This
# script is the safe half of both.
#
# On the retry after adding this guard, all eight bundles went through on
# attempt 1 in ~1 minute each -- so the hang was transient proxy state that
# only an external freeze detector could break out of.
#
# Usage:
#   hf_upload_stallguard.sh <repo_id> <path_in_repo_prefix> <file> [file ...]
#
# Example:
#   hf_upload_stallguard.sh vinccniv/sa8797p-qwen3-w8a16-bundles \
#       2026-08-16-regime/bundles /path/to/*.tar.gz
#
# Env: STALL_SECS (default 180), MAX_ATTEMPTS (default 4), POLL (default 20).
#
# NOTE: this script deliberately does NOT read or write repo visibility.
# Visibility on these repos is switched often and deliberately by the user; read
# it live yourself before and after if it matters, and never "restore" a
# remembered value.
set -uo pipefail

HERE=$(cd "$(dirname "$0")" && pwd)
HF=${1:?repo_id}; PREFIX=${2:?path-in-repo prefix}; shift 2
[ $# -gt 0 ] || { echo "no files given"; exit 2; }

STALL_SECS=${STALL_SECS:-180}
MAX_ATTEMPTS=${MAX_ATTEMPTS:-4}
POLL=${POLL:-20}
LOGD=${LOGD:-$(mktemp -d)}
mkdir -p "$LOGD"
echo "logs: $LOGD   stall=${STALL_SECS}s attempts=$MAX_ATTEMPTS"

progress () {   # highest MB seen so far in a log
  grep -ao "[0-9]\+MB /" "$1" 2>/dev/null | grep -o "[0-9]\+" | sort -n | tail -1
}

upload_one () {
  local f="$1" n; n=$(basename "$f")
  local log="$LOGD/$n.log"

  for attempt in $(seq 1 "$MAX_ATTEMPTS"); do
    : > "$log"
    python3 "$HERE/hf_upload_file.py" "$f" "$HF" "$PREFIX/$n" >> "$log" 2>&1 &
    local pid=$! last=-1 frozen=0

    while kill -0 "$pid" 2>/dev/null; do
      sleep "$POLL"
      local cur; cur=$(progress "$log"); cur=${cur:-0}
      if [ "$cur" -gt "$last" ]; then
        last=$cur; frozen=0
      else
        frozen=$((frozen + POLL))
        if [ "$frozen" -ge "$STALL_SECS" ]; then
          echo "  STALLED at ${cur}MB for ${frozen}s -- killing attempt $attempt"
          kill -9 "$pid" 2>/dev/null; pkill -9 -P "$pid" 2>/dev/null
          break
        fi
      fi
    done
    wait "$pid" 2>/dev/null

    if grep -q "UPLOAD_DONE" "$log"; then
      echo "  OK on attempt $attempt ($(grep -o 'verified server-side: [0-9,]*' "$log" | tail -1))"
      return 0
    fi
    echo "  attempt $attempt did not complete; retrying"
  done
  echo "  *** GIVING UP on $n after $MAX_ATTEMPTS attempts"
  return 1
}

fail=0
for f in "$@"; do
  [ -f "$f" ] || { echo "SKIP (missing): $f"; continue; }
  echo "=== [$(date +%H:%M:%S)] $(basename "$f") ($(ls -l "$f" | awk '{print $5}') B)"
  upload_one "$f" || fail=$((fail + 1))
done

echo "=== [$(date +%H:%M:%S)] done; $fail failed"
[ "$fail" -eq 0 ]
