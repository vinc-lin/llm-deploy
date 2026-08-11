#!/usr/bin/env bash
# Push a folder to HF through the local proxy (127.0.0.1:17890), which drops
# long-lived upload streams: the HF client then hangs forever on CLOSE-WAIT
# sockets instead of timing out. This wraps `hf upload-large-folder`
# (resumable: hashing + committed files are cached) and restarts it whenever
# it stalls, until it exits 0.
#
# Stall detection, checked every 20s:
#   (a) the process has >=1 TCP connection and ALL of them are CLOSE-WAIT
#       for 2 consecutive checks, or
#   (b) the last progress line in the log is unchanged for STALL_SECS (180).
#
# Usage: hf_upload_watchdog.sh <repo_id> <local_dir> [logfile]
set -u
REPO=${1:?repo id}
DIR=${2:?local dir}
LOG=${3:-$HOME/llm-local/hf-upload-watchdog.log}
STALL_SECS=${STALL_SECS:-180}
MAX_ATTEMPTS=${MAX_ATTEMPTS:-30}

attempt=0
while (( attempt < MAX_ATTEMPTS )); do
  attempt=$((attempt + 1))
  echo "[watchdog] attempt $attempt $(date '+%F %T')" >> "$LOG"
  hf upload-large-folder "$REPO" "$DIR" --repo-type model --num-workers 4 \
      >> "$LOG" 2>&1 &
  UP=$!

  last_line=""; last_change=$(date +%s); dead_socket_checks=0
  while kill -0 "$UP" 2>/dev/null; do
    sleep 20
    # (a) socket check: every connection of the uploader is half-closed
    socks=$(ss -tnp 2>/dev/null | grep "pid=$UP," || true)
    if [[ -n "$socks" ]] && ! grep -qv "^CLOSE-WAIT" <<< "$socks"; then
      dead_socket_checks=$((dead_socket_checks + 1))
    else
      dead_socket_checks=0
    fi
    # (b) progress-line freeze (strip \r redraws first)
    line=$(tr '\r' '\n' < "$LOG" | grep -v '^\[watchdog\]' | grep . | tail -1)
    if [[ "$line" != "$last_line" ]]; then
      last_line=$line; last_change=$(date +%s)
    fi
    if (( dead_socket_checks >= 2 )) || (( $(date +%s) - last_change > STALL_SECS )); then
      echo "[watchdog] stall detected (sockets=$dead_socket_checks, frozen=$(( $(date +%s) - last_change ))s) — restarting" >> "$LOG"
      kill "$UP" 2>/dev/null; sleep 2; kill -9 "$UP" 2>/dev/null
      break
    fi
  done

  wait "$UP"; rc=$?
  if (( rc == 0 )); then
    echo "[watchdog] upload completed OK on attempt $attempt" >> "$LOG"
    exit 0
  fi
  sleep 5
done
echo "[watchdog] GAVE UP after $MAX_ATTEMPTS attempts" >> "$LOG"
exit 1
