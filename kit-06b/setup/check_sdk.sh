#!/usr/bin/env bash
# Verify this host can build: the exact SDK version, every tool and library the
# build touches, both interpreters, the checkpoint, and enough disk.
#
# The version check is not pedantry -- quantization behaviour is SDK-specific, so
# a different SDK invalidates every measured number the kit's README quotes,
# including the md5 acceptance gate.
set -euo pipefail
# shellcheck source-path=SCRIPTDIR source=../env.sh
source "$(dirname "$0")/../env.sh"

WANT_SDK=2.48.40.260702
fail=0
note() { echo "  $1"; }
bad()  { echo "  MISSING: $1"; fail=1; }

echo "== host =="
note "$(hostname) · $(nproc) cores · $(free -g 2>/dev/null | awk '/^Mem:/{print $2" GB RAM"}')"
note "KIT_ROOT=$KIT_ROOT"
note "KIT_DATA=$KIT_DATA"
note "KIT_OUT=$KIT_OUT"

echo "== SDK =="
[ -d "$QAIRT_SDK" ] || { echo "ABORT: QAIRT_SDK not found: $QAIRT_SDK" >&2; exit 1; }
note "path: $QAIRT_SDK"
case "$QAIRT_SDK" in
  *"$WANT_SDK"*) note "version: $WANT_SDK OK" ;;
  *) echo "  WARN: expected $WANT_SDK -- the kit's measured numbers, including the"
     echo "        md5 gate, are specific to that SDK" ;;
esac

echo "== x86 build tools =="
for t in qairt-converter qnn-context-binary-generator \
         qnn-context-binary-utility qairt-dlc-info; do
    if [ -x "$QAIRT_SDK/bin/x86_64-linux-clang/$t" ]; then note "$t"
    else bad "bin/x86_64-linux-clang/$t"; fi
done

echo "== x86 backend libraries =="
for l in libQnnModelDlc.so libQnnHtp.so; do
    if [ -f "$QAIRT_SDK/lib/x86_64-linux-clang/$l" ]; then note "$l"
    else bad "lib/x86_64-linux-clang/$l"; fi
done

echo "== device payload (bundled, never executed here) =="
for l in libGenie.so libQnnHtp.so libQnnSystem.so libQnnHtpPrepare.so \
         libQnnHtpNetRunExtensions.so libQnnHtpV81Stub.so; do
    if [ -f "$QAIRT_SDK/lib/aarch64-android/$l" ]; then note "$l"
    else bad "lib/aarch64-android/$l"; fi
done
if [ -f "$QAIRT_SDK/lib/hexagon-v81/unsigned/libQnnHtpV81Skel.so" ]; then
    note "libQnnHtpV81Skel.so"
else bad "lib/hexagon-v81/unsigned/libQnnHtpV81Skel.so"; fi
if [ -f "$QAIRT_SDK/bin/aarch64-android/genie-t2t-run" ]; then note "genie-t2t-run"
else bad "bin/aarch64-android/genie-t2t-run"; fi

echo "== interpreters =="
if [ -x "$PY_DEPLOY" ]; then note "PY_DEPLOY $("$PY_DEPLOY" -V 2>&1)"
else bad "PY_DEPLOY ($PY_DEPLOY) -- run setup/make_envs.sh"; fi
if [ -x "$PY_QAIRT" ]; then note "PY_QAIRT $("$PY_QAIRT" -V 2>&1)"
else bad "PY_QAIRT ($PY_QAIRT) -- run setup/make_envs.sh"; fi
note "QUANT_DEVICE=$QUANT_DEVICE"

echo "== checkpoint =="
# Verified, never fetched: some build hosts cannot reach Hugging Face at all
# (tank resolves it IPv6-only with no route), so a missing checkpoint is a
# staging problem to report, not something to paper over with a download.
if [ -f "$MODEL/config.json" ] && [ -f "$MODEL/tokenizer.json" ]; then
    note "Qwen3-0.6B at $MODEL"
else
    bad "Qwen3-0.6B at $MODEL (config.json + tokenizer.json)"
    echo "        If this host cannot reach Hugging Face, stage it from one that can:"
    echo "        rsync -a <src>/models/Qwen3-0.6B/ $MODEL/"
fi

echo "== disk =="
note "guarding $(disk_guard_target)"
if disk_guard 20; then note "at least 20 GB free"; else fail=1; fi

if (( fail == 0 )); then
    echo "SDK CHECK PASSED"
else
    echo "SDK CHECK FAILED" >&2
    exit 1
fi
