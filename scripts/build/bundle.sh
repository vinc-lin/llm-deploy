#!/usr/bin/env bash
# Assemble a push-ready flat device bundle (summary §1.3 layout):
# all .so + genie-t2t-run + ctx-bin + tokenizer + dialog JSON + backend ext
# config in ONE directory, tarred for adb push (large pushes: push tar, extract
# on device, per summary §1.3 USB-disconnect caveat).
#
# Usage: bundle.sh <name> <ctxbin_path> [dialog_json]
set -euo pipefail
source "$(dirname "$0")/../env.sh"

NAME=${1:?bundle name}
CTXBIN=${2:?path to ctx-bin .bin}
DIALOG=${3:-$LLMDEPLOY_ROOT/configs/genie_dialog_qwen3_0.6b.json}

OUT=$LLMDEPLOY_DATA/bundles/$NAME
mkdir -p "$OUT"

A=$QAIRT_SDK/lib/aarch64-android
H=$QAIRT_SDK/lib/hexagon-v81/unsigned

# The 7 device libraries (summary §1.3) — flat, no lib/ subdir
cp "$A/libGenie.so" "$A/libQnnHtp.so" "$A/libQnnSystem.so" \
   "$A/libQnnHtpPrepare.so" "$A/libQnnHtpNetRunExtensions.so" \
   "$A/libQnnHtpV81Stub.so" "$H/libQnnHtpV81Skel.so" "$OUT/"

cp "$QAIRT_SDK/bin/aarch64-android/genie-t2t-run" "$OUT/"
cp "$CTXBIN" "$OUT/"
cp "$LLMDEPLOY_DATA/models/Qwen3-0.6B/tokenizer.json" "$OUT/"
cp "$DIALOG" "$OUT/genie_dialog.json"
cp "$LLMDEPLOY_ROOT/configs/htp_backend_ext_config.json" "$OUT/"

# dialog must reference the actual ctx-bin filename
BIN_NAME=$(basename "$CTXBIN")
python3 - "$OUT/genie_dialog.json" "$BIN_NAME" <<'PYEOF'
import json, sys
p, bin_name = sys.argv[1], sys.argv[2]
d = json.load(open(p))
d["dialog"]["engine"]["model"]["binary"]["ctx-bins"] = [bin_name]
json.dump(d, open(p, "w"), indent=2)
PYEOF

ls -lh "$OUT"
tar -C "$LLMDEPLOY_DATA/bundles" -czf "$LLMDEPLOY_DATA/bundles/$NAME.tar.gz" "$NAME"
ls -lh "$LLMDEPLOY_DATA/bundles/$NAME.tar.gz"
echo "BUNDLE READY: $LLMDEPLOY_DATA/bundles/$NAME.tar.gz"
echo "On device: adb push $NAME.tar.gz /data/local/tmp/ && adb shell 'cd /data/local/tmp && tar xzf $NAME.tar.gz'"
echo "Run:       cd /data/local/tmp/$NAME && LD_LIBRARY_PATH=. ./genie-t2t-run -c genie_dialog.json -p '<prompt>'"
