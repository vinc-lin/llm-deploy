#!/usr/bin/env bash
# ctx-bin generation (summary §2.2 step 5). Run: scripts/build/ctxbin.sh <dlc_dir> <out_dir> <binary_name>
# Caveats (validated remotely):
#  - CWD MUST be the directory containing htp_config.json / htp_backend_config.json
#  - --binary_file value must NOT include .bin (auto-appended)
#  - --model libQnnModelDlc.so with --dlc_path (never DLC as --model)
set -euo pipefail
source "$(dirname "$0")/../env.sh"

DLC_DIR=$(realpath "${1:?dlc dir}")
OUT_DIR=$(realpath -m "${2:?output dir}")
BIN_NAME=${3:-qwen3_06b_w8a16_ctx}   # no .bin suffix!
mkdir -p "$OUT_DIR"

cd "$LLMDEPLOY_ROOT/configs"
"$QAIRT_SDK/bin/x86_64-linux-clang/qnn-context-binary-generator" \
  --model "$QAIRT_SDK/lib/x86_64-linux-clang/libQnnModelDlc.so" \
  --dlc_path "$DLC_DIR/prefill.dlc,$DLC_DIR/decode.dlc" \
  --backend "$QAIRT_SDK/lib/x86_64-linux-clang/libQnnHtp.so" \
  --output_dir "$OUT_DIR" \
  --binary_file "$BIN_NAME" \
  --config_file htp_config.json

ls -lh "$OUT_DIR"
"$QAIRT_SDK/bin/x86_64-linux-clang/qnn-context-binary-utility" \
  --context_binary "$OUT_DIR/$BIN_NAME.bin" \
  --json_file "$OUT_DIR/$BIN_NAME.info.json" || true
