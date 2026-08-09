#!/usr/bin/env bash
# DLC conversion (summary §2.2 step 4). Run: scripts/build/convert.sh <onnx_dir> <dlc_dir> <encodings> <cl_prefill> <ctx>
# Converter caveats (validated remotely): --target_backend HTP must be UPPERCASE;
# --float_fallback / --target_soc_model / --input_list DO NOT exist on this converter.
set -euo pipefail
source "$(dirname "$0")/../env.sh"

ONNX_DIR=${1:?onnx dir}
DLC_DIR=${2:?dlc out dir}
ENCODINGS=${3:?filtered encodings json}
CL_PREFILL=${4:-128}
CTX=${5:-1024}
PAST=$((CTX + CL_PREFILL - 1))
TOTAL=$((PAST + 1))
NLAYERS=${NLAYERS:-28}
NKV=${NKV:-8}
HEADDIM=${HEADDIM:-128}

mkdir -p "$DLC_DIR"

echo "== prefill =="
"$PY_QAIRT" "$QAIRT_SDK/bin/x86_64-linux-clang/qairt-converter" \
  --input_network "$ONNX_DIR/prefill.onnx" \
  --output_path "$DLC_DIR/prefill.dlc" \
  --quantization_overrides "$ENCODINGS" \
  --float_bitwidth 16 --target_backend HTP \
  -d input_ids "1,$CL_PREFILL" \
  -d attention_mask "1,1,$CL_PREFILL,$CL_PREFILL" \
  -d position_ids_cos "1,$CL_PREFILL,64" \
  -d position_ids_sin "1,$CL_PREFILL,64"

echo "== decode =="
DECODE_DIMS=(-d input_ids "1,1" \
  -d attention_mask "1,1,1,$TOTAL" \
  -d position_ids_cos "1,1,64" \
  -d position_ids_sin "1,1,64")
for i in $(seq 0 $((NLAYERS - 1))); do
  DECODE_DIMS+=(-d "past_key_${i}_in" "1,$NKV,$PAST,$HEADDIM")
  DECODE_DIMS+=(-d "past_value_${i}_in" "1,$NKV,$PAST,$HEADDIM")
done
"$PY_QAIRT" "$QAIRT_SDK/bin/x86_64-linux-clang/qairt-converter" \
  --input_network "$ONNX_DIR/decode.onnx" \
  --output_path "$DLC_DIR/decode.dlc" \
  --quantization_overrides "$ENCODINGS" \
  --float_bitwidth 16 --target_backend HTP \
  "${DECODE_DIMS[@]}"

ls -lh "$DLC_DIR"
