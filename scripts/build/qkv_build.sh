#!/usr/bin/env bash
# QKV-fused W8A16 build with encodings surgery (summary §4.1 unlock).
# Requires a completed NON-FUSED full_build (donor for Q-branch INT16 encodings).
#
# Usage: qkv_build.sh <name> <donor_prefill_quant_dir> [cl] [ctx] [--fuse-gate-up]
#   e.g. qkv_build.sh qwen3-0.6b-w8a16-fuseqkv   $DATA/work/quant/qwen3-0.6b-w8a16-prefill 128 1024
#        qkv_build.sh qwen3-0.6b-w8a16-fuseqkvgu $DATA/work/quant/qwen3-0.6b-w8a16-prefill 128 1024 --fuse-gate-up
set -euo pipefail
source "$(dirname "$0")/../env.sh"

NAME=${1:?name}
DONOR=${2:?donor prefill quant dir (non-fused)}
CL=${3:-128}
CTX=${4:-1024}
shift 4 || true
EXTRA=("$@")

MODEL=${MODEL:-$LLMDEPLOY_DATA/models/Qwen3-0.6B}
QP=$LLMDEPLOY_DATA/work/quant/$NAME-prefill
QD=$LLMDEPLOY_DATA/work/quant/$NAME-decode
DLC=$LLMDEPLOY_DATA/work/dlc/$NAME
CTXBIN=$LLMDEPLOY_DATA/work/ctxbin/$NAME
PY=$LLMDEPLOY_DATA/envs/qwen3-deploy/bin/python
CONVERTER="$QAIRT_SDK/bin/x86_64-linux-clang/qairt-converter"
PAST=$((CTX + CL - 1))
TOTAL=$((PAST + 1))

echo "== [1/8] fused prefill quantization =="
$PY "$LLMDEPLOY_ROOT/scripts/quant/quantize_aimet.py" --model "$MODEL" \
    --cl-prefill "$CL" --fuse-qkv --out "$QP" \
    ${QUANT_DEVICE:+--device "$QUANT_DEVICE"} "${EXTRA[@]}"

echo "== [2/8] fused decode export =="
$PY "$LLMDEPLOY_ROOT/scripts/quant/quantize_aimet.py" --model "$MODEL" \
    --cl-prefill "$CL" --ctx "$CTX" --fuse-qkv --export-decode "$QP" --out "$QD" \
    ${QUANT_DEVICE:+--device "$QUANT_DEVICE"} "${EXTRA[@]}"

echo "== [3/8] filter =="
$PY "$LLMDEPLOY_ROOT/scripts/quant/filter_aimet_w8a16.py" "$QP/model.encodings"

echo "== [4/8] rename =="
$PY "$LLMDEPLOY_ROOT/scripts/quant/rename_aimet_io.py" \
    --model "$QP/model.onnx" --encodings "$QP/model_filtered.encodings" --layers 28

echo "== [5/8] QKV surgery (donor: $DONOR) =="
$PY "$LLMDEPLOY_ROOT/scripts/export/qkv_surgery.py" \
    --fused-onnx "$QP/model_renamed.onnx" \
    --fused-encodings "$QP/model_filtered_renamed.encodings" \
    --donor-onnx "$DONOR/model_renamed.onnx" \
    --donor-encodings "$DONOR/model_filtered_renamed.encodings" \
    --out "$QP/model_surgery.encodings"
ENC=$QP/model_surgery.encodings

echo "== [6/8] rename decode + overlap check =="
$PY "$LLMDEPLOY_ROOT/scripts/quant/rename_aimet_io.py" \
    --model "$QD/model.onnx" --encodings "$ENC" --layers 28 --with-past

echo "== [7/8] convert prefill + decode =="
mkdir -p "$DLC"
$PY_QAIRT "$CONVERTER" --input_network "$QP/model_renamed.onnx" \
    --output_path "$DLC/prefill.dlc" --quantization_overrides "$ENC" \
    --float_bitwidth 16 --target_backend HTP \
    -d input_ids "1,$CL" -d attention_mask "1,$CL,$CL" \
    -d position_ids_cos "1,$CL,64" -d position_ids_sin "1,$CL,64"
DIMS=(-d input_ids "1,1" -d attention_mask "1,1,$TOTAL"
      -d position_ids_cos "1,1,64" -d position_ids_sin "1,1,64")
for i in $(seq 0 27); do
  DIMS+=(-d "past_key_${i}_in" "1,8,128,$PAST" -d "past_value_${i}_in" "1,8,$PAST,128")
done
$PY_QAIRT "$CONVERTER" --input_network "$QD/model_renamed.onnx" \
    --output_path "$DLC/decode.dlc" --quantization_overrides "$ENC" \
    --float_bitwidth 16 --target_backend HTP "${DIMS[@]}"

echo "== [8/8] ctx-bin =="
cd "$LLMDEPLOY_ROOT/configs"
qnn-context-binary-generator \
    --model "$QAIRT_SDK/lib/x86_64-linux-clang/libQnnModelDlc.so" \
    --dlc_path "$DLC/prefill.dlc,$DLC/decode.dlc" \
    --backend "$QAIRT_SDK/lib/x86_64-linux-clang/libQnnHtp.so" \
    --output_dir "$CTXBIN" --binary_file "${NAME}_ctx" \
    --config_file htp_config.json
echo "QKV BUILD COMPLETE: $CTXBIN/${NAME}_ctx.bin"
