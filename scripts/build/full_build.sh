#!/usr/bin/env bash
# Full local W8A16 build: quantize -> rename -> convert -> two-graph ctx-bin.
# Every stage validated individually on 2026-08-10 (see docs/LOCAL_ENV.md).
#
# Usage: full_build.sh <name> [cl_prefill] [ctx] [extra quantize flags...]
#   e.g. full_build.sh qwen3-0.6b-w8a16 128 1024
#        full_build.sh qwen3-0.6b-w8a16-fusegu 128 1024 --fuse-gate-up
set -euo pipefail
source "$(dirname "$0")/../env.sh"

NAME=${1:?name}
CL=${2:-128}
CTX=${3:-1024}
shift; shift || true; shift || true
EXTRA_FLAGS=("$@")

MODEL=${MODEL:-$LLMDEPLOY_DATA/models/Qwen3-0.6B}   # override: MODEL=.../Qwen3-1.7B full_build.sh ...

# --input-embeds (passed through to quantize_aimet) turns this into the LUT
# probe build: the tower is fed pre-computed hidden states from an external LUT
# instead of token ids, which is the Qwen3-VL feed shape on a model that is
# known-good on device. Detected here the same way --quant-head is, because it
# has to change TWO later stages as well -- the I/O rename and the converter's
# input dims -- and a flag that only reached the quantizer would silently build
# a graph whose first input is still called input_ids, which qualla would then
# drive as token ids (nsp-model.cpp:668 matches the name literally).
EMBEDS=0
for f in "$@"; do [ "$f" = "--input-embeds" ] && EMBEDS=1; done
# An embeddings-fed inputs_embeds must NOT land FLOAT_16. Genie fills it from the
# float32 LUT and then calls quantizeInput, whose tensorOffset is an ELEMENT
# offset for UFIXED_8/16 and FLOAT_32 but a BYTE offset for FLOAT_16 -- so the
# pad fill of a partially-filled prefill chunk writes halfway into the real
# prompt and overwrites its back half (the 2026-08-15 text-garbage defect).
# Giving the tensor a 16-bit INT activation encoding makes the converter type it
# uFxp_16, the correct branch. See scripts/quant/graft_input_encoding.py.
# On by default so the defect cannot be rebuilt by accident; EMBEDS_FP16_IN=1
# reproduces the broken control on purpose.
GRAFT_EMBEDS=0
if [ "$EMBEDS" = "1" ] && [ -z "${EMBEDS_FP16_IN:-}" ]; then GRAFT_EMBEDS=1; fi
QP=$LLMDEPLOY_DATA/work/quant/$NAME-prefill
QD=$LLMDEPLOY_DATA/work/quant/$NAME-decode
DLC=$LLMDEPLOY_DATA/work/dlc/$NAME
CTXBIN=$LLMDEPLOY_DATA/work/ctxbin/$NAME
PY=$LLMDEPLOY_DATA/envs/qwen3-deploy/bin/python
CONVERTER="$QAIRT_SDK/bin/x86_64-linux-clang/qairt-converter"
PAST=$((CTX + CL - 1))
TOTAL=$((PAST + 1))

# Export+quantize is deterministic and by far the most expensive part, so skip it
# when its outputs are already on disk -- same idiom (and same FORCE_EXPORT
# escape) as ladekv_build.sh, and the same reason: re-deriving a byte-identical
# artifact costs ~20 min and buys nothing. This is what makes a converter-flag
# change (e.g. grafting an I/O encoding) a cheap re-convert instead of a rebuild.
if [[ -f "$QP/model_renamed.onnx" && -f "$QD/model_renamed.onnx" && -z "${FORCE_EXPORT:-}" ]]; then
  echo "== [1-4/7] SKIP quantize+rename ($QD/model_renamed.onnx exists; FORCE_EXPORT=1 to redo) =="
  ENC=$QP/model_filtered_renamed.encodings
  [[ -f $ENC ]] || { echo "MISSING $ENC — rerun with FORCE_EXPORT=1"; exit 1; }
else
disk_guard 20
echo "== [1/7] AIMET W8A16 prefill quantization (CL=$CL) =="
$PY "$LLMDEPLOY_ROOT/scripts/quant/quantize_aimet.py" --model "$MODEL" \
    --cl-prefill "$CL" --out "$QP" ${QUANT_DEVICE:+--device "$QUANT_DEVICE"} "${EXTRA_FLAGS[@]}"

disk_guard 20
echo "== [2/7] decode export with prefill encodings (CTX=$CTX) =="
$PY "$LLMDEPLOY_ROOT/scripts/quant/quantize_aimet.py" --model "$MODEL" \
    --cl-prefill "$CL" --ctx "$CTX" --export-decode "$QP" --out "$QD" \
    ${QUANT_DEVICE:+--device "$QUANT_DEVICE"} "${EXTRA_FLAGS[@]}"

echo "== [3/7] encodings filter =="
# --quant-head keeps lm_head's 8-bit weight quantizer; the filter strips every
# lm_head encoding by default, so it must be told to spare that one param.
HEAD_FLAG=()
for f in "${EXTRA_FLAGS[@]}"; do
    [ "$f" = "--quant-head" ] && HEAD_FLAG=(--keep-head-weight)
done
$PY "$LLMDEPLOY_ROOT/scripts/quant/filter_aimet_w8a16.py" "$QP/model.encodings" "${HEAD_FLAG[@]}"

echo "== [4/7] canonical I/O rename =="
# --vl-text is the "first input is inputs_embeds" switch; n-deepstack 0 because
# a plain Qwen3 tower has no deepstack inputs. The name is the contract.
RENAME_FLAGS=()
[ "$EMBEDS" = "1" ] && RENAME_FLAGS=(--vl-text --n-deepstack 0)
$PY "$LLMDEPLOY_ROOT/scripts/quant/rename_aimet_io.py" \
    --model "$QP/model.onnx" --encodings "$QP/model_filtered.encodings" --layers 28 \
    "${RENAME_FLAGS[@]}"
$PY "$LLMDEPLOY_ROOT/scripts/quant/rename_aimet_io.py" \
    --model "$QD/model.onnx" --encodings "$QP/model_filtered.encodings" --layers 28 --with-past \
    "${RENAME_FLAGS[@]}"
ENC=$QP/model_filtered_renamed.encodings
fi

if [ "$GRAFT_EMBEDS" = "1" ]; then
    echo "== [4b/7] graft the inputs_embeds activation encoding (uFxp_16 I/O) =="
    $PY "$LLMDEPLOY_ROOT/scripts/quant/graft_input_encoding.py" \
        --encodings "$ENC" --tensor inputs_embeds \
        --lut "${LUT_DIR:-$LLMDEPLOY_DATA/work/lut/qwen3-0.6b}"
fi

# Hidden size read from the checkpoint rather than hardcoded: the same probe
# should work for 1.7B without editing this script.
HID=$($PY -c "import json,sys; print(json.load(open('$MODEL/config.json'))['hidden_size'])")
if [ "$EMBEDS" = "1" ]; then
    IN0_P=(-d inputs_embeds "1,1,$CL,$HID")
    IN0_D=(-d inputs_embeds "1,1,1,$HID")
else
    IN0_P=(-d input_ids "1,$CL")
    IN0_D=(-d input_ids "1,1")
fi

disk_guard
echo "== [5/7] convert prefill -> DLC =="
mkdir -p "$DLC"
$PY_QAIRT "$CONVERTER" --input_network "$QP/model_renamed.onnx" \
    --output_path "$DLC/prefill.dlc" --quantization_overrides "$ENC" \
    --float_bitwidth 16 --target_backend HTP \
    "${IN0_P[@]}" -d attention_mask "1,$CL,$CL" \
    -d position_ids_cos "1,$CL,64" -d position_ids_sin "1,$CL,64"

disk_guard
echo "== [6/7] convert decode -> DLC (single source of truth: prefill encodings) =="
DIMS=("${IN0_D[@]}" -d attention_mask "1,1,$TOTAL"
      -d position_ids_cos "1,1,64" -d position_ids_sin "1,1,64")
for i in $(seq 0 27); do
  DIMS+=(-d "past_key_${i}_in" "1,8,128,$PAST" -d "past_value_${i}_in" "1,8,$PAST,128")
done
$PY_QAIRT "$CONVERTER" --input_network "$QD/model_renamed.onnx" \
    --output_path "$DLC/decode.dlc" --quantization_overrides "$ENC" \
    --float_bitwidth 16 --target_backend HTP "${DIMS[@]}"

disk_guard
echo "== [7/7] two-graph ctx-bin (vtcm 16, unsigned PD, v81) =="
cd "$LLMDEPLOY_ROOT/configs"
qnn-context-binary-generator \
    --model "$QAIRT_SDK/lib/x86_64-linux-clang/libQnnModelDlc.so" \
    --dlc_path "$DLC/prefill.dlc,$DLC/decode.dlc" \
    --backend "$QAIRT_SDK/lib/x86_64-linux-clang/libQnnHtp.so" \
    --output_dir "$CTXBIN" --binary_file "${NAME}_ctx" \
    --config_file htp_config.json
qnn-context-binary-utility --context_binary "$CTXBIN/${NAME}_ctx.bin" \
    --json_file "$CTXBIN/info.json"
$PY - <<PYEOF
import json
d = json.load(open("$CTXBIN/info.json"))
print("GRAPHS:", [g["info"]["graphName"] for g in d["info"]["graphs"]])
PYEOF
ls -lh "$CTXBIN"
echo "FULL BUILD COMPLETE: $CTXBIN/${NAME}_ctx.bin"
