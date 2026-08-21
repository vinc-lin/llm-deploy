#!/usr/bin/env bash
# The 2-shard 0.6B: reproduce the 4B's STRUCTURE at 0.6B scale.
#
# WHY
# ---
# The Qwen3-VL-4B's decode step 1 is wrong while its decode GRAPHS are proven
# correct on device (Test J A1). After Tests J/K/L the candidate list is down to
# the last structural difference between the 0.6B that works and the 4B that
# does not:
#
#     0.6B (works)      one ctx-bin,  in-graph embedding lookup
#     0.6B lutprobe     one ctx-bin,  LUT-fed inputs_embeds     <- Test L
#     THIS BUILD        TWO ctx-bins, LUT-fed inputs_embeds     <- the 4B's shape
#     VL-4B (broken)    TWO ctx-bins, LUT-fed inputs_embeds
#
# If this build reproduces the 4B's "first token right, then garbage", the bug
# is in the split and we can bisect it HOST-side on a 0.6B that rebuilds in
# minutes instead of hours on tank. If it generates correctly, the split is
# exonerated at 0.6B and the 4B's fault needs something the 0.6B does not have
# (scale, MRoPE, 36 layers, 2560 hidden).
#
# WHAT IT REUSES
# --------------
# Nothing is re-quantized. The LUT probe's own AIMET exports already carry
# `inputs_embeds` as the first input and 28 layers, so this cuts THOSE at the
# layer-13/14 seam and converts the halves. That keeps the calibration
# identical to the single-bin probe, which is the point: one variable, the split.
#
#   work/quant/qwen3-0.6b-w8a16-lutprobe-prefillkv128/model_renamed.onnx  (AR=128, past-KV)
#   work/quant/qwen3-0.6b-w8a16-lutprobe-decode/model_renamed.onnx        (AR=1)
#   work/quant/qwen3-0.6b-w8a16-lutprobe-prefill/model_filtered_renamed.encodings
#
# The ctx-bin generation is `vl_text_ctxbin_split.sh` unchanged -- it is already
# parameterised by LAYERS/HIDDEN/NKV/HEAD_DIM/NDEEP, and NDEEP=0 makes its
# deepstack loop a no-op. Two overrides matter and are NOT defaults:
#   EMBED_LUT_DIR  -> the 0.6B LUT, not the VL one (1024-wide, different range)
#   VIT_INFO       -> a path that does not exist, so no image_features range is
#                     unioned into the inputs_embeds encoding. A text-only tower
#                     has none, and covering +-11.6 would waste the whole grid.
#
# Usage: lutsplit_06b_build.sh [name] [cl] [ctx] [split_at]
set -euo pipefail
source "$(dirname "$0")/../env.sh"

NAME=${1:-qwen3-06b-lutsplit}
CL=${2:-128}
CTX=${3:-512}
SPLIT=${4:-14}
LAYERS=${LAYERS:-28}
SEAM=${SEAM:-/layers.$((SPLIT - 1))/Add_1_output_0}

SRC=${SRC:-$LLMDEPLOY_DATA/work/quant}
PROBE=${PROBE:-qwen3-0.6b-w8a16-lutprobe}
ONNX=$LLMDEPLOY_DATA/work/onnx/$NAME
ENCDIR=$LLMDEPLOY_DATA/work/quant/$NAME-enc
DLC=$LLMDEPLOY_DATA/work/dlc/$NAME
CTXBIN=$LLMDEPLOY_DATA/work/ctxbin/$NAME

PRE_ONNX=$SRC/$PROBE-prefillkv$CL/model_renamed.onnx
DEC_ONNX=$SRC/$PROBE-decode/model_renamed.onnx
ENC_WHOLE=$SRC/$PROBE-prefill/model_filtered_renamed.encodings

for f in "$PRE_ONNX" "$DEC_ONNX" "$ENC_WHOLE"; do
    [ -f "$f" ] || { echo "MISSING prerequisite: $f"; exit 1; }
done

echo "== [1/4] cut the AIMET exports at $SEAM (layers 0-$((SPLIT - 1)) | $SPLIT-$((LAYERS - 1))) =="
disk_guard 20
for spec in "prefill:$PRE_ONNX" "decode:$DEC_ONNX"; do
    tag=${spec%%:*}; onnx=${spec#*:}
    if [ -f "$ONNX/${tag}_0/${tag}_0.onnx" ] && [ -z "${FORCE_SPLIT:-}" ]; then
        echo "   SKIP $tag (exists; FORCE_SPLIT=1 to redo)"
        continue
    fi
    "$PY_DEPLOY" "$LLMDEPLOY_ROOT/scripts/quant/split_aimet_onnx.py" \
        --onnx "$onnx" --seam "$SEAM" --split-at "$SPLIT" --layers "$LAYERS" \
        --out-dir "$ONNX" --tag "$tag" --with-past
done

echo "== [2/4] split the encodings at the same seam =="
mkdir -p "$ENCDIR"
"$PY_DEPLOY" "$LLMDEPLOY_ROOT/scripts/quant/split_encodings.py" \
    --encodings "$ENC_WHOLE" --split-at "$SPLIT" --layers "$LAYERS" \
    --out-dir "$ENCDIR"

echo "== [3/4] convert + generate the two ctx-bins =="
# PREFILL_PAST=$CTX gives the past-KV prefill. A bertcache prefill (mask
# [1,AR,AR]) is FATAL in a split tower: shard 0 has no logits, classifies
# DECODER_PREFILL, its expected CL is rewritten to the cache-group max, and the
# mask then fails validateModel -- the node never loads. REFERENCE.md 3.6.
LAYERS=$LAYERS NDEEP=0 HIDDEN=${HIDDEN:-1024} NKV=${NKV:-8} HEAD_DIM=${HEAD_DIM:-128} \
PREFILL_PAST=$CTX \
ONNX=$ONNX ENCDIR=$ENCDIR DLC=$DLC CTXBIN=$CTXBIN \
EMBED_LUT_DIR=${EMBED_LUT_DIR:-$LLMDEPLOY_DATA/work/lut/qwen3-0.6b} \
VIT_INFO=${VIT_INFO:-/nonexistent-no-vision-tower} \
    bash "$LLMDEPLOY_ROOT/scripts/build/vl_text_ctxbin_split.sh" \
        "$NAME" "$CL" "$CTX" "$SPLIT"

echo "== [4/4] gate: shard 0's inputs_embeds must be uFxp_16, not FLOAT_16 =="
# This is the defect that voided Test K's K1. Gate it here so the bundle cannot
# ship the wrong dtype a second time.
"$PY_DEPLOY" "$LLMDEPLOY_ROOT/scripts/validate/lint_embedding_dtype.py" \
    --ctxbin-info "$CTXBIN/1_of_2/info.json" \
    --lut-params "${EMBED_LUT_DIR:-$LLMDEPLOY_DATA/work/lut/qwen3-0.6b}/embedding_lut_params.json"

echo ""
echo "ctx-bins:"
du -h "$CTXBIN"/*/*.bin 2>/dev/null || ls -la "$CTXBIN"
echo ""
echo "next: bundle it with the 0.6B LUT + tokenizer + libs and a 2-ctx-bin"
echo "      dialog config, then run the same prompts as Test L."
