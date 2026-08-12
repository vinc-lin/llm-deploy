#!/usr/bin/env bash
# Stage 2 / Task 5: AIMET W8A16 quantization of the Qwen3-VL text tower.
#
# Same shape as full_build.sh's quantization half, with the VL differences all
# carried by --vl-text: embeddings-in (inputs_embeds, no in-graph token table),
# three deepstack inputs, 36 layers, and a MULTIMODAL calibration set (real ViT
# features spliced onto the image-token positions).
#
# Ends after the canonical I/O rename. DLC conversion and the ctx-bin are Task 6
# and deliberately not started here -- the artifacts this script leaves
# ($QP/model_renamed.onnx, $QD/model_renamed.onnx, $ENC) are exactly its inputs.
#
# Usage: vl_text_build.sh <name> [cl_prefill] [ctx] [extra quantize flags...]
#   e.g. vl_text_build.sh qwen3vl-4b-w8a16 128 2048
set -euo pipefail
source "$(dirname "$0")/../env.sh"

NAME=${1:?name}
CL=${2:-128}
CTX=${3:-2048}
shift; shift || true; shift || true
EXTRA_FLAGS=("$@")

MODEL=${MODEL:-$LLMDEPLOY_DATA/models/Qwen3-VL-4B-Instruct}
VIT=${VIT:-$LLMDEPLOY_DATA/work/onnx/qwen3vl-4b-vit/vit.onnx}
# calibration windows depend on the checkpoint and on AR, not on the quant
# variant, so the cache is shared across variants of the same (model, AR)
CALIB=${CALIB:-$LLMDEPLOY_DATA/work/quant/qwen3vl-4b-calib-ar$CL.npz}
QP=$LLMDEPLOY_DATA/work/quant/$NAME-prefill
QD=$LLMDEPLOY_DATA/work/quant/$NAME-decode
PY=$PY_DEPLOY
LAYERS=${LAYERS:-36}
NDEEP=${NDEEP:-3}
# 4B fp32 does not fit in 8 GB of VRAM; CPU is not a preference here
DEVICE=${QUANT_DEVICE:-cpu}
PAST=$((CTX + CL - 1))
Q=(--vl-text --n-deepstack "$NDEEP" --vl-calib "$CALIB" --lean-export
   --device "$DEVICE" "${EXTRA_FLAGS[@]}")

disk_guard
echo "== [1/5] multimodal calibration windows (AR=$CL) =="
if [ -f "$CALIB" ]; then
  echo "reusing $CALIB"
else
  $PY "$LLMDEPLOY_ROOT/scripts/quant/vl_calib_build.py" --model "$MODEL" \
      --vit-onnx "$VIT" --out "$CALIB" --ar "$CL" --n-deepstack "$NDEEP"
fi

# The old "~8.6GB each" here was the 0.6B number, not the 4B one. Measured
# 0.6B: 8.5GB written per graph, of which only 2.9GB is load-bearing; --lean-
# export (in $Q above) drops the rest. Scaled by the 5.35x parameter ratio the
# 4B text tower writes ~31GB per graph and retains ~16GB, and the decode run
# needs the prefill dir still on disk -- so guard for the pair, not one graph.
#
# WARNING: on a 47GB-RAM + 16GB-swap box the 4B export does NOT complete --
# OOM-killed twice at anon-rss 37.4GB and 45.4GB with swap exhausted, both
# times inside _create_onnx_model_with_markers. AIMET's legacy sim.export
# holds four fp32 copies of the graph at once (sim.model + model_to_export +
# the marker deepcopy + the in-memory ONNX proto = 4 x 15.0GB at 4B).
# Calibration and --eval complete fine; it is only the export that does not
# fit. See docs/LOCAL_ENV.md.
disk_guard 40
echo "== [2/5] AIMET W8A16 prefill quantization (CL=$CL) =="
/usr/bin/time -v $PY "$LLMDEPLOY_ROOT/scripts/quant/quantize_aimet.py" \
    --model "$MODEL" --cl-prefill "$CL" --out "$QP" --eval "${Q[@]}"

disk_guard 40
echo "== [3/5] decode export with prefill encodings (CTX=$CTX, past=$PAST) =="
# Single encodings lineage: the decode graph adopts the prefill run's encodings
# verbatim, so every shared tensor -- above all the KV path -- keeps byte-
# identical quant params. Mixed encodings are a fatal Genie load error.
/usr/bin/time -v $PY "$LLMDEPLOY_ROOT/scripts/quant/quantize_aimet.py" \
    --model "$MODEL" --cl-prefill "$CL" --ctx "$CTX" --export-decode "$QP" \
    --out "$QD" "${Q[@]}"

echo "== [4/5] encodings filter =="
$PY "$LLMDEPLOY_ROOT/scripts/quant/filter_aimet_w8a16.py" "$QP/model.encodings"

echo "== [5/5] canonical I/O rename (layers=$LAYERS) =="
$PY "$LLMDEPLOY_ROOT/scripts/quant/rename_aimet_io.py" \
    --model "$QP/model.onnx" --encodings "$QP/model_filtered.encodings" \
    --layers "$LAYERS" --vl-text --n-deepstack "$NDEEP"
$PY "$LLMDEPLOY_ROOT/scripts/quant/rename_aimet_io.py" \
    --model "$QD/model.onnx" --encodings "$QP/model_filtered.encodings" \
    --layers "$LAYERS" --vl-text --n-deepstack "$NDEEP" --with-past
ENC=$QP/model_filtered_renamed.encodings

du -sh "$QP" "$QD"
echo "QUANTIZATION COMPLETE"
echo "  prefill onnx : $QP/model_renamed.onnx"
echo "  decode  onnx : $QD/model_renamed.onnx"
echo "  encodings    : $ENC   (single lineage for BOTH graphs)"
