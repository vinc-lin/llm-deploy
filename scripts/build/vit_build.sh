#!/usr/bin/env bash
# Qwen3-VL vision tower: ONNX -> FP16 DLC -> single-graph ctx-bin.
#
# No AIMET stage: the ViT ships FP16 (spec section 4), so there are no
# encodings and no calibration set. FP16 is requested at conversion time via
# --float_bitwidth 16, the same flag the text pipeline uses for its
# non-quantised tensors.
#
# Config note: configs/htp_backend_config.json is text-model specific -- its
# graph_names are prefill/decode/verify32, so none of its graph tuning would
# bind to this graph, and context.weight_sharing_enabled only means anything
# for a multi-graph ctx-bin. Rather than mutate a config the text builds
# depend on, this script emits a ViT-specific pair of configs into the build
# directory and runs the generator from there (htp_config.json resolves
# config_file_path relative to cwd). Device/graph values (v81, unsigned PD,
# vtcm_mb 16, O=3, hvx_threads 4) are carried over unchanged; the text-only
# extras (weight sharing, extended_udma, sparse_weights_compression) are
# dropped.
#
# Usage: vit_build.sh [name]
set -euo pipefail
source "$(dirname "$0")/../env.sh"

NAME=${1:-qwen3vl-4b-vit-fp16}
ONNX=$LLMDEPLOY_DATA/work/onnx/qwen3vl-4b-vit/vit.onnx
DLC=$LLMDEPLOY_DATA/work/dlc/$NAME
CTXBIN=$LLMDEPLOY_DATA/work/ctxbin/$NAME
CONVERTER="$QAIRT_SDK/bin/x86_64-linux-clang/qairt-converter"
GRAPH=vit            # ctx-bin graph name == DLC basename

[ -f "$ONNX" ] || { echo "missing $ONNX -- run export_qwen3vl_vit.py first"; exit 1; }

echo "== [1/3] convert ViT -> FP16 DLC =="
mkdir -p "$DLC"
$PY_QAIRT "$CONVERTER" --input_network "$ONNX" \
    --output_path "$DLC/$GRAPH.dlc" \
    --float_bitwidth 16 --target_backend HTP \
    -d pixel_values "1024,1536"

echo "== [2/3] single-graph ctx-bin (vtcm 16, unsigned PD, v81) =="
mkdir -p "$CTXBIN"
cat > "$CTXBIN/htp_backend_config.json" <<JSONEOF
{
  "graphs": [
    {
      "graph_names": [
        "$GRAPH"
      ],
      "O": 3,
      "vtcm_mb": 16,
      "hvx_threads": 4,
      "fp16_relaxed_precision": 0
    }
  ],
  "devices": [
    {
      "dsp_arch": "v81",
      "soc_model": 0,
      "pd_session": "unsigned",
      "cores": [
        {
          "core_id": 0,
          "perf_profile": "burst",
          "rpc_control_latency": 100,
          "rpc_polling_time": 9999
        }
      ]
    }
  ]
}
JSONEOF
cat > "$CTXBIN/htp_config.json" <<JSONEOF
{
  "backend_extensions": {
    "shared_library_path": "libQnnHtpNetRunExtensions.so",
    "config_file_path": "htp_backend_config.json"
  }
}
JSONEOF
cd "$CTXBIN"
qnn-context-binary-generator \
    --model "$QAIRT_SDK/lib/x86_64-linux-clang/libQnnModelDlc.so" \
    --dlc_path "$DLC/$GRAPH.dlc" \
    --backend "$QAIRT_SDK/lib/x86_64-linux-clang/libQnnHtp.so" \
    --output_dir "$CTXBIN" --binary_file "${NAME}_ctx" \
    --config_file htp_config.json

echo "== [3/3] dump graph info =="
qnn-context-binary-utility --context_binary "$CTXBIN/${NAME}_ctx.bin" \
    --json_file "$CTXBIN/info.json"
$PY_DEPLOY - <<PYEOF
import json
d = json.load(open("$CTXBIN/info.json"))
for g in d["info"]["graphs"]:
    print("GRAPH:", g["info"]["graphName"])
PYEOF
ls -lh "$CTXBIN"
echo "VIT BUILD COMPLETE: $CTXBIN/${NAME}_ctx.bin"
