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
# dropped. Measurements and precedent: docs/NOTES-vit-htp-config.md.
#
# Usage: vit_build.sh [name]
set -euo pipefail
source "$(dirname "$0")/../env.sh"

NAME=${1:-qwen3vl-4b-vit-fp16}
ONNX=$LLMDEPLOY_DATA/work/onnx/qwen3vl-4b-vit/vit.onnx
DLC=$LLMDEPLOY_DATA/work/dlc/$NAME
CTXBIN=$LLMDEPLOY_DATA/work/ctxbin/$NAME
CONVERTER="$QAIRT_SDK/bin/x86_64-linux-clang/qairt-converter"
GRAPH=vit            # ctx-bin graph name == DLC basename -- ASSERTED in step 3,
                     # because the backend-extension config keys on this name
OPT_LEVEL=3          # these three are what the generated config exists to secure;
VTCM_MB=16           # step 3 reads them back out of the finalized binary and
HVX_THREADS=4        # fails the build if the config did not bind

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
      "O": $OPT_LEVEL,
      "vtcm_mb": $VTCM_MB,
      "hvx_threads": $HVX_THREADS,
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

echo "== [3/3] dump graph info and verify the config bound =="
qnn-context-binary-utility --context_binary "$CTXBIN/${NAME}_ctx.bin" \
    --json_file "$CTXBIN/info.json"
$PY_DEPLOY - <<PYEOF
import json
import sys

d = json.load(open("$CTXBIN/info.json"))
graphs = d["info"]["graphs"]
for g in graphs:
    print("GRAPH:", g["info"]["graphName"])

# A backend-extension config that fails to bind does not error -- the graph just
# compiles with defaults (O=0, vtcm 4 MB, 0 HVX threads) and the log stays clean.
# That exact silent regression shipped once already: verify32 omitted from
# graph_names, 4 MB VTCM + 24 MB spill on device, LADE SIGSEGV
# (docs/BUILD_GUIDE.md section 5.4b). With no device, this readback is the only
# place it is detectable, so it is a build failure, not a warning.
errs = []
if len(graphs) != 1:
    errs.append("expected 1 graph, found %d: %r"
                % (len(graphs), [g["info"]["graphName"] for g in graphs]))
else:
    info = graphs[0]["info"]
    if info["graphName"] != "$GRAPH":
        errs.append("graphName is %r, expected %r -- graph_names in "
                    "htp_backend_config.json keys on that string, so the config "
                    "bound to nothing" % (info["graphName"], "$GRAPH"))
    blob = info.get("graphBlobInfo", {}).get("info", {})
    for key, want in (("optimizationLevel", $OPT_LEVEL),
                      ("vtcmSize", $VTCM_MB),
                      ("numHvxThreads", $HVX_THREADS)):
        if blob.get(key) != want:
            errs.append("%s is %r, expected %r" % (key, blob.get(key), want))

if errs:
    print("BUILD REJECTED: htp_backend_config.json did not take effect")
    for e in errs:
        print("  - " + e)
    print("  see docs/NOTES-vit-htp-config.md")
    sys.exit(1)

blob = graphs[0]["info"]["graphBlobInfo"]["info"]
print("CONFIG BOUND: O=%d vtcm=%d MB hvx_threads=%d"
      % (blob["optimizationLevel"], blob["vtcmSize"], blob["numHvxThreads"]))
PYEOF
ls -lh "$CTXBIN"
echo "VIT BUILD COMPLETE: $CTXBIN/${NAME}_ctx.bin"
