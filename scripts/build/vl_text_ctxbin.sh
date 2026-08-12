#!/usr/bin/env bash
# Stage 2 / Task 6: Qwen3-VL text tower -> two W8A16 DLCs -> weight-shared ctx-bin.
#
# Split out from vl_text_build.sh on purpose: quantizing the 4B tower costs ~45
# minutes and 63 GB of RAM, so a conversion retry must not drag that along. This
# script consumes exactly what vl_text_build.sh leaves behind and nothing else.
#
# Both graphs convert against the SAME encodings file, taken from the PREFILL
# directory. That is not a tidiness preference -- every tensor the two graphs
# share, above all the KV path, has to carry byte-identical quant params or
# Genie fails the ctx-bin load outright. full_build.sh does the same thing.
#
# Config note: configs/htp_backend_config.json lists graph_names
# prefill/decode/verify32 and is wired for the 0.6B/1.7B text builds. Ours are
# also called prefill/decode, so it would *appear* to bind -- but it also
# carries text-model tuning (extended_udma, sparse_weights_compression) that is
# unverified for a 4B VL tower. Rather than mutate a config three shipped builds
# depend on, this script emits its own pair into the build directory and runs
# the generator from there (htp_config.json resolves config_file_path relative
# to cwd). Precedent and measurements: docs/NOTES-vit-htp-config.md.
#
# Usage: vl_text_ctxbin.sh [name] [cl_prefill] [ctx]
#   e.g. vl_text_ctxbin.sh qwen3vl-4b-w8a16 128 2048
set -euo pipefail
source "$(dirname "$0")/../env.sh"

NAME=${1:-qwen3vl-4b-w8a16}
CL=${2:-128}
CTX=${3:-2048}

QP=$LLMDEPLOY_DATA/work/quant/$NAME-prefill
QD=$LLMDEPLOY_DATA/work/quant/$NAME-decode
DLC=$LLMDEPLOY_DATA/work/dlc/$NAME
CTXBIN=$LLMDEPLOY_DATA/work/ctxbin/$NAME
ENC=$QP/model_filtered_renamed.encodings
CONVERTER="$QAIRT_SDK/bin/x86_64-linux-clang/qairt-converter"

LAYERS=${LAYERS:-36}
NDEEP=${NDEEP:-3}
HIDDEN=${HIDDEN:-2560}
NKV=${NKV:-8}
HEAD_DIM=${HEAD_DIM:-128}
PAST=$((CTX + CL - 1))       # 2175
TOTAL=$((PAST + 1))          # 2176

# Graph names are baked in at conversion time from the --output_path basename,
# dots included: converting to decode.dlc.new yields a graph called decode_dlc,
# and renaming the file afterwards does NOT change it. They must match
# graph_names below exactly, so convert straight to the final filename.
OPT_LEVEL=3          # these three are what the generated config exists to secure;
VTCM_MB=16           # step 3 reads them back out of the finalized binary and
HVX_THREADS=4        # fails the build if the config did not bind

for f in "$QP/model_renamed.onnx" "$QD/model_renamed.onnx" "$ENC"; do
    [ -f "$f" ] || { echo "missing $f -- run vl_text_build.sh first"; exit 1; }
done

disk_guard 20
echo "== [1/3] convert prefill -> W8A16 DLC (AR=$CL) =="
mkdir -p "$DLC"
$PY_QAIRT "$CONVERTER" --input_network "$QP/model_renamed.onnx" \
    --output_path "$DLC/prefill.dlc" --quantization_overrides "$ENC" \
    --float_bitwidth 16 --target_backend HTP \
    -d inputs_embeds "1,1,$CL,$HIDDEN" \
    -d attention_mask "1,$CL,$CL" \
    -d position_ids_cos "1,$CL,64" \
    -d position_ids_sin "1,$CL,64" \
    $(for i in $(seq 0 $((NDEEP - 1))); do
          echo -n "-d deepstack_visual_embed_$i 1,1,$CL,$HIDDEN "
      done)

disk_guard 20
echo "== [2/3] convert decode -> W8A16 DLC (AR=1, past=$PAST) =="
# Same $ENC as prefill: single lineage, byte-identical KV quant params.
DIMS=(-d inputs_embeds "1,1,1,$HIDDEN"
      -d attention_mask "1,1,$TOTAL"
      -d position_ids_cos "1,1,64"
      -d position_ids_sin "1,1,64")
for i in $(seq 0 $((NDEEP - 1))); do
    DIMS+=(-d "deepstack_visual_embed_$i" "1,1,1,$HIDDEN")
done
for i in $(seq 0 $((LAYERS - 1))); do
    DIMS+=(-d "past_key_${i}_in"   "1,$NKV,$HEAD_DIM,$PAST"
           -d "past_value_${i}_in" "1,$NKV,$PAST,$HEAD_DIM")
done
$PY_QAIRT "$CONVERTER" --input_network "$QD/model_renamed.onnx" \
    --output_path "$DLC/decode.dlc" --quantization_overrides "$ENC" \
    --float_bitwidth 16 --target_backend HTP "${DIMS[@]}"

ls -lh "$DLC"

disk_guard 20
echo "== [3/3] two-graph weight-shared ctx-bin (vtcm $VTCM_MB, unsigned PD, v81) =="
mkdir -p "$CTXBIN"
# weight_sharing_enabled matters here in a way it did not for the single-graph
# ViT: prefill and decode hold the same ~4 GB of weights, so without sharing the
# binary roughly doubles.
cat > "$CTXBIN/htp_backend_config.json" <<JSONEOF
{
  "graphs": [
    {
      "graph_names": [
        "prefill",
        "decode"
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
  ],
  "context": {
    "weight_sharing_enabled": true
  }
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
    --dlc_path "$DLC/prefill.dlc,$DLC/decode.dlc" \
    --backend "$QAIRT_SDK/lib/x86_64-linux-clang/libQnnHtp.so" \
    --output_dir "$CTXBIN" --binary_file "${NAME}_ctx" \
    --config_file htp_config.json

qnn-context-binary-utility --context_binary "$CTXBIN/${NAME}_ctx.bin" \
    --json_file "$CTXBIN/info.json"

$PY_DEPLOY - <<PYEOF
import json
import sys

d = json.load(open("$CTXBIN/info.json"))
graphs = d["info"]["graphs"]
by_name = {g["info"]["graphName"]: g["info"] for g in graphs}
for n in by_name:
    print("GRAPH:", n)

# A backend-extension config that fails to bind does not error -- the graph just
# compiles with defaults (O=0, vtcm 4 MB, 0 HVX threads) and the log stays clean.
# That exact silent regression shipped once already: verify32 omitted from
# graph_names, 4 MB VTCM + 24 MB spill on device, LADE SIGSEGV
# (docs/BUILD_GUIDE.md section 5.4b). With no device, this readback is the only
# place it is detectable, so it is a build failure, not a warning.
errs = []
want_graphs = {"prefill", "decode"}
if set(by_name) != want_graphs:
    errs.append("graphs are %r, expected %r -- graph_names keys on these strings"
                % (sorted(by_name), sorted(want_graphs)))

for name, info in sorted(by_name.items()):
    blob = info.get("graphBlobInfo", {}).get("info", {})
    for key, want in (("optimizationLevel", $OPT_LEVEL),
                      ("vtcmSize", $VTCM_MB),
                      ("numHvxThreads", $HVX_THREADS)):
        if blob.get(key) != want:
            errs.append("%s: %s is %r, expected %r" % (name, key, blob.get(key), want))

# Weight sharing is the difference between one ~4 GB weight set and two. It is
# reported under graphBlobInfoV2, not graphBlobInfo.
shared = 0
for name, info in by_name.items():
    v2 = info.get("graphBlobInfoV2", {})
    shared = max(shared, (v2.get("info", v2) or {}).get("sharedWeightsSize", 0) or 0)
if shared <= 0:
    errs.append("sharedWeightsSize is %r -- weight sharing did NOT engage, so "
                "the binary carries both graphs' weights separately" % shared)

# The 2026-08-11 device-garbage contract: qualla reads logits row n_process-1
# assuming all-position output. A [1,1,vocab] prefill head passes load
# validation silently and then reads out of bounds.
pf = by_name.get("prefill", {})
logits = [t["info"]["dimensions"] for t in pf.get("graphOutputs", [])
          if t["info"]["name"] == "logits"]
if logits and list(logits[0][:2]) != [1, $CL]:
    errs.append("prefill logits are %r, expected all-position [1, $CL, vocab]"
                % (logits[0],))

for name, info in sorted(by_name.items()):
    ins = {t["info"]["name"] for t in info.get("graphInputs", [])}
    if "inputs_embeds" not in ins:
        errs.append("%s: no 'inputs_embeds' input -- Genie keys on that exact "
                    "name to select the embeddings path" % name)
    print("  %-8s inputs=%d outputs=%d"
          % (name, len(info.get("graphInputs", [])), len(info.get("graphOutputs", []))))

if errs:
    print("BUILD REJECTED:")
    for e in errs:
        print("  - " + e)
    print("  see docs/NOTES-vit-htp-config.md")
    sys.exit(1)

blob = by_name["prefill"]["graphBlobInfo"]["info"]
print("CONFIG BOUND: O=%d vtcm=%d MB hvx_threads=%d  sharedWeights=%.2f GB"
      % (blob["optimizationLevel"], blob["vtcmSize"], blob["numHvxThreads"],
         shared / 2**30))
PYEOF

ls -lh "$CTXBIN"
echo "CTXBIN COMPLETE: $CTXBIN/${NAME}_ctx.bin"
