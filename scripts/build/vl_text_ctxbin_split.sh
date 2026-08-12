#!/usr/bin/env bash
# Qwen3-VL text tower, SPLIT build: 4 DLCs -> 2 weight-shared ctx-bins.
#
# The unsplit tower cannot ship: qnn-context-binary-generator enforces a hard
# 3.5 GiB per-graph serialization limit and the 4B tower estimates 4.18 GiB.
# That estimate is weights-only -- proven by generating from the prefill graph
# alone, which has no past-KV at all and reported the byte-identical figure --
# so shrinking the context does not help. See docs/NOTES-genie-splits.md.
#
# Layout (names are the wiring: Genie assigns split index by SORTED graph name
# within each (AR, CL) group, and the name is baked in from the DLC basename):
#
#   ctx-bin 1_of_2   prefill_0 + decode_0   layers 0-17    out: last_hidden_states
#   ctx-bin 2_of_2   prefill_1 + decode_1   layers 18-35   in:  last_hidden_states
#
# Encodings come from splitting ONE whole-tower calibration (eval 4/4), so the
# two chunks cannot disagree about the boundary tensor.
#
# Usage: vl_text_ctxbin_split.sh [name] [cl_prefill] [ctx] [split_at]
set -euo pipefail
source "$(dirname "$0")/../env.sh"

NAME=${1:-qwen3vl-4b-w8a16}
CL=${2:-128}
CTX=${3:-2048}
SPLIT=${4:-18}

# AIMET's own export, cut at the layer seam by split_aimet_onnx.py. NOT the
# torch re-export: torch.onnx.export renames Linear weights, so only 72/198
# param encodings would match and most layers would convert as FP16.
ONNX=${ONNX:-$LLMDEPLOY_DATA/work/onnx/qwen3vl-4b-aimet-split}
ENCDIR=$LLMDEPLOY_DATA/work/quant/$NAME-split-enc
DLC=$LLMDEPLOY_DATA/work/dlc/$NAME-split
CTXBIN=$LLMDEPLOY_DATA/work/ctxbin/$NAME-split
CONVERTER="$QAIRT_SDK/bin/x86_64-linux-clang/qairt-converter"

LAYERS=${LAYERS:-36}
NDEEP=${NDEEP:-3}
HIDDEN=${HIDDEN:-2560}
NKV=${NKV:-8}
HEAD_DIM=${HEAD_DIM:-128}
PAST=$((CTX + CL - 1))
TOTAL=$((PAST + 1))

OPT_LEVEL=3          # these three are what the generated config exists to secure;
VTCM_MB=16           # step 3 reads them back out of each finalized binary and
HVX_THREADS=4        # fails the build if the config did not bind

for g in prefill_0 decode_0 prefill_1 decode_1; do
    [ -f "$ONNX/$g/$g.onnx" ] || { echo "missing $ONNX/$g/$g.onnx -- run split_aimet_onnx.py"; exit 1; }
done
for c in 0 1; do
    [ -f "$ENCDIR/chunk$c.encodings" ] || { echo "missing $ENCDIR/chunk$c.encodings -- run split_encodings.py"; exit 1; }
done

mkdir -p "$DLC"

# head_dims <seq> <is_first> -> the four non-KV inputs shared by every graph.
# The first chunk takes Genie's rank-4 inputs_embeds; a later chunk takes the
# rank-3 boundary tensor. Genie connects splits by NAME and does not check that
# their shapes agree, so both sides are rank-3 by construction.
head_dims() {
    local seq=$1 first=$2
    if [ "$first" = "1" ]; then
        echo -n "-d inputs_embeds 1,1,$seq,$HIDDEN "
    else
        echo -n "-d last_hidden_states 1,$seq,$HIDDEN "
    fi
    echo -n "-d attention_mask 1,$seq,$3 -d position_ids_cos 1,$seq,64 -d position_ids_sin 1,$seq,64 "
}

convert() {
    local graph=$1 seq=$2 past=$3 first=$4 start=$5 end=$6 chunk=$7 total=$8
    local args=()
    # shellcheck disable=SC2046
    args+=($(head_dims "$seq" "$first" "$total"))
    if [ "$first" = "1" ]; then
        for i in $(seq 0 $((NDEEP - 1))); do
            args+=(-d "deepstack_visual_embed_$i" "1,1,$seq,$HIDDEN")
        done
    fi
    if [ "$past" -gt 0 ]; then
        # KV keeps GLOBAL layer indices: renumbering a later chunk to 0..n-1
        # would collide with the first chunk's names, and same-named tensors
        # across splits are THE SAME BUFFER -- the caches would silently alias.
        for i in $(seq "$start" $((end - 1))); do
            args+=(-d "past_key_${i}_in"   "1,$NKV,$HEAD_DIM,$past"
                   -d "past_value_${i}_in" "1,$NKV,$past,$HEAD_DIM")
        done
    fi
    disk_guard 20
    echo "== convert $graph (S=$seq, past=$past, layers $start-$((end - 1))) =="
    $PY_QAIRT "$CONVERTER" --input_network "$ONNX/$graph/$graph.onnx" \
        --output_path "$DLC/$graph.dlc" \
        --quantization_overrides "$ENCDIR/chunk$chunk.encodings" \
        --float_bitwidth 16 --target_backend HTP "${args[@]}"
}

convert prefill_0 "$CL" 0      1 0       "$SPLIT" 0 "$CL"
convert decode_0  1     "$PAST" 1 0      "$SPLIT" 0 "$TOTAL"
convert prefill_1 "$CL" 0      0 "$SPLIT" "$LAYERS" 1 "$CL"
convert decode_1  1     "$PAST" 0 "$SPLIT" "$LAYERS" 1 "$TOTAL"
ls -lh "$DLC"

# One ctx-bin per chunk, holding that chunk's two AR variants so its weights are
# shared between them. Splitting prefill/decode into SEPARATE binaries instead
# would defeat that and double the payload.
for c in 0 1; do
    part=$((c + 1))
    OUT=$CTXBIN/${part}_of_2
    mkdir -p "$OUT"
    cat > "$OUT/htp_backend_config.json" <<JSONEOF
{
  "graphs": [
    {
      "graph_names": [
        "prefill_$c",
        "decode_$c"
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
    cat > "$OUT/htp_config.json" <<JSONEOF
{
  "backend_extensions": {
    "shared_library_path": "libQnnHtpNetRunExtensions.so",
    "config_file_path": "htp_backend_config.json"
  }
}
JSONEOF
    disk_guard 20
    echo "== ctx-bin ${part}_of_2 (prefill_$c + decode_$c) =="
    cd "$OUT"
    qnn-context-binary-generator \
        --model "$QAIRT_SDK/lib/x86_64-linux-clang/libQnnModelDlc.so" \
        --dlc_path "$DLC/prefill_$c.dlc,$DLC/decode_$c.dlc" \
        --backend "$QAIRT_SDK/lib/x86_64-linux-clang/libQnnHtp.so" \
        --output_dir "$OUT" --binary_file "${NAME}_${part}_of_2" \
        --config_file htp_config.json
    qnn-context-binary-utility --context_binary "$OUT/${NAME}_${part}_of_2.bin" \
        --json_file "$OUT/info.json"
done

echo "== verify both binaries =="
$PY_DEPLOY - <<PYEOF
import json, sys

errs = []
for c in (0, 1):
    part = c + 1
    p = f"$CTXBIN/{part}_of_2/info.json"
    d = json.load(open(p))
    graphs = {g["info"]["graphName"]: g["info"] for g in d["info"]["graphs"]}
    want = {f"prefill_{c}", f"decode_{c}"}
    print(f"{part}_of_2: {sorted(graphs)}")
    if set(graphs) != want:
        errs.append(f"{part}_of_2 graphs {sorted(graphs)} != {sorted(want)}")

    # A backend-extension config that fails to bind does not error -- the graph
    # compiles with defaults (O=0, vtcm 4 MB) and the log stays clean. That
    # silent regression shipped once already (BUILD_GUIDE section 5.4b, LADE
    # SIGSEGV). With no device this readback is the only place it is detectable.
    for name, info in sorted(graphs.items()):
        blob = info.get("graphBlobInfo", {}).get("info", {})
        for key, wanted in (("optimizationLevel", $OPT_LEVEL),
                            ("vtcmSize", $VTCM_MB),
                            ("numHvxThreads", $HVX_THREADS)):
            if blob.get(key) != wanted:
                errs.append(f"{name}: {key} is {blob.get(key)!r}, expected {wanted!r}")

    shared = 0
    for info in graphs.values():
        v2 = info.get("graphBlobInfoV2", {})
        shared = max(shared, (v2.get("info", v2) or {}).get("sharedWeightsSize", 0) or 0)
    if shared <= 0:
        errs.append(f"{part}_of_2 sharedWeightsSize {shared} -- weight sharing did not engage")

    for name, info in sorted(graphs.items()):
        ins = {t["info"]["name"] for t in info.get("graphInputs", [])}
        outs = {t["info"]["name"]: t["info"]["dimensions"]
                for t in info.get("graphOutputs", [])}
        # First split must expose inputs_embeds; last must emit logits. Genie
        # checks exactly this (validateModel 1a and 2).
        if c == 0:
            if "inputs_embeds" not in ins:
                errs.append(f"{name}: no inputs_embeds (Genie requires it in the first split)")
            if "last_hidden_states" not in outs:
                errs.append(f"{name}: no last_hidden_states output to hand to split 2")
        else:
            if "last_hidden_states" not in ins:
                errs.append(f"{name}: no last_hidden_states input from split 1")
            if "logits" not in outs:
                errs.append(f"{name}: no logits (Genie requires it in the last split)")
            if name.startswith("prefill") and list(outs["logits"][:2]) != [1, $CL]:
                errs.append(f"{name}: logits {outs['logits']} not all-position [1, $CL, vocab]")
        print(f"  {name:10s} in={len(ins):3d} out={len(outs):3d}")
    print(f"  sharedWeights={shared/2**30:.2f} GB")

if errs:
    print("BUILD REJECTED:")
    for e in errs:
        print("  - " + e)
    sys.exit(1)
print("BOTH BINARIES VERIFIED")
PYEOF

du -sh "$CTXBIN"/*
echo "SPLIT CTXBIN COMPLETE: $CTXBIN"
