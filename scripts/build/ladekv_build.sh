#!/usr/bin/env bash
# Build the "ladekv" variant: replace the bertcache prefill (AR-128 CL-128,
# no past-KV inputs) with a PAST-KV prefill graph (AR=128, CL=1152, past=1024,
# all-position logits) and stitch prefill + decode + verify32 into a 3-graph
# ctx-bin.
#
# Why: qualla's lhd-dec (dialog type "lade") SIGSEGVs in libGenie.so when its
# warmup/verification batches land while the only prompt-capable graph is a
# no-past-KV bertcache prefill (device reports 2026-08-10/11, fault x0 =
# 0x6b8b4567 uninitialized-memory sentinel). SDK lade sample models use
# past-KV prefill graphs. Trade-off: basic mode loses the ~23 tok/s bertcache
# early phase (generation goes through AR-1 decode from token 1); lade mode
# targets 2-4x on all decode. Also enables >128-token prompts via chunking.
#
# Reuses decode.dlc + verify32.dlc and the baseline prefill quant encodings
# (adoption keeps all cross-graph scales bit-identical) — only the prefill
# graph is new. The new prefill DLC lives in its own dir so it can keep the
# filename prefill.dlc (graph names derive from DLC filenames).
#
# Requires completed full_build.sh + lade_build.sh for <name>.
# Usage: ladekv_build.sh <name> [cl_prefill] [ctx] [ar_prefill]
#   e.g. ladekv_build.sh qwen3-0.6b-w8a16 128 1024 128
#
# Fused variants (see docs/PLAN_0.6B_max_tps.md): FUSE_FLAGS and ENC_SRC, same
# contract as lade_build.sh — FUSE_FLAGS is appended to the quantize_aimet.py
# export (the wrapper's structure depends on it) and ENC_SRC replaces the
# encodings used for rename + conversion with the fused build's
# model_surgery.encodings. All three graphs in the ctx-bin must share it.
set -euo pipefail
source "$(dirname "$0")/../env.sh"

NAME=${1:?name}
CL=${2:-128}
CTX=${3:-1024}
AR=${4:-128}          # past-KV prefill AR
VAR=${VERIFY_AR:-32}  # existing verify graph AR
FUSE=()
[[ -n "${FUSE_FLAGS:-}" ]] && read -r -a FUSE <<< "$FUSE_FLAGS"

# --input-embeds inside FUSE_FLAGS turns this into the LUT probe's past-KV
# prefill: the tower is fed hidden states from an external LUT. Detected here
# because, as in full_build.sh, it has to change the I/O rename and the
# converter dims too -- a flag that only reached the quantizer would build a
# graph still called input_ids, which qualla drives as token ids
# (nsp-model.cpp:668 matches the name literally).
EMBEDS=0
for f in "${FUSE[@]:-}"; do [ "$f" = "--input-embeds" ] && EMBEDS=1; done
# Same contract as full_build.sh: an embeddings-fed inputs_embeds must stay
# FLOAT_32, because qualla's only implemented float path is fp32 lut -> fp32
# input (dialog.cpp:678 copies raw bytes; basic.cpp:161's fp32->fp16 case is an
# empty `// TODO`). Must be set HERE too -- this script converts the past-KV
# prefill separately, and a flag applied only in full_build.sh would ship a
# correct decode graph beside a prefill still declaring FLOAT_16.
PRES=()
if [ "$EMBEDS" = "1" ] && [ -z "${EMBEDS_FP16_IN:-}" ]; then
    PRES=(--preserve_io_datatype inputs_embeds)
fi
# NO_VERIFY=1 builds prefill+decode only. verify32 exists for LADE, which is
# parked as a 30% regression and unused in basic mode, so the probe omits it
# rather than paying for an export that changes nothing it measures.
NO_VERIFY=${NO_VERIFY:-0}

MODEL=${MODEL:-$LLMDEPLOY_DATA/models/Qwen3-0.6B}
QP=$LLMDEPLOY_DATA/work/quant/$NAME-prefill
QKV=$LLMDEPLOY_DATA/work/quant/$NAME-prefillkv$AR
DLC=$LLMDEPLOY_DATA/work/dlc/$NAME
DLCKV=$LLMDEPLOY_DATA/work/dlc/$NAME-ladekv
CTXBIN=$LLMDEPLOY_DATA/work/ctxbin/$NAME-ladekv
PY=$LLMDEPLOY_DATA/envs/qwen3-deploy/bin/python
CONVERTER="$QAIRT_SDK/bin/x86_64-linux-clang/qairt-converter"
PAST=$((CTX + CL - AR))   # 1024 for AR=128
TOTAL=$((CTX + CL))       # 1152

REQ=("$QP/model_torch.encodings" "$QP/model_filtered.encodings" "$DLC/decode.dlc")
[ "$NO_VERIFY" = "1" ] || REQ+=("$DLC/verify$VAR.dlc")
for f in "${REQ[@]}"; do
  [[ -f $f ]] || { echo "MISSING prerequisite: $f (run full_build.sh${NO_VERIFY:+ }$([ "$NO_VERIFY" = 1 ] || echo "+ lade_build.sh") $NAME first)"; exit 1; }
done

if [[ -n "${ENC_SRC:-}" ]]; then
  [[ -f $ENC_SRC ]] || { echo "MISSING ENC_SRC: $ENC_SRC"; exit 1; }
  ENC_IN=$ENC_SRC        # already renamed; rename's own output is discarded
  ENC=$ENC_SRC
  echo "== encodings override: $ENC (fused lineage) =="
else
  ENC_IN=$QP/model_filtered.encodings
  ENC=$QP/model_filtered_renamed.encodings
fi

if [[ -f "$QKV/model_renamed.onnx" && -z "${FORCE_EXPORT:-}" ]]; then
  echo "== [1-2/5] SKIP export+rename ($QKV/model_renamed.onnx exists; FORCE_EXPORT=1 to redo) =="
else
  disk_guard 20
  echo "== [1/5] AIMET past-KV prefill export (AR=$AR, past=$PAST) ${FUSE[*]:-} =="
  $PY "$LLMDEPLOY_ROOT/scripts/quant/quantize_aimet.py" --model "$MODEL" \
      --cl-prefill "$CL" --ctx "$CTX" --decode-ar "$AR" \
      --export-decode "$QP" --out "$QKV" ${QUANT_DEVICE:+--device "$QUANT_DEVICE"} \
      "${FUSE[@]}"

  disk_guard
  echo "== [2/5] canonical I/O rename =="
  RENAME_FLAGS=()
  [ "$EMBEDS" = "1" ] && RENAME_FLAGS=(--vl-text --n-deepstack 0)
  $PY "$LLMDEPLOY_ROOT/scripts/quant/rename_aimet_io.py" \
      --model "$QKV/model.onnx" --encodings "$ENC_IN" \
      --layers 28 --with-past "${RENAME_FLAGS[@]}"
fi

if [ "$EMBEDS" = "1" ]; then
    echo "== [3/5] FP parity: LUT-fed (parity_ladekv_read.py feeds TOKEN IDS and"
    echo "         cannot drive an embeddings-in graph; parity_lutprobe.py is"
    echo "         the equivalent gate and reads the LUT at the runtime's own"
    echo "         byte offsets) =="
    $PY "$LLMDEPLOY_ROOT/scripts/validate/parity_lutprobe.py" \
        --onnx "$QKV/model_renamed.onnx" \
        --lut "${LUT_DIR:-$LLMDEPLOY_DATA/work/lut/qwen3-0.6b}" \
        --model "$MODEL"
else
    echo "== [3/5] FP parity: qualla feed pattern incl. chunking (AR=$AR) =="
    $PY "$LLMDEPLOY_ROOT/scripts/validate/parity_ladekv_read.py" \
        --model "$MODEL" --onnx "$QKV/model_renamed.onnx" \
        --ar "$AR" --ctx "$CTX"
fi

disk_guard
echo "== [4/5] convert past-KV prefill -> DLC =="
mkdir -p "$DLCKV"
HID=$($PY -c "import json; print(json.load(open('$MODEL/config.json'))['hidden_size'])")
if [ "$EMBEDS" = "1" ]; then
  DIMS=(-d inputs_embeds "1,1,$AR,$HID" -d attention_mask "1,$AR,$TOTAL"
        -d position_ids_cos "1,$AR,64" -d position_ids_sin "1,$AR,64")
else
  DIMS=(-d input_ids "1,$AR" -d attention_mask "1,$AR,$TOTAL"
      -d position_ids_cos "1,$AR,64" -d position_ids_sin "1,$AR,64")
fi
for i in $(seq 0 27); do
  DIMS+=(-d "past_key_${i}_in" "1,8,128,$PAST" -d "past_value_${i}_in" "1,8,$PAST,128")
done
$PY_QAIRT "$CONVERTER" --input_network "$QKV/model_renamed.onnx" \
    --output_path "$DLCKV/prefill.dlc" --quantization_overrides "$ENC" \
    --float_bitwidth 16 --target_backend HTP "${PRES[@]}" "${DIMS[@]}"

INFO=$DLCKV/prefill_info.txt
"$QAIRT_SDK/bin/x86_64-linux-clang/qairt-dlc-info" -i "$DLCKV/prefill.dlc" > "$INFO" 2>/dev/null
grep -E "^\| (logits|past_key_0_in)" "$INFO" || true
grep -qE "^\| logits +\| 1,$AR,151936" "$INFO" || {
  echo "FAIL: prefill logits shape is not [1,$AR,151936]"; exit 1; }
grep -qE "^\| past_key_0_in" "$INFO" || {
  echo "FAIL: prefill has no past-KV inputs — wrong graph exported"; exit 1; }

disk_guard
# Unlike ctxbin_variant.sh this build reads the SHARED configs/ backend config,
# so the config is not an argument and cannot be hashed from one. Fold its
# content hash into the recipe instead -- otherwise editing configs/ would leave
# the key unchanged and the registry would claim a stale bin is current.
if [ "$NO_VERIFY" = "1" ]; then
  DLC_LIST="$DLCKV/prefill.dlc,$DLC/decode.dlc"; GRAPH_LIST="prefill,decode"
else
  DLC_LIST="$DLCKV/prefill.dlc,$DLC/decode.dlc,$DLC/verify$VAR.dlc"
  GRAPH_LIST="prefill,decode,verify$VAR"
fi
coord_guard "ladekv $NAME" \
    "$DLC_LIST" \
    "$GRAPH_LIST" \
    "{\"config_md5\": \"$(md5sum "$LLMDEPLOY_ROOT/configs/htp_backend_config.json" | cut -d' ' -f1)\"}" \
    20
echo "== [5/5] ctx-bin ($GRAPH_LIST) =="
cd "$LLMDEPLOY_ROOT/configs"
qnn-context-binary-generator \
    --model "$QAIRT_SDK/lib/x86_64-linux-clang/libQnnModelDlc.so" \
    --dlc_path "$DLC_LIST" \
    --backend "$QAIRT_SDK/lib/x86_64-linux-clang/libQnnHtp.so" \
    --output_dir "$CTXBIN" --binary_file "${NAME}-ladekv_ctx" \
    --config_file htp_config.json
qnn-context-binary-utility --context_binary "$CTXBIN/${NAME}-ladekv_ctx.bin" \
    --json_file "$CTXBIN/info.json"
$PY - <<PYEOF
import json
d = json.load(open("$CTXBIN/info.json"))
for g in d["info"]["graphs"]:
    gi = g["info"]
    print("GRAPH:", gi["graphName"],
          "inputs:", len(gi.get("graphInputs", [])),
          "outputs:", len(gi.get("graphOutputs", [])))
PYEOF
ls -lh "$CTXBIN"
coord_done "${NAME}-ladekv" "$CTXBIN/${NAME}-ladekv_ctx.bin" ctxbin "prefill-kv+decode+verify$VAR"
echo "LADEKV BUILD COMPLETE: $CTXBIN/${NAME}-ladekv_ctx.bin"
