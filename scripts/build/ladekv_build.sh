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
set -euo pipefail
source "$(dirname "$0")/../env.sh"

NAME=${1:?name}
CL=${2:-128}
CTX=${3:-1024}
AR=${4:-128}          # past-KV prefill AR
VAR=${VERIFY_AR:-32}  # existing verify graph AR

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

for f in "$QP/model_torch.encodings" "$QP/model_filtered.encodings" \
         "$DLC/decode.dlc" "$DLC/verify$VAR.dlc"; do
  [[ -f $f ]] || { echo "MISSING prerequisite: $f (run full_build.sh + lade_build.sh $NAME first)"; exit 1; }
done

disk_guard() {  # standing constraint: never let Windows C: run dry
  local free_gb
  free_gb=$(df --output=avail -BG /mnt/c | tail -1 | tr -dc 0-9)
  if (( free_gb < 6 )); then
    echo "ABORT: C: free space ${free_gb}GB < 6GB"; exit 1
  fi
}

if [[ -f "$QKV/model_renamed.onnx" && -z "${FORCE_EXPORT:-}" ]]; then
  echo "== [1-2/5] SKIP export+rename ($QKV/model_renamed.onnx exists; FORCE_EXPORT=1 to redo) =="
else
  disk_guard
  echo "== [1/5] AIMET past-KV prefill export (AR=$AR, past=$PAST) =="
  $PY "$LLMDEPLOY_ROOT/scripts/quant/quantize_aimet.py" --model "$MODEL" \
      --cl-prefill "$CL" --ctx "$CTX" --decode-ar "$AR" \
      --export-decode "$QP" --out "$QKV" ${QUANT_DEVICE:+--device "$QUANT_DEVICE"}

  disk_guard
  echo "== [2/5] canonical I/O rename =="
  $PY "$LLMDEPLOY_ROOT/scripts/quant/rename_aimet_io.py" \
      --model "$QKV/model.onnx" --encodings "$QP/model_filtered.encodings" \
      --layers 28 --with-past
fi
ENC=$QP/model_filtered_renamed.encodings

echo "== [3/5] FP parity: qualla feed pattern incl. chunking (AR=$AR) =="
$PY "$LLMDEPLOY_ROOT/scripts/validate/parity_ladekv_read.py" \
    --model "$MODEL" --onnx "$QKV/model_renamed.onnx" \
    --ar "$AR" --ctx "$CTX"

disk_guard
echo "== [4/5] convert past-KV prefill -> DLC =="
mkdir -p "$DLCKV"
DIMS=(-d input_ids "1,$AR" -d attention_mask "1,$AR,$TOTAL"
      -d position_ids_cos "1,$AR,64" -d position_ids_sin "1,$AR,64")
for i in $(seq 0 27); do
  DIMS+=(-d "past_key_${i}_in" "1,8,128,$PAST" -d "past_value_${i}_in" "1,8,$PAST,128")
done
$PY_QAIRT "$CONVERTER" --input_network "$QKV/model_renamed.onnx" \
    --output_path "$DLCKV/prefill.dlc" --quantization_overrides "$ENC" \
    --float_bitwidth 16 --target_backend HTP "${DIMS[@]}"

INFO=$DLCKV/prefill_info.txt
"$QAIRT_SDK/bin/x86_64-linux-clang/qairt-dlc-info" -i "$DLCKV/prefill.dlc" > "$INFO" 2>/dev/null
grep -E "^\| (logits|past_key_0_in)" "$INFO" || true
grep -qE "^\| logits +\| 1,$AR,151936" "$INFO" || {
  echo "FAIL: prefill logits shape is not [1,$AR,151936]"; exit 1; }
grep -qE "^\| past_key_0_in" "$INFO" || {
  echo "FAIL: prefill has no past-KV inputs — wrong graph exported"; exit 1; }

disk_guard
echo "== [5/5] three-graph ctx-bin (prefill-kv + decode + verify$VAR) =="
cd "$LLMDEPLOY_ROOT/configs"
qnn-context-binary-generator \
    --model "$QAIRT_SDK/lib/x86_64-linux-clang/libQnnModelDlc.so" \
    --dlc_path "$DLCKV/prefill.dlc,$DLC/decode.dlc,$DLC/verify$VAR.dlc" \
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
echo "LADEKV BUILD COMPLETE: $CTXBIN/${NAME}-ladekv_ctx.bin"
