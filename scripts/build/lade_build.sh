#!/usr/bin/env bash
# Add an AR=32 verification graph to an existing baseline build and produce a
# 3-graph ctx-bin (prefill CL=128 + decode AR=1 + verify AR=32) for Genie
# lookahead decoding (dialog type "lade" -> qualla lhd-dec).
#
# Requires a completed full_build.sh run for <name>: reuses its prefill quant
# dir (model_torch.encodings for scale adoption, model_filtered.encodings for
# rename) and its prefill/decode DLCs. The verify graph converts against the
# SAME prefill encodings file, so all cross-graph scales stay bit-identical.
#
# Runtime contract (pinned from SDK qualla source, see docs/NOTES-genie-io.md):
#   - lhd-dec batches up to (ngram-1)*(window+gcap) tokens per step; the lade
#     config MUST keep that <= 32 so batches route to this graph and not the
#     prefill graph (no past-KV inputs — cannot serve incremental verification).
#   - engine detects AR = numElements(input_ids), CL = mask last dim.
#   - verify graph must emit logits for ALL 32 positions.
#
# Usage: lade_build.sh <name> [cl_prefill] [ctx] [ar]
#   e.g. lade_build.sh qwen3-0.6b-w8a16 128 1024 32
#
# Fused variants (docs/MAX_TPS_QWEN3_0.6B.md §3) need two overrides, both
# no-ops when unset:
#   FUSE_FLAGS  appended verbatim to every quantize_aimet.py call. The export
#               wrapper's structure comes from --fuse-qkv/--fuse-gate-up, so
#               without this the verify graph is exported UNFUSED against a
#               fused donor's encodings.
#   ENC_SRC     the encodings file used for the rename + conversion. A fused
#               build's truth is qkv_build.sh's model_surgery.encodings (donor
#               q_proj INT16 encoding grafted onto the Q split), not
#               model_filtered_renamed.encodings. Already in renamed-I/O space,
#               so it is passed through unchanged — same as qkv_build.sh §6/§7.
#   e.g. FUSE_FLAGS="--fuse-qkv --fuse-gate-up" \
#        ENC_SRC=$DATA/work/quant/<name>-prefill/model_surgery.encodings \
#        lade_build.sh qwen3-0.6b-w8a16-fuseqkvgu 128 1024 32
set -euo pipefail
source "$(dirname "$0")/../env.sh"

NAME=${1:?name}
CL=${2:-128}
CTX=${3:-1024}
AR=${4:-32}
FUSE=()
[[ -n "${FUSE_FLAGS:-}" ]] && read -r -a FUSE <<< "$FUSE_FLAGS"

MODEL=${MODEL:-$LLMDEPLOY_DATA/models/Qwen3-0.6B}
QP=$LLMDEPLOY_DATA/work/quant/$NAME-prefill
QV=$LLMDEPLOY_DATA/work/quant/$NAME-verify$AR
DLC=$LLMDEPLOY_DATA/work/dlc/$NAME
CTXBIN=$LLMDEPLOY_DATA/work/ctxbin/$NAME-lade
PY=$LLMDEPLOY_DATA/envs/qwen3-deploy/bin/python
CONVERTER="$QAIRT_SDK/bin/x86_64-linux-clang/qairt-converter"
PAST=$((CTX + CL - AR))
TOTAL=$((CTX + CL))

for f in "$QP/model_torch.encodings" "$QP/model_filtered.encodings" \
         "$DLC/prefill.dlc" "$DLC/decode.dlc"; do
  [[ -f $f ]] || { echo "MISSING prerequisite: $f (run full_build.sh $NAME first)"; exit 1; }
done

# Cross-graph rule: every DLC in one ctx-bin must convert against the SAME
# encodings file, or the KV quant params differ and Genie fails the load.
if [[ -n "${ENC_SRC:-}" ]]; then
  [[ -f $ENC_SRC ]] || { echo "MISSING ENC_SRC: $ENC_SRC"; exit 1; }
  ENC_IN=$ENC_SRC        # already renamed; rename's own output is discarded
  ENC=$ENC_SRC
  echo "== encodings override: $ENC (fused lineage) =="
else
  ENC_IN=$QP/model_filtered.encodings
  ENC=$QP/model_filtered_renamed.encodings
fi

disk_guard 20
echo "== [1/4] AIMET verify-graph export (AR=$AR, past=$PAST) ${FUSE[*]:-} =="
$PY "$LLMDEPLOY_ROOT/scripts/quant/quantize_aimet.py" --model "$MODEL" \
    --cl-prefill "$CL" --ctx "$CTX" --decode-ar "$AR" \
    --export-decode "$QP" --out "$QV" ${QUANT_DEVICE:+--device "$QUANT_DEVICE"} \
    "${FUSE[@]}"

disk_guard
echo "== [2/4] canonical I/O rename (verify) =="
$PY "$LLMDEPLOY_ROOT/scripts/quant/rename_aimet_io.py" \
    --model "$QV/model.onnx" --encodings "$ENC_IN" \
    --layers 28 --with-past

disk_guard
echo "== [3/4] convert verify -> DLC (prefill encodings; AR=$AR) =="
DIMS=(-d input_ids "1,$AR" -d attention_mask "1,$AR,$TOTAL"
      -d position_ids_cos "1,$AR,64" -d position_ids_sin "1,$AR,64")
for i in $(seq 0 27); do
  DIMS+=(-d "past_key_${i}_in" "1,8,128,$PAST" -d "past_value_${i}_in" "1,8,$PAST,128")
done
$PY_QAIRT "$CONVERTER" --input_network "$QV/model_renamed.onnx" \
    --output_path "$DLC/verify$AR.dlc" --quantization_overrides "$ENC" \
    --float_bitwidth 16 --target_backend HTP "${DIMS[@]}"

disk_guard
echo "== [4/4] three-graph ctx-bin (prefill + decode + verify$AR) =="
cd "$LLMDEPLOY_ROOT/configs"
qnn-context-binary-generator \
    --model "$QAIRT_SDK/lib/x86_64-linux-clang/libQnnModelDlc.so" \
    --dlc_path "$DLC/prefill.dlc,$DLC/decode.dlc,$DLC/verify$AR.dlc" \
    --backend "$QAIRT_SDK/lib/x86_64-linux-clang/libQnnHtp.so" \
    --output_dir "$CTXBIN" --binary_file "${NAME}-lade_ctx" \
    --config_file htp_config.json
qnn-context-binary-utility --context_binary "$CTXBIN/${NAME}-lade_ctx.bin" \
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
echo "LADE BUILD COMPLETE: $CTXBIN/${NAME}-lade_ctx.bin"
