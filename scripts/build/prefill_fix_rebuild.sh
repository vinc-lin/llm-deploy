#!/usr/bin/env bash
# Regenerate ONLY the prefill graph of an existing variant build with the
# fixed all-position-logits head (2026-08-11 device-garbage root cause:
# Genie's basic dialog samples logits row n_process-1 of an all-position
# tensor; our old prefill emitted last-token-only -> out-of-bounds read).
#
# Reuses the variant's OWN previous calibration via --adopt-encodings, so
# every scale stays bit-identical to the existing decode/verify DLCs and
# only prefill.dlc needs reconverting. Ctx-bin + bundle are NOT done here.
#
# Usage: prefill_fix_rebuild.sh <name> [cl] [ctx] [extra quantize flags...]
#   e.g. prefill_fix_rebuild.sh qwen3-0.6b-w8a16 128 1024
#        prefill_fix_rebuild.sh qwen3-0.6b-w8a16-fusegu 128 1024 --fuse-gate-up
#        SURGERY=1 DONOR=qwen3-0.6b-w8a16 prefill_fix_rebuild.sh \
#            qwen3-0.6b-w8a16-fuseqkvgu 128 1024 --fuse-qkv --fuse-gate-up
set -euo pipefail
source "$(dirname "$0")/../env.sh"

NAME=${1:?name}
CL=${2:-128}
CTX=${3:-1024}
shift; shift || true; shift || true
EXTRA_FLAGS=("$@")

MODEL=${MODEL:-$LLMDEPLOY_DATA/models/Qwen3-0.6B}
QP=$LLMDEPLOY_DATA/work/quant/$NAME-prefill
DLC=$LLMDEPLOY_DATA/work/dlc/$NAME
PY=$LLMDEPLOY_DATA/envs/qwen3-deploy/bin/python
CONVERTER="$QAIRT_SDK/bin/x86_64-linux-clang/qairt-converter"

for f in "$QP/model_torch.encodings" "$DLC/decode.dlc"; do
  [[ -f $f ]] || { echo "MISSING prerequisite: $f"; exit 1; }
done

disk_guard 20
echo "== [1/4] prefill re-export (all-position logits, adopt encodings) =="
$PY "$LLMDEPLOY_ROOT/scripts/quant/quantize_aimet.py" --model "$MODEL" \
    --cl-prefill "$CL" --out "$QP" --adopt-encodings "$QP" \
    ${QUANT_DEVICE:+--device "$QUANT_DEVICE"} "${EXTRA_FLAGS[@]}"

echo "== [2/4] filter + rename =="
$PY "$LLMDEPLOY_ROOT/scripts/quant/filter_aimet_w8a16.py" "$QP/model.encodings"
$PY "$LLMDEPLOY_ROOT/scripts/quant/rename_aimet_io.py" \
    --model "$QP/model.onnx" --encodings "$QP/model_filtered.encodings" --layers 28

ENC=$QP/model_filtered_renamed.encodings
if [[ ${SURGERY:-0} == 1 ]]; then
  DONORQP=$LLMDEPLOY_DATA/work/quant/${DONOR:?SURGERY=1 needs DONOR=<baseline name>}-prefill
  echo "== [2b] QKV encodings surgery (donor: $DONORQP) =="
  $PY "$LLMDEPLOY_ROOT/scripts/export/qkv_surgery.py" \
      --fused-onnx "$QP/model_renamed.onnx" \
      --fused-encodings "$QP/model_filtered_renamed.encodings" \
      --donor-onnx "$DONORQP/model_renamed.onnx" \
      --donor-encodings "$DONORQP/model_filtered_renamed.encodings" \
      --out "$QP/model_surgery.encodings"
  ENC=$QP/model_surgery.encodings
fi

disk_guard
echo "== [3/4] convert prefill -> DLC =="
$PY_QAIRT "$CONVERTER" --input_network "$QP/model_renamed.onnx" \
    --output_path "$DLC/prefill.dlc" --quantization_overrides "$ENC" \
    --float_bitwidth 16 --target_backend HTP \
    -d input_ids "1,$CL" -d attention_mask "1,$CL,$CL" \
    -d position_ids_cos "1,$CL,64" -d position_ids_sin "1,$CL,64"

echo "== [4/4] verify logits shape in converted DLC =="
INFO=$(mktemp)
"$QAIRT_SDK/bin/x86_64-linux-clang/qairt-dlc-info" -i "$DLC/prefill.dlc" > "$INFO" 2>/dev/null
grep -E "^\| logits" "$INFO" || true
if grep -qE "^\| logits +\| 1,$CL,151936" "$INFO"; then
  echo "PREFILL FIX OK: $NAME logits [1,$CL,151936]"
else
  echo "FAIL: $NAME prefill logits shape is not [1,$CL,151936]"; rm -f "$INFO"; exit 1
fi
rm -f "$INFO"
