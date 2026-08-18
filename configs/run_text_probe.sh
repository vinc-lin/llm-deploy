#!/bin/sh
# v5 probe A -- run the shipped text ctx-bins under qnn-net-run, no Genie.
#
# Answers one question: does the ctx-bin itself compute the right logits?
#   correct -> converter/ctx-bin exonerated, the fault is Genie's feed path
#   wrong   -> converter/ctx-bin is at fault, and a rebuild is justified
#
# Run from inside the bundle directory:
#   sh run_text_probe.sh 2>&1 | tee text_probe.log
#
# POSIX sh on purpose -- Android's /system/bin/sh is not bash.

set -e
export LD_LIBRARY_PATH=.
OUT=text_probe_out
BIN0=qwen3vl-4b-w8a16_1_of_2.bin
BIN1=qwen3vl-4b-w8a16_2_of_2.bin

# Both ctx-bins hold [prefill_N, decode_N] in that order, so the decode graph is
# the SECOND entry. qnn-net-run takes one input-list per graph, comma separated,
# and "__" skips a graph -- hence "__,<list>" runs decode only. Skipping prefill
# matters for more than speed: prefill would need its own 128-row inputs, and a
# missing-input error there would look like a decode failure.
SKIP=__

# Every past-KV tensor is [1,8,128,2175] or [1,8,2175,128] fp16 -- both are
# 4,454,400 bytes, so ONE zero file serves all 72 inputs across both shards.
# Generated here rather than shipped: 72 x 4.4 MB would add 320 MB to the
# bundle to express "zeros".
ZERO=zeros_past.raw
if [ ! -f "$ZERO" ]; then
    echo "== generating $ZERO (4454400 bytes of zeros)"
    dd if=/dev/zero of="$ZERO" bs=4454400 count=1 2>/dev/null
fi
ZSZ=$(wc -c < "$ZERO")
if [ "$ZSZ" -ne 4454400 ]; then
    echo "FATAL: $ZERO is $ZSZ bytes, expected 4454400"; exit 1
fi

mkdir -p "$OUT"

# Emit one qnn-net-run input line: "name:=file name:=file ...".
# $1 case dir, $2 graph (decode_0|decode_1), $3 first layer index, $4 hidden-in file
emit_list() {
    _dir=$1; _graph=$2; _base=$3; _hid=$4
    _line="$_hid"
    for t in attention_mask position_ids_cos position_ids_sin; do
        _line="$_line ${t}:=${_dir}/${_graph}/${t}.raw"
    done
    if [ "$_graph" = "decode_0" ]; then
        for k in 0 1 2; do
            _line="$_line deepstack_visual_embed_${k}:=${_dir}/${_graph}/deepstack_visual_embed_${k}.raw"
        done
    fi
    _i=$_base
    _end=$((_base + 18))
    while [ "$_i" -lt "$_end" ]; do
        _line="$_line past_key_${_i}_in:=${ZERO} past_value_${_i}_in:=${ZERO}"
        _i=$((_i + 1))
    done
    echo "$_line"
}

for CASE in $(cat probe_cases.txt); do
    echo ""
    echo "===================== case $CASE ====================="

    # ---- shard 0: inputs_embeds -> last_hidden_states --------------------
    emit_list "$CASE" decode_0 0 "inputs_embeds:=${CASE}/decode_0/inputs_embeds.raw" \
        > "$OUT/${CASE}_d0_list.txt"
    echo "== shard 0 (decode_0)"
    ./qnn-net-run --retrieve_context "$BIN0" --backend libQnnHtp.so \
        --input_list "${SKIP},$OUT/${CASE}_d0_list.txt" \
        --config_file netrun_htp_config.json \
        --use_native_input_files --use_native_output_files \
        --output_dir "$OUT/${CASE}_d0"
    echo "   exit=$?"

    D0OUT=$(find "$OUT/${CASE}_d0" -name "last_hidden_states*" | head -1)
    if [ -z "$D0OUT" ]; then
        echo "FATAL: shard 0 produced no last_hidden_states -- stop here and report"
        ls -R "$OUT/${CASE}_d0" || true
        exit 1
    fi
    echo "   shard0 out: $D0OUT ($(wc -c < "$D0OUT") bytes, expect 5120)"

    # ---- shard 1, CHAINED: fed the DEVICE's own shard-0 output ------------
    emit_list "$CASE" decode_1 18 "last_hidden_states:=${D0OUT}" \
        > "$OUT/${CASE}_d1chain_list.txt"
    echo "== shard 1 (decode_1) CHAINED on the device's shard-0 output"
    ./qnn-net-run --retrieve_context "$BIN1" --backend libQnnHtp.so \
        --input_list "${SKIP},$OUT/${CASE}_d1chain_list.txt" \
        --config_file netrun_htp_config.json \
        --use_native_input_files --use_native_output_files \
        --output_dir "$OUT/${CASE}_d1chain"
    echo "   exit=$?"

    # ---- shard 1, ISOLATED: fed the HOST reference boundary ---------------
    # The pair is the point. Chained tells you what the device really produces;
    # isolated tells you whether shard 1 is correct GIVEN a known-good input.
    # Chained wrong + isolated right => shard 0 is the culprit. Both wrong =>
    # shard 1 (or both). Neither run alone can make that call.
    emit_list "$CASE" decode_1 18 \
        "last_hidden_states:=${CASE}/decode_1/last_hidden_states.raw" \
        > "$OUT/${CASE}_d1iso_list.txt"
    echo "== shard 1 (decode_1) ISOLATED on the host reference boundary"
    ./qnn-net-run --retrieve_context "$BIN1" --backend libQnnHtp.so \
        --input_list "${SKIP},$OUT/${CASE}_d1iso_list.txt" \
        --config_file netrun_htp_config.json \
        --use_native_input_files --use_native_output_files \
        --output_dir "$OUT/${CASE}_d1iso"
    echo "   exit=$?"
done

echo ""
echo "===================== done ====================="
echo "pull the whole $OUT/ directory back to the host and run:"
echo "  scripts/validate/compare_text_probe.py --kit <kit> --results $OUT"
ls -R "$OUT" | head -40
