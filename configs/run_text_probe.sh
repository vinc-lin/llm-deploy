#!/bin/sh
# v5 probe A -- run the shipped text ctx-bins under qnn-net-run, no Genie.
#
# Answers one question: does the ctx-bin itself compute the right values?
#   correct -> converter/ctx-bin exonerated; the fault is Genie's feed path
#   wrong   -> converter/ctx-bin is at fault, and a rebuild is justified
#
#   sh run_text_probe.sh 2>&1 | tee text_probe.log
#
# POSIX sh on purpose: Android's /system/bin/sh is not bash.

set -e
export LD_LIBRARY_PATH=.
OUT=text_probe_out
BIN0=qwen3vl-4b-w8a16_1_of_2.bin
BIN1=qwen3vl-4b-w8a16_2_of_2.bin

# Every past-KV tensor is [1,8,128,2175] or [1,8,2175,128] fp16 -- both are
# 4,454,400 bytes, so ONE zero file serves all 72 inputs across both shards.
# The cache is empty AND fully masked in every case, so its contents cannot
# affect any result; shipping 72 real files would add 320 MB to say "zeros".
ZERO=zeros_past.raw
if [ ! -f "$ZERO" ]; then
    echo "== generating $ZERO (4454400 bytes)"
    dd if=/dev/zero of="$ZERO" bs=4454400 count=1 2>/dev/null
fi
ZSZ=$(wc -c < "$ZERO")
[ "$ZSZ" -eq 4454400 ] || { echo "FATAL: $ZERO is $ZSZ, expected 4454400"; exit 1; }

mkdir -p "$OUT"

# One qnn-net-run input line: "name:=file name:=file ...".
#   $1 case  $2 graph  $3 layer base  $4 layer count  $5 leading "name:=file"
#   $6 space-separated deepstack input names ("" for shard 1)
emit() {
    _c=$1; _g=$2; _base=$3; _n=$4; _line=$5; _deep=$6
    for t in attention_mask position_ids_cos position_ids_sin; do
        _line="$_line ${t}:=${_c}/${_g}/${t}.raw"
    done
    for d in $_deep; do
        _line="$_line ${d}:=${_c}/${_g}/${d}.raw"
    done
    _i=$_base; _end=$((_base + _n))
    while [ "$_i" -lt "$_end" ]; do
        _line="$_line past_key_${_i}_in:=${ZERO} past_value_${_i}_in:=${ZERO}"
        _i=$((_i + 1))
    done
    echo "$_line"
}

# The ctx-bins hold [prefill_N, decode_N] in that order, so a decode case is
# the SECOND graph and a prefill case the FIRST. qnn-net-run takes one input
# list per graph, comma separated, and "__" skips one. Running only the graph
# under test matters for more than speed: the other graph would demand its own
# correctly shaped inputs, and that failure would look like ours.
lists() {   # $1 = list file for the graph under test
    if [ "$GRAPH_IDX" -eq 0 ]; then echo "$1,__"; else echo "__,$1"; fi
}

run_one() {  # $1 label  $2 ctx-bin  $3 list file  $4 output dir
    echo "== $1"
    ./qnn-net-run --retrieve_context "$2" --backend libQnnHtp.so \
        --input_list "$(lists "$3")" \
        --config_file netrun_htp_config.json \
        --use_native_input_files --use_native_output_files \
        --output_dir "$4"
    echo "   exit=$?"
}

for CASE in $(cat probe_cases.txt); do
    echo ""
    echo "===================== case $CASE ====================="
    # shellcheck disable=SC1090
    . "./$CASE/case.env"
    echo "   kind=$KIND AR=$AR rows=$N_REAL graphs=$G0/$G1 graph_idx=$GRAPH_IDX"

    # ---- shard 0 ---------------------------------------------------------
    emit "$CASE" "$G0" "$LAYER_BASE_0" "$LAYER_N_0" \
         "inputs_embeds:=${CASE}/${G0}/inputs_embeds.raw" "$DEEP" \
         > "$OUT/${CASE}_s0.txt"
    run_one "shard 0 ($G0)" "$BIN0" "$OUT/${CASE}_s0.txt" "$OUT/${CASE}_s0"

    S0=$(find "$OUT/${CASE}_s0" -name "last_hidden_states*" | head -1)
    if [ -z "$S0" ]; then
        echo "FATAL: shard 0 produced no last_hidden_states -- stop and report"
        ls -R "$OUT/${CASE}_s0" || true
        exit 1
    fi
    echo "   shard0 out: $S0 ($(wc -c < "$S0") bytes, expect $HIDDEN_BYTES)"

    # ---- shard 1 CHAINED: fed the device's own shard-0 output -------------
    emit "$CASE" "$G1" "$LAYER_BASE_1" "$LAYER_N_1" \
         "last_hidden_states:=${S0}" "" > "$OUT/${CASE}_s1chain.txt"
    run_one "shard 1 ($G1) CHAINED on device shard-0 output" \
            "$BIN1" "$OUT/${CASE}_s1chain.txt" "$OUT/${CASE}_s1chain"

    # ---- shard 1 ISOLATED: fed the host reference boundary ----------------
    # The pair is the point. Chained shows what the device really produces;
    # isolated shows whether shard 1 is right GIVEN a known-good input.
    # chained wrong + isolated right => shard 0 is the culprit. Both wrong =>
    # shard 1. Neither run alone can make that call.
    emit "$CASE" "$G1" "$LAYER_BASE_1" "$LAYER_N_1" \
         "last_hidden_states:=${CASE}/${G1}/last_hidden_states.raw" "" \
         > "$OUT/${CASE}_s1iso.txt"
    run_one "shard 1 ($G1) ISOLATED on host reference boundary" \
            "$BIN1" "$OUT/${CASE}_s1iso.txt" "$OUT/${CASE}_s1iso"
done

echo ""
echo "===================== done ====================="
echo "pull the whole $OUT/ directory back to the host, then run:"
echo "  scripts/validate/compare_text_probe.py --kit <kit> --results $OUT"
du -sh "$OUT" 2>/dev/null || true
