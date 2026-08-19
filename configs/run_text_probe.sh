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

# Within one graph, key [1,8,D,PAST] and value [1,8,PAST,D] are the same byte
# count, so ONE zero file serves all 72 past inputs of that case. But the size
# is PER CASE: the past-KV prefill carries PAST=2048 and decode PAST=2175
# (each plus its AR makes 2176), so the two cases need different files. The
# cache is empty AND fully masked everywhere, so contents cannot affect any
# result -- shipping real files would add ~320 MB to express "zeros".
mkdir -p "$OUT"

ensure_zero() {   # $1 = byte count -> echoes the filename
    _z="zeros_past_$1.raw"
    if [ ! -f "$_z" ] || [ "$(wc -c < "$_z")" -ne "$1" ]; then
        rm -f "$_z"
        dd if=/dev/zero of="$_z" bs="$1" count=1 2>/dev/null
    fi
    [ "$(wc -c < "$_z")" -eq "$1" ] || { echo "FATAL: cannot make $_z" >&2; exit 1; }
    echo "$_z"
}

# One qnn-net-run input line: "name:=file name:=file ...".
#   $1 case  $2 graph  $3 layer base  $4 layer count  $5 leading "name:=file"
#   $6 space-separated deepstack input names ("" for shard 1)  $7 zero file
emit() {
    _c=$1; _g=$2; _base=$3; _n=$4; _line=$5; _deep=$6; ZERO=$7
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

# Every run also writes the 36 per-layer KV tensors, and Test F reads shard 0's
# as the layer scan. They are cheap to keep: these graphs emit only the NEW
# AR-wide slice, not the concatenated cache, so a prefill tap is 8x128x128 fp16
# = 256 KB and a decode tap 2 KB -- about 9 MB per prefill run rather than the
# 160 MB the full-cache shape would have cost. Nothing needs pruning.

for CASE in $(cat probe_cases.txt); do
    echo ""
    echo "===================== case $CASE ====================="
    # shellcheck disable=SC1090
    . "./$CASE/case.env"
    echo "   kind=$KIND AR=$AR rows=$N_REAL graphs=$G0/$G1 graph_idx=$GRAPH_IDX"
    ZEROF=$(ensure_zero "$PAST_BYTES")
    echo "   past-KV zero file: $ZEROF ($PAST_BYTES bytes x 72 inputs)"

    # ---- shard 0 ---------------------------------------------------------
    emit "$CASE" "$G0" "$LAYER_BASE_0" "$LAYER_N_0" \
         "inputs_embeds:=${CASE}/${G0}/inputs_embeds.raw" "$DEEP" "$ZEROF" \
         > "$OUT/${CASE}_s0.txt"
    run_one "shard 0 ($G0)" "$BIN0" "$OUT/${CASE}_s0.txt" "$OUT/${CASE}_s0"

    # Pick shard 0's output DETERMINISTICALLY and prove it is the right file.
    # `find ... | head -1` returns directory order, while compare_text_probe.py
    # scores sorted()[0]. If the directory ever holds more than one match, the
    # file FED to shard 1 need not be the file SCORED as shard 0's output -- and
    # the symptom of that is precisely "shard 0 matches the host at cosine 1.0000
    # yet chaining it gives a different argmax". Sort, require exactly one, and
    # make the size a hard failure rather than a printed remark.
    _hits=$(find "$OUT/${CASE}_s0" -type f -name 'last_hidden_states*' | sort)
    _n=$(printf '%s\n' "$_hits" | grep -c . || true)
    if [ -z "$_hits" ]; then
        echo "FATAL: shard 0 produced no last_hidden_states -- stop and report"
        ls -R "$OUT/${CASE}_s0" || true
        exit 1
    fi
    if [ "$_n" -ne 1 ]; then
        echo "FATAL: $_n files match last_hidden_states* in $OUT/${CASE}_s0 --"
        echo "       ambiguous, and the comparator may score a different one."
        printf '%s\n' "$_hits" | sed 's/^/         /'
        exit 1
    fi
    S0=$_hits
    _sz=$(wc -c < "$S0")
    if [ "$_sz" -ne "$HIDDEN_BYTES" ]; then
        echo "FATAL: shard0 out $S0 is $_sz bytes, expected $HIDDEN_BYTES."
        echo "       Wrong tensor, wrong dtype, or a truncated write. Stop."
        exit 1
    fi
    echo "   shard0 out: $S0 ($_sz bytes == HIDDEN_BYTES, OK)"
    # Record what was actually fed, so the host can prove it scored the same file.
    printf '%s\n' "$S0" > "$OUT/${CASE}_s0_fed.txt"

    # ---- shard 1 CHAINED: fed the device's own shard-0 output -------------
    emit "$CASE" "$G1" "$LAYER_BASE_1" "$LAYER_N_1" \
         "last_hidden_states:=${S0}" "" "$ZEROF" > "$OUT/${CASE}_s1chain.txt"
    run_one "shard 1 ($G1) CHAINED on device shard-0 output" \
            "$BIN1" "$OUT/${CASE}_s1chain.txt" "$OUT/${CASE}_s1chain"

    # ---- shard 1 ISOLATED: fed the host reference boundary ----------------
    # The pair is the point. Chained shows what the device really produces;
    # isolated shows whether shard 1 is right GIVEN a known-good input.
    # chained wrong + isolated right => shard 0 is the culprit. Both wrong =>
    # shard 1. Neither run alone can make that call.
    emit "$CASE" "$G1" "$LAYER_BASE_1" "$LAYER_N_1" \
         "last_hidden_states:=${CASE}/${G1}/last_hidden_states.raw" "" "$ZEROF" \
         > "$OUT/${CASE}_s1iso.txt"
    run_one "shard 1 ($G1) ISOLATED on host reference boundary" \
            "$BIN1" "$OUT/${CASE}_s1iso.txt" "$OUT/${CASE}_s1iso"
done

echo ""
echo "===================== done ====================="
echo "pull the whole $OUT/ directory back to the host, then run:"
echo "  scripts/validate/compare_text_probe.py --kit <kit> --results $OUT"
du -sh "$OUT" 2>/dev/null || true
