#!/bin/sh
# Test L -- run the 0.6B LUT probe's ctx-bin under qnn-net-run, bypassing Genie.
#
# Test K ran this bundle through Genie and it was wrong from the first token.
# This runs the SAME bin with the inputs supplied as files instead of by Genie's
# LUT feed, so the two can be told apart.
#
#   sh run_lutprobe_kit.sh 2>&1 | tee lutprobe_kit.log
#   (host)  adb pull /data/local/tmp/lutprobe/probe_out ./probe_out_l
#           $PY_DEPLOY scripts/validate/analyze_lutprobe_kit.py --kit <kit> --out probe_out_l
#
# The argmax comparison happens on the HOST: the logits row is 151936 wide and
# there is no sane way to argmax that in shell. This script's job is to produce
# the bytes; analyze_lutprobe_kit.py reads them.
set -u

BIN=${BIN:-qwen3-0.6b-w8a16-lutprobe-ladekv_ctx.bin}
OUT=${OUT:-probe_out}
# Relative to the kit directory, not an absolute /data/local/tmp path: the kit
# has to work wherever it was pushed, and an absolute default fails silently
# (dd writes nothing, every past input then points at a file that is not there).
ZERO=${ZERO:-./_zero.raw}

[ -f "$BIN" ] || { echo "FATAL: $BIN not found -- run from the kit directory"; exit 1; }

# One shared all-zero file, reused for every past-KV input of a prefill case.
# Both past_key [1,8,128,512] and past_value [1,8,512,128] hold 8*128*512
# elements, so a single file of PAST_BYTES serves both.
mk_zero() {   # $1 = bytes
    if [ ! -f "$ZERO" ] || [ "$(wc -c < "$ZERO")" -ne "$1" ]; then
        echo "   creating $1-byte zero file at $ZERO"
        dd if=/dev/zero of="$ZERO" bs="$1" count=1 2>/dev/null
    fi
    # Verify rather than assume: if this file is wrong every past-KV input of
    # every prefill case points at it, and the failure would look like the bin's.
    if [ ! -f "$ZERO" ] || [ "$(wc -c < "$ZERO")" -ne "$1" ]; then
        echo "FATAL: could not create a $1-byte zero file at $ZERO" >&2
        exit 1
    fi
}

# One qnn-net-run input line: "name:=file name:=file ...".
input_line() {   # $1 = case dir, $2 = graph subdir
    _d="$1/$2"
    _line=""
    for _t in inputs_embeds attention_mask position_ids_cos position_ids_sin; do
        _line="$_line ${_t}:=${_d}/${_t}.raw"
    done
    _i=0
    while [ "$_i" -lt "$N_PAST_PAIRS" ]; do
        if [ "$PAST_MODE" = "files" ]; then
            _k="${_d}/past_key_${_i}_in.raw"
            _v="${_d}/past_value_${_i}_in.raw"
            for _f in "$_k" "$_v"; do
                if [ ! -f "$_f" ]; then
                    echo "FATAL: $_f missing -- did you untar past_kv.tar.gz?" >&2
                    exit 1
                fi
                _sz=$(wc -c < "$_f")
                if [ "$_sz" -ne "$PAST_BYTES" ]; then
                    echo "FATAL: $_f is $_sz bytes, expected $PAST_BYTES" >&2
                    exit 1
                fi
            done
            _line="$_line past_key_${_i}_in:=${_k} past_value_${_i}_in:=${_v}"
        else
            _line="$_line past_key_${_i}_in:=${ZERO} past_value_${_i}_in:=${ZERO}"
        fi
        _i=$((_i + 1))
    done
    echo "$_line"
}

# The bin holds [prefill, decode] in that order, so a prefill case is graph 0
# and a decode case graph 1. "__" skips a graph: running only the one under test
# matters for more than speed -- the other would demand its own correctly shaped
# inputs, and that failure would look like ours.
lists() { if [ "$GRAPH_IDX" -eq 0 ]; then echo "$1,__"; else echo "__,$1"; fi; }

rm -rf "$OUT"; mkdir -p "$OUT"

for CASE in $(cat probe_cases.txt); do
    echo ""
    echo "===================== $CASE ====================="
    [ -f "$CASE/case.env" ] || { echo "SKIP: no $CASE/case.env"; continue; }
    # Reset before sourcing: these are sourced in a loop into one shell, so a
    # case that does not set PAST_MODE would inherit the previous case's value
    # and feed a real cache where it meant an empty one.
    PAST_MODE=zero
    . "./$CASE/case.env"
    echo "   kind=$KIND graph=$GRAPH_IDX past=$PAST_MODE expect_argmax=$EXPECT_ARGMAX"

    [ "$PAST_MODE" = "zero" ] && mk_zero "$PAST_BYTES"

    LIST="$OUT/${CASE}_input.txt"
    input_line "$CASE" "$KIND" > "$LIST"

    ./qnn-net-run --retrieve_context "$BIN" --backend libQnnHtp.so \
        --input_list "$(lists "$LIST")" \
        --config_file netrun_htp_config.json \
        --use_native_input_files --use_native_output_files \
        --output_dir "$OUT/$CASE"
    echo "   exit=$?"
done

echo ""
echo "done -> $OUT   (pull it and run analyze_lutprobe_kit.py on the host)"
