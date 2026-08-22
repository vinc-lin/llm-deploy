#!/bin/sh
# Stage O2 -- the 4B engine-knob sweep. Run from /data/local/tmp/v5.
#
# One run per config, each feeding the SAME 20 exact token ids (the templated
# 2+2 prompt). The verdict per run is binary and printed inline:
#
#   PASS  = output is `4` and then generation STOPS (EOS honoured -- 2 tokens).
#           That knob fixes the decode defect.
#   FAIL  = `4` followed by anything else (the known repetition loop).
#
# The control (o2a) MUST fail -- it is byte-equal to the shipping config. If it
# passes, something else changed on the board and the sweep is uninterpretable.
set -u
TOK=${TOK:-testn/n1_2plus2_p20.tok}
[ -f "$TOK" ] || { echo "FATAL: $TOK not found -- push the testn kit first"; exit 1; }

for c in o2a_ctrl o2b_gswitch o2c_poll o2d_async o2e_mmapb o2f_nommap o2g_all; do
    CFG="testo/genie_dialog_qwen3vl_4b_${c}.json"
    [ -f "$CFG" ] || { echo "SKIP $c: $CFG missing"; continue; }
    echo ""
    echo "===================== $c ====================="
    ./genie-t2t-run -c "$CFG" -tok "$TOK" \
        --profile "o2_${c}.json" 2>&1 | tee "o2_${c}.txt"
    # Verdict: strip the [BEGIN]/[END] markers; a PASS is '4' alone.
    body=$(sed -n 's/.*\[BEGIN\]://p' "o2_${c}.txt" | tr -d '[:space:]' | sed 's/\[END\].*//')
    if [ "$body" = "4" ]; then
        echo ">>> $c: PASS -- '4' then stop. THIS KNOB FIXES DECODE. <<<"
    else
        echo ">>> $c: FAIL (output continues past '4') <<<"
    fi
done
echo ""
echo "sweep done. If any run PASSED, confirm it with the weather prompt:"
echo "  ./genie-t2t-run -c testo/genie_dialog_qwen3vl_4b_<passing>.json \\"
echo "      -tok testn/n1_weather_p18.tok --profile o2_confirm.json | tee o2_confirm.txt"
echo "then go straight to Stage O6 (the full e2e run) with that config."
