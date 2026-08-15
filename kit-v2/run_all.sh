#!/system/bin/sh
# SA8797P decode-regime session, kit v2 -- run this on the device.
#
#   sh run_all.sh [results_dir]
#
# Supersedes kit/run_all.sh. Two things changed and both matter:
#
#   1. FIVE reps per arm, not three. The 2026-08-15 session measured one arm
#      (pastkv2g) at 23.43 / 44.54 / 29.34 tok/s on a single binary. That spread
#      is larger than every effect this session is chasing, so three reps cannot
#      resolve anything. Every raw value is kept; report the MEDIAN.
#
#   2. Every arm is labelled pure or BLENDED. A bundle containing a graph whose
#      AR equals its CL keeps generating THROUGH that graph until the KV cache
#      passes AR, so its tok/s is a time-weighted blend of two rates -- and it
#      comes out flatteringly fast. Never compare a blended arm against a pure
#      one. This kit ships only pure bundles; the label is here so it stays true
#      if someone adds an arm.
#
# Arms run in PRIORITY ORDER: stopping early still leaves the decisive
# measurements done. Any arm whose bundle is absent is skipped, so pulling a
# subset is fine.
set -u

HERE=$(cd "$(dirname "$0")" && pwd)
mkdir -p "${1:-results}"
RESULTS=$(cd "${1:-results}" && pwd)   # must be ABSOLUTE: arms run inside cd "$BUNDLE"
MANIFEST="$RESULTS/MANIFEST.txt"
: > "$MANIFEST"

REPS=5

log() { echo "$@" | tee -a "$MANIFEST"; }

temp() {
    # best-effort thermal reading; several of these exist depending on kernel
    for z in /sys/class/thermal/thermal_zone*/temp; do
        [ -r "$z" ] && echo "$z=$(cat "$z" 2>/dev/null)"
    done 2>/dev/null | head -4 | tr '\n' ' '
}

log "kit v2 run started: $(date)"
log "device: $(getprop ro.product.model 2>/dev/null || echo unknown)"
log "reps per arm: $REPS"
log ""

# run_arm <arm_id> <bundle_dir> <dialog_json> <prompt_file> <topology> [reps]
run_arm() {
    ARM=$1; BUNDLE=$2; DIALOG=$3; PROMPT=$4; TOPO=$5; N=${6:-$REPS}

    if [ ! -d "$BUNDLE" ]; then
        log "SKIP $ARM: bundle $BUNDLE not present"
        return 0
    fi
    if [ ! -f "$BUNDLE/$DIALOG" ]; then
        log "SKIP $ARM: $BUNDLE/$DIALOG not present"
        return 0
    fi

    OUT="$RESULTS/$ARM"
    mkdir -p "$OUT"
    cp "$BUNDLE/$DIALOG" "$OUT/dialog_used.json" 2>/dev/null
    cp "$BUNDLE/htp_backend_ext_config.json" "$OUT/" 2>/dev/null
    echo "$TOPO" > "$OUT/TOPOLOGY"

    log "RUN  $ARM [$TOPO]: $BUNDLE ($DIALOG) prompt=$(basename "$PROMPT") reps=$N"
    log "     temp_before: $(temp)"

    # warm-up, discarded: cold init is ~1.8-2.0 s vs ~800 ms warm
    ( cd "$BUNDLE" && LD_LIBRARY_PATH=. ./genie-t2t-run \
        -c "$DIALOG" --prompt_file "$PROMPT" ) > "$OUT/warmup.txt" 2>&1

    i=1
    while [ $i -le $N ]; do
        ( cd "$BUNDLE" && LD_LIBRARY_PATH=. ./genie-t2t-run \
            -c "$DIALOG" --prompt_file "$PROMPT" \
            --profile "$OUT/profile_r$i.json" ) > "$OUT/stdout_r$i.txt" 2>&1
        RC=$?
        [ $RC -ne 0 ] && log "     rep$i EXIT=$RC  <-- see $OUT/stdout_r$i.txt"
        i=$((i + 1))
    done

    log "     temp_after : $(temp)"
    log "     done -> $OUT"
    # Cool-down between arms. Thermal state is a prime suspect for the
    # 2026-08-15 variance, so never run two arms back to back.
    sleep 30
}

P_TECH="$HERE/prompts/technical.txt"
P_SIMPLE="$HERE/prompts/simple.txt"
P_STRUCT="$HERE/prompts/structured.txt"

log "=== P0b -- RE-BASELINE. Every delta below is computed against this. ==="
run_arm p0_rebaseline    qwen3_06b_w8a16_gqafix_ladekv        genie_dialog_basic.json "$P_TECH" pure

log ""
log "=== P4 -- THE PAIR THAT DECIDES THE PLAN. Run BOTH; the verdict is in the ordering. ==="
log "    byte model: qh +17.9% / cl512 +10.1%  compute model: qh +3.6% / cl512 +26.0%"
run_arm p4_qh_ladekv     qwen3_06b_w8a16_gqafix_qh_ladekv     genie_dialog_basic.json "$P_TECH" pure
run_arm p4_cl512_ladekv  qwen3_06b_w8a16_gqafix_cl512_ladekv  genie_dialog_basic.json "$P_TECH" pure

log ""
log "=== P5 -- THE NULL TEST. hvx8 changes zero DDR bytes; byte model predicts 0.0%. ==="
log "    Compare hvx8 against p5_ctrl, NOT against p0_rebaseline (same config path)."
run_arm p5_ctrl          qwen3_06b_w8a16_gqafix_ctrl_ladekv   genie_dialog_basic.json "$P_TECH" pure
run_arm p5_hvx8          qwen3_06b_w8a16_gqafix_hvx8_ladekv   genie_dialog_basic.json "$P_TECH" pure

log ""
log "=== P2 -- rep-variance isolation (8 reps, temperature logged each side) ==="
run_arm p2_pastkv2g_var  qwen3_06b_w8a16_gqafix_pastkv2g      genie_dialog_basic.json "$P_TECH" pure 8

log ""
log "=== P6 -- cheap ctx-bin-only knobs, ~5 min each ==="
# udma: first real A/B ever -- the key sat in the wrong config section in every
# previous build and was silently ignored. Binary confirmed changed offline.
run_arm p6_udma          qwen3_06b_w8a16_gqafix_udma_ladekv   genie_dialog_basic.json "$P_TECH" pure
run_arm p6_socmodel72    qwen3_06b_w8a16_gqafix_socmodel72_ladekv genie_dialog_basic.json "$P_TECH" pure
# dlbc and wpack produced byte-identical binaries offline -- low expectation.
run_arm p6_dlbc          qwen3_06b_w8a16_gqafix_dlbc_ladekv   genie_dialog_basic.json "$P_TECH" pure
run_arm p6_wpack         qwen3_06b_w8a16_gqafix_wpack_ladekv  genie_dialog_basic.json "$P_TECH" pure

log ""
log "=== P7 -- fusion, combined with the GQA fix for the first time ==="
run_arm p7_fuseqkvgu     qwen3_06b_w8a16_gqafix_fuseqkvgu_ladekv genie_dialog_basic.json "$P_TECH" pure

log ""
log "=== P3 -- prompt distribution on the baseline ==="
run_arm p3_basic_simple      qwen3_06b_w8a16_gqafix_ladekv genie_dialog_basic.json "$P_SIMPLE" pure
run_arm p3_basic_structured  qwen3_06b_w8a16_gqafix_ladekv genie_dialog_basic.json "$P_STRUCT" pure

log ""
log "kit v2 run finished: $(date)"
log ""
log "NOT in this script:"
log "  P0 -- pull /data/local/tmp/results off the device FIRST (/data is 98-99% full)"
log "  P1 -- the decode-only cycle profile: uses qnn-net-run, not genie-t2t-run."
log "        Run verify_profile_inputs.py BEFORE it. See runsheet.md section P1."
echo ""
echo "Send back the whole '$RESULTS' directory, including all $REPS raw rep values per arm."
