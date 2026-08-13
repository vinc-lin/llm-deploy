#!/system/bin/sh
# SA8797P decode-throughput test kit — run this on the device.
#
#   sh run_all.sh [results_dir]
#
# Runs the arms in runsheet.md in PRIORITY ORDER, so if you run out of time and
# stop early, the most decisive measurements are already done. Every arm is
# skipped automatically if its bundle is not present, so you can pull a subset.
#
# Protocol (from the 2026-08-13 report, kept identical so results compare):
#   - one warm-up run per bundle, discarded (cold init is ~1.8-2.0 s vs ~800 ms)
#   - 3 reps per arm, greedy, identical prompt across arms
#   - never compare init->first-logits against TTFT; both are reported, labelled
#
# Everything lands in $RESULTS: per-rep --profile JSON and stdout, plus a
# MANIFEST listing what actually ran. Send the whole directory back.
set -u

RESULTS=${1:-results}
HERE=$(cd "$(dirname "$0")" && pwd)
mkdir -p "$RESULTS"
MANIFEST="$RESULTS/MANIFEST.txt"
: > "$MANIFEST"

log() { echo "$@" | tee -a "$MANIFEST"; }

log "kit run started: $(date)"
log "device: $(getprop ro.product.model 2>/dev/null || echo unknown)"
log ""

# run_arm <arm_id> <bundle_dir> <dialog_json> <prompt_file>
run_arm() {
    ARM=$1; BUNDLE=$2; DIALOG=$3; PROMPT=$4

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

    log "RUN  $ARM: $BUNDLE [$DIALOG] prompt=$(basename "$PROMPT")"

    # warm-up, discarded
    ( cd "$BUNDLE" && LD_LIBRARY_PATH=. ./genie-t2t-run \
        -c "$DIALOG" --prompt_file "$PROMPT" ) > "$OUT/warmup.txt" 2>&1

    i=1
    while [ $i -le 3 ]; do
        ( cd "$BUNDLE" && LD_LIBRARY_PATH=. ./genie-t2t-run \
            -c "$DIALOG" --prompt_file "$PROMPT" \
            --profile "$OUT/profile_r$i.json" ) > "$OUT/stdout_r$i.txt" 2>&1
        RC=$?
        if [ $RC -ne 0 ]; then
            log "     rep$i EXIT=$RC  <-- non-zero; see $OUT/stdout_r$i.txt"
            # 139 = SIGSEGV. Record and keep going; do not abort the session.
        fi
        i=$((i + 1))
    done
    log "     done -> $OUT"
}

P_TECH="$HERE/prompts/technical.txt"
P_SIMPLE="$HERE/prompts/simple.txt"
P_STRUCT="$HERE/prompts/structured.txt"

# ---------------------------------------------------------------- priority 2
# B7b: the headline. Does removing GQA replication move AR-1 throughput?
log "=== PRIORITY 2 — B7b: post-fix throughput (the headline) ==="
run_arm p2_gqafix_local_basic   qwen3_06b_w8a16_gqafix_local   genie_dialog.json       "$P_TECH"
run_arm p2_gqafix_ladekv_basic  qwen3_06b_w8a16_gqafix_ladekv  genie_dialog_basic.json "$P_TECH"
run_arm p2_gqafix_ladekv_lade   qwen3_06b_w8a16_gqafix_ladekv  genie_dialog.json       "$P_TECH"

# ---------------------------------------------------------------- priority 3
# A1: splits the qh regression from a real 3-graph penalty. ONE run, on a
# bundle you already have. If you run nothing else, run this.
log ""
log "=== PRIORITY 3 — A1: basic on the PRE-FIX plain ladekv bin ==="
run_arm p3_a1_ladekv_basic      qwen3_06b_w8a16_ladekv         genie_dialog_basic.json "$P_TECH"

# ---------------------------------------------------------------- priority 4
# Where do the bytes bind? Only meaningful if p2 showed a byte-bound result.
log ""
log "=== PRIORITY 4 — byte-bound branch: W8 head and shorter context ==="
run_arm p4_gqafix_qh_basic      qwen3_06b_w8a16_gqafix_qh      genie_dialog.json       "$P_TECH"
run_arm p4_gqafix_cl512_basic   qwen3_06b_w8a16_gqafix_cl512   genie_dialog.json       "$P_TECH"

# ---------------------------------------------------------------- priority 5
log ""
log "=== PRIORITY 5 — config trims ==="
run_arm p5_gqafix_dlbc_basic    qwen3_06b_w8a16_gqafix_dlbc    genie_dialog.json       "$P_TECH"
run_arm p5_fuseqkvgu_lade       qwen3_06b_w8a16_fuseqkvgu_ladekv genie_dialog.json     "$P_TECH"

# ---------------------------------------------------------------- priority 6
# LADE acceptance is prompt-dependent (report §6.2). Three prompt classes on
# ONE binary is the acceptance map; it decides whether LADE ships at all.
log ""
log "=== PRIORITY 6 — LADE acceptance map (3 prompt classes, one binary) ==="
run_arm p6_lade_simple          qwen3_06b_w8a16_gqafix_ladekv  genie_dialog.json       "$P_SIMPLE"
run_arm p6_lade_structured      qwen3_06b_w8a16_gqafix_ladekv  genie_dialog.json       "$P_STRUCT"
run_arm p6_basic_simple         qwen3_06b_w8a16_gqafix_ladekv  genie_dialog_basic.json "$P_SIMPLE"
run_arm p6_basic_structured     qwen3_06b_w8a16_gqafix_ladekv  genie_dialog_basic.json "$P_STRUCT"

# ---------------------------------------------------------------- priority 7
log ""
log "=== PRIORITY 7 — TTFT product metric (hybrid prefill) ==="
run_arm p7_hybrid_ttft          qwen3_06b_w8a16_gqafix_hybrid  genie_dialog.json       "$P_TECH"

log ""
log "kit run finished: $(date)"
log ""
log "Priority 1 (B7a, the decode-only cycle profile) is NOT in this script --"
log "it uses qnn-net-run, not genie-t2t-run. See runsheet.md section B7a."
echo ""
echo "Send back the whole '$RESULTS' directory."
