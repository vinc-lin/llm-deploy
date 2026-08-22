#!/usr/bin/env bash
# Materialise the Test J device package.
#
# Test J is the first probe fed PRODUCTION-SHAPED input: chat-templated windows
# with real ViT features, instead of the bare token ids Tests B/C/E/F used. It
# needs NO new ctx-bins -- it reruns the same shard-0 graphs on the same two bins
# the v5 session shipped, so the package is the kit, the runner, the analyzer and
# the docs (~25 MB), and it MERGES into the 03_vl4b_v5/ folder already on device:
#
#   adb push testj/. /data/local/tmp/v5/
#
# Nothing here is over 1 GB, so it uploads in one commit and the
# upload-large-folder COMMIT stall does not apply.
#
#   qwen3vl_testj_session/
#     TEST_J_decode_step.md   <- they read this
#     README.md                             <- the 60-second version
#     testj/                                <- push this into the v5 folder
#       probe_cases.txt  run_text_probe.sh  r0_text/ r1_image/ r2_chunk0/
#     host_refs/                            <- host side; NOT pushed to device
#     analyze_realistic_probe.py            <- run after the pull
#     MANIFEST.md                           <- md5s, incl. the bins they must have
#
# Usage: assemble.sh [outdir]
set -euo pipefail
source "$(dirname "$0")/../../scripts/env.sh"

OUT=${1:-$LLMDEPLOY_DATA/bundles/qwen3vl_testj_session}
HERE=$(cd "$(dirname "$0")" && pwd)
KIT=${KIT:-$LLMDEPLOY_DATA/work/text_probe_j}
# The bins the device must ALREADY have. Not shipped -- only verified, so the
# operator can prove the 4.5 GB on their device is the bytes these references
# were computed against.
VL_BINS=${VL_BINS:-$LLMDEPLOY_DATA/work/ctxbin/qwen3vl-4b-w8a16-gqa-splitkv-u16in}

CASES="j0_2plus2_s1 j1_weather_s1 j2_weather_s2"

disk_guard 2
[ -d "$KIT" ] || { echo "FATAL: no kit at $KIT -- build it on tank first:"; \
    echo "  build_text_probe_kit.py --suite r --windows <calib.npz> ..."; exit 1; }

rm -rf "$OUT"
mkdir -p "$OUT/testj"

echo "== docs =="
cp "$LLMDEPLOY_ROOT/docs/TEST_J_decode_step.md" "$OUT/"
cp "$LLMDEPLOY_ROOT/docs/DEVICE_SESSION_PROTOCOL.md" "$OUT/"
cp "$LLMDEPLOY_ROOT/docs/RUNBOOK_e2e_qwen3vl_4b.md" "$OUT/"
cp "$HERE/RESULTS_TEMPLATE.md" "$HERE/collect_j.sh" "$OUT/"
cp "$LLMDEPLOY_ROOT/scripts/validate/analyze_realistic_probe.py" "$OUT/"
# The retraction belongs with the test: it is why this test exists at all, and
# an operator reading only the guide would not know the previous four results
# were measuring an artifact.
cp "$LLMDEPLOY_ROOT/docs/ROOTCAUSE_qwen3vl_4b_boundary_gain.md" "$OUT/"

echo "== probe kit =="
for c in $CASES; do
    [ -d "$KIT/$c" ] || { echo "FATAL: kit is missing case $c"; exit 1; }
    cp -r "$KIT/$c" "$OUT/testj/"
done
cp "$KIT/cases.json" "$OUT/testj/"
printf '%s\n' $CASES > "$OUT/testj/probe_cases.txt"
cp "$LLMDEPLOY_ROOT/configs/run_text_probe.sh" "$OUT/testj/"

# The device never reads the .npy references -- they are the HOST side of the
# comparison. Shipping them would roughly double the push for nothing.
echo "== host-side references (not pushed to the device) =="
mkdir -p "$OUT/host_refs"
for c in $CASES; do
    mkdir -p "$OUT/host_refs/$c"
    cp -r "$KIT/$c/ref" "$OUT/host_refs/$c/"
done
cp "$KIT/cases.json" "$OUT/host_refs/"
find "$OUT/testj" -name 'ref' -type d -prune -exec rm -rf {} +

# c1_chunk1 and c2_chunk2 each ship a REAL cross-chunk cache: 72 tensors of
# NKV*D*PAST fp16 = 294 MB per case. They are 94% and 88% zero respectively
# (128 and 256 populated positions of 2048), so they compress to 19 MB and
# 36 MB. c0_chunk0's cache is all zeros and ships nothing at all -- the runner's
# shared zero file serves it, saving 294 MB on a case whose cache bytes carry no
# information.
if ls "$OUT"/testj/j*/decode_*/past_*.raw >/dev/null 2>&1; then
    echo "== compressing the cross-chunk KV caches =="
    RAW=$(du -sm "$OUT"/testj/j*/decode_* | awk '{s+=$1} END {print s}')
    ( cd "$OUT/testj" && tar czf past_kv.tar.gz j*/decode_*/past_*.raw \
      && rm -f j*/decode_*/past_*.raw )
    GZ=$(du -sm "$OUT/testj/past_kv.tar.gz" | cut -f1)
    # SPLIT IT SMALL. Measured 2026-08-20 against this proxy: the 231 kit files
    # (LFS-tracked too -- `.gitattributes` has *.raw -- up to ~4 MB each) commit
    # in 105 s, a single 53 MB stream fails every time, and 8 MB parts still
    # fail. The working size is the one the rest of the kit already proves, so
    # parts are 2 MB. The md5 of the whole is recorded because a silently
    # truncated cache fed to the graph would read as a device defect.
    KVMD5=$(md5sum "$OUT/testj/past_kv.tar.gz" | cut -d" " -f1)
    ( cd "$OUT/testj" && split -b 2M -d -a 3 past_kv.tar.gz past_kv.tar.gz.part- \
      && rm -f past_kv.tar.gz )
    NPART=$(ls "$OUT/testj"/past_kv.tar.gz.part-* | wc -l)
    echo "   ${RAW} MB of KV -> ${GZ} MB compressed -> ${NPART} parts of 2 MB"
    echo "   assembled md5: $KVMD5"
    printf '%s  past_kv.tar.gz\n' "$KVMD5" > "$OUT/testj/past_kv.tar.gz.md5"
fi

# Stage A2 uses the Test H kit. Ship it inside this package rather than making
# the operator assemble a session from two downloads -- the whole point of a
# session bundle is that it is one thing.
TESTH=${TESTH:-$LLMDEPLOY_DATA/bundles/qwen3vl_testh_session}
if [ -d "$TESTH/testh" ]; then
    echo "== Stage A2 kit (Test H, cross-chunk prefill) =="
    cp -r "$TESTH/testh" "$OUT/"
    mkdir -p "$OUT/host_refs_testh"
    cp -r "$TESTH/host_refs/." "$OUT/host_refs_testh/"
    echo "   $(ls "$OUT/testh"/past_kv.tar.gz.part-* 2>/dev/null | wc -l) KV parts + $(ls "$OUT/testh" | wc -l) entries"
else
    echo "   WARNING: no Test H package at $TESTH -- Stage A2 will be missing"
fi

echo "== MANIFEST.md =="
{
    echo "# Test J — manifest"
    echo
    echo "Generated by \`assemble.sh\`."
    echo
    echo "## ctx-bins — NOT shipped; verify these are what you already have"
    echo
    echo "Test J reruns the graphs Tests B/C/E/F ran, on the bins the v5 session"
    echo "shipped. If these md5s do not match your device, stop and report it:"
    echo "every reference here was computed against these exact bytes."
    echo
    echo '| file | bytes | md5 |'
    echo '|---|---:|---|'
    for n in 1 2; do
        f=$VL_BINS/${n}_of_2/qwen3vl-4b-w8a16_${n}_of_2.bin
        [ -f "$f" ] || continue
        printf '| `%s` | %s | `%s` |\n' "qwen3vl-4b-w8a16_${n}_of_2.bin" \
            "$(stat -c%s "$f")" "$(md5sum "$f" | cut -d' ' -f1)"
    done
    echo
    echo "## cases — all chat-templated, real ViT features, deepstack zeroed"
    echo
    echo '| case | window | split | real rows | reference rows | host row RMS |'
    echo '|---|---|---|---:|---|---|'
    "$PY_DEPLOY" - "$KIT/cases.json" <<'PY'
import json, sys
for m in json.load(open(sys.argv[1])):
    rms = " / ".join(f"{v:.3f}" for v in m["ref_row_rms"])
    rows = ",".join(str(r) for r in m["real_rows"])
    print(f'| `{m["case"]}` | {m.get("window")} | {m.get("window_split")} | '
          f'{m.get("n_token_rows")} | {rows} | {rms} |')
PY
    echo
    echo "Each case hands the decode graph a HOST-BUILT version of exactly the"
    echo "state Genie should have produced at that step. **The headline number is"
    echo "the logits argmax, not the boundary gain.**"
    echo
    echo '| case | prompt | step | cache in | expected argmax |'
    echo '|---|---|---:|---|---:|'
    echo '| `j0_2plus2_s1` | templated 2+2 | 1 | 20, all prefill | **151645** `<\|im_end\|>` |'
    echo '| `j1_weather_s1` | templated weather | 1 | 18, all prefill | **9104** ` weather` |'
    echo '| `j2_weather_s2` | templated weather | 2 | 19, one written by DECODE | **4344** ` changes` |'
    echo
    echo "\`j2\` is the recurrence -- its cache contains a row the decode graph"
    echo "wrote, which is the only part of the decode path never run on device."
    echo
    echo "Genie produced 2939 \`ention\` where j0 expects 151645, and 3279"
    echo "\`aged\` where j1 expects 9104. Ground truth is the real split graphs"
    echo "with the full KV recurrence (host_generate_check.py):"
    echo "\`2+2 -> [19, 151645]\` and \`weather -> [91169, 9104, 4344, ...]\`."
    echo
    echo "## testj/ contents"
    echo
    echo '| file | bytes | md5 |'
    echo '|---|---:|---|'
    (cd "$OUT/testj" && find . -type f | sort | while read -r f; do
        printf '| `%s` | %s | `%s` |\n' "${f#./}" \
            "$(stat -c%s "$f")" "$(md5sum "$f" | cut -d' ' -f1)"
    done)
} > "$OUT/MANIFEST.md"

cat > "$OUT/README.md" <<'EOF'
# Test J — the final session

**Everything still standing between us and end-to-end, in one package.**
No rebuild, no new ctx-bins. **~30 minutes of board time**, four stages.

## Read these two, in this order

| file | what it is |
|---|---|
| `TEST_J_decode_step.md` | **what** the tests are and **why** — the state of play, the one open defect, and what each outcome means |
| `DEVICE_SESSION_PROTOCOL.md` | **how** to run each one, what to capture, and how to record it |

Then fill in `RESULTS_TEMPLATE.md` and send it with the tarball from
`collect_j.sh`.

## Where we are

Root cause found and confirmed: the 4B is calibrated on **chat-templated**
prompts and every earlier test sent **raw** text. Templating flips the first
generated token to **correct**.

That left **one defect and three unrun paths**:

| | |
|---|---|
| ❌ **decode step 1** | Genie's first decode call is wrong. `2+2` should give `4` then `<\|im_end\|>` — **two tokens** — and instead loops. The graphs are fine (host does it correctly; Test G ran a decode step through *both* shards on device at 1/1) |
| ❓ cross-chunk prefill | never run on device, and the 273-token image prompt needs three chunks |
| ❓ image pipeline post-fix | never run since the `uFxp_16` fix, because the old guide gated it behind a raw-prompt test |
| ❓ timing | no init/TTFT/decode number exists for this tower |

## The four stages

| stage | what | time |
|---|---|---|
| **A1** | decode-step probe (`testj/`) — expects argmax **151645 / 9104 / 4344** | 5 min |
| **A2** | cross-chunk prefill (`testh/`) — unblocks the image path | 3 min |
| **B** | Genie text, templated + raw control — verbatim output | 5 min |
| **C** | image pipeline + six photographs — **judge the FIRST WORD only** | 12 min |
| **D** | timing | 5 min |

**Stage C is worth running even though decode is broken**: the first generated
token comes from *prefill*, so a first word that matches the picture validates
the whole **image → ViT → splice → prefill** path independently of the decode
defect.

## Two traps that have already cost sessions

1. **Check the shard-0 md5 first.** It must be
   `f031e3a7563bf16f2d5ca98a71b357f6`. `qwen3vl_4b_e2e_pipeline_v5/` shipped a
   stale pre-fix copy (`065056ba…`); it has been replaced, but check anyway.
2. **Never feed a `*_u16.raw` image** — Genie stages the file as float32
   regardless of dtype, so it over-reads and `SIGSEGV`s. Every image must be
   `*_fp32.raw` at **6,295,552 bytes**.

## If time runs short

Value order: **A1** (decides graph vs Genie) → **A2** (unblocks the image path)
→ **B1** (one command, confirms A against Genie) → **C1** → the rest.
Stopping after A1 + B1 is still a useful session.
EOF

echo
du -sh "$OUT"/* 2>/dev/null
echo
echo "Test J package complete: $OUT"
