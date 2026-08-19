#!/usr/bin/env bash
# Materialise the complete, self-contained v5 device-session folder.
#
# The repo holds only scripts/configs/docs, so the shippable folder is built
# under $LLMDEPLOY_DATA/bundles/. Everything the device team needs ends up in one
# directory, in the order the guide runs them:
#
#   qwen3vl_v5_session/
#     TESTING_GUIDE.md      <- they read this
#     RESULTS_TEMPLATE.md   <- they fill this in
#     MANIFEST.md           <- md5s + the dtype gate's verdict per bundle
#     prompts/              <- the 127/128/129-token triad
#     collect.sh            <- one tarball of everything, device-side
#     01_probe_06b_fp16in/  <- broken control  (inputs_embeds FLOAT_16)
#     02_probe_06b_u16in/   <- the fix         (inputs_embeds UFIXED_16)
#     03_vl4b_v5/           <- Qwen3-VL-4B     (inputs_embeds UFIXED_16)
#
# Every bundle is gated by lint_embedding_dtype.py before it is allowed in, and
# the gate's verdict is written into MANIFEST.md. 01 is EXPECTED to fail that
# gate -- it is the control -- so its failure is recorded rather than fatal.
#
# Usage: assemble.sh [outdir]
set -euo pipefail
source "$(dirname "$0")/../../scripts/env.sh"

OUT=${1:-$LLMDEPLOY_DATA/bundles/qwen3vl_v5_session}
HERE=$(cd "$(dirname "$0")" && pwd)
STAGE_CTRL=$LLMDEPLOY_DATA/hf-staging-lutprobe/qwen3_06b_lutprobe
STAGE_V5=$LLMDEPLOY_DATA/hf-staging-v5/qwen3vl_4b_e2e_pipeline_v5
NEW_06B=$LLMDEPLOY_DATA/work/ctxbin/qwen3-0.6b-w8a16-lutprobe-ladekv
VL_FIX=${VL_FIX:-$LLMDEPLOY_DATA/work/ctxbin/qwen3vl-4b-w8a16-gqa-splitkv-u16in}
# The qnn-net-run probe kit MUST be built against the ctx-bins that will execute:
# it writes each input in that tensor's declared native encoding. The v5 session
# shipped a kit built for the previous FLOAT_16 bins, which silently fed fp16 bit
# patterns into a UFIXED_POINT_16 input (same byte count, no error) and produced
# near-zero cosines that read as "the ctx-bins are numerically wrong". Overlaid
# here rather than inherited from STAGE_V5 so it cannot go stale again.
PROBE_KIT=${PROBE_KIT:-$LLMDEPLOY_DATA/work/probe_kit_u16}

disk_guard 10
mkdir -p "$OUT"
MISSING=()

echo "== docs, prompts, collector =="
cp "$HERE/TESTING_GUIDE.md" "$HERE/RESULTS_TEMPLATE.md" "$HERE/collect.sh" "$OUT/"
cp "$LLMDEPLOY_ROOT/docs/ISSUE_qwen3vl_4b_text_numerics.md" "$OUT/"
cp -r "$HERE/prompts" "$OUT/"

# ---------------------------------------------------------------- 01 control
echo "== 01_probe_06b_fp16in (broken control) =="
if [ -d "$STAGE_CTRL" ]; then
    D=$OUT/01_probe_06b_fp16in
    mkdir -p "$D"
    # Hard-link rather than copy: these are the same bytes as the staging dir,
    # and the LUT alone is 622 MB. Falls back to cp across filesystems.
    for f in "$STAGE_CTRL"/*; do
        ln -f "$f" "$D/$(basename "$f")" 2>/dev/null || cp "$f" "$D/"
    done
    # The staging copy predates the _comment removal, and it is now HARD-LINKED
    # into this bundle -- writing through the link would corrupt staging too.
    # Unlink first, then install the current config from the repo.
    rm -f "$D/genie_dialog_qwen3_0.6b_lutprobe.json"
    cp "$LLMDEPLOY_ROOT/configs/genie_dialog_qwen3_0.6b_lutprobe.json" "$D/"
    cp "$LLMDEPLOY_ROOT/configs/genie_dialog_qwen3_0.6b_lutprobe.notes.md" "$D/" 2>/dev/null || true
else
    MISSING+=("01: $STAGE_CTRL")
fi

# ---------------------------------------------------------------- 02 the fix
echo "== 02_probe_06b_u16in (the fix) =="
if [ -f "$NEW_06B/qwen3-0.6b-w8a16-lutprobe-ladekv_ctx.bin" ]; then
    D=$OUT/02_probe_06b_u16in
    mkdir -p "$D"
    # Runtime libs, tokenizer and LUT are identical to the control; hard-link
    # them so the folder does not carry a second 622 MB copy of the same LUT.
    for f in genie-t2t-run libGenie.so libQnnHtp.so libQnnHtpNetRunExtensions.so \
             libQnnHtpPrepare.so libQnnHtpV81Skel.so libQnnHtpV81Stub.so \
             libQnnSystem.so tokenizer.json embedding_float32_lut.bin \
             embedding_lut_params.json htp_backend_ext_config.json; do
        [ -f "$STAGE_CTRL/$f" ] && ln -f "$STAGE_CTRL/$f" "$D/$f" 2>/dev/null \
            || cp "$STAGE_CTRL/$f" "$D/$f"
    done
    cp "$NEW_06B/qwen3-0.6b-w8a16-lutprobe-ladekv_ctx.bin" "$D/"
    cp "$NEW_06B/info.json" "$D/qwen3-0.6b-w8a16-lutprobe-ladekv_ctx.info.json"
    cp "$LLMDEPLOY_ROOT/configs/genie_dialog_qwen3_0.6b_lutprobe_u16in.json" "$D/"
    cp "$LLMDEPLOY_ROOT/configs/genie_dialog_qwen3_0.6b_lutprobe_u16in.notes.md" "$D/" 2>/dev/null || true
else
    MISSING+=("02: $NEW_06B/*_ctx.bin")
fi

# ---------------------------------------------------------------- 03 the 4B
echo "== 03_vl4b_v5 (Qwen3-VL-4B) =="
if [ -d "$STAGE_V5" ]; then
    D=$OUT/03_vl4b_v5
    mkdir -p "$D"
    cp -r "$STAGE_V5"/. "$D/"
    # Swap in the rebuilt TEXT ctx-bins if they exist; the ViT bin and LUT are
    # unchanged since v3 and are deliberately left alone.
    if [ -d "$VL_FIX" ]; then
        for n in 1 2; do
            src=$VL_FIX/${n}_of_2/qwen3vl-4b-w8a16_${n}_of_2.bin
            [ -f "$src" ] && cp "$src" "$D/" && \
                cp "$VL_FIX/${n}_of_2/info.json" "$D/qwen3vl-4b-w8a16_${n}_of_2.info.json"
        done
    else
        MISSING+=("03: text ctx-bins not rebuilt yet ($VL_FIX)")
    fi
    if [ -d "$PROBE_KIT" ]; then
        rm -rf "$D/decode1tok" "$D/prefill4tok"
        cp -r "$PROBE_KIT"/decode1tok "$PROBE_KIT"/prefill4tok \
              "$PROBE_KIT"/cases.json "$PROBE_KIT"/probe_cases.txt "$D/"
    else
        MISSING+=("03: probe kit not rebuilt for these bins ($PROBE_KIT)")
    fi
else
    MISSING+=("03: $STAGE_V5")
fi

# ---------------------------------------------------------------- manifest
echo "== MANIFEST.md =="
{
    echo "# v5 session — manifest"
    echo
    echo "Generated by \`assemble.sh\`. Verify with \`md5sum -c\` after pushing if"
    echo "a transfer looks suspect."
    echo
    for b in 01_probe_06b_fp16in 02_probe_06b_u16in 03_vl4b_v5; do
        [ -d "$OUT/$b" ] || continue
        echo "## $b"
        echo
        # The dtype gate. 01 is the control and is SUPPOSED to fail it.
        # A VL bundle also ships the ViT bin's info.json, which has no
        # inputs_embeds at all -- picking it just makes the gate assert. Choose
        # by CONTENT: the shard that actually declares the tensor.
        info=$(grep -l '"inputs_embeds"' "$OUT/$b"/*info.json 2>/dev/null | head -1)
        if [ -n "$info" ]; then
            echo '```'
            "$PY_DEPLOY" "$LLMDEPLOY_ROOT/scripts/validate/lint_embedding_dtype.py" \
                --lut-datatype float32 --ctxbin-info "$info" 2>&1 \
                | sed 's/^/  /' || true
            echo '```'
            echo
        fi
        echo '| file | bytes | md5 |'
        echo '|---|---:|---|'
        (cd "$OUT/$b" && find . -maxdepth 1 -type f | sort | while read -r f; do
            printf '| `%s` | %s | `%s` |\n' "${f#./}" \
                "$(stat -c%s "$f")" "$(md5sum "$f" | cut -d' ' -f1)"
        done)
        echo
    done
} > "$OUT/MANIFEST.md"

echo
du -sh "$OUT"/* 2>/dev/null
if [ ${#MISSING[@]} -gt 0 ]; then
    echo
    echo "INCOMPLETE -- these parts are not in the folder:"
    printf '  - %s\n' "${MISSING[@]}"
    echo "The folder is still usable for whatever IS present; re-run after building the rest."
    exit 2
fi
echo
echo "v5 session folder complete: $OUT"
