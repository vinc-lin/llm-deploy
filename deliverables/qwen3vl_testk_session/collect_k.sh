#!/bin/sh
# Collect whatever Test K produced into one tarball.
#
# Deliberately tolerant: a stage you skipped contributes nothing rather than
# failing the run. A half-full tarball beats a clean exit code and no data.
#
# Run on the DEVICE:
#     sh /data/local/tmp/collect_k.sh
# then on the host:
#     adb pull /data/local/tmp/testk_results.tar.gz .
#
# K1 writes into two different directories (the probe and its control), so this
# gathers from both -- that pairing IS the measurement and a tarball with only
# one half cannot be read.

set -u
OUT=/data/local/tmp/testk_results
LUT=${LUT:-/data/local/tmp/lutprobe}
CTRL=${CTRL:-/data/local/tmp/gqafix}
V5=${V5:-/data/local/tmp/v5}

rm -rf "$OUT"
mkdir -p "$OUT/k1_probe" "$OUT/k1_control" "$OUT/k2_image" "$OUT/k3_timing"

copy() {   # copy <dest> <file>...   -- missing files are skipped, not fatal
    d=$1; shift
    for f in "$@"; do
        [ -f "$f" ] && cp "$f" "$d/" 2>/dev/null && echo "  + $f"
    done
}

echo "== K1a probe =="
copy "$OUT/k1_probe" "$LUT"/k1a_*.txt "$LUT"/k1a_*.json

echo "== K1b control =="
copy "$OUT/k1_control" "$CTRL"/k1b_*.txt "$CTRL"/k1b_*.json

echo "== K2 image =="
copy "$OUT/k2_image" "$V5"/k2a_*.log "$V5"/k2b_*.log

echo "== K3 timing =="
copy "$OUT/k3_timing" "$V5"/k3_timing.json

echo "== provenance =="
{
    echo "date: $(date)"
    echo "fingerprint: $(getprop ro.build.fingerprint 2>/dev/null)"
    echo
    echo "-- md5 --"
    for f in "$LUT/qwen3-0.6b-w8a16-lutprobe-ladekv_ctx.bin" \
             "$LUT/embedding_float32_lut.bin" \
             "$V5/qwen3vl-4b-w8a16_1_of_2.bin"; do
        [ -f "$f" ] && md5sum "$f"
    done
    echo
    echo "-- prompt_seg2.txt as left on device (size tells us which variant) --"
    [ -f "$V5/prompt_seg2.txt" ] && wc -c "$V5/prompt_seg2.txt"
    echo
    echo "-- max-num-tokens --"
    grep -h max-num-tokens "$LUT"/*.json "$V5"/genie_dialog_qwen3vl_4b.json 2>/dev/null
} > "$OUT/PROVENANCE.txt" 2>&1
echo "  + PROVENANCE.txt"

n=$(find "$OUT" -type f | wc -l)
cd /data/local/tmp && tar czf testk_results.tar.gz testk_results
echo
echo "collected $n files -> /data/local/tmp/testk_results.tar.gz"
[ "$n" -lt 3 ] && echo "WARNING: that is very few files -- check the paths above \
(LUT=$LUT CTRL=$CTRL V5=$V5) and re-run with the right ones exported."
exit 0
