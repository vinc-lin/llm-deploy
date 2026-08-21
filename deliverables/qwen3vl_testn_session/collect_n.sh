#!/bin/sh
# Collect whatever Test N produced into one tarball.
#
# Tolerant by design: a stage you skipped contributes nothing rather than
# failing the run. A half-full tarball beats a clean exit code and no data.
#
#     sh /data/local/tmp/v5/collect_n.sh
#     (host) adb pull /data/local/tmp/testn_results.tar.gz .
#
# The state dumps (state_p*.bin) are EXCLUDED on purpose -- they can be
# hundreds of MB and travel separately. Their sizes go into PROVENANCE.txt.
set -u
OUT=/data/local/tmp/testn_results
V5=${V5:-/data/local/tmp/v5}
LUT=${LUT:-/data/local/tmp/lutprobe}
SPLIT=${SPLIT:-/data/local/tmp/lutsplit}

rm -rf "$OUT"
mkdir -p "$OUT/n1_ladder" "$OUT/n2_testl" "$OUT/n3_testm" "$OUT/n4_image" "$OUT/n5_timing"

copy() { d=$1; shift; for f in "$@"; do
    [ -f "$f" ] && cp "$f" "$d/" 2>/dev/null && echo "  + $f"; done; }

echo "== N1 ladder + -e + state-dump logs =="
copy "$OUT/n1_ladder" "$V5"/n1_*.txt "$V5"/n1_*.json "$V5"/n1b_*.txt \
     "$V5"/n1b_*.json "$V5"/n1c_*.txt "$V5"/n1c_*.json

echo "== N2 Test L =="
copy "$OUT/n2_testl" "$LUT"/l0_*.txt "$LUT"/l0_*.json "$LUT"/testl/lutprobe_kit.log
[ -d "$LUT/testl/probe_out" ] && cp -r "$LUT/testl/probe_out" "$OUT/n2_testl/" \
    2>/dev/null && echo "  + probe_out/"

echo "== N3 Test M =="
copy "$OUT/n3_testm" "$SPLIT"/m_*.txt "$SPLIT"/m_*.json

echo "== N4 image =="
copy "$OUT/n4_image" "$V5"/n4_*.log

echo "== N5 timing =="
copy "$OUT/n5_timing" "$V5"/n5_*.json

echo "== provenance =="
{
    echo "date: $(date)"
    echo "fingerprint: $(getprop ro.build.fingerprint 2>/dev/null)"
    echo
    echo "-- md5 --"
    for f in "$V5/qwen3vl-4b-w8a16_1_of_2.bin" "$V5/qwen3vl-4b-w8a16_2_of_2.bin" \
             "$V5/testn/n1_2plus2_p21.tok" \
             "$LUT/qwen3-0.6b-w8a16-lutprobe-ladekv_ctx.bin" \
             "$SPLIT/qwen3-06b-lutsplit_1_of_2.bin" \
             "$SPLIT/qwen3-06b-lutsplit_2_of_2.bin"; do
        [ -f "$f" ] && md5sum "$f"
    done
    echo
    echo "-- state dumps (travel separately; sizes recorded here) --"
    ls -l "$V5"/state_p*.bin 2>/dev/null || echo "(none)"
    for f in "$V5"/state_p*.bin; do
        [ -f "$f" ] && { echo "first 64 bytes of $f:"; xxd -l 64 "$f" 2>/dev/null \
            || od -A x -t x1 -N 64 "$f"; }
    done
    echo
    echo "-- config state --"
    grep -h max-num-tokens "$V5"/genie_dialog_qwen3vl_4b.json 2>/dev/null
    [ -f "$V5/prompt_seg2.txt" ] && wc -c "$V5/prompt_seg2.txt"
} > "$OUT/PROVENANCE.txt" 2>&1
echo "  + PROVENANCE.txt"

n=$(find "$OUT" -type f | wc -l)
cd /data/local/tmp && tar czf testn_results.tar.gz testn_results
echo ""
echo "collected $n files -> /data/local/tmp/testn_results.tar.gz"
[ "$n" -lt 3 ] && echo "WARNING: very few files -- check V5=$V5 LUT=$LUT SPLIT=$SPLIT"
exit 0
