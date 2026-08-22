#!/bin/sh
# Collect whatever Test L produced into one tarball.
#
# Tolerant by design: a stage you skipped contributes nothing rather than
# failing the run. A half-full tarball beats a clean exit code and no data.
#
#   sh /data/local/tmp/lutprobe/collect_l.sh
#   (host) adb pull /data/local/tmp/testl_results.tar.gz .
set -u
OUT=/data/local/tmp/testl_results
LUT=${LUT:-/data/local/tmp/lutprobe}
KIT=${KIT:-$LUT/testl}

rm -rf "$OUT"; mkdir -p "$OUT/l0_genie" "$OUT/l1l2_netrun"

copy() { d=$1; shift; for f in "$@"; do
    [ -f "$f" ] && cp "$f" "$d/" 2>/dev/null && echo "  + $f"; done; }

echo "== L0 (Genie) =="
copy "$OUT/l0_genie" "$LUT"/l0_*.txt "$LUT"/l0_*.json

echo "== L1/L2 (qnn-net-run) =="
copy "$OUT/l1l2_netrun" "$KIT"/lutprobe_kit.log
if [ -d "$KIT/probe_out" ]; then
    cp -r "$KIT/probe_out" "$OUT/l1l2_netrun/" 2>/dev/null && echo "  + probe_out/"
fi

echo "== provenance =="
{
    echo "date: $(date)"
    echo "fingerprint: $(getprop ro.build.fingerprint 2>/dev/null)"
    echo
    echo "-- md5 (9720e46e... = corrected; 880a6abd... = the FLOAT_16 bin) --"
    for f in "$LUT/qwen3-0.6b-w8a16-lutprobe-ladekv_ctx.bin" \
             "$LUT/embedding_float32_lut.bin"; do
        [ -f "$f" ] && md5sum "$f"
    done
    echo
    echo "-- max-num-tokens --"
    grep -h max-num-tokens "$LUT"/*.json 2>/dev/null
    echo
    echo "-- did the decode KV get extracted? --"
    for c in l2a_decode_s1 l2b_decode_s2; do
        n=$(ls "$KIT/$c/decode" 2>/dev/null | wc -l)
        echo "$c: $n file(s) in decode/ (expect 60)"
    done
} > "$OUT/PROVENANCE.txt" 2>&1
echo "  + PROVENANCE.txt"

n=$(find "$OUT" -type f | wc -l)
cd /data/local/tmp && tar czf testl_results.tar.gz testl_results
echo ""
echo "collected $n files -> /data/local/tmp/testl_results.tar.gz"
[ "$n" -lt 3 ] && echo "WARNING: very few files -- check LUT=$LUT KIT=$KIT and \
re-run with the right paths exported."
exit 0
