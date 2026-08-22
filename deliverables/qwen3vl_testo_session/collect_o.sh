#!/bin/sh
# Collect Test O's outputs into one tarball. Tolerant: a skipped stage
# contributes nothing rather than failing the run.
#
#     sh /data/local/tmp/v5/collect_o.sh
#     (host) adb pull /data/local/tmp/testo_results.tar.gz .
#
# The state dumps travel separately (testo_state_dumps.tar.gz, made at O1).
set -u
OUT=/data/local/tmp/testo_results
V5=${V5:-/data/local/tmp/v5}
SPLIT=${SPLIT:-/data/local/tmp/lutsplit}

rm -rf "$OUT"
mkdir -p "$OUT/o1_dumps_meta" "$OUT/o2_sweep" "$OUT/o3_lutsplit" \
         "$OUT/o4_restore" "$OUT/o5_logcat" "$OUT/o6_e2e"

copy() { d=$1; shift; for f in "$@"; do
    [ -f "$f" ] && cp "$f" "$d/" 2>/dev/null && echo "  + $f"; done; }

echo "== O1 (metadata only -- payloads travel separately) =="
copy "$OUT/o1_dumps_meta" "$V5"/o1_*.txt "$V5"/o1_*.json
for d in "$V5"/state_p20 "$V5"/state_p21 "$V5"/state_w18 "$V5"/state_w19; do
    [ -d "$d" ] && { cp "$d/dialog.json" "$OUT/o1_dumps_meta/$(basename "$d").dialog.json" 2>/dev/null \
        && echo "  + $(basename "$d")/dialog.json"; }
done

echo "== O2 =="
copy "$OUT/o2_sweep" "$V5"/o2_*.txt "$V5"/o2_*.json "$V5"/o2_sweep.log

echo "== O3 =="
copy "$OUT/o3_lutsplit" "$SPLIT"/o3_*.txt "$SPLIT"/o3_*.json

echo "== O4 =="
copy "$OUT/o4_restore" "$V5"/o4_*.txt "$V5"/o4_*.json

echo "== O5 =="
copy "$OUT/o5_logcat" "$V5"/o5_fail.txt "$V5"/o5_logcat.txt

echo "== O6 =="
copy "$OUT/o6_e2e" "$V5"/e2e_*.txt "$V5"/e2e_*.json "$V5"/e2e_*.log

echo "== provenance =="
{
    echo "date: $(date)"
    echo "fingerprint: $(getprop ro.build.fingerprint 2>/dev/null)"
    echo
    echo "-- md5 --"
    for f in "$V5/qwen3vl-4b-w8a16_1_of_2.bin" "$V5/qwen3vl-4b-w8a16_2_of_2.bin" \
             "$SPLIT/qwen3-06b-lutsplit_1_of_2.bin"; do
        [ -f "$f" ] && md5sum "$f"
    done
    echo
    echo "-- state dump sizes --"
    for d in "$V5"/state_p20 "$V5"/state_p21 "$V5"/state_w18 "$V5"/state_w19; do
        [ -d "$d" ] && ls -l "$d"
    done
    echo
    echo "-- configs on device --"
    ls "$V5/testo" 2>/dev/null
} > "$OUT/PROVENANCE.txt" 2>&1
echo "  + PROVENANCE.txt"

n=$(find "$OUT" -type f | wc -l)
cd /data/local/tmp && tar czf testo_results.tar.gz testo_results
echo ""
echo "collected $n files -> /data/local/tmp/testo_results.tar.gz"
[ "$n" -lt 3 ] && echo "WARNING: very few files -- check V5=$V5 SPLIT=$SPLIT"
exit 0
