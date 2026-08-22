#!/bin/sh
# Gather every artefact the v5 session produces into one tarball.
# Device-side, POSIX sh (the board has no bash). Run from anywhere:
#     sh /data/local/tmp/v5/collect.sh
# Writes /data/local/tmp/v5_results.tar.gz
#
# Deliberately tolerant: a test you skipped just contributes nothing. It never
# fails the whole collection because one file is missing -- a half-full tarball
# pulled off the board beats a clean exit code and no data.
set -u
ROOT=${ROOT:-/data/local/tmp}
OUT=$ROOT/v5_results.tar.gz
STAGE=$ROOT/v5_results

rm -rf "$STAGE"; mkdir -p "$STAGE"

take() {   # take <dest-subdir> <path>...
    d="$STAGE/$1"; shift
    mkdir -p "$d"
    for p in "$@"; do
        [ -e "$p" ] && cp -r "$p" "$d/" 2>/dev/null && echo "  + $p"
    done
}

echo "collecting test 1 (fp16-in control)"
take test1 "$ROOT"/test1_fp16in.log "$ROOT"/p16/prof_fp16in_*.json

echo "collecting test 2 (uFxp_16-in fix)"
take test2 "$ROOT"/test2_u16in.log "$ROOT"/p32/prof_u16in_*.json

echo "collecting test 3 + 4 (4B)"
take vl4b "$ROOT"/v5/v5_t2t_*.json "$ROOT"/v5/v5_pipeline.log \
          "$ROOT"/v5/v5_cold.json "$ROOT"/v5/v5_warm*.json "$ROOT"/v5/v5_timing.log

echo "collecting test 5 (qnn-net-run probe, if run)"
take probe "$ROOT"/v5/text_probe.log "$ROOT"/v5/text_probe_out

echo "collecting environment"
{
    echo "date: $(date 2>/dev/null)"
    echo "--- ls of each bundle ---"
    for d in p16 p32 v5; do
        echo "== $ROOT/$d"; ls -l "$ROOT/$d" 2>/dev/null
    done
    echo "--- getprop ---"
    getprop ro.product.model 2>/dev/null
    getprop ro.build.version.release 2>/dev/null
} > "$STAGE/environment.txt" 2>&1

# logcat is best-effort: it is the single most useful thing on a crash and the
# most annoying thing to be asked for afterwards.
logcat -d > "$STAGE/logcat.txt" 2>/dev/null && echo "  + logcat"
[ -d /data/tombstones ] && cp -r /data/tombstones "$STAGE/" 2>/dev/null

tar czf "$OUT" -C "$ROOT" v5_results 2>/dev/null
echo
echo "wrote $OUT"
ls -l "$OUT" 2>/dev/null
echo
echo "now, on the host:  adb pull $OUT ."
