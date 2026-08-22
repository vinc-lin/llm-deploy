#!/bin/sh
# Gather every artefact the Test J session produces into one tarball.
# Device-side, POSIX sh (the board has no bash):
#     sh /data/local/tmp/v5/collect_j.sh
# Writes /data/local/tmp/testj_results.tar.gz
#
# Deliberately tolerant: a stage you skipped contributes nothing rather than
# failing the collection. A half-full tarball pulled off the board beats a clean
# exit code and no data.
set -u
ROOT=${ROOT:-/data/local/tmp}
STAGE=$ROOT/testj_results
OUT=$ROOT/testj_results.tar.gz
rm -rf "$STAGE"; mkdir -p "$STAGE"

take() {   # take <dest-subdir> <path>...
    d="$STAGE/$1"; shift
    mkdir -p "$d"
    for p in "$@"; do
        [ -e "$p" ] && cp -r "$p" "$d/" 2>/dev/null && echo "  + $p"
    done
}

echo "A1 decode-step probe"
take a1_decode "$ROOT"/v5/text_probe_j.log "$ROOT"/v5/text_probe_out

echo "A2 cross-chunk prefill"
take a2_chunks "$ROOT"/v5h/text_probe_h.log "$ROOT"/v5h/text_probe_out

echo "B Genie text"
take b_text "$ROOT"/v5/b1_templated.txt "$ROOT"/v5/b2_weather.txt "$ROOT"/v5/b3_raw.txt \
            "$ROOT"/v5/b1_templated.json "$ROOT"/v5/b2_weather.json "$ROOT"/v5/b3_raw.json

echo "C image pipeline"
take c_image "$ROOT"/v5/c1_pipeline.log "$ROOT"/v5/c2_wx_*.log

echo "D timing"
take d_timing "$ROOT"/v5/d1_timing.json

echo "environment + preconditions"
{
    echo "date: $(date 2>/dev/null)"
    echo "fingerprint: $(getprop ro.build.fingerprint 2>/dev/null)"
    echo "--- ctx-bin md5 (shard 0 MUST be f031e3a7563bf16f2d5ca98a71b357f6) ---"
    md5sum "$ROOT"/v5/qwen3vl-4b-w8a16_*.bin 2>/dev/null
    echo "--- image blob sizes (each MUST be 6295552) ---"
    ls -l "$ROOT"/v5/*_fp32.raw 2>/dev/null
} > "$STAGE/env.txt" 2>&1
echo "  + env.txt"

( cd "$ROOT" && tar czf "$OUT" testj_results ) && echo "" && echo "wrote $OUT"
ls -l "$OUT" 2>/dev/null
echo "pull it with:  adb pull $OUT ."
