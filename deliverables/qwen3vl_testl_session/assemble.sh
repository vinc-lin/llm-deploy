#!/usr/bin/env bash
# Materialise the Test L device package, and fix the bundle Test K ran.
#
# Test K ran the LUT probe through Genie and it was wrong from the first token.
# The cause turned out to be ours: the bundle on the Hub shipped the PRE-GRAFT
# ctx-bin, whose `inputs_embeds` is FLOAT_16 -- the dtype that makes
# setupInputEmbeddings' pad write land in the middle of the real prompt
# (nsp-model.cpp:3144 vs :1813). So this script does two things:
#
#   1. replaces the bin (and its info.json) with the grafted uFxp_16 build,
#      and re-runs lint_embedding_dtype.py so the swap cannot go unverified
#   2. adds the Test L kit -- the same bin under qnn-net-run, which is what
#      separates "the bin is wrong" from "Genie's feed is wrong"
#
# The decode cases carry a 639-wide KV cache with ~13 valid positions, so they
# are almost entirely zeros and gzip crushes 142 MB to a few MB. They ship as a
# tarball for that reason, and because the runner size-checks each file it is
# obvious when the untar was skipped.
#
# Usage: assemble.sh [outdir]
set -euo pipefail
source "$(dirname "$0")/../../scripts/env.sh"

OUT=${1:-$LLMDEPLOY_DATA/bundles/qwen3_06b_lutprobe_v2}
HERE=$(cd "$(dirname "$0")" && pwd)
ROOT=$(cd "$HERE/../.." && pwd)
SRC=${SRC:-/home/vinc/llm-local/hf-staging-lutprobe/qwen3_06b_lutprobe}
CTX=${CTX:-$LLMDEPLOY_DATA/work/ctxbin/qwen3-0.6b-w8a16-lutprobe-ladekv}
KIT=${KIT:-$LLMDEPLOY_DATA/work/lutprobe_kit}
BIN=qwen3-0.6b-w8a16-lutprobe-ladekv_ctx.bin

rm -rf "$OUT"; mkdir -p "$OUT"

echo "== [1/5] bundle files (libs, LUT, tokenizer, configs) from the Test K bundle"
for f in "$SRC"/*; do
    b=$(basename "$f")
    case "$b" in
        "$BIN"|*.info.json) continue ;;          # replaced below
    esac
    cp -a "$f" "$OUT/$b"
done

echo "== [2/5] the CORRECTED ctx-bin (uFxp_16 inputs_embeds)"
cp -a "$CTX/$BIN" "$OUT/$BIN"
cp -a "$CTX/info.json" "$OUT/${BIN%.bin}.info.json"
echo "   old (Test K, FLOAT_16): 880a6abdec4a64b67b275ec817c054ca"
echo "   new: $(md5sum "$OUT/$BIN" | cut -c1-32)"

echo "== [3/5] gate the swap -- this is the whole point of the rebuild"
"$PY_DEPLOY" "$ROOT/scripts/validate/lint_embedding_dtype.py" \
    --ctxbin-info "$OUT/${BIN%.bin}.info.json" \
    --lut-params  "$OUT/embedding_lut_params.json" | tee "$OUT/_lint.txt"
grep -q '^PASS' "$OUT/_lint.txt" || { echo "FATAL: gate did not pass"; exit 1; }
rm -f "$OUT/_lint.txt"

echo "== [4/5] Test L kit"
mkdir -p "$OUT/testl"
cp -a "$KIT/probe_cases.txt" "$OUT/testl/"
cp -a "$ROOT/configs/run_lutprobe_kit.sh" "$OUT/testl/"
cp -a "$ROOT/configs/htp_backend_ext_config.json" "$OUT/testl/" 2>/dev/null || \
    cp -a "$SRC/htp_backend_ext_config.json" "$OUT/testl/"
cat > "$OUT/testl/netrun_htp_config.json" <<'JSON'
{
  "backend_extensions": {
    "shared_library_path": "libQnnHtpNetRunExtensions.so",
    "config_file_path": "htp_backend_ext_config.json"
  }
}
JSON
for c in $(cat "$KIT/probe_cases.txt"); do
    mkdir -p "$OUT/testl/$c"
    cp -a "$KIT/$c/case.env" "$OUT/testl/$c/"
    cp -a "$KIT/$c/ref" "$OUT/testl/$c/"
    if [ -d "$KIT/$c/prefill" ]; then cp -a "$KIT/$c/prefill" "$OUT/testl/$c/"; fi
done

echo "== [5/5] decode KV as a tarball (mostly zeros -- gzip crushes it)"
( cd "$KIT" && tar czf "$OUT/testl/past_kv.tar.gz" \
      l2a_decode_s1/decode l2b_decode_s2/decode )
( cd "$OUT/testl" && md5sum past_kv.tar.gz > past_kv.tar.gz.md5 )
echo "   $(du -h "$OUT/testl/past_kv.tar.gz" | cut -f1) (from $(du -sh "$KIT" | cut -f1) raw)"

cp -a "$ROOT/docs/TEST_L_ctxbin_vs_genie.md" "$OUT/" 2>/dev/null || true
cp -a "$HERE/RESULTS_TEMPLATE.md" "$HERE/collect_l.sh" "$OUT/" 2>/dev/null || true
cp -a "$ROOT/docs/DEVICE_TEST_INDEX.md" "$OUT/" 2>/dev/null || true

echo ""
echo "package -> $OUT   ($(du -sh "$OUT" | cut -f1), $(find "$OUT" -type f | wc -l) files)"
