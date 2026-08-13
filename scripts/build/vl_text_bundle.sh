#!/usr/bin/env bash
# Flat, push-ready device bundle for the Qwen3-VL-4B text tower (2-split W8A16).
#
# Flat by design: Genie's loader resolves the runtime .so files from the bundle
# root, not from a lib/ subdirectory (docs/BUILD_GUIDE.md).
#
# Two ctx-bins, not one: the tower does not fit a single context binary
# (3.5 GiB per-graph serialization limit vs 4.18 GiB needed). Genie takes a LIST
# of ctx-bins and assigns split index by SORTED graph name within each (AR, CL)
# group, so prefill_0/decode_0 are split 1 and prefill_1/decode_1 split 2.
# See docs/NOTES-genie-splits.md.
#
# The embedding LUT ships too: this graph takes inputs_embeds, not input_ids,
# because the runtime has to splice visual features into the sequence, so it
# owns the token lookup. float32 -- a fixed-point LUT silently no-ops against an
# FP16 graph input (see scripts/export/extract_embed_lut.py).
#
# Usage: vl_text_bundle.sh [bundlename] [ctxbin_name]
set -euo pipefail
source "$(dirname "$0")/../env.sh"

BUNDLE=${1:-qwen3vl_4b_text_w8a16}
NAME=${2:-qwen3vl-4b-w8a16}
CTXDIR=$LLMDEPLOY_DATA/work/ctxbin/$NAME-split
LUT=$LLMDEPLOY_DATA/work/lut/qwen3vl-4b
MODEL=${MODEL:-$LLMDEPLOY_DATA/models/Qwen3-VL-4B-Instruct}
OUT=$LLMDEPLOY_DATA/bundles/$BUNDLE

for p in 1 2; do
    [ -f "$CTXDIR/${p}_of_2/${NAME}_${p}_of_2.bin" ] || {
        echo "missing ctx-bin ${p}_of_2 -- run vl_text_ctxbin_split.sh first"; exit 1; }
done
[ -f "$LUT/embedding_float32_lut.bin" ] || { echo "missing $LUT/embedding_float32_lut.bin"; exit 1; }

disk_guard 12
rm -rf "$OUT"; mkdir -p "$OUT"

for p in 1 2; do
    cp "$CTXDIR/${p}_of_2/${NAME}_${p}_of_2.bin" "$OUT/"
done
cp "$LUT/embedding_float32_lut.bin" "$OUT/"
cp "$LUT/embedding_lut_params.json" "$OUT/"
cp "$LLMDEPLOY_ROOT/configs/genie_dialog_qwen3vl_4b.json" "$OUT/genie_dialog.json"
cp "$LLMDEPLOY_ROOT/configs/htp_backend_ext_config_vltext.json" "$OUT/"
cp "$MODEL/tokenizer.json" "$OUT/"

for f in libGenie.so libQnnHtp.so libQnnHtpV81Stub.so libQnnHtpPrepare.so \
         libQnnSystem.so libQnnHtpV81Skel.so libQnnHtpNetRunExtensions.so; do
    src=$(find "$QAIRT_SDK/lib" -name "$f" -path "*aarch64-android*" | head -1)
    [ -n "$src" ] || { echo "MISSING SDK lib: $f"; exit 1; }
    cp "$src" "$OUT/"
done
for b in genie-t2t-run genie-app; do
    cp "$QAIRT_SDK/bin/aarch64-android/$b" "$OUT/" || { echo "MISSING $b"; exit 1; }
done

# A config naming a file that is not in the bundle is a silent runtime failure,
# so every internal reference is resolved against what was actually copied.
$PY_DEPLOY - <<PYEOF
import json, sys, os
out = "$OUT"
d = json.load(open(os.path.join(out, "genie_dialog.json")))["dialog"]
refs = [d["tokenizer"]["path"], d["embedding"]["lut-path"],
        d["engine"]["backend"]["extensions"]] + d["engine"]["model"]["binary"]["ctx-bins"]
missing = [r for r in refs if not os.path.exists(os.path.join(out, r))]
for r in refs:
    print(("  OK   " if r not in missing else "  MISS ") + r)
if missing:
    print("BUNDLE REJECTED: config references files not in the bundle:", missing)
    sys.exit(1)

# Every graph in every ctx-bin must be listed in graph_names, or it silently
# compiles with HTP defaults (O=0, 4 MB VTCM). Checked here against the actual
# binaries rather than assumed.
ext = json.load(open(os.path.join(out, d["engine"]["backend"]["extensions"])))
listed = {n for g in ext["graphs"] for n in g["graph_names"]}
have = set()
for p in (1, 2):
    info = json.load(open("$CTXDIR/%d_of_2/info.json" % p))
    have |= {g["info"]["graphName"] for g in info["info"]["graphs"]}
unbound = have - listed
print("  graphs in ctx-bins:", sorted(have))
print("  graph_names       :", sorted(listed))
if unbound:
    print("BUNDLE REJECTED: unbound graphs (would use HTP defaults):", sorted(unbound))
    sys.exit(1)
print("BUNDLE CONSISTENT")
PYEOF

tar -C "$LLMDEPLOY_DATA/bundles" -czf "$OUT.tar.gz" "$BUNDLE"
ls -lh "$OUT"
du -sh "$OUT" "$OUT.tar.gz"
echo "BUNDLE COMPLETE: $OUT"
