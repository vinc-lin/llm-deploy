#!/usr/bin/env bash
# Assemble the flat, push-ready END-TO-END Qwen3-VL-4B pipeline bundle:
# W8A16 ViT ctx-bin + 2-split W8A16 text ctx-bins + embedding LUT + the three
# node configs + the genie-app pipeline script + the runtime .so set + a
# preprocessed sample image and its prompt segments.
#
# Flat by design: Genie's loader resolves the runtime .so files AND every
# config-referenced path from the bundle root, not from subdirectories
# (docs/BUILD_GUIDE.md §1).
#
# Sources are the already-gated artifacts -- the Stage 2 text bundle
# (vl_text_bundle.sh, which ran its own consistency check) and the W8A16 ViT
# ctx-bin directory (vit_build_quant.sh, which verified UFIXED_16 IO) -- so
# every binary byte shipped here is one that passed its own gate. NOTE: the
# ViT is the W8A16 tower, NOT the fp16 one vit_bundle.sh still defaults to; a
# stock Genie pipeline cannot drive an fp16 image encoder at all
# (setupInputFP16 is an empty stub, nsp-image-model.cpp:526-530).
#
# Three things are GENERATED here rather than copied, each for a reason:
#
#   * info.json per ctx-bin. Genie binds HTP tuning blocks by graph NAME, and
#     an unbound graph silently gets backend defaults (O=0, 4 MB VTCM) -- a
#     failure that already shipped once in this repo and ended in a device
#     SIGSEGV. The names are re-read from the FINAL binaries with
#     qnn-context-binary-utility so the lint checks what ships, not what a
#     build log once said. The bundle is flat, so each info.json carries its
#     binary's stem instead of all three being called info.json.
#   * prompt_seg{1,2}.txt, derived from the tokenizer's own chat template.
#     Written as bytes with NO trailing newline: genie-app reads them with
#     rdbuf() (main.cpp:1151-1157), so a trailing newline would land in the
#     prompt. They are files and not `node set text` arguments because a
#     quoted argument cannot carry a real newline -- genie-app's split() pushes
#     the character after an escape verbatim (main.cpp:655-658) and nothing
#     unescapes later (main.cpp:1140-1142), so "\n" becomes the two characters
#     '\' and 'n'.
#   * sample_image.{png,raw,json}. The scene is the red circle + blue square
#     the E2E parity gate used (parity_e2e_vl.py:make_image), so the device's
#     caption is directly comparable to that gate's recorded Tier-A text. The
#     raw blob is quantized against the SHIPPED ctx-bin's own info.json, not
#     against model.encodings: Genie does no preprocessing, it feeds the file
#     as an opaque blob, so those bytes must already be exactly what this
#     graph's pixel_values tensor expects.
#
# Usage: vl_pipeline_bundle.sh [bundlename]
set -euo pipefail
source "$(dirname "$0")/../env.sh"

BUNDLE=${1:-qwen3vl_4b_e2e_pipeline}
TEXT=$LLMDEPLOY_DATA/bundles/qwen3vl_4b_text_w8a16
VIT_CTXDIR=$LLMDEPLOY_DATA/work/ctxbin/qwen3vl-4b-vit-w8a16
VIT_BUNDLE=$LLMDEPLOY_DATA/bundles/qwen3vl_4b_vit_fp16
MODEL=${MODEL:-$LLMDEPLOY_DATA/models/Qwen3-VL-4B-Instruct}
OUT=$LLMDEPLOY_DATA/bundles/$BUNDLE
UTIL=$QAIRT_SDK/bin/x86_64-linux-clang/qnn-context-binary-utility

VIT_BIN=qwen3vl-4b-vit-w8a16_ctx.bin
TEXT_BINS=(qwen3vl-4b-w8a16_1_of_2.bin qwen3vl-4b-w8a16_2_of_2.bin)

# Where the two TEXT ctx-bins come from. Defaults to the gated Stage 2 text
# bundle; point it at a variant build (the past-KV prefill rebuild lands in
# work/ctxbin/<name>-splitkv/{1,2}_of_2/) to swap only the text tower while
# every other byte still comes from its own gated source. The FILENAMES do not
# change -- the node config references them.
TEXT_CTX=${TEXT_CTX:-$TEXT}

# Everything taken from the gated text bundle: ctx-bins, LUT, tokenizer, the
# text tower's htp extensions, both genie drivers and the 7 runtime libraries.
# genie-app is the driver for THIS bundle (it is the only prebuilt binary
# exposing GenieNode_*/GeniePipeline_* and the image-encoder roles);
# genie-t2t-run ships alongside it purely as the text-only triage path.
TEXT_FILES=(
  "${TEXT_BINS[@]}"
  embedding_float32_lut.bin embedding_lut_params.json tokenizer.json
  htp_backend_ext_config_vltext.json
  genie-app genie-t2t-run
  libGenie.so libQnnHtp.so libQnnSystem.so libQnnHtpPrepare.so
  libQnnHtpNetRunExtensions.so libQnnHtpV81Stub.so libQnnHtpV81Skel.so
)
CONFIGS=(
  genie_image_encoder_qwen3vl.json
  genie_text_encoder_qwen3vl.json
  genie_text_generator_qwen3vl_4b.json
  genie_pipeline_qwen3vl.script
  # Fallback: one file swap if the primary somehow still fails at load. It
  # filters the prefill graphs out before they become variants, at the cost of
  # all prefill (~30 s TTFT instead of ~3-4 s). lint check 10 proves it differs
  # from the primary by exactly the two select-graphs keys.
  genie_text_generator_qwen3vl_4b_decodeonly.json
  genie_pipeline_qwen3vl_decodeonly.script
)

# Weather/road test kit: per-image blob + sidecar + jpg + its own script, all
# FLAT like the rest of the bundle. Built by scripts/pipeline/build_test_kit.py.
KIT=${KIT:-$LLMDEPLOY_DATA/work/kit}

# src_of <file> -> the directory that file is taken from
src_of() {
    for b in "${TEXT_BINS[@]}"; do
        [ "$1" = "$b" ] && { echo "$TEXT_CTX"; return; }
    done
    echo "$TEXT"
}

MISSING=0
for f in "${TEXT_FILES[@]}"; do
    s=$(src_of "$f")
    [ -f "$s/$f" ] || { echo "ERROR: missing $s/$f" >&2; MISSING=1; }
done
for f in "${CONFIGS[@]}"; do
    [ -f "$LLMDEPLOY_ROOT/configs/$f" ] || {
        echo "ERROR: missing $LLMDEPLOY_ROOT/configs/$f" >&2; MISSING=1; }
done
[ -f "$VIT_CTXDIR/$VIT_BIN" ] || {
    echo "ERROR: missing $VIT_CTXDIR/$VIT_BIN -- run vit_build_quant.sh first" >&2
    MISSING=1; }
[ -f "$VIT_BUNDLE/htp_backend_ext_config_vit.json" ] || {
    echo "ERROR: missing $VIT_BUNDLE/htp_backend_ext_config_vit.json" >&2; MISSING=1; }
[ -x "$UTIL" ] || { echo "ERROR: missing $UTIL" >&2; MISSING=1; }
[ -d "$MODEL" ] || { echo "ERROR: missing model dir $MODEL" >&2; MISSING=1; }
[ "$MISSING" -eq 0 ] || { echo "ERROR: sources incomplete (above); aborting" >&2; exit 1; }

# ~6.3 GB of copies plus a ~5 GB tarball. Sized to the step, not to a flat
# floor: $LLMDEPLOY_DATA is on the ext4.vhdx on C:, and a failed vhdx grow does
# NOT surface as ENOSPC -- the guest still reports free space, the host write
# fails, every mmap'd page takes SIGBUS and the VM hard-crashes.
disk_guard 16

rm -rf "$OUT"; mkdir -p "$OUT"    # stale binaries must never leak into a bundle

echo "== [1/5] copy gated artifacts =="
[ "$TEXT_CTX" = "$TEXT" ] || echo "   text ctx-bins from $TEXT_CTX"
for f in "${TEXT_FILES[@]}"; do cp "$(src_of "$f")/$f" "$OUT/"; done
cp "$VIT_CTXDIR/$VIT_BIN" "$OUT/"
cp "$VIT_BUNDLE/htp_backend_ext_config_vit.json" "$OUT/"
for f in "${CONFIGS[@]}"; do cp "$LLMDEPLOY_ROOT/configs/$f" "$OUT/"; done
# Docs ship WITH the bundle and are tracked in the repo, so what the device
# team reads is version-controlled rather than hand-written into a build dir.
cp "$LLMDEPLOY_ROOT/docs/BUNDLE_README_qwen3vl_4b_e2e.md" "$OUT/README.md"
cp "$LLMDEPLOY_ROOT/docs/DEVICE_TEST_qwen3vl_e2e.md" "$OUT/DEVICE_TEST.md"
if [ -d "$KIT" ] && compgen -G "$KIT/wx_*.raw" >/dev/null; then
    cp "$KIT"/wx_*.{raw,json,jpg,script} "$KIT/TEST_IMAGES.md" "$OUT/"
    echo "   test kit: $(ls "$KIT"/wx_*.raw | wc -l) image(s) from $KIT"
else
    echo "   WARNING: no test kit at $KIT -- lint check 11 will fail" >&2
fi

echo "== [2/5] re-read graph names from the final ctx-bins =="
for b in "$VIT_BIN" "${TEXT_BINS[@]}"; do
    "$UTIL" --context_binary "$OUT/$b" --json_file "$OUT/${b%.bin}.info.json" >/dev/null
    echo "   ${b%.bin}.info.json"
done

echo "== [3/5] prompt segments from the tokenizer's chat template =="
PROMPT=${PROMPT:-"Describe this image in one sentence."} \
OUT="$OUT" MODEL="$MODEL" $PY_DEPLOY - <<'PYEOF'
import os
from pathlib import Path
from transformers import AutoProcessor

out = Path(os.environ["OUT"])
prompt = os.environ["PROMPT"]
proc = AutoProcessor.from_pretrained(os.environ["MODEL"])
messages = [{"role": "user", "content": [
    {"type": "image"}, {"type": "text", "text": prompt}]}]
text = proc.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

# The processor expands the single <|image_pad|> to one per merged ViT row at
# call time; the two halves around it are exactly what the accumulator needs
# before and after the image embeddings.
parts = text.split("<|image_pad|>")
assert len(parts) == 2, f"expected exactly one <|image_pad|> in {text!r}"
seg1, seg2 = parts
assert seg1.endswith("<|vision_start|>"), repr(seg1)
assert seg2.startswith("<|vision_end|>"), repr(seg2)

# write_bytes, never echo or print: rdbuf() takes the file verbatim, so ONE
# added newline is a corrupted prompt (genie-app main.cpp:1151-1157). Note the
# rule is "nothing beyond the segment", not "no newline at the end": seg2's own
# last character IS a newline, because the chat template ends
# "<|im_start|>assistant\n" and the model is primed on that token.
assert not seg1.endswith("\n"), repr(seg1)
assert seg2.endswith("<|im_start|>assistant\n"), repr(seg2)
for name, seg in (("prompt_seg1.txt", seg1), ("prompt_seg2.txt", seg2)):
    (out / name).write_bytes(seg.encode("utf-8"))
    print(f"   {name}: {len(seg.encode('utf-8'))} bytes {seg!r}")
PYEOF

echo "== [4/5] sample image (red circle + blue square, the E2E gate's scene) =="
OUT="$OUT" $PY_DEPLOY - <<'PYEOF'
import os
from PIL import Image, ImageDraw

img = Image.new("RGB", (512, 512), "white")
d = ImageDraw.Draw(img)
d.ellipse((96, 96, 288, 288), fill=(220, 30, 30))
d.rectangle((300, 300, 460, 460), fill=(30, 60, 220))
img.save(os.path.join(os.environ["OUT"], "sample_image.png"))
PYEOF
$PY_DEPLOY "$LLMDEPLOY_ROOT/scripts/pipeline/preprocess_image.py" \
    --model "$MODEL" --image "$OUT/sample_image.png" \
    --out "$OUT/sample_image.raw" \
    --encodings "$OUT/${VIT_BIN%.bin}.info.json"

echo "== [5/5] Gate 2: static contract lint =="
# Fails the build on any violation: an unshippable bundle must not reach a tar.
$PY_DEPLOY "$LLMDEPLOY_ROOT/scripts/validate/lint_pipeline_bundle.py" \
    --bundle "$OUT" --model "$MODEL"

tar -C "$LLMDEPLOY_DATA/bundles" -czf "$OUT.tar.gz" "$BUNDLE"
ls -l "$OUT"
echo "files: $(find "$OUT" -type f | wc -l)"
du -sh "$OUT" "$OUT.tar.gz"
echo "BUNDLE COMPLETE: $OUT"
echo "On device: adb push $BUNDLE.tar.gz /data/local/tmp/ && \\"
echo "  adb shell 'cd /data/local/tmp && tar xzf $BUNDLE.tar.gz && cd $BUNDLE && \\"
echo "             chmod +x genie-app && LD_LIBRARY_PATH=. ./genie-app -s genie_pipeline_qwen3vl.script'"
