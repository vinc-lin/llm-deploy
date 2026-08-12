#!/usr/bin/env bash
# Assemble a push-ready flat device bundle for the Qwen3-VL-4B vision tower
# (Stage 1). Same flat layout as bundle.sh (BUILD_GUIDE §1): every .so, the
# ctx-bin, and every JSON live in ONE directory, because Genie's loader resolves
# runtime .so files from the bundle root, not from a lib/ subdirectory.
#
# Usage: vit_bundle.sh [name] [ctxbin_path]
#
# ---------------------------------------------------------------------------
# Two things this script deliberately does differently from bundle.sh:
#
# 1. It ships configs/htp_backend_ext_config_vit.json, NOT the shared
#    configs/htp_backend_ext_config.json. `graph_names` is a name-keyed
#    selector: the shared file lists ["prefill","decode","verify32"], and our
#    graph is named "vit". Shipping the shared file verbatim would bind the vit
#    graph to NO tuning block, silently falling back to defaults (O=0, 4 MB
#    VTCM, 0 HVX threads) at runtime. That exact failure is on record in
#    BUILD_GUIDE §5.4 / reports/qwen3-0.6b-w8a16-ladekv-test-report.md, where an
#    omitted `verify32` gave 4 MB VTCM + 24 MB spill instead of 16 MB + 0 and
#    ended in a SIGSEGV. The compile-time half of this is handled in
#    vit_build.sh; this is the runtime half.
#
# 2. The ViT runtime config uses perf_profile "burst", not the shared file's
#    "llm_decode_burst". llm_decode_burst is tuned for the memory-latency-bound
#    AR=1 autoregressive decode loop. The vision tower is a one-shot,
#    compute-bound FP16 encoder pass, so the generic maximum-performance profile
#    is the right fit -- and it is what the build-time config that vit_build.sh
#    generated already uses, which keeps compile-time and runtime tuning aligned.
# ---------------------------------------------------------------------------
set -euo pipefail
source "$(dirname "$0")/../env.sh"

NAME=${1:-qwen3vl_4b_vit_fp16}
CTXBIN=${2:-$LLMDEPLOY_DATA/work/ctxbin/qwen3vl-4b-vit-fp16/qwen3vl-4b-vit-fp16_ctx.bin}

ENCODER_CFG=$LLMDEPLOY_ROOT/configs/genie_image_encoder_qwen3vl.json
EXT_CFG=$LLMDEPLOY_ROOT/configs/htp_backend_ext_config_vit.json

[ -f "$CTXBIN" ] || { echo "ERROR: ctx-bin not found: $CTXBIN" >&2
                      echo "       build it first with scripts/build/vit_build.sh" >&2
                      exit 1; }
[ -f "$ENCODER_CFG" ] || { echo "ERROR: missing $ENCODER_CFG" >&2; exit 1; }
[ -f "$EXT_CFG" ]     || { echo "ERROR: missing $EXT_CFG" >&2; exit 1; }

OUT=$LLMDEPLOY_DATA/bundles/$NAME
rm -rf "$OUT"   # stale binaries from a previous bundling must not leak in
mkdir -p "$OUT"

A=$QAIRT_SDK/lib/aarch64-android
H=$QAIRT_SDK/lib/hexagon-v81/unsigned
B=$QAIRT_SDK/bin/aarch64-android

# The 7 device libraries (BUILD_GUIDE §1) -- flat, no lib/ subdir.
# Every path is asserted: a missing .so must be a hard failure, not a warning
# that scrolls past and leaves an unloadable bundle on the device.
SRCS=(
  "$A/libGenie.so"
  "$A/libQnnHtp.so"
  "$A/libQnnSystem.so"
  "$A/libQnnHtpPrepare.so"
  "$A/libQnnHtpNetRunExtensions.so"
  "$A/libQnnHtpV81Stub.so"
  "$H/libQnnHtpV81Skel.so"
  # Driver. genie-t2t-run is text-only (GenieDialog_* only) and genie-t2e-run is
  # a text-to-embedding tool (GenieEmbedding_* only); neither can drive an image
  # encoder. genie-app is the one that can: it exposes the GenieNode_* and
  # GeniePipeline_* APIs as a scripted command language, including
  #   node config create / node create / node set image | node set data /
  #   node execute / node get data
  # and it knows the GENIE_NODE_IMAGE_ENCODER_* tensor roles. It is driven with
  # `genie-app -s SCRIPT`. See the Stage 3 note at the bottom of this file.
  "$B/genie-app"
)

MISSING=0
for f in "${SRCS[@]}"; do
  [ -f "$f" ] || { echo "ERROR: missing SDK file: $f" >&2; MISSING=1; }
done
[ "$MISSING" -eq 0 ] || { echo "ERROR: $QAIRT_SDK is missing required files (above); aborting" >&2; exit 1; }

cp "${SRCS[@]}" "$OUT/"
cp "$CTXBIN"     "$OUT/"
cp "$ENCODER_CFG" "$OUT/genie_image_encoder.json"
cp "$EXT_CFG"     "$OUT/"

# The encoder config addresses the ctx-bin and the extensions file by bare
# filename, resolved relative to the bundle root. Assert every internal
# reference actually landed in the bundle -- a dangling reference here is a
# silent runtime failure on a device we cannot attach a debugger to.
python3 - "$OUT" <<'PYEOF'
import json, os, sys
out = sys.argv[1]
cfg_name = "genie_image_encoder.json"
cfg = json.load(open(os.path.join(out, cfg_name)))
eng = cfg["image-encoder"]["engine"]
refs = list(eng["model"]["binary"]["ctx-bins"]) + [eng["backend"]["extensions"]]
bad = [r for r in refs if not os.path.isfile(os.path.join(out, r))]
for r in refs:
    print(f"  {'OK  ' if r not in bad else 'DANGLING'} {cfg_name} -> {r}")
if bad:
    sys.exit(f"ERROR: {cfg_name} references files not present in bundle: {bad}")
ext = json.load(open(os.path.join(out, eng["backend"]["extensions"])))
names = [n for g in ext["graphs"] for n in g["graph_names"]]
print(f"  graph_names = {names}")
if "vit" not in names:
    sys.exit("ERROR: 'vit' not in graph_names -- graph would run with HTP defaults")
PYEOF

ls -l "$OUT"
tar -C "$LLMDEPLOY_DATA/bundles" -czf "$LLMDEPLOY_DATA/bundles/$NAME.tar.gz" "$NAME"
ls -lh "$LLMDEPLOY_DATA/bundles/$NAME.tar.gz"
echo "BUNDLE READY: $LLMDEPLOY_DATA/bundles/$NAME.tar.gz"
echo "On device: adb push $NAME.tar.gz /data/local/tmp/ && adb shell 'cd /data/local/tmp && tar xzf $NAME.tar.gz'"
cat <<'EOF'
Run:       cd /data/local/tmp/<name> && LD_LIBRARY_PATH=. ./genie-app -s vit.txt

  genie-app is a script interpreter (`-s FILE`), not a flag-driven CLI. The
  SDK's own image-encoder example is examples/Genie/genie-app/scripts/glm-4v;
  the vision-tower-only subset of it is:

    pipeline config create pcfg
    pipeline create        pipe pcfg
    node config create     vitcfg genie_image_encoder.json
    node create            vit    vitcfg
    pipeline add           pipe vit
    node set image         vit GENIE_NODE_IMAGE_ENCODER_IMAGE_INPUT pixel_values.raw
    pipeline execute       pipe
    node free              vit
    pipeline free          pipe

  `node set image` does NO image decoding or preprocessing -- it reads the file
  as an opaque byte blob straight into GenieNode_setData. So pixel_values.raw
  must already be the host-preprocessed FP16 [1024,1536] tensor in Qwen3-VL's
  patch layout, exactly as fed to the parity check.

  KNOWN STAGE 3 CONSTRAINT: Genie's ImageEncoder node exposes exactly one
  output, GENIE_NODE_IMAGE_ENCODER_EMBEDDING_OUTPUT (see the SDK's
  Genie/src/pipeline/ImageEncoder.cpp -- one m_data buffer, published to the
  pipeline under the single tensor name "image_embeddings"). Our graph emits
  four: image_features plus deepstack_visual_embed_0/1/2. The three deepstack
  tensors have no Genie node IO name and therefore no route out of a stock
  ImageEncoder node. Wiring ViT -> LLM through GeniePipeline will get
  image_features only; reading all four needs qnn-net-run against the ctx-bin
  (using the build-time htp_config.json/htp_backend_config.json next to it) or a
  custom QNN driver. Resolve this before committing to a Stage 3 design.

  Our graph's single input is named `pixel_values`, which is in ImageEncoder's
  accepted input-name map, so the node itself will construct without tripping
  "ImageEncoder meet unsupported input layer of model".
EOF
