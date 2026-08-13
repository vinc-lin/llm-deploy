#!/usr/bin/env bash
# Qwen3-VL vision tower, W8A16: quantized ONNX -> FIXED-POINT DLC -> ctx-bin.
#
# The quantized sibling of vit_build.sh, and the one that ships. The fp16 tower
# that vit_build.sh produces cannot be driven by a stock GeniePipeline on QAIRT
# 2.48.40.260702 at all:
#
#   - QnnNspImageModel::setupInputFP16 is an empty stub -- it discards the pixel
#     blob and returns success (nsp-image-model.cpp:526-530);
#   - the requantize dispatch table has no Float16 entries in either direction,
#     so moving image features into the embedding accumulator throws
#     (quantization/src/Quantization.cpp:163-192).
#
# Both gaps close ONLY if the graph's I/O is QNN_DATATYPE_UFIXED_POINT_16, which
# dispatches to a real setupInput<uint16_t> copy (nsp-image-model.cpp:541-543)
# and has a {UFixed16, Float32} converter entry. So "the I/O really is UFIXED_16"
# is not a nice-to-have here -- it is the whole reason the tower was quantized,
# and step 3 fails the build if the converter did not deliver it. This repo has
# precedent for a converter silently folding a quantization away (s4 -> f16 on
# the W4A16 dead end, CLAUDE.md), which is exactly why it is asserted and not
# eyeballed.
#
# CONVERSION FLAGS
#   --quantization_overrides <model.encodings>
#       replaces vit_build.sh's --float_bitwidth-only float conversion. This is
#       what makes the graph fixed-point at all. AIMET wrote 16-bit asymmetric
#       INT activation encodings for pixel_values and all four outputs, so the
#       converter types those five graph I/O tensors uFxp_16.
#   --float_bitwidth 16
#       kept, and NOT redundant: the converter reports a long "OPs fallback to
#       float" list (all 24 attention blocks -- rotary, the two MatMuls, Softmax,
#       and attn.proj), because AIMET's htp_quantsim_config_v81 places no
#       activation quantizers there. Those tensors have to land in fp16, the only
#       float type HTP executes; the 32-bit default would be a graph the backend
#       cannot run. Same flag, same reason, as the text pipeline (convert.sh).
#   NOT --preserve_io_datatype
#       that flag pins graph I/O to the SOURCE model's dtype, i.e. float32 --
#       precisely the failure mode this build exists to avoid. The converter
#       default (preserve_io=[layout], datatype free) is what lets the encodings
#       retype the I/O.
#   No --config io.yaml is needed either: the yaml route (Desired Model
#       Parameters: DataType/QuantParams) can force a dtype onto I/O, but here the
#       encodings already produce UFIXED_16 with the right scale/offset, and a
#       second hand-maintained copy of those numbers is a divergence waiting to
#       happen. Step 3 asserts the DLC's I/O quant params equal model.encodings'.
#
# WHY THE QKV ENCODING IS FILTERED OUT FIRST (step 1a)
# aimet-torch only quantizes MODULE outputs. Everything inside
# Qwen3VLVisionAttention.forward is functional code -- rotary, the two MatMuls,
# Softmax -- so none of it carries an activation encoding and the converter puts
# the whole attention island in fp16. That is fine, and is exactly what the text
# tower ships (its k_proj/v_proj outputs are Float_16 for the same reason).
# The ViT differs in one detail that breaks the build: its QKV is ONE fused
# nn.Linear, so its output IS a module output and IS encoded -- and V reaches
# `attn @ v` through nothing but Reshape/Transpose/Split/Squeeze, all of which
# merely inherit their input's dtype. The converter dequantizes the Q/K path
# (their .float() Casts are ops it must fall back) but leaves V quantized, and
# then asks the backend for a MatMul it does not have:
#
#   [ERROR] Unsupported input/output datatypes requested for the HTP Op 'MatMul'
#           in the node '/blocks.0/attn/MatMul_1'
#     in[0]:QNN_DATATYPE_FLOAT_16  in[1]:QNN_DATATYPE_UFIXED_POINT_16
#     out[0]:QNN_DATATYPE_FLOAT_16
#   ... QnnBackend_validateOpConfig failed 3110 / ComposeGraphs Failed
#
# There is no converter flag that inserts the missing dequantize. Dropping the
# activation encoding on the fused QKV output puts the whole attention island in
# fp16 uniformly, which composes -- and it is a strictly HIGHER-precision graph
# than the quantsim that was measured, since it removes a quantizer rather than
# adding one. The QKV weights stay INT8 per-channel (only the ACTIVATION encoding
# is dropped) and every Gemm, LayerNorm and residual outside attention stays
# W8A16.
#
# This is not a new idea, it is the text tower's rule verbatim. REFERENCE.md
# section 4 already records "QKV grafts the donor q_proj INT16 encoding onto the
# Q split with K/V splits FP16, done at the ENCODINGS level", and
# scripts/export/qkv_surgery.py states it as step 1: "qkv_proj output tensor: NO
# encoding (stays FP16)". The ViT needs only that step, and no donor graft: its
# Q/K reach attention through .float() Casts that fall back to float anyway, so
# there is nothing an INT16 Q encoding could buy.
#
# NAMING -- two independent strings, do not conflate them:
#   * the GRAPH name is baked in at conversion time from the --output_path
#     BASENAME, dots included (converting to vit.dlc.new would yield graph
#     "vit_dlc"), and renaming the file afterwards does NOT change it. It must be
#     "vit", because configs/htp_backend_ext_config_vit.json -- the RUNTIME config
#     that ships in the bundle -- keys its tuning on graph_names ["vit"].
#   * the CTX-BIN FILE name is --binary_file and is unrelated to the graph name.
#     It must be qwen3vl-4b-vit-w8a16_ctx.bin, which is what
#     configs/genie_image_encoder_qwen3vl.json references.
#
# The build-time htp_backend_config.json below is generated private to the build
# dir for the reason vit_build.sh spells out: graph_names is a name-keyed
# selector, so a non-matching name binds the tuning to nothing and the graph
# silently compiles with backend defaults (O=0, 4 MB VTCM). perf_profile stays
# "burst" (not llm_decode_burst): the ViT is a one-shot compute-bound pass.
#
# Usage: vit_build_quant.sh [name] [quant_dir]
set -euo pipefail
source "$(dirname "$0")/../env.sh"

NAME=${1:-qwen3vl-4b-vit-w8a16}
QUANT=${2:-$LLMDEPLOY_DATA/work/quant/qwen3vl-4b-vit}
ONNX=$QUANT/model.onnx
ENC=$QUANT/model.encodings
DLC=$LLMDEPLOY_DATA/work/dlc/$NAME
CTXBIN=$LLMDEPLOY_DATA/work/ctxbin/$NAME
CONVERTER="$QAIRT_SDK/bin/x86_64-linux-clang/qairt-converter"
GRAPH=vit            # ctx-bin graph name == DLC basename -- ASSERTED in step 3
OPT_LEVEL=3          # these three are what the generated config exists to secure;
VTCM_MB=16           # step 3 reads them back out of the finalized binary and
HVX_THREADS=4        # fails the build if the config did not bind

[ -f "$ONNX" ] || { echo "missing $ONNX -- run quantize_vit_aimet.py first"; exit 1; }
[ -f "$ENC" ] || { echo "missing $ENC -- run quantize_vit_aimet.py first"; exit 1; }

echo "== [1a/3] filter the fused-QKV activation encodings (see header) =="
mkdir -p "$DLC"
FILTERED=$DLC/model.filtered.encodings
$PY_DEPLOY - <<'PYEOF' "$ENC" "$FILTERED"
import json
import re
import sys

src, dst = sys.argv[1], sys.argv[2]
d = json.loads(open(src).read())
acts, params = d["activation_encodings"], d["param_encodings"]
assert isinstance(acts, list), f"unexpected encodings schema: {type(acts)}"

QKV_ACT = re.compile(r"^/blocks/blocks\.\d+/attn/qkv/Gemm_output_0$")
QKV_W = re.compile(r"^blocks\.\d+\.attn\.qkv\.weight$")
n_blocks = sum(1 for e in params if QKV_W.match(e["name"]))
dropped = [e["name"] for e in acts if QKV_ACT.match(e["name"])]
assert n_blocks > 0, "no blocks.N.attn.qkv.weight param encodings -- wrong model?"
assert len(dropped) == n_blocks, (
    f"expected one fused-QKV activation encoding per block ({n_blocks}), found "
    f"{len(dropped)}. The graph changed shape; re-derive the filter before "
    "shipping, do not convert as-is -- an unfiltered fused-QKV output makes the "
    "HTP backend reject /blocks.0/attn/MatMul_1 (fp16 x ufixed16).")
d["activation_encodings"] = [e for e in acts if not QKV_ACT.match(e["name"])]
open(dst, "w").write(json.dumps(d, indent=2))
print(f"  dropped {len(dropped)} fused-QKV activation encodings of {len(acts)}; "
      f"{len(d['activation_encodings'])} remain, {len(params)} param encodings kept "
      "(QKV weights stay INT8 per-channel)")
PYEOF

echo "== [1/3] convert quantized ViT -> UFIXED_16 DLC =="
# 442 MB DLC out of a 1.65 GB external-data ONNX, plus the converter's own
# simplification copies. Sized to the step, per CLAUDE.md.
disk_guard 20
$PY_QAIRT "$CONVERTER" --input_network "$ONNX" \
    --output_path "$DLC/$GRAPH.dlc" \
    --quantization_overrides "$FILTERED" \
    --float_bitwidth 16 --target_backend HTP \
    -d pixel_values "1024,1536"

# Fail here rather than after a 9-minute ctx-bin generation: the DLC is where the
# fixed-point I/O either happened or did not.
qairt-dlc-info -i "$DLC/$GRAPH.dlc" > "$DLC/dlc-info.txt"
$PY_DEPLOY - <<'PYEOF' "$DLC/dlc-info.txt"
import re
import sys

text = open(sys.argv[1]).read()
want = {"pixel_values": "APP_WRITE"}
for n in ("image_features", "deepstack_visual_embed_0", "deepstack_visual_embed_1",
          "deepstack_visual_embed_2"):
    want[n] = "APP_READ"
DIMS = {"pixel_values": "1024,1536"}
errs = []
for name, kind in want.items():
    # The leading boundary matters: without it this would also match a longer
    # converter-invented tensor that merely ENDS in one of these names.
    m = re.search(r"(?:^|[\s|])" + re.escape(name) +
                  r" \(data type: (\w+); tensor dimension: "
                  r"\[([^]]*)\]; tensor type: (\w+)\)", text)
    if not m:
        errs.append(f"{name}: not a graph I/O tensor of the DLC")
        continue
    dtype, dims, ttype = m.groups()
    print(f"  DLC I/O {name:26s} {dtype:9s} [{dims}] {ttype}")
    if dtype != "uFxp_16":
        errs.append(f"{name}: DLC dtype is {dtype}, expected uFxp_16 "
                    "(QNN_DATATYPE_UFIXED_POINT_16)")
    if ttype != kind:
        errs.append(f"{name}: tensor type {ttype}, expected {kind}")
    if dims != DIMS.get(name, "256,2560"):
        errs.append(f"{name}: dims [{dims}], expected [{DIMS.get(name, '256,2560')}] "
                    "-- wrong -d, or the graph is not the 32x32-patch ViT")
if errs:
    print("BUILD REJECTED: converter did not emit fixed-point graph I/O")
    for e in errs:
        print("  - " + e)
    print("  A float-I/O vision tower cannot be driven by GeniePipeline at all:")
    print("  setupInputFP16 is an empty stub and the requant table has no Float16.")
    sys.exit(1)
PYEOF

echo "== [2/3] single-graph ctx-bin (vtcm 16, unsigned PD, v81) =="
disk_guard 6
mkdir -p "$CTXBIN"
cat > "$CTXBIN/htp_backend_config.json" <<JSONEOF
{
  "graphs": [
    {
      "graph_names": [
        "$GRAPH"
      ],
      "O": $OPT_LEVEL,
      "vtcm_mb": $VTCM_MB,
      "hvx_threads": $HVX_THREADS,
      "fp16_relaxed_precision": 0
    }
  ],
  "devices": [
    {
      "dsp_arch": "v81",
      "soc_model": 0,
      "pd_session": "unsigned",
      "cores": [
        {
          "core_id": 0,
          "perf_profile": "burst",
          "rpc_control_latency": 100,
          "rpc_polling_time": 9999
        }
      ]
    }
  ]
}
JSONEOF
cat > "$CTXBIN/htp_config.json" <<JSONEOF
{
  "backend_extensions": {
    "shared_library_path": "libQnnHtpNetRunExtensions.so",
    "config_file_path": "htp_backend_config.json"
  }
}
JSONEOF
cd "$CTXBIN"
qnn-context-binary-generator \
    --model "$QAIRT_SDK/lib/x86_64-linux-clang/libQnnModelDlc.so" \
    --dlc_path "$DLC/$GRAPH.dlc" \
    --backend "$QAIRT_SDK/lib/x86_64-linux-clang/libQnnHtp.so" \
    --output_dir "$CTXBIN" --binary_file "${NAME}_ctx" \
    --config_file htp_config.json

echo "== [3/3] dump graph info and verify dtypes, encodings and tuning =="
qnn-context-binary-utility --context_binary "$CTXBIN/${NAME}_ctx.bin" \
    --json_file "$CTXBIN/info.json"
$PY_DEPLOY - <<'PYEOF' "$CTXBIN/info.json" "$ENC" "$GRAPH" "$OPT_LEVEL" "$VTCM_MB" "$HVX_THREADS"
import json
import sys

info_path, enc_path, GRAPH = sys.argv[1], sys.argv[2], sys.argv[3]
OPT_LEVEL, VTCM_MB, HVX_THREADS = (int(x) for x in sys.argv[4:7])
OUTS = ["image_features", "deepstack_visual_embed_0", "deepstack_visual_embed_1",
        "deepstack_visual_embed_2"]

d = json.load(open(info_path))
graphs = d["info"]["graphs"]
for g in graphs:
    print("GRAPH:", g["info"]["graphName"])

errs = []
# --- exactly one graph, named `vit` -------------------------------------------
# A backend-extension config that fails to bind does not error -- the graph just
# compiles with defaults (O=0, vtcm 4 MB, 0 HVX threads) and the log stays clean.
# That exact silent regression shipped once already: verify32 omitted from
# graph_names, 4 MB VTCM + 24 MB spill on device, LADE SIGSEGV
# (docs/BUILD_GUIDE.md section 5.4b). With no device, this readback is the only
# place it is detectable, so it is a build failure, not a warning.
if len(graphs) != 1:
    errs.append("expected 1 graph, found %d: %r"
                % (len(graphs), [g["info"]["graphName"] for g in graphs]))
    print("BUILD REJECTED:", errs[0])
    sys.exit(1)

info = graphs[0]["info"]
if info["graphName"] != GRAPH:
    errs.append("graphName is %r, expected %r -- graph_names in "
                "htp_backend_ext_config_vit.json keys on that string, so the "
                "runtime config would bind to nothing"
                % (info["graphName"], GRAPH))

# --- fixed-point I/O, with the encodings intact --------------------------------
enc = json.loads(open(enc_path).read())
acts = enc["activation_encodings"]
by_name = {e["name"]: e for e in acts} if isinstance(acts, list) else dict(acts)


def want_quant(name):
    e = by_name.get(name)
    assert e is not None, f"{name} has no activation encoding in {enc_path}"
    if isinstance(e, list):
        e = e[0]
    scale, offset = e["scale"], e["offset"]
    return (scale[0] if isinstance(scale, list) else scale,
            offset[0] if isinstance(offset, list) else offset)


def check_io(t, name, kind):
    ti = t["info"]
    dtype = ti.get("dataType")
    print(f"  ctx-bin {kind:9s} {name:26s} {dtype} {ti.get('dimensions')} "
          f"{json.dumps(ti.get('quantizeParams'))}")
    if dtype != "QNN_DATATYPE_UFIXED_POINT_16":
        errs.append(f"{name}: ctx-bin dataType is {dtype}, expected "
                    "QNN_DATATYPE_UFIXED_POINT_16 -- GeniePipeline cannot drive "
                    "float I/O (setupInputFP16 is a stub; the requant table has "
                    "no Float16 entries)")
        return
    qp = ti.get("quantizeParams") or {}
    # qnn-context-binary-utility names this member "scaleOffset" on 2.48.40;
    # other releases have emitted "scaleOffsetEncoding". Accept either, but
    # never fall through to "no check" -- a silently skipped encodings check is
    # how a wrong scale reaches a device.
    so = qp.get("scaleOffset") or qp.get("scaleOffsetEncoding")
    if qp.get("quantizationEncoding") != "QNN_QUANTIZATION_ENCODING_SCALE_OFFSET" or not so:
        errs.append(f"{name}: quantizeParams carry no scale/offset: {qp!r}")
        return
    ws, wo = want_quant(name)
    # The ctx-bin stores the scale as float32; model.encodings is the float64
    # text of that same float32, so compare with a tight relative tolerance
    # rather than for exact equality. The offset is integral and must match.
    if abs(so["scale"] - ws) > 1e-6 * abs(ws):
        errs.append(f"{name}: scale {so['scale']!r} != encodings {ws!r}")
    if int(so["offset"]) != int(wo):
        errs.append(f"{name}: offset {so['offset']!r} != encodings {wo!r}")


ins = info.get("graphInputs") or []
outs = info.get("graphOutputs") or []
if [t["info"]["name"] for t in ins] != ["pixel_values"]:
    errs.append("graph inputs are %r, expected ['pixel_values']"
                % [t["info"]["name"] for t in ins])
else:
    check_io(ins[0], "pixel_values", "input")

got_outs = {t["info"]["name"]: t for t in outs}
if set(got_outs) != set(OUTS):
    errs.append("graph outputs are %r, expected %r" % (sorted(got_outs), sorted(OUTS)))
for name in OUTS:                      # by name: the ctx-bin's own order is
    t = got_outs.get(name)             # deepstack_* first, image_features last
    if t is not None:
        check_io(t, name, "output")

# --- HTP tuning actually bound -------------------------------------------------
blob = info.get("graphBlobInfo", {}).get("info", {})
for key, want in (("optimizationLevel", OPT_LEVEL),
                  ("vtcmSize", VTCM_MB),
                  ("numHvxThreads", HVX_THREADS)):
    if blob.get(key) != want:
        errs.append("%s is %r, expected %r" % (key, blob.get(key), want))

if errs:
    print("BUILD REJECTED")
    for e in errs:
        print("  - " + e)
    print("  see docs/NOTES-vit-htp-config.md and CLAUDE.md (hard contracts)")
    sys.exit(1)

print("IO VERIFIED: pixel_values + 4 outputs are QNN_DATATYPE_UFIXED_POINT_16 "
      "with the model.encodings scale/offset")
print("CONFIG BOUND: O=%d vtcm=%d MB hvx_threads=%d spill_fill=%d bytes"
      % (blob["optimizationLevel"], blob["vtcmSize"], blob["numHvxThreads"],
         blob.get("spillFillBufferSize", -1)))
PYEOF
ls -lh "$CTXBIN"
echo "VIT QUANT BUILD COMPLETE: $CTXBIN/${NAME}_ctx.bin"
echo "next: scripts/validate/parity_vit_quant.py --dlc $DLC/$GRAPH.dlc --info $CTXBIN/info.json"
