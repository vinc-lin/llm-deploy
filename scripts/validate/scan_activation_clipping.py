#!/usr/bin/env python
"""Which activations exceed their CALIBRATED encoding range on a given row?

WHY THIS EXISTS. Test F (2026-08-20) proved the Qwen3-VL-4B boundary gain is
triggered by the attention-sink condition, and the first reading of that result
was "fp16 saturation in input_layernorm". That reading is wrong, and the
encodings say so: these activations are `bw:16, dtype:INT, PER_TENSOR`
asymmetric with a CALIBRATED range, so fp16's 65504 never enters. The failure
mode actually available is range CLIPPING -- which is a build-side artefact we
choose, not a hardware limit.

AIMET is run with `QuantScheme.post_training_tf_enhanced`, an MSE-optimal range
search that DELIBERATELY discards outliers: sacrificing one extreme value buys
finer resolution for the other 2559. On an attention-sink row -- whose massive
activations sit orders of magnitude above the bulk -- that is exactly the wrong
trade, and the calibrated range comes out short.

Measured on `fp_ctrl_pre`, sink row 0 vs normal row 1:

    layers.0/mlp/down_proj   actual 9.00  vs calibrated 5.482   1.64x over
    layers.1/mlp/Mul_1       actual 61.40 vs calibrated 59.272  1.04x over

while the same tensors on row 1 peak at 1.325 and 4.033 -- comfortably inside.

Run `sim_activation_clipping.py` next: it clamps to these ranges and shows the
clipping reproduces the device's 1.39x.

  scan_activation_clipping.py <kit> <encodings> [case]
  PROBE_ONNX=... to point at a graph other than the VL-4B past-KV prefill.
"""
import json
import os
import sys
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort

KIT = Path(sys.argv[1])
ENC = Path(sys.argv[2])
CASE = sys.argv[3] if len(sys.argv) > 3 else "fp_ctrl_pre"
ONNX = Path(os.environ.get(
    "PROBE_ONNX",
    Path(os.environ["LLMDEPLOY_DATA"]) /
    "work/onnx/qwen3vl-4b-gqa-aimet-splitkv/prefill_0/prefill_0.onnx"))

meta = json.loads((KIT / CASE / "ref" / "meta.json").read_text())
rows = meta["real_rows"]
sink, normal = rows[0], rows[1]

# ---- calibrated ranges ----------------------------------------------------
enc = json.loads(ENC.read_text())["activation_encodings"]
rng = {}
for e in enc:
    if e.get("dtype") != "INT":
        continue
    sc, off = e["scale"][0], e["offset"][0]
    qmax = (1 << e["bw"]) - 1
    rng[e["name"]] = (off * sc, (qmax + off) * sc, e["bw"])

# ---- rebuild the exact device feed ---------------------------------------
m = onnx.load(str(ONNX), load_external_data=False)
have = {o.name for o in m.graph.output}
produced = {o for n in m.graph.node for o in n.output}
want = [n for n in rng if n in produced]
for n in want:
    if n not in have:
        m.graph.output.append(onnx.helper.make_tensor_value_info(
            n, onnx.TensorProto.FLOAT, None))
print(f"tapping {len(want)} of {len(rng)} encoded tensors")

# The weights live in a sibling `prefill_0.data`, referenced by a RELATIVE path,
# so the modified graph has to be written into the SAME directory or ORT cannot
# resolve it. Serialising to bytes loses that resolution entirely.
tapped = ONNX.with_name(ONNX.stem + "_tapped.onnx")
onnx.save(m, str(tapped))
so = ort.SessionOptions()
so.intra_op_num_threads = int(os.environ.get("OMP_NUM_THREADS", "40"))
try:
    sess = ort.InferenceSession(str(tapped), so,
                                providers=["CPUExecutionProvider"])
finally:
    tapped.unlink(missing_ok=True)
shapes = {i.name: i.shape for i in sess.get_inputs()}


def rd(name, g=None):
    g = g or meta["kind"] + "_0"
    p = KIT / CASE / g / f"{name}.raw"
    raw = p.read_bytes()
    spec = meta["input_dtypes"].get(name)
    if spec == "QNN_DATATYPE_UFIXED_POINT_16":
        # dequantise with the bin's own inputs_embeds encoding
        sc, off = 0.00035394279984757304, -32927
        # scale is a python float -> numpy promotes to float64 and ORT
        # rejects the feed. Force float32 back.
        return (((np.frombuffer(raw, "<u2").astype(np.float32) + off) * sc)
                .astype(np.float32))
    return np.frombuffer(raw, "<f2").astype(np.float32)


feeds = {}
for t in ("inputs_embeds", "attention_mask", "position_ids_cos",
          "position_ids_sin"):
    feeds[t] = rd(t).reshape(shapes[t])
for d in meta["deep_names"]:
    dn = d + "_p" if d + "_p" in shapes else d
    feeds[dn] = np.zeros([x if isinstance(x, int) else 1
                          for x in shapes[dn]], np.float32)
for n, s in shapes.items():
    if n.startswith("past_"):
        feeds[n] = np.zeros([x if isinstance(x, int) else 1 for x in s],
                            np.float32)

outs = sess.run(want, feeds)

# ---- compare --------------------------------------------------------------
print(f"\ncase {CASE}: sink row {sink}, normal row {normal}\n")
print(f"{'tensor':62s} {'calib range':>22s} {'sink |max|':>11s} "
      f"{'over':>7s} {'norm |max|':>11s}")
bad = []
for n, v in zip(want, outs):
    lo, hi, bw = rng[n]
    a = np.asarray(v)
    if a.ndim < 2 or a.shape[-2] < max(rows) + 1:
        continue                      # not an [.., AR, ..] tensor
    s = float(np.abs(a[..., sink, :]).max())
    o = float(np.abs(a[..., normal, :]).max())
    lim = max(abs(lo), abs(hi))
    over = s / lim if lim else float("inf")
    flag = "  <== CLIPS" if over > 1.0 else ""
    if over > 1.0:
        bad.append((n, over, s, lim))
    print(f"{n[:62]:62s} [{lo:9.3f},{hi:9.3f}] {s:11.2f} {over:7.2f}x "
          f"{o:11.3f}{flag}")

print("\n=== tensors whose SINK-row value exceeds the calibrated range ===")
if not bad:
    print("  none -- every encoded tensor fits. Clipping is NOT the mechanism.")
for n, o, s, lim in sorted(bad, key=lambda x: -x[1]):
    print(f"  {o:8.2f}x over   actual {s:12.2f} vs calibrated {lim:10.2f}   {n}")
