#!/usr/bin/env python
"""Does clamping to the CALIBRATED encoding ranges reproduce the device's
row-0 gain? Host only.

Companion to `scan_activation_clipping.py`, which finds WHICH tensors clip.
This one proves the clipping is the CAUSE, by clamping selected tensors to
their calibrated [min, max] inside the real graph and measuring the resulting
boundary against the untouched reference.

MEASURED 2026-08-20, case fp_ctrl_pre, row 0 (device: 1.38959x, resid 0.447%,
cos 0.999990):

    variant                                        gain    resid       cos
    none        (control)                        1.0013   0.002%  1.000000
    idx:0       layers.0/mlp/down_proj alone     0.9295   0.169%  0.999999
    idx:1       layers.1/mlp/Mul_1 alone         1.0613   0.089%  1.000000
    idx:0,1     BOTH                             1.3914   0.416%  0.999991
    overrange   all five over-range tensors      1.3937   0.417%  0.999991
    all         all 180 encoded tensors          1.3505   0.384%  0.999993

Rows 1-3 stay within 0.6% in every variant, matching the device.

NOTE THE INTERACTION. Neither tensor alone does it -- 0.93x and 1.06x multiply
to ~0.99 -- yet together they give 1.39x. Clipping layer 0's down_proj changes
the residual entering layer 1, which changes what layer 1's SwiGLU product
does. So no single tensor's clipping ratio predicts the boundary error, and
looking for one culprit would have missed this.

Use it to VALIDATE A FIX too: recalibrate, re-run `scan_activation_clipping.py`
to confirm nothing on a sink row exceeds its range, then re-run this and expect
the `overrange` gain to fall to ~1.00.

  sim_activation_clipping.py <kit> <encodings> <case> <variant>
     variant: none | l0down | overrange | all | idx:0,1
"""
import gc
import json
import os
import sys
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort
from onnx import helper, numpy_helper

KIT, ENC, CASE, VARIANT = Path(sys.argv[1]), Path(sys.argv[2]), sys.argv[3], sys.argv[4]
ONNX = Path(os.environ.get(
    "PROBE_ONNX",
    Path(os.environ["LLMDEPLOY_DATA"]) /
    "work/onnx/qwen3vl-4b-gqa-aimet-splitkv/prefill_0/prefill_0.onnx"))

meta = json.loads((KIT / CASE / "ref" / "meta.json").read_text())
rows = meta["real_rows"]

rng = {}
for e in json.loads(ENC.read_text())["activation_encodings"]:
    if e.get("dtype") != "INT":
        continue
    sc, off = e["scale"][0], e["offset"][0]
    rng[e["name"]] = (off * sc, ((1 << e["bw"]) - 1 + off) * sc)

OVER = ["/layers/layers.0/mlp/down_proj/MatMul_output_0",
        "/layers.1/mlp/Mul_1_output_0",
        "/layers/layers.0/mlp/gate_proj/MatMul_output_0",
        "/layers/layers.2/mlp/up_proj/MatMul_output_0",
        "/layers/layers.2/mlp/gate_proj/MatMul_output_0"]

m = onnx.load(str(ONNX), load_external_data=False)
produced = {o for n in m.graph.node for o in n.output}
if VARIANT == "none":
    targets = []
elif VARIANT == "l0down":
    targets = OVER[:1]
elif VARIANT == "overrange":
    targets = OVER
elif VARIANT == "all":
    targets = [n for n in rng if n in produced]
elif VARIANT.startswith("idx:"):
    targets = [OVER[int(i)] for i in VARIANT[4:].split(",")]
else:
    raise SystemExit(f"unknown variant {VARIANT}")
targets = [t for t in targets if t in produced and t in rng]

for t in targets:
    lo, hi = rng[t]
    node = next(n for n in m.graph.node if t in n.output)
    idx = list(node.output).index(t)
    pre = t + "_preclip"
    node.output[idx] = pre
    lo_n, hi_n = t + "_lo", t + "_hi"
    m.graph.initializer.extend([
        numpy_helper.from_array(np.array(lo, np.float32), lo_n),
        numpy_helper.from_array(np.array(hi, np.float32), hi_n)])
    m.graph.node.append(helper.make_node("Clip", [pre, lo_n, hi_n], [t],
                                         name=t + "_clipnode"))
# Clip nodes were appended at the end; ONNX requires topological order.
order, emitted, pending = [], set(i.name for i in m.graph.initializer) | \
    set(i.name for i in m.graph.input), list(m.graph.node)
while pending:
    progress = False
    rest = []
    for n in pending:
        if all(i in emitted or i == "" for i in n.input):
            order.append(n)
            emitted.update(n.output)
            progress = True
        else:
            rest.append(n)
    if not progress:
        raise SystemExit("could not topologically sort after clip insertion")
    pending = rest
del m.graph.node[:]
m.graph.node.extend(order)

print(f"variant {VARIANT}: clamped {len(targets)} tensor(s)")
tapped = ONNX.with_name(f"{ONNX.stem}_clamp_{VARIANT}.onnx")
onnx.save(m, str(tapped))
so = ort.SessionOptions()
so.intra_op_num_threads = int(os.environ.get("OMP_NUM_THREADS", "40"))
try:
    sess = ort.InferenceSession(str(tapped), so, providers=["CPUExecutionProvider"])
finally:
    tapped.unlink(missing_ok=True)
shapes = {i.name: i.shape for i in sess.get_inputs()}


def rd(name, g=None):
    g = g or meta["kind"] + "_0"
    raw = (KIT / CASE / g / f"{name}.raw").read_bytes()
    if meta["input_dtypes"].get(name) == "QNN_DATATYPE_UFIXED_POINT_16":
        sc, off = 0.00035394279984757304, -32927
        return (((np.frombuffer(raw, "<u2").astype(np.float32) + off) * sc)
                .astype(np.float32))
    return np.frombuffer(raw, "<f2").astype(np.float32)


feeds = {t: rd(t).reshape(shapes[t]) for t in
         ("inputs_embeds", "attention_mask", "position_ids_cos", "position_ids_sin")}
for d in meta["deep_names"]:
    dn = d + "_p" if d + "_p" in shapes else d
    feeds[dn] = np.zeros([x if isinstance(x, int) else 1 for x in shapes[dn]], np.float32)
for n, s in shapes.items():
    if n.startswith("past_"):
        feeds[n] = np.zeros([x if isinstance(x, int) else 1 for x in s], np.float32)

hid = sess.run(["last_hidden_states"], feeds)[0].reshape(-1, 2560)
del sess
gc.collect()

ref = np.load(KIT / CASE / "ref" / "last_hidden_states.npy")
print(f"\n{'row':>5} {'ref RMS':>10} {'got RMS':>10} {'gain':>8} {'resid':>8} {'cos':>10}")
for i, r in enumerate(rows):
    a = ref[i].astype(np.float64)
    b = hid[r].astype(np.float64)
    g = float(b @ a / (a @ a))
    resid = float(np.linalg.norm(b - g * a) / max(np.linalg.norm(b), 1e-30))
    c = float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b)))
    ra = float(np.sqrt((a ** 2).mean()))
    rb = float(np.sqrt((b ** 2).mean()))
    print(f"{r:>5} {ra:10.3f} {rb:10.3f} {g:8.4f} {resid:8.3%} {c:10.6f}")
