#!/usr/bin/env python
"""Does ANY encoded activation exceed its calibrated range on a REALISTIC input?

MEASURED 2026-08-20: worst over-range on a HELD-OUT eval window is 1.082x, and
row-0 ratios run 0.08-0.99x -- row 0 never clips on a realistic window. The
calibrated ranges generalise. Compare the Test F probe's row 0 at 1.64x over.

Also measured here: the calibrated ranges sit essentially ON the calibration
maximum (5.4819 vs 5.4805 observed; 3.0768 vs 3.0771), so tf_enhanced did NOT
discard outliers and a min-max requantize would produce the same ranges. That
killed the proposed fix before it was built.


Scope question for the boundary defect. The Test F probe reproduces it, but the
probe's row 0 is a plain word token at position 0 -- something a chat-templated
production prompt never contains, since those always begin <|im_start|>. So the
probe may be exercising a condition that never occurs in real use.

This scans all 180 encoded tensors over every calibration AND held-out eval
window and reports anything over range. Eval windows are the ones that matter:
the calib split had its ranges fitted to it and cannot exceed by construction.
"""
import json
import os
import sys
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort

CALIB, ENC = Path(sys.argv[1]), Path(sys.argv[2])
ONNX = Path(os.environ["LLMDEPLOY_DATA"]) / \
    "work/onnx/qwen3vl-4b-gqa-aimet-splitkv/prefill_0/prefill_0.onnx"

rng = {}
for e in json.loads(ENC.read_text())["activation_encodings"]:
    if e.get("dtype") == "INT":
        sc, off = e["scale"][0], e["offset"][0]
        rng[e["name"]] = max(abs(off * sc), abs(((1 << e["bw"]) - 1 + off) * sc))

z = np.load(CALIB)
emb, mask, cos, sin = z["embeds"], z["mask"], z["cos"], z["sin"]
deep, nval, desc, splits = z["deep"], z["n_valid"], z["desc"], z["split"]
N, AR = emb.shape[0], int(z["ar"])

m = onnx.load(str(ONNX), load_external_data=False)
have = {o.name for o in m.graph.output}
produced = {o for n in m.graph.node for o in n.output}
want = [t for t in rng if t in produced]
for t in want:
    if t not in have:
        m.graph.output.append(onnx.helper.make_tensor_value_info(
            t, onnx.TensorProto.FLOAT, None))
tapped = ONNX.with_name("prefill_0_scanall.onnx")
onnx.save(m, str(tapped))
so = ort.SessionOptions()
so.intra_op_num_threads = int(os.environ.get("OMP_NUM_THREADS", "40"))
try:
    sess = ort.InferenceSession(str(tapped), so, providers=["CPUExecutionProvider"])
finally:
    tapped.unlink(missing_ok=True)
shapes = {i.name: i.shape for i in sess.get_inputs()}
TOTAL = shapes["attention_mask"][-1]
PAST, MV = TOTAL - AR, -100.0
print(f"{N} windows, AR={AR}, tapping {len(want)} encoded tensors")

past_feeds = {n: np.zeros([x if isinstance(x, int) else 1 for x in s], np.float32)
              for n, s in shapes.items() if n.startswith("past_")}
deep_names = [n for n in shapes if n.startswith("deepstack_")]

worst = {}       # tensor -> (ratio, split, desc, row0_ratio)
for i in range(N):
    mk = np.full((1, AR, TOTAL), MV, np.float32)
    mk[0, :, PAST:] = np.where(mask[i][0] >= 0, 0.0, MV)
    f = {"inputs_embeds": emb[i].astype(np.float32).reshape(shapes["inputs_embeds"]),
         "attention_mask": mk,
         "position_ids_cos": cos[i].astype(np.float32).reshape(shapes["position_ids_cos"]),
         "position_ids_sin": sin[i].astype(np.float32).reshape(shapes["position_ids_sin"]),
         **past_feeds}
    for k, dn in enumerate(deep_names):
        f[dn] = deep[i][k].astype(np.float32).reshape(shapes[dn])
    outs = sess.run(want, f)
    nv = int(nval[i])
    for t, v in zip(want, outs):
        a = np.abs(np.asarray(v))
        if a.ndim < 2 or a.shape[-2] < nv:
            continue
        r = float(a[..., :nv, :].max()) / rng[t]
        r0 = float(a[..., 0, :].max()) / rng[t]
        if t not in worst or r > worst[t][0]:
            worst[t] = (r, str(splits[i]), str(desc[i]), r0)
    print(f"  {i+1}/{N} {str(splits[i]):5s} {str(desc[i])[:44]}", flush=True)

print("\n=== tensors over their calibrated range on a realistic window ===")
over = {t: v for t, v in worst.items() if v[0] > 1.001}
if not over:
    print("  NONE. Every encoded activation fits on every calibration and")
    print("  held-out eval window. The clipping the Test F probe triggers does")
    print("  NOT fire on chat-templated production prompts.")
for t, (r, sp, d, r0) in sorted(over.items(), key=lambda x: -x[1][0])[:20]:
    print(f"  {r:6.3f}x  (row0 {r0:6.3f}x)  [{sp}]  {t[:56]}   {d[:34]}")

print("\n=== closest to the edge (top 10, whether or not over) ===")
for t, (r, sp, d, r0) in sorted(worst.items(), key=lambda x: -x[1][0])[:10]:
    print(f"  {r:6.3f}x  [{sp:5s}]  {t[:60]}")
ev = {t: v for t, v in worst.items() if v[1] == "eval"}
print(f"\nmax ratio on HELD-OUT eval windows: "
      f"{max((v[0] for v in ev.values()), default=float('nan')):.3f}x")
