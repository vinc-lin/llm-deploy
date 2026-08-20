#!/usr/bin/env python
"""Does the boundary defect fire on a REALISTIC chat-templated prompt? NO.

THIS IS THE SCRIPT THAT FALSIFIED THE 2026-08-20 ROOT CAUSE. Keep it wired into
any future claim about the activation path.

MEASURED, six realistic windows, every encoded activation clamped:

    window                                   row0 gain   row1 gain
    EVAL img100 'What is happening in...'       0.9990      1.0000
    EVAL text 'The capital of France is'        0.9990      1.0000
    calib img0-chunk0[0:128]                    0.9990      1.0000

    worst |gain-1| over rows 0-3:  0.0347   (device on the probe: 0.38959)

The real attention sink on a chat-templated prompt sits at ROW 1 with RMS 220.3
-- larger than the probe's synthetic row-0 sink at 107.2 -- and comes through at
gain 1.0000. The calibration covers the model's genuine massive activations; it
simply does not cover the probe's artificial one.


The Test F probe reproduces a 1.39x row-0 gain, and clamping to the calibrated
activation ranges reproduces that to 0.3% -- so the clamp simulation is a
faithful stand-in for the device and this needs no hardware.

But the probe's row 0 is a plain word token at position 0, which a production
prompt never has: those are chat-templated and begin <|im_start|>. So the open
question is whether the defect is a property of the model or an artifact of the
probe. This runs the SAME clamp simulation on real calibration/eval windows --
chat-templated prompts with real ViT features -- and reports the per-row
boundary gain.

  realistic_clamp.py <calib.npz> <encodings> [n_windows]
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

CALIB, ENC = Path(sys.argv[1]), Path(sys.argv[2])
NW = int(sys.argv[3]) if len(sys.argv) > 3 else 6
ONNX = Path(os.environ["LLMDEPLOY_DATA"]) / \
    "work/onnx/qwen3vl-4b-gqa-aimet-splitkv/prefill_0/prefill_0.onnx"

rng = {}
for e in json.loads(ENC.read_text())["activation_encodings"]:
    if e.get("dtype") == "INT":
        sc, off = e["scale"][0], e["offset"][0]
        rng[e["name"]] = (off * sc, ((1 << e["bw"]) - 1 + off) * sc)

z = np.load(CALIB)
emb, mask, cos, sin = z["embeds"], z["mask"], z["cos"], z["sin"]
deep, nval, desc, splits = z["deep"], z["n_valid"], z["desc"], z["split"]
AR = int(z["ar"])
# prefer held-out eval windows, and windows that actually start a sequence
order = sorted(range(emb.shape[0]),
               key=lambda i: (str(splits[i]) != "eval", -int(nval[i])))[:NW]


def build(clamped):
    m = onnx.load(str(ONNX), load_external_data=False)
    produced = {o for n in m.graph.node for o in n.output}
    if clamped:
        for t in [x for x in rng if x in produced]:
            lo, hi = rng[t]
            node = next(n for n in m.graph.node if t in n.output)
            node.output[list(node.output).index(t)] = t + "_pre"
            m.graph.initializer.extend([
                numpy_helper.from_array(np.array(lo, np.float32), t + "_lo"),
                numpy_helper.from_array(np.array(hi, np.float32), t + "_hi")])
            m.graph.node.append(helper.make_node(
                "Clip", [t + "_pre", t + "_lo", t + "_hi"], [t], name=t + "_c"))
        order_, emitted, pending = [], set(i.name for i in m.graph.initializer) | \
            set(i.name for i in m.graph.input), list(m.graph.node)
        while pending:
            rest, prog = [], False
            for n in pending:
                if all(i in emitted or i == "" for i in n.input):
                    order_.append(n); emitted.update(n.output); prog = True
                else:
                    rest.append(n)
            if not prog:
                raise SystemExit("topo sort failed")
            pending = rest
        del m.graph.node[:]
        m.graph.node.extend(order_)
    p = ONNX.with_name(f"prefill_0_rc_{int(clamped)}.onnx")
    onnx.save(m, str(p))
    so = ort.SessionOptions()
    so.intra_op_num_threads = int(os.environ.get("OMP_NUM_THREADS", "40"))
    try:
        s = ort.InferenceSession(str(p), so, providers=["CPUExecutionProvider"])
    finally:
        p.unlink(missing_ok=True)
    return s


def feeds_for(sess, i):
    sh = {x.name: x.shape for x in sess.get_inputs()}
    total = sh["attention_mask"][-1]
    past = total - AR
    mk = np.full((1, AR, total), -100.0, np.float32)
    mk[0, :, past:] = np.where(mask[i][0] >= 0, 0.0, -100.0)
    f = {"inputs_embeds": emb[i].astype(np.float32).reshape(sh["inputs_embeds"]),
         "attention_mask": mk,
         "position_ids_cos": cos[i].astype(np.float32).reshape(sh["position_ids_cos"]),
         "position_ids_sin": sin[i].astype(np.float32).reshape(sh["position_ids_sin"])}
    for n, s in sh.items():
        if n.startswith("past_"):
            f[n] = np.zeros([x if isinstance(x, int) else 1 for x in s], np.float32)
    for k, dn in enumerate([n for n in sh if n.startswith("deepstack_")]):
        f[dn] = deep[i][k].astype(np.float32).reshape(sh[dn])
    return f


print("running unclamped reference ...", flush=True)
s0 = build(False)
ref = {i: s0.run(["last_hidden_states"], feeds_for(s0, i))[0].reshape(-1, 2560)
       for i in order}
del s0
gc.collect()
print("running with every encoded activation clamped ...", flush=True)
s1 = build(True)
got = {i: s1.run(["last_hidden_states"], feeds_for(s1, i))[0].reshape(-1, 2560)
       for i in order}
del s1
gc.collect()

print(f"\n{'window':44s} {'row':>4s} {'ref RMS':>10s} {'gain':>8s} {'cos':>10s}")
worst = 0.0
for i in order:
    nv = int(nval[i])
    for r in list(range(min(4, nv))):
        a = ref[i][r].astype(np.float64)
        b = got[i][r].astype(np.float64)
        if not (a @ a):
            continue
        g = float(b @ a / (a @ a))
        c = float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b)))
        tag = f"[{splits[i]}] {str(desc[i])[:34]}" if r == 0 else ""
        flag = "  <== MIS-SCALED" if abs(g - 1) > 0.05 else ""
        print(f"{tag:44s} {r:>4d} {float(np.sqrt((a**2).mean())):10.3f} "
              f"{g:8.4f} {c:10.6f}{flag}")
        worst = max(worst, abs(g - 1))
print(f"\nworst |gain-1| over {len(order)} realistic windows, rows 0-3: {worst:.4f}")
print("device measured 0.38959 on the Test F probe's row 0, for comparison.")
