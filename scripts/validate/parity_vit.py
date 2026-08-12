#!/usr/bin/env python
"""Gate 2: exported ONNX vs HF Qwen3VLVisionModel, via ONNX Runtime.

Catches breakage introduced by the torch->ONNX trace itself (constant folding,
op lowering) that Gate 1 cannot see.

Uses seed=1 while the exporter traced with seed=0: a graph that accidentally
baked the traced input in as a constant would pass at seed 0 and fail here.

Run:
  $PY_DEPLOY scripts/validate/parity_vit.py \
      --model $LLMDEPLOY_DATA/models/Qwen3-VL-4B-Instruct \
      --onnx  $LLMDEPLOY_DATA/work/onnx/qwen3vl-4b-vit/vit.onnx
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import onnxruntime as ort
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "export"))
from modeling_vit_export import make_pixel_values, output_names  # noqa: E402

TOL = 2e-3
COS_MIN = 0.9999


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--onnx", required=True)
    args = ap.parse_args()

    from transformers import Qwen3VLForConditionalGeneration
    hf = Qwen3VLForConditionalGeneration.from_pretrained(
        args.model, dtype=torch.float32, attn_implementation="eager"
    ).eval()
    vision = hf.model.visual

    pixel_values, grid_thw = make_pixel_values(args.model, seed=1)
    with torch.no_grad():
        ref = vision(pixel_values, grid_thw)
    refs = [ref.pooler_output.numpy()] + [d.numpy() for d in ref.deepstack_features]

    sess = ort.InferenceSession(args.onnx, providers=["CPUExecutionProvider"])
    got = sess.run(None, {"pixel_values": pixel_values.numpy()})

    names = output_names(len(ref.deepstack_features))
    assert len(got) == len(refs), f"expected {len(refs)} outputs, got {len(got)}"
    # sess.run(None, ...) returns outputs in the graph's declared order, which is
    # what pairs `got` with `refs` below. Assert that order rather than trust it:
    # a reordered graph would otherwise silently compare the wrong tensors.
    sess_names = [o.name for o in sess.get_outputs()]
    assert sess_names == names, f"graph output order {sess_names} != {names}"

    worst_d, worst_cos = 0.0, 1.0
    for name, a, b in zip(names, got, refs):
        assert a.shape == b.shape, f"{name}: shape {a.shape} != {b.shape}"
        # Must fire BEFORE the reduction below: max(0.0, nan) is 0.0 and
        # min(1.0, nan) is 1.0, so a NaN reaching it is silently dropped and the
        # gate exits 0 on a graph emitting garbage. Downstream fp16 conversion
        # makes overflow-to-NaN a live failure mode, so it has to be loud.
        assert np.isfinite(a).all(), f"{name}: ONNX output has non-finite values"
        assert np.isfinite(b).all(), f"{name}: HF reference has non-finite values"
        d = float(np.abs(a - b).max())
        # float64 for the reduction: in float32 over 655k elements this returns
        # values like 1.00000012, which is mathematically impossible for a cosine.
        af, bf = a.ravel().astype(np.float64), b.ravel().astype(np.float64)
        cos = float(np.dot(af, bf) / (np.linalg.norm(af) * np.linalg.norm(bf)))
        print(f"  {name:28s} max|d|={d:.3e} cos={cos:.8f}")
        worst_d, worst_cos = max(worst_d, d), min(worst_cos, cos)

    assert worst_d < TOL, f"ONNX diverges from HF: max|d|={worst_d:.3e} >= {TOL}"
    assert worst_cos > COS_MIN, f"ONNX cosine {worst_cos:.8f} <= {COS_MIN}"
    print(f"PASS: ONNX matches HF (max|d|={worst_d:.3e}, min cos={worst_cos:.8f})")


if __name__ == "__main__":
    main()
