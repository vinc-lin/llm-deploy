#!/usr/bin/env python
"""Gate 1: static-shape ViT wrapper vs HF Qwen3VLVisionModel (torch level).

Proves the wrapper's constant-folding (position embeddings, rotary tables,
Conv3d->Linear patch embed) is mathematically identical to HF's dynamic path.

Run:
  $PY_DEPLOY scripts/validate/parity_vit_wrapper.py \
      --model $LLMDEPLOY_DATA/models/Qwen3-VL-4B-Instruct
"""
import argparse
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "export"))
from modeling_vit_export import ExportQwen3VLViT, make_pixel_values  # noqa: E402

TOL = 1e-4


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    args = ap.parse_args()

    from transformers import Qwen3VLForConditionalGeneration
    hf = Qwen3VLForConditionalGeneration.from_pretrained(
        args.model, dtype=torch.float32, attn_implementation="eager"
    ).eval()
    vision = hf.model.visual

    pixel_values, grid_thw = make_pixel_values(args.model, seed=0)
    assert tuple(grid_thw[0].tolist()) == (1, 32, 32), f"unexpected grid {grid_thw}"
    assert pixel_values.shape == (1024, 1536), f"unexpected pixels {pixel_values.shape}"

    with torch.no_grad():
        ref = vision(pixel_values, grid_thw)
        ref_main = ref.pooler_output
        ref_deep = ref.deepstack_features

        wrapper = ExportQwen3VLViT(vision, grid_thw).eval()
        got = wrapper(pixel_values)

    assert len(got) == 1 + len(ref_deep), f"expected {1+len(ref_deep)} outputs, got {len(got)}"

    worst = 0.0
    for name, a, b in [("image_features", got[0], ref_main)] + [
        (f"deepstack_visual_embed_{i}", got[1 + i], ref_deep[i]) for i in range(len(ref_deep))
    ]:
        assert a.shape == b.shape, f"{name}: shape {a.shape} != {b.shape}"
        d = (a - b).abs().max().item()
        cos = torch.nn.functional.cosine_similarity(
            a.flatten().unsqueeze(0), b.flatten().unsqueeze(0)
        ).item()
        print(f"  {name:28s} shape={tuple(a.shape)} max|d|={d:.3e} cos={cos:.8f}")
        worst = max(worst, d)

    assert worst < TOL, f"wrapper diverges from HF: max|d|={worst:.3e} >= {TOL}"
    print(f"PASS: wrapper matches HF (worst max|d|={worst:.3e})")


if __name__ == "__main__":
    main()
