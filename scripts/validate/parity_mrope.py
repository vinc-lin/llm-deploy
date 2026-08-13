#!/usr/bin/env python
"""Gate 0 (Stage 3): host-side interleaved MRoPE tables vs HF, exact.

The device graphs take precomputed cos/sin [1, S, 64]; for image prompts the
host must build them from 3-D (t,h,w) positions with Qwen3-VL's INTERLEAVED
mrope layout. A blocked (Qwen2-VL-style) layout is numerically wrong for
image spans yet identical for pure text -- exactly the kind of bug that only
shows up on device. This gate pins our tables to HF's own rotary embedding on
a real templated image prompt, and mutation-tests the interleave.

Run:
  $PY_DEPLOY scripts/validate/parity_mrope.py \
      --model $LLMDEPLOY_DATA/models/Qwen3-VL-4B-Instruct
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "export"))
from modeling_export import mrope_tables, rope_tables, rope_theta_of  # noqa: E402

TOL = 1e-5  # same math, different op order; fp32 gives ~1e-7


def make_image():
    from PIL import Image, ImageDraw
    img = Image.new("RGB", (512, 512), "white")
    d = ImageDraw.Draw(img)
    d.ellipse((96, 96, 288, 288), fill=(220, 30, 30))
    return img


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    args = ap.parse_args()

    from transformers import AutoProcessor, Qwen3VLForConditionalGeneration
    proc = AutoProcessor.from_pretrained(
        args.model, min_pixels=512 * 512, max_pixels=512 * 512)
    hf = Qwen3VLForConditionalGeneration.from_pretrained(
        args.model, dtype=torch.float32, attn_implementation="eager").eval()
    cfg = hf.config.text_config

    messages = [{"role": "user", "content": [
        {"type": "image"}, {"type": "text", "text": "Describe this image."}]}]
    text = proc.apply_chat_template(messages, tokenize=False,
                                    add_generation_prompt=True)
    inputs = proc(text=[text], images=[make_image()], return_tensors="pt")
    ids, grid = inputs.input_ids, inputs.image_grid_thw
    assert grid.tolist() == [[1, 32, 32]], grid.tolist()
    n = ids.shape[1]

    # transformers 5.14 requires mm_token_type_ids (text=0/image=1/video=2),
    # returned by the processor since return_mm_token_type_ids defaults True.
    pos3, _ = hf.model.get_rope_index(
        ids, mm_token_type_ids=inputs.mm_token_type_ids,
        image_grid_thw=grid, attention_mask=torch.ones_like(ids))
    assert pos3.shape == (3, 1, n), pos3.shape
    pos3 = pos3[:, 0]                                    # [3, n]
    # the prompt must actually exercise 3-D positions or the gate is vacuous
    assert not torch.equal(pos3[0], pos3[1]), "no image span in positions?"

    x = torch.zeros(1, n, cfg.hidden_size)
    ref_cos, ref_sin = hf.model.language_model.rotary_emb(x, pos3[:, None])
    half = cfg.head_dim // 2                             # 64
    ref_cos, ref_sin = ref_cos[0, :, :half], ref_sin[0, :, :half]  # [n, 64]

    cos, sin = mrope_tables(pos3, cfg.head_dim, rope_theta_of(cfg))
    d = max(float((cos[0] - ref_cos).abs().max()),
            float((sin[0] - ref_sin).abs().max()))
    print(f"  interleaved tables vs HF rotary: max|d|={d:.3e} (n={n})")
    assert d < TOL, f"mrope_tables diverges from HF: {d:.3e}"

    # degenerate case: flat positions must reproduce rope_tables exactly
    flat = torch.arange(n)
    c1, s1 = mrope_tables(flat[None].expand(3, -1), cfg.head_dim,
                          rope_theta_of(cfg))
    c2, s2 = rope_tables(flat, cfg.head_dim, rope_theta_of(cfg))
    assert torch.equal(c1, c2) and torch.equal(s1, s2), \
        "text-only mrope != rope_tables"

    # mutation: a BLOCKED layout must disagree on this prompt
    sec = (24, 20, 20)
    ang = pos3.to(torch.float32)[:, :, None] * (
        1.0 / (rope_theta_of(cfg) ** (torch.arange(0, half).float() / half)))
    blocked = torch.cat([ang[i, :, sum(sec[:i]):sum(sec[:i + 1])]
                         for i in range(3)], dim=-1)
    db = float((blocked.cos() - ref_cos).abs().max())
    print(f"  blocked-layout control: max|d|={db:.3e} (must be >> {TOL})")
    assert db > 1e-2, "blocked layout matches HF?! gate is vacuous"

    print("PASS: mrope_tables matches HF interleaved MRoPE")


if __name__ == "__main__":
    main()
