#!/usr/bin/env python
"""Gate 1 (Stage 2): Qwen3-VL text wrapper vs HF, torch level.

Covers the two new modes together, because they are only ever used together:
embeddings-in (no in-graph embed lookup) and additive deepstack after layers
0/1/2. Deepstack is fed non-zero on purpose -- an all-zero probe would pass
even if the wrapper ignored the inputs entirely.

Run:
  $PY_DEPLOY scripts/validate/parity_vl_text_wrapper.py \
      --model $LLMDEPLOY_DATA/models/Qwen3-VL-4B-Instruct
"""
import argparse
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "export"))
from modeling_export import ExportQwen3, causal_mask, rope_tables, rope_theta_of  # noqa: E402

TOL = 5e-3
S = 16


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    args = ap.parse_args()

    from transformers import Qwen3VLForConditionalGeneration
    hf = Qwen3VLForConditionalGeneration.from_pretrained(
        args.model, dtype=torch.float32, attn_implementation="eager"
    ).eval()
    lm = hf.model.language_model
    cfg = hf.config.text_config
    n_deep = 3

    torch.manual_seed(0)
    embeds = torch.randn(1, S, cfg.hidden_size) * 0.02
    deep = [torch.randn(1, S, cfg.hidden_size) * 0.02 for _ in range(n_deep)]

    with torch.no_grad():
        ref = lm(
            inputs_embeds=embeds,
            attention_mask=torch.ones(1, S, dtype=torch.long),
            deepstack_visual_embeds=[d[0] for d in deep],
            visual_pos_masks=torch.ones(1, S, dtype=torch.bool),
        ).last_hidden_state
        ref_logits = hf.lm_head(ref)

    m = ExportQwen3.from_hf_vl_text(hf, use_past=False, n_deepstack=n_deep)
    mask = causal_mask(S, S)
    cos, sin = rope_tables(torch.arange(S), cfg.head_dim, rope_theta_of(cfg))
    with torch.no_grad():
        ours = m(embeds, mask, cos, sin, *deep)[0]

    assert torch.isfinite(ours).all(), "wrapper produced non-finite logits"
    assert torch.isfinite(ref_logits).all(), "HF reference produced non-finite logits"
    assert ours.shape == ref_logits.shape, f"{ours.shape} != {ref_logits.shape}"
    d = (ours - ref_logits).abs().max().item()
    print(f"  logits max|d| = {d:.3e}  shape={tuple(ours.shape)}")
    assert d < TOL, f"wrapper diverges from HF: {d:.3e} >= {TOL}"

    with torch.no_grad():
        zeroed = m(embeds, mask, cos, sin, *[torch.zeros_like(x) for x in deep])[0]
    dz = (zeroed - ours).abs().max().item()
    print(f"  zero-deepstack delta = {dz:.3e} (must be >> 0)")
    assert dz > 1e-3, "deepstack inputs are ignored by the wrapper"

    print(f"PASS: VL text wrapper matches HF (max|d|={d:.3e})")


if __name__ == "__main__":
    main()
