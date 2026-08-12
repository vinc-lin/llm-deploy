#!/usr/bin/env python
"""Gate: chunked (split) Qwen3-VL text export == the unsplit export.

The 4B tower does not fit one ctx-bin (3.5 GiB per-graph serialization limit),
so the layer stack is split across two. Genie wires the chunks together
IMPLICITLY BY TENSOR NAME and explicitly does NOT check that the boundary
shapes agree ("Missing check : Shape of tensor between splits match up",
nsp-graph.cpp:229). Nothing downstream will catch a bad seam, so it is checked
here: run the chunks in sequence and require the result to match the whole
model exactly.

Also pins the invariants the ctx-bin layout depends on -- which chunk owns the
embedding path, norm and lm_head, and that deepstack cannot be attached to a
non-first chunk (it applies to global layers 0..n-1).

Run:
  $PY_DEPLOY scripts/validate/parity_vl_text_split.py \
      --model $LLMDEPLOY_DATA/models/Qwen3-VL-4B-Instruct
"""
import argparse
import gc
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "export"))
from modeling_export import ExportQwen3, causal_mask, rope_tables, rope_theta_of  # noqa: E402

S = 8
SPLIT_AT = 18


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--split-at", type=int, default=SPLIT_AT)
    args = ap.parse_args()

    from transformers import Qwen3VLForConditionalGeneration
    hf = Qwen3VLForConditionalGeneration.from_pretrained(
        args.model, dtype=torch.float32, attn_implementation="eager"
    ).eval()
    cfg = hf.config.text_config
    L, split = cfg.num_hidden_layers, args.split_at
    n_deep = 3

    torch.manual_seed(0)
    embeds = torch.randn(1, S, cfg.hidden_size) * 0.02
    deep = [torch.randn(1, S, cfg.hidden_size) * 0.02 for _ in range(n_deep)]
    mask = causal_mask(S, S)
    cos, sin = rope_tables(torch.arange(S), cfg.head_dim, rope_theta_of(cfg))

    # --- reference: the whole tower in one graph
    whole = ExportQwen3.from_hf_vl_text(hf, use_past=False, n_deepstack=n_deep)
    with torch.no_grad():
        ref = whole(embeds, mask, cos, sin, *deep)
    ref_logits, ref_kv = ref[0], ref[1:]
    assert len(ref_kv) == 2 * L, f"expected {2*L} KV outputs, got {len(ref_kv)}"
    del whole
    gc.collect()

    # --- chunked: [0, split) then [split, L)
    c0 = ExportQwen3.from_hf_vl_text(hf, use_past=False, n_deepstack=n_deep,
                                     layer_range=(0, split))
    c1 = ExportQwen3.from_hf_vl_text(hf, use_past=False, n_deepstack=0,
                                     layer_range=(split, L))

    # Deepstack targets global layers 0..n-1, so attaching it to a chunk that
    # does not start at 0 would silently add visual features at the wrong depth.
    # Must be rejected specifically, not by some incidental exception.
    try:
        ExportQwen3.from_hf_vl_text(hf, use_past=False, n_deepstack=3,
                                    layer_range=(split, L))
    except ValueError as e:
        assert "deepstack" in str(e), f"wrong ValueError: {e}"
    else:
        raise AssertionError("deepstack on a non-first chunk must be rejected")

    del hf
    gc.collect()

    # Structural invariants the two-ctx-bin layout relies on.
    assert c0.embed_tokens is None, "chunk0 must not carry the token table (embeddings-in)"
    assert c0.norm is None and c0.lm_head is None, \
        "chunk0 must not own norm/lm_head -- 389 M params of dead payload in its ctx-bin"
    assert c1.norm is not None and c1.lm_head is not None, "chunk1 must own norm + lm_head"
    assert len(c0.layers) == split and len(c1.layers) == L - split

    with torch.no_grad():
        out0 = c0(embeds, mask, cos, sin, *deep)
        hidden, kv0 = out0[0], out0[1:]
        out1 = c1(hidden, mask, cos, sin)
        got_logits, kv1 = out1[0], out1[1:]

    assert torch.isfinite(got_logits).all(), "chunked logits are non-finite"
    assert torch.isfinite(ref_logits).all(), "reference logits are non-finite"
    assert got_logits.shape == ref_logits.shape, f"{got_logits.shape} != {ref_logits.shape}"
    assert len(kv0) == 2 * split, f"chunk0 emitted {len(kv0)} KV, expected {2*split}"
    assert len(kv1) == 2 * (L - split), f"chunk1 emitted {len(kv1)} KV, expected {2*(L-split)}"

    d = (got_logits - ref_logits).abs().max().item()
    print(f"  boundary tensor  : {tuple(hidden.shape)}  (last_hidden_states)")
    print(f"  logits max|d|    : {d:.3e}  shape={tuple(got_logits.shape)}")

    # Every KV output must survive the split unchanged, in global layer order.
    chunked_kv = list(kv0) + list(kv1)
    assert len(chunked_kv) == len(ref_kv), \
        f"chunked emitted {len(chunked_kv)} KV tensors, whole model {len(ref_kv)}"
    worst_kv, worst_i = 0.0, -1
    for i, (a, b) in enumerate(zip(chunked_kv, ref_kv)):
        assert a.shape == b.shape, f"KV {i}: {a.shape} != {b.shape}"
        dv = (a - b).abs().max().item()
        if dv > worst_kv:
            worst_kv, worst_i = dv, i
    print(f"  KV max|d|        : {worst_kv:.3e}  (worst of {2*L}, index {worst_i})")

    assert d == 0.0, f"chunked logits differ from whole model: {d:.3e}"
    assert worst_kv == 0.0, f"chunked KV differs at output {worst_i}: {worst_kv:.3e}"
    print(f"PASS: split at layer {split} is exactly equivalent to the whole tower")


if __name__ == "__main__":
    main()
