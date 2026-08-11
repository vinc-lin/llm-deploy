#!/usr/bin/env python
"""Parity: prefill.onnx under Genie's ACTUAL runtime read pattern.

Mimics qualla's basic dialog exactly (nsp-model.cpp):
  - tokens LEFT-aligned at window slots 0..n-1, pad after (setupInput, non-ENCODER)
  - attention mask rows n..S-1 fully masked (buildMask)
  - RoPE positions arange(n) for valid slots, 0 for pad slots
  - logits sampled at ROW n-1 (getLogits: "assumes right-padded input")

Checks, for several prompt lengths:
  1. argmax(logits[0, n-1]) == HF greedy next token   (the device-critical read)
  2. rows 0..n-1 match HF per-position logits (max|dlogits| reported)

A last-token-only prefill graph FAILS check 1 structurally (only 1 row exists);
this script is the regression guard for the 2026-08-11 device-garbage root cause.

Usage:
  parity_qualla_read.py --onnx .../work/onnx/parity-fix --cl-prefill 128
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import onnxruntime as ort
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "export"))
from modeling_export import MASK_VALUE, causal_mask, rope_tables, rope_theta_of  # noqa: E402

DATA = Path("/home/vinc/llm-local")

PROMPTS = [
    "What is 2+2? Answer with one number.",
    "The capital of France is",
    "def fibonacci(n):",
    "解释一下什么是注意力机制。",
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=str(DATA / "models/Qwen3-0.6B"))
    ap.add_argument("--onnx", required=True, help="dir containing prefill.onnx")
    ap.add_argument("--cl-prefill", type=int, default=128)
    args = ap.parse_args()

    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(args.model)
    hf = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=torch.float32).eval()
    theta = rope_theta_of(hf.config)
    hd = hf.config.head_dim
    S = args.cl_prefill

    sess = ort.InferenceSession(str(Path(args.onnx) / "prefill.onnx"),
                                providers=["CPUExecutionProvider"])
    out_shape = sess.get_outputs()[0].shape
    print(f"prefill logits shape: {out_shape}")
    assert out_shape[1] == S, (
        f"logits covers {out_shape[1]} position(s), need all {S} — "
        "graph still has the last-token-only head")

    failures = 0
    for p in PROMPTS:
        ids = tok(p, return_tensors="pt").input_ids[:, :S]
        n = ids.shape[1]

        # qualla-style left-aligned feed
        padded = torch.zeros(1, S, dtype=torch.int32)
        padded[0, :n] = ids[0]
        mask = causal_mask(S, S)          # rank-3 [1, S, S]
        mask[:, n:, :] = MASK_VALUE       # pad rows fully masked
        pos = torch.cat([torch.arange(n), torch.zeros(S - n, dtype=torch.long)])
        cos, sin = rope_tables(pos, hd, theta)

        outs = sess.run(None, {
            "input_ids": padded.numpy(),
            "attention_mask": mask.numpy(),
            "position_ids_cos": cos.numpy(),
            "position_ids_sin": sin.numpy(),
        })
        onnx_logits = torch.from_numpy(outs[0][0])   # [S, V]

        with torch.no_grad():
            ref = hf(ids.to(torch.long)).logits[0]   # [n, V]

        # 1. the device-critical read: row n-1
        am_dev, am_ref = onnx_logits[n - 1].argmax().item(), ref[n - 1].argmax().item()
        # 2. all valid rows
        diff = (onnx_logits[:n] - ref).abs().max().item()
        ok = am_dev == am_ref
        failures += 0 if ok else 1
        print(f"[{'OK ' if ok else 'FAIL'}] n={n:3d} row{n-1} argmax dev={am_dev} ref={am_ref} "
              f"({tok.decode([am_dev])!r} vs {tok.decode([am_ref])!r})  "
              f"max|dlogits| rows0..{n-1} = {diff:.2e}  {p[:30]!r}")

    if failures:
        sys.exit(f"{failures}/{len(PROMPTS)} qualla-read checks FAILED")
    print(f"all {len(PROMPTS)} qualla-read checks passed")


if __name__ == "__main__":
    main()
