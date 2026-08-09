#!/usr/bin/env python
"""Parity: exported ONNX graphs vs HF torch reference.

Test 1 (prefill): last-token logits argmax must match torch; max|dlogits| reported.
Test 2 (decode chain): prefill once, then N greedy decode steps through the
static decode graph (right-aligned sliding KV window) must reproduce the torch
greedy token sequence exactly.

Uses SMALL export dims for speed; run export first with matching dims, e.g.:
  export_qwen3.py --cl-prefill 32 --ctx 96 --out .../work/onnx/parity-small
  parity_onnx.py --onnx .../work/onnx/parity-small --cl-prefill 32 --ctx 96
"""
import argparse
import os
import sys
from pathlib import Path

import numpy as np
import onnxruntime as ort
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "export"))
from modeling_export import MASK_VALUE, causal_mask, rope_tables  # noqa: E402

DATA = Path(os.environ.get("LLMDEPLOY_DATA", "/home/vinc/llm-local"))


def run_prefill(sess, ids_padded, n_valid, cl_prefill, cfg_head_dim, rope_theta):
    """ids_padded: [1, cl_prefill] int32, right-aligned (front zeros masked)."""
    pad = cl_prefill - n_valid
    mask = causal_mask(cl_prefill, cl_prefill)  # rank-3 [1, S, S]
    mask[:, :, :pad] = MASK_VALUE  # padded front never attended
    # positions: padded slots get position 0 (masked anyway), valid tokens 0..n-1
    pos = torch.cat([torch.zeros(pad, dtype=torch.long), torch.arange(n_valid)])
    cos, sin = rope_tables(pos, cfg_head_dim, rope_theta)
    feeds = {
        "input_ids": ids_padded.numpy(),
        "attention_mask": mask.numpy(),
        "position_ids_cos": cos.numpy(),
        "position_ids_sin": sin.numpy(),
    }
    outs = sess.run(None, feeds)
    return outs  # [logits, kv...]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=str(DATA / "models/Qwen3-0.6B"))
    ap.add_argument("--onnx", required=True)
    ap.add_argument("--cl-prefill", type=int, required=True)
    ap.add_argument("--ctx", type=int, required=True)
    ap.add_argument("--steps", type=int, default=8)
    args = ap.parse_args()

    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(args.model)
    hf = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=torch.float32).eval()
    cfg = hf.config
    L, n_kv, hd = cfg.num_hidden_layers, cfg.num_key_value_heads, cfg.head_dim

    onnx_dir = Path(args.onnx)
    so = ort.SessionOptions()
    sess_p = ort.InferenceSession(str(onnx_dir / "prefill.onnx"), so, providers=["CPUExecutionProvider"])
    sess_d = ort.InferenceSession(str(onnx_dir / "decode.onnx"), so, providers=["CPUExecutionProvider"])

    prompt = "The capital of France is"
    ids = tok(prompt, return_tensors="pt").input_ids
    n = ids.shape[1]
    assert n <= args.cl_prefill

    # ---- torch reference ----
    with torch.no_grad():
        ref_logits = hf(ids).logits[0, -1]
        ref_gen = hf.generate(ids, max_new_tokens=args.steps, do_sample=False,
                              temperature=None, top_p=None, top_k=None)[0, n:].tolist()

    # ---- Test 1: prefill parity ----
    ids_padded = torch.zeros(1, args.cl_prefill, dtype=torch.int32)
    ids_padded[0, -n:] = ids[0]
    outs = run_prefill(sess_p, ids_padded, n, args.cl_prefill, hd, cfg.rope_theta)
    onnx_logits = torch.from_numpy(outs[0][0, -1])
    diff = (onnx_logits - ref_logits).abs().max().item()
    am_ref, am_onnx = ref_logits.argmax().item(), onnx_logits.argmax().item()
    print(f"[prefill] max|dlogits|={diff:.2e}  argmax ref={am_ref} onnx={am_onnx} "
          f"{'OK' if am_ref == am_onnx else 'MISMATCH'}")
    assert am_ref == am_onnx

    # ---- Test 2: decode chain with right-aligned sliding window ----
    # Genie KV contract: K transposed [1, n_kv, D, P], V [1, n_kv, P, D];
    # graph outputs only the new slice — the host (here: us; on-device: Genie)
    # maintains the sliding cache.
    P = args.ctx + args.cl_prefill - 1  # decode graph past length
    # init caches: zeros front (masked), prefill kv (length cl_prefill) at the back
    caches = []
    for i in range(L):
        k = np.zeros((1, n_kv, hd, P), dtype=np.float32)
        v = np.zeros((1, n_kv, P, hd), dtype=np.float32)
        k[:, :, :, -args.cl_prefill:] = outs[1 + 2 * i]
        v[:, :, -args.cl_prefill:, :] = outs[2 + 2 * i]
        caches.append((k, v))

    valid = n  # valid kv entries (right-aligned)
    cur = torch.tensor([[am_onnx]], dtype=torch.int32)  # first generated token
    produced = [am_onnx]
    for step in range(1, args.steps):
        total = P + 1
        mask = torch.full((1, 1, total), MASK_VALUE, dtype=torch.float32)
        mask[:, :, -(valid + 1):] = 0.0
        pos = torch.tensor([n + step - 1])
        cos, sin = rope_tables(pos, hd, cfg.rope_theta)
        feeds = {
            "input_ids": cur.numpy(),
            "attention_mask": mask.numpy(),
            "position_ids_cos": cos.numpy(),
            "position_ids_sin": sin.numpy(),
        }
        for i, (k, v) in enumerate(caches):
            feeds[f"past_key_{i}_in"] = k
            feeds[f"past_value_{i}_in"] = v
        outs_d = sess_d.run(None, feeds)
        nxt = int(np.argmax(outs_d[0][0, -1]))
        produced.append(nxt)
        cur = torch.tensor([[nxt]], dtype=torch.int32)
        # slide window: drop oldest column, append the new slice at the back
        caches = [
            (np.concatenate([caches[i][0][:, :, :, 1:], outs_d[1 + 2 * i]], axis=3),
             np.concatenate([caches[i][1][:, :, 1:, :], outs_d[2 + 2 * i]], axis=2))
            for i in range(L)
        ]
        valid += 1

    ok = produced == ref_gen[: len(produced)]
    print(f"[decode]  onnx tokens: {produced}")
    print(f"[decode]  ref  tokens: {ref_gen[:len(produced)]}")
    print(f"[decode]  {'OK — greedy sequences identical' if ok else 'MISMATCH'}")
    assert ok
    print("PARITY PASSED")


if __name__ == "__main__":
    main()
