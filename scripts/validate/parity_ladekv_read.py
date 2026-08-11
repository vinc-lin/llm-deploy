#!/usr/bin/env python
"""Parity: past-KV prefill graph under qualla's ACTUAL runtime feed pattern.

Mimics qualla's strategy for an AR=128 / CL=1152 / past=1024 past-KV prefill
(contract pinned from SDK source, see docs/NOTES-genie-io.md):
  - tokens LEFT-aligned at slots 0..n-1, pad token after (nsp-model.cpp:1994-2016)
  - concat KV layout: mask columns [0,past) = past region, [past,CL) = new
    region; chunk row i allows past cols [0,n_valid_kv) plus new cols
    past..past+i, everything else masked (attention-mask.cpp:72-79)
  - RoPE positions iota(n_past+i) for valid rows, 0 for pad rows
  - logits sampled at ROW n_process-1 (nsp-model.cpp:3294-3295)
  - prompts > AR are chunked AR at a time with growing n_past; new-slice KV
    outputs are appended into the past buffers (kvmanager.cpp:430-465)

qualla feeds fp16(-1000) as the masked value; our encodings calibrate the mask
input at MASK_VALUE=-100 so the on-device quantizer clips -1000 -> -100, which
still zeroes attention. The harness feeds MASK_VALUE, matching what the graph
effectively sees.

Checks (FP32 ONNX vs HF):
  1. short prompts (single chunk): argmax(logits[n-1]) == HF greedy + row diffs
  2. chunked prompts (129 and ~200 tokens, two chunks): chunk-1 KV outs feed
     chunk-2 past inputs; argmax of chunk-2's row (n-AR-1) == HF greedy

Usage:
  parity_ladekv_read.py --onnx .../model_renamed.onnx [--ar 128] [--ctx 1024]
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import onnxruntime as ort
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "export"))
from modeling_export import MASK_VALUE, rope_tables, rope_theta_of  # noqa: E402

DATA = Path("/home/vinc/llm-local")
PAD_TOKEN = 151645  # qualla pad-token defaults to first eos-token (context.cpp:51)

SHORT_PROMPTS = [
    "What is 2+2? Answer with one number.",
    "The capital of France is",
    "def fibonacci(n):",
    "解释一下什么是注意力机制。",
]
LONG_TEXT = (
    "The Hexagon tensor processor executes large language models by streaming "
    "quantized weights from DDR through vector tightly coupled memory while "
    "HVX threads perform the matrix multiplications for each transformer "
    "layer in sequence. "
) * 20


def run_chunk(sess, layers, n_kv, hd, theta, AR, PAST, TOTAL,
              ids_chunk, n_past, past_k, past_v, n_valid_kv):
    """Feed one qualla-style chunk; return (logits [AR,V], new past_k/v)."""
    n = ids_chunk.shape[0]
    padded = torch.full((1, AR), PAD_TOKEN, dtype=torch.int32)
    padded[0, :n] = ids_chunk

    mask = torch.full((1, AR, TOTAL), MASK_VALUE, dtype=torch.float32)
    for i in range(n):
        mask[0, i, :n_valid_kv] = 0.0            # valid past span
        mask[0, i, PAST:PAST + i + 1] = 0.0      # causal new span
    pos = torch.cat([torch.arange(n_past, n_past + n),
                     torch.zeros(AR - n, dtype=torch.long)])
    cos, sin = rope_tables(pos, hd, theta)

    feeds = {
        "input_ids": padded.numpy(),
        "attention_mask": mask.numpy(),
        "position_ids_cos": cos.numpy(),
        "position_ids_sin": sin.numpy(),
    }
    for i in range(layers):
        feeds[f"past_key_{i}_in"] = past_k[i]
        feeds[f"past_value_{i}_in"] = past_v[i]

    out_names = ["logits"] + [f"past_{t}_{i}_out"
                              for i in range(layers) for t in ("key", "value")]
    outs = dict(zip(out_names, sess.run(out_names, feeds)))

    # append new-slice KV (valid rows 0..n-1) into the past buffers
    for i in range(layers):
        past_k[i][:, :, :, n_valid_kv:n_valid_kv + n] = \
            outs[f"past_key_{i}_out"][:, :, :, :n]
        past_v[i][:, :, n_valid_kv:n_valid_kv + n, :] = \
            outs[f"past_value_{i}_out"][:, :, :n, :]
    return torch.from_numpy(outs["logits"][0])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=str(DATA / "models/Qwen3-0.6B"))
    ap.add_argument("--onnx", required=True, help="path to renamed prefill ONNX")
    ap.add_argument("--ar", type=int, default=128)
    ap.add_argument("--ctx", type=int, default=1024)
    ap.add_argument("--layers", type=int, default=28)
    args = ap.parse_args()

    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(args.model)
    hf = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=torch.float32).eval()
    theta, hd = rope_theta_of(hf.config), hf.config.head_dim
    n_kv = hf.config.num_key_value_heads
    AR = args.ar
    PAST = args.ctx                # concat: past region = CL - AR
    TOTAL = args.ctx + AR          # CL

    sess = ort.InferenceSession(args.onnx, providers=["CPUExecutionProvider"])
    shp = {o.name: o.shape for o in sess.get_outputs()}
    assert shp["logits"][1] == AR, f"logits {shp['logits']} not all-position"
    assert any(i.name == "past_key_0_in" for i in sess.get_inputs()), \
        "graph has no past-KV inputs — wrong ONNX"

    def fresh_past():
        k = [np.zeros((1, n_kv, hd, PAST), dtype=np.float32) for _ in range(args.layers)]
        v = [np.zeros((1, n_kv, PAST, hd), dtype=np.float32) for _ in range(args.layers)]
        return k, v

    failures = 0

    print("== single-chunk prompts ==")
    for p in SHORT_PROMPTS:
        ids = tok(p, return_tensors="pt").input_ids[0, :AR]
        n = ids.shape[0]
        pk, pv = fresh_past()
        logits = run_chunk(sess, args.layers, n_kv, hd, theta, AR, PAST, TOTAL,
                           ids, 0, pk, pv, 0)
        with torch.no_grad():
            ref = hf(ids[None].to(torch.long)).logits[0]
        am_dev, am_ref = logits[n - 1].argmax().item(), ref[n - 1].argmax().item()
        diff = (logits[:n] - ref).abs().max().item()
        ok = am_dev == am_ref
        failures += 0 if ok else 1
        print(f"[{'OK ' if ok else 'FAIL'}] n={n:3d} row{n-1} argmax dev={am_dev} "
              f"ref={am_ref} ({tok.decode([am_dev])!r} vs {tok.decode([am_ref])!r})  "
              f"max|dlogits| = {diff:.2e}  {p[:30]!r}")

    print("== chunked prompts (two chunks through growing past-KV) ==")
    long_ids = tok(LONG_TEXT, return_tensors="pt").input_ids[0]
    for n_total in (AR + 1, min(AR + 72, 2 * AR)):
        ids = long_ids[:n_total]
        pk, pv = fresh_past()
        run_chunk(sess, args.layers, n_kv, hd, theta, AR, PAST, TOTAL,
                  ids[:AR], 0, pk, pv, 0)
        n2 = n_total - AR
        logits2 = run_chunk(sess, args.layers, n_kv, hd, theta, AR, PAST, TOTAL,
                            ids[AR:], AR, pk, pv, AR)
        with torch.no_grad():
            ref = hf(ids[None].to(torch.long)).logits[0]
        am_dev, am_ref = logits2[n2 - 1].argmax().item(), ref[-1].argmax().item()
        diff = (logits2[:n2] - ref[AR:]).abs().max().item()
        ok = am_dev == am_ref
        failures += 0 if ok else 1
        print(f"[{'OK ' if ok else 'FAIL'}] n={n_total} chunk2 row{n2-1} argmax "
              f"dev={am_dev} ref={am_ref} ({tok.decode([am_dev])!r} vs "
              f"{tok.decode([am_ref])!r})  max|dlogits| chunk2 = {diff:.2e}")

    if failures:
        sys.exit(f"{failures} qualla-read checks FAILED")
    print("all qualla-read checks passed (single-chunk + chunked)")


if __name__ == "__main__":
    main()
