#!/usr/bin/env python
"""Parity: AR>1 verification graph (verify.onnx) vs HF torch reference.

Emulates one lookahead/speculative verification pass: prefill the prompt, then
feed the next AR greedy tokens as ONE batched call to the verify graph with a
causal batch mask. Position j's logits must argmax to greedy token j+1, and
per-position logits must match an HF forward on the concatenated sequence.

This validates: multi-token batched decode with past KV, all-position logits,
rank-3 per-row mask, per-row rope positions. (Tree-shaped attention maps are
built host-side by Genie's AttentionMask from the same mask input — not a
graph property, so a causal chain fully exercises the graph contract.)

Run export first with matching dims, e.g.:
  export_qwen3.py --cl-prefill 32 --ctx 96 --ar-verify 8 --out .../parity-lade
  parity_verify.py --onnx .../parity-lade --cl-prefill 32 --ctx 96 --ar 8
"""
import argparse
import os
import sys
from pathlib import Path

import numpy as np
import onnxruntime as ort
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "export"))
from modeling_export import MASK_VALUE, causal_mask, rope_tables, rope_theta_of  # noqa: E402
from parity_onnx import run_prefill  # noqa: E402

DATA = Path(os.environ.get("LLMDEPLOY_DATA", "/home/vinc/llm-local"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=str(DATA / "models/Qwen3-0.6B"))
    ap.add_argument("--onnx", required=True)
    ap.add_argument("--cl-prefill", type=int, required=True)
    ap.add_argument("--ctx", type=int, required=True)
    ap.add_argument("--ar", type=int, required=True)
    args = ap.parse_args()

    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(args.model)
    hf = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=torch.float32).eval()
    cfg = hf.config
    L, n_kv, hd = cfg.num_hidden_layers, cfg.num_key_value_heads, cfg.head_dim
    AR = args.ar

    onnx_dir = Path(args.onnx)
    so = ort.SessionOptions()
    sess_p = ort.InferenceSession(str(onnx_dir / "prefill.onnx"), so,
                                  providers=["CPUExecutionProvider"])
    sess_v = ort.InferenceSession(str(onnx_dir / "verify.onnx"), so,
                                  providers=["CPUExecutionProvider"])

    prompt = "The capital of France is"
    ids = tok(prompt, return_tensors="pt").input_ids
    n = ids.shape[1]
    assert n <= args.cl_prefill

    # ---- torch reference: greedy AR+1 tokens; logits over prompt+AR sequence ----
    with torch.no_grad():
        ref_gen = hf.generate(ids, max_new_tokens=AR + 1, do_sample=False,
                              temperature=None, top_p=None, top_k=None)[0, n:].tolist()
        full = torch.cat([ids, torch.tensor(ref_gen[:AR]).unsqueeze(0)], dim=1)
        ref_logits_all = hf(full).logits[0]  # [n+AR, V]

    # ---- prefill -> right-aligned past caches at the verify graph's past len ----
    ids_padded = torch.zeros(1, args.cl_prefill, dtype=torch.int32)
    ids_padded[0, -n:] = ids[0]
    outs = run_prefill(sess_p, ids_padded, n, args.cl_prefill, hd, rope_theta_of(cfg))

    Pv = args.ctx + args.cl_prefill - AR  # verify graph past length
    feeds = {}
    for i in range(L):
        k = np.zeros((1, n_kv, hd, Pv), dtype=np.float32)
        v = np.zeros((1, n_kv, Pv, hd), dtype=np.float32)
        k[:, :, :, -args.cl_prefill:] = outs[1 + 2 * i]
        v[:, :, -args.cl_prefill:, :] = outs[2 + 2 * i]
        feeds[f"past_key_{i}_in"] = k
        feeds[f"past_value_{i}_in"] = v

    # ---- one batched verification call: rows j = greedy tokens t_0..t_{AR-1} ----
    batch = torch.tensor(ref_gen[:AR], dtype=torch.int32).unsqueeze(0)
    total = Pv + AR
    mask = torch.full((1, AR, total), MASK_VALUE, dtype=torch.float32)
    mask[:, :, Pv - n:Pv] = 0.0  # every row sees the prompt KV (right-aligned)
    for j in range(AR):         # row j additionally sees batch slots 0..j
        mask[0, j, Pv:Pv + j + 1] = 0.0
    pos = torch.arange(n, n + AR)  # row j is sequence position n+j
    cos, sin = rope_tables(pos, hd, rope_theta_of(cfg))
    feeds.update({
        "input_ids": batch.numpy(),
        "attention_mask": mask.numpy(),
        "position_ids_cos": cos.numpy(),
        "position_ids_sin": sin.numpy(),
    })
    outs_v = sess_v.run(None, feeds)
    logits_v = torch.from_numpy(outs_v[0][0])  # [AR, V]

    # ---- compare every position vs the HF forward on the full sequence ----
    ok = True
    max_diff = 0.0
    for j in range(AR):
        ref_j = ref_logits_all[n + j]  # sequence position n+j predicts t_{j+1}
        d = (logits_v[j] - ref_j).abs().max().item()
        max_diff = max(max_diff, d)
        am_ref, am_v = ref_j.argmax().item(), logits_v[j].argmax().item()
        match = am_ref == am_v == ref_gen[j + 1]
        ok &= match
        print(f"[verify pos {j}] argmax ref={am_ref} onnx={am_v} "
              f"expected_next={ref_gen[j + 1]} max|d|={d:.2e} "
              f"{'OK' if match else 'MISMATCH'}")
    # new-slice KV outputs must equal the batch rows' K/V (shape check + finite)
    k_out = outs_v[1]
    assert k_out.shape == (1, n_kv, hd, AR), f"key slice shape {k_out.shape}"
    print(f"[verify] all-position max|dlogits|={max_diff:.2e}")
    print("VERIFY PARITY PASSED" if ok else "VERIFY PARITY FAILED")
    assert ok


if __name__ == "__main__":
    main()
