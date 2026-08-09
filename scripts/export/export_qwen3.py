#!/usr/bin/env python
"""Export Qwen3 to prefill.onnx + decode.onnx for the QAIRT/Genie pipeline.

Usage:
  $PY_DEPLOY scripts/export/export_qwen3.py \
      --model $LLMDEPLOY_DATA/models/Qwen3-0.6B \
      --out   $LLMDEPLOY_DATA/work/onnx/qwen3-0.6b \
      --cl-prefill 128 --ctx 1024 [--fuse-gate-up] [--fuse-qkv]

Emits prefill.onnx (S=cl-prefill, no past, last-token logits) and
decode.onnx (S=1, past=ctx+cl_prefill-1, full logits [1,1,V]).
I/O names follow the position_ids_cos/sin convention (pos-id-dim 64);
adjust via --io-names if SDK Genie docs dictate different names.
"""
import argparse
import sys
from pathlib import Path

import onnx
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from modeling_export import ExportQwen3, causal_mask, rope_tables, rope_theta_of  # noqa: E402

OPSET = 17


def export_graph(model, cfg, out_path, seq, past_len):
    n_kv, head_dim = cfg.num_key_value_heads, cfg.head_dim
    total = past_len + seq
    input_ids = torch.zeros(1, seq, dtype=torch.int32)
    mask = causal_mask(seq, total)
    cos, sin = rope_tables(torch.arange(past_len, past_len + seq), head_dim, rope_theta_of(cfg))

    names_in = ["input_ids", "attention_mask", "position_ids_cos", "position_ids_sin"]
    args = [input_ids, mask, cos, sin]
    names_out = ["logits"]
    for i in range(cfg.num_hidden_layers):
        names_out += [f"past_key_{i}_out", f"past_value_{i}_out"]
    if past_len > 0:
        for i in range(cfg.num_hidden_layers):
            names_in += [f"past_key_{i}_in", f"past_value_{i}_in"]
            # Genie contract: keys transposed [1, n_kv, D, P], values [1, n_kv, P, D]
            args += [torch.zeros(1, n_kv, head_dim, past_len),
                     torch.zeros(1, n_kv, past_len, head_dim)]

    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.onnx.export(
        model,
        tuple(args),
        str(out_path),
        input_names=names_in,
        output_names=names_out,
        opset_version=OPSET,
        do_constant_folding=True,
        dynamo=False,
    )
    m = onnx.load(str(out_path), load_external_data=False)
    onnx.checker.check_model(str(out_path))
    print(f"  {out_path.name}: {len(m.graph.node)} nodes, opset {OPSET}, "
          f"inputs={len(names_in)}, outputs={len(names_out)}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--cl-prefill", type=int, default=128)
    ap.add_argument("--ctx", type=int, default=1024)
    ap.add_argument("--fuse-gate-up", action="store_true")
    ap.add_argument("--fuse-qkv", action="store_true")
    ap.add_argument("--parity-check", action="store_true",
                    help="compare wrapper logits vs HF forward before export")
    args = ap.parse_args()

    from transformers import AutoModelForCausalLM
    hf = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=torch.float32).eval()
    cfg = hf.config
    out = Path(args.out)

    if args.parity_check:
        m = ExportQwen3.from_hf(hf, args.fuse_gate_up, args.fuse_qkv, use_past=False)
        ids = torch.randint(0, cfg.vocab_size, (1, 16), dtype=torch.int32)
        mask = causal_mask(16, 16)
        cos, sin = rope_tables(torch.arange(16), cfg.head_dim, rope_theta_of(cfg))
        with torch.no_grad():
            ours = m(ids, mask, cos, sin)[0]
            ref = hf(ids.to(torch.long)).logits
        diff = (ours - ref).abs().max().item()
        print(f"wrapper-vs-HF max|dlogits| = {diff:.2e}")
        assert diff < 5e-3, "wrapper does not match HF forward"

    # decode past length: window is ctx + cl_prefill - 1 so that after one
    # decode step total == ctx + cl_prefill (== ctx-bin max CL, e.g. 1024+128=1152)
    past_len = args.ctx + args.cl_prefill - 1

    print("exporting prefill...")
    prefill = ExportQwen3.from_hf(hf, args.fuse_gate_up, args.fuse_qkv,
                                  use_past=False, logits_last_only=True)
    with torch.no_grad():
        export_graph(prefill, cfg, out / "prefill.onnx", args.cl_prefill, 0)

    print("exporting decode...")
    decode = ExportQwen3.from_hf(hf, args.fuse_gate_up, args.fuse_qkv,
                                 use_past=True, logits_last_only=False)
    with torch.no_grad():
        export_graph(decode, cfg, out / "decode.onnx", 1, past_len)


if __name__ == "__main__":
    main()
