#!/usr/bin/env python
"""QKV-fusion quantization surgery (summary §4.1 main unlock).

Problem (from summary §3.1): with fused qkv_proj, Q/K/V share one output tensor
and one output quantizer — Q needs INT16 (attention MatMul), K/V must stay FP16
across the prefill/decode graph boundary. A single choice gives either error
5005 (cross-graph INT16 scale mismatch) or garbage (all-FP16 Q).

Local realization: ONNX opset 17 cannot express INT16 QuantizeLinear, and
qairt-converter consumes quantization via the encodings file keyed by tensor
name. The fused export already contains Split(Q,K,V) after qkv_proj, so the
"qkv → DQ → Split → Quantize(Q only)" surgery becomes an ENCODINGS transform:

  1. qkv_proj output tensor: NO encoding (stays FP16)
  2. Q-split output tensor:  INT16 encoding, harvested from a NON-FUSED donor
     calibration's q_proj output encoding (same layer)
  3. K/V-split outputs:      NO encoding (stay FP16)

Usage:
  qkv_surgery.py --fused-onnx work/quant/fuseqkv/model.onnx \
                 --fused-encodings work/quant/fuseqkv/model_filtered.encodings \
                 --donor-encodings work/quant/nonfused/model_filtered.encodings \
                 --donor-onnx work/quant/nonfused/model.onnx \
                 --out work/quant/fuseqkv/model_surgery.encodings
"""
import argparse
import json
import re
from collections import defaultdict
from pathlib import Path

import onnx

LAYER_RE = re.compile(r"layers[./](\d+)[./]self_attn")


def weight_consumers(model):
    """Map initializer-name-substring -> list of (layer_idx, node)."""
    out = defaultdict(list)
    for node in model.graph.node:
        if node.op_type not in ("MatMul", "Gemm", "Conv"):
            continue
        for inp in node.input:
            m = LAYER_RE.search(inp)
            for tag in ("qkv_proj", "q_proj"):
                if tag in inp and m:
                    out[tag].append((int(m.group(1)), node))
    return out


def find_split_outputs(model, matmul_node):
    """Follow the qkv MatMul output through pass-through ops to the Split node."""
    produced = matmul_node.output[0]
    passthrough = {"Add", "Reshape", "Cast", "Identity"}
    frontier = {produced}
    for _ in range(6):  # bounded walk
        for node in model.graph.node:
            if any(i in frontier for i in node.input):
                if node.op_type == "Split":
                    return list(node.output)
                if node.op_type in passthrough:
                    frontier = frontier | set(node.output)
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fused-onnx", required=True)
    ap.add_argument("--fused-encodings", required=True)
    ap.add_argument("--donor-onnx", required=True)
    ap.add_argument("--donor-encodings", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    fused = onnx.load(args.fused_onnx, load_external_data=False)
    donor = onnx.load(args.donor_onnx, load_external_data=False)
    enc = json.loads(Path(args.fused_encodings).read_text())
    donor_enc = json.loads(Path(args.donor_encodings).read_text())

    # Normalize both AIMET schemas to a name-keyed dict; remember list order
    def as_dict(sec):
        if isinstance(sec, dict):
            return dict(sec), "dict"
        return {e["name"]: e for e in sec}, "list"

    acts, acts_fmt = as_dict(enc.get("activation_encodings", []))
    donor_acts, _ = as_dict(donor_enc.get("activation_encodings", []))

    fused_map = weight_consumers(fused)
    donor_map = weight_consumers(donor)
    donor_q_out = {layer: node.output[0] for layer, node in donor_map.get("q_proj", [])}

    n_ok = 0
    for layer, node in sorted(fused_map.get("qkv_proj", [])):
        splits = find_split_outputs(fused, node)
        if not splits or len(splits) != 3:
            print(f"layer {layer}: Split(Q,K,V) not found after qkv MatMul — SKIP")
            continue
        q_out, k_out, v_out = splits
        # 1) strip any encoding on the shared qkv tensor + K/V branches
        for t in (node.output[0], k_out, v_out):
            acts.pop(t, None)
        # 2) graft donor q_proj INT16 encoding onto the Q branch
        d = donor_q_out.get(layer)
        if d is None or d not in donor_acts:
            print(f"layer {layer}: donor q_proj encoding missing (donor tensor {d}) — SKIP")
            continue
        grafted = dict(donor_acts[d])
        if "name" in grafted:
            grafted["name"] = q_out
        acts[q_out] = grafted
        n_ok += 1

    if acts_fmt == "list":
        enc["activation_encodings"] = list(acts.values())
    else:
        enc["activation_encodings"] = acts
    Path(args.out).write_text(json.dumps(enc, indent=2))
    total = len(fused_map.get("qkv_proj", []))
    print(f"surgery applied on {n_ok}/{total} layers -> {args.out}")
    if n_ok != total:
        raise SystemExit("INCOMPLETE SURGERY — inspect skipped layers")


if __name__ == "__main__":
    main()
