#!/usr/bin/env python
"""Restore canonical graph I/O names on an AIMET-exported ONNX.

AIMET's quantsim.export re-exports the model with mangled I/O names
(onnx::Cast_0, t.10601, numeric outputs). Genie requires canonical names
(input_ids/attention_mask/position_ids_cos/position_ids_sin, logits,
past_key_i_out/past_value_i_out) baked into the ctx-bin.

Mapping is POSITIONAL (torch.onnx.export preserves forward-arg / return order):
  inputs : input_ids, attention_mask, position_ids_cos, position_ids_sin,
           [past_key_0_in, past_value_0_in, ...]
  outputs: logits, past_key_0_out, past_value_0_out, ...

A consumer-pattern sanity check guards the positional assumption:
  - input_ids feeds a Cast/Gather; attention_mask feeds many Add nodes
    (one per layer); cos/sin feed Concat (rope duplication).

Also rewrites matching names inside the encodings file (list or dict schema).
Writes <model>_renamed.onnx next to the input and <encodings>_renamed.encodings.
"""
import argparse
import json
from pathlib import Path

import onnx


def canonical_names(n_layers, with_past):
    ins = ["input_ids", "attention_mask", "position_ids_cos", "position_ids_sin"]
    if with_past:
        for i in range(n_layers):
            ins += [f"past_key_{i}_in", f"past_value_{i}_in"]
    outs = ["logits"]
    for i in range(n_layers):
        outs += [f"past_key_{i}_out", f"past_value_{i}_out"]
    return ins, outs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--encodings", required=True)
    ap.add_argument("--layers", type=int, default=28)
    ap.add_argument("--with-past", action="store_true")
    args = ap.parse_args()

    m = onnx.load(args.model, load_external_data=False)
    g = m.graph
    ins, outs = canonical_names(args.layers, args.with_past)
    assert len(g.input) == len(ins), f"graph has {len(g.input)} inputs, expected {len(ins)}"
    assert len(g.output) == len(outs), f"graph has {len(g.output)} outputs, expected {len(outs)}"

    # --- sanity: consumer patterns for the first four inputs ---
    consumers = {}
    for node in g.node:
        for i in node.input:
            consumers.setdefault(i, []).append(node.op_type)
    c0 = consumers.get(g.input[0].name, [])
    c1 = consumers.get(g.input[1].name, [])
    c2 = consumers.get(g.input[2].name, [])
    c3 = consumers.get(g.input[3].name, [])
    assert any(op in ("Cast", "Gather") for op in c0), f"input0 consumers {c0[:5]} not Cast/Gather"
    # mask feeds Add directly, or Unsqueeze-per-layer when forward does mask.unsqueeze(1)
    n_mask_ops = c1.count("Add") + c1.count("Unsqueeze")
    assert n_mask_ops >= args.layers // 2, f"input1 consumers {c1[:5]} — not the mask?"
    assert "Concat" in c2 and "Concat" in c3, f"inputs 2/3 consumers {c2[:3]}/{c3[:3]} lack Concat — not cos/sin?"

    rename = {}
    for vi, new in zip(list(g.input), ins):
        rename[vi.name] = new
    for vi, new in zip(list(g.output), outs):
        rename[vi.name] = new

    def fix(name):
        return rename.get(name, name)

    for vi in list(g.input) + list(g.output) + list(g.value_info):
        vi.name = fix(vi.name)
    for node in g.node:
        node.input[:] = [fix(x) for x in node.input]
        node.output[:] = [fix(x) for x in node.output]

    out_model = Path(args.model).with_name(Path(args.model).stem + "_renamed.onnx")
    onnx.save(m, str(out_model))

    enc = json.loads(Path(args.encodings).read_text())
    n_renamed = 0
    for section in ("activation_encodings", "param_encodings"):
        sec = enc.get(section)
        if sec is None:
            continue
        if isinstance(sec, dict):
            enc[section] = {fix(k): v for k, v in sec.items()}
            n_renamed += sum(1 for k in sec if k in rename)
        else:
            for e in sec:
                if e.get("name") in rename:
                    e["name"] = rename[e["name"]]
                    n_renamed += 1
    out_enc = Path(args.encodings).with_name(Path(args.encodings).stem + "_renamed.encodings")
    out_enc.write_text(json.dumps(enc, indent=2))

    print(f"renamed {len(rename)} graph I/O tensors; {n_renamed} encoding entries touched")
    print(f"wrote {out_model}\nwrote {out_enc}")


if __name__ == "__main__":
    main()
