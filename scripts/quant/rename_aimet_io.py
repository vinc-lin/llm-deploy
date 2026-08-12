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

--vl-text switches to the Qwen3-VL text tower's I/O: the first input is
`inputs_embeds` (embeddings-in — the name is matched literally by qualla
nsp-model.cpp:668 to select InputType::EMBEDDINGS) and n_deepstack
`deepstack_visual_embed_i` inputs sit between cos/sin and the past-KV tail, in
ExportQwen3.forward's argument order. Without the flag nothing changes.
"""
import argparse
import json
from pathlib import Path

import onnx


def canonical_names(n_layers, with_past, vl_text=False, n_deepstack=0):
    first = "inputs_embeds" if vl_text else "input_ids"
    ins = [first, "attention_mask", "position_ids_cos", "position_ids_sin"]
    ins += [f"deepstack_visual_embed_{i}" for i in range(n_deepstack if vl_text else 0)]
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
    ap.add_argument("--vl-text", action="store_true",
                    help="Qwen3-VL text tower: inputs_embeds + deepstack inputs")
    ap.add_argument("--n-deepstack", type=int, default=3)
    args = ap.parse_args()

    m = onnx.load(args.model, load_external_data=False)
    g = m.graph
    ins, outs = canonical_names(args.layers, args.with_past, args.vl_text, args.n_deepstack)
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
    if args.vl_text:
        # embeddings-in: [1,1,AR,H] is squeezed to [1,AR,H] and lands in the
        # residual stream / first RMSNorm — never a token lookup.
        assert any(op in ("Squeeze", "Reshape", "Mul", "Add") for op in c0), \
            f"input0 consumers {c0[:5]} not the embeddings path"
    else:
        assert any(op in ("Cast", "Gather") for op in c0), f"input0 consumers {c0[:5]} not Cast/Gather"
    # mask feeds Add directly, or Unsqueeze-per-layer when forward does mask.unsqueeze(1)
    n_mask_ops = c1.count("Add") + c1.count("Unsqueeze")
    assert n_mask_ops >= args.layers // 2, f"input1 consumers {c1[:5]} — not the mask?"
    assert "Concat" in c2 and "Concat" in c3, f"inputs 2/3 consumers {c2[:3]}/{c3[:3]} lack Concat — not cos/sin?"
    if args.vl_text:
        # each deepstack tensor is added to the residual stream after its layer
        for k in range(args.n_deepstack):
            ck = consumers.get(g.input[4 + k].name, [])
            assert any(op in ("Add", "Squeeze", "Reshape") for op in ck), \
                f"deepstack input {k} consumers {ck[:5]} — not a residual add?"

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
