#!/usr/bin/env python
"""Split AIMET's exported ONNX into two chunks at a layer seam.

Why not split the torch export instead: torch.onnx.export renames Linear
weights to onnx__MatMul_<n>, so a separately-exported graph matches only the
norm weights -- measured 72/198 param and 18/180 activation encodings matched,
meaning 126 weight encodings silently do not apply and those layers convert as
FP16. AIMET's own export keeps the module names and matches 198/198 and
180/180. Conversion therefore has to start from AIMET's ONNX, and the split has
to be a graph surgery on it rather than a re-export.

That also preserves the calibration: the whole-tower run scored --eval 4/4 and
its encodings stay valid, because cutting the graph moves no arithmetic. And it
sidesteps having to reconcile the boundary tensor's quant params across two
independent calibration runs (Genie requires identically-named tensors across
splits to agree -- validateModel check 4).

The seam tensor is renamed to `last_hidden_states` on both sides. Genie
special-cases that name as a recognised graph output (nsp-graph.cpp:231);
leaving it as `/layers.N/Add_1_output_0` would make the first chunk's output
set unrecognised during graph-type classification.

Usage:
  split_aimet_onnx.py --onnx <aimet model_renamed.onnx> --seam '/layers.17/Add_1_output_0' \\
      --split-at 18 --layers 36 --out-dir <dir> [--with-past]
"""
import argparse
import gc
from pathlib import Path

import onnx
from onnx.utils import Extractor

BOUNDARY = "last_hidden_states"


def rename_tensor(model, old, new):
    """Rename a tensor everywhere it appears: graph inputs, outputs, value_info
    and every node reference. Missing any one of these leaves a dangling edge
    that onnx.checker reports far from the cause."""
    for vi in list(model.graph.input) + list(model.graph.output) + list(model.graph.value_info):
        if vi.name == old:
            vi.name = new
    for n in model.graph.node:
        for i, t in enumerate(n.input):
            if t == old:
                n.input[i] = new
        for i, t in enumerate(n.output):
            if t == old:
                n.output[i] = new
    return model


def save(model, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    # >2 GiB protobuf ceiling: weights must live beside the graph.
    onnx.save(model, str(path), save_as_external_data=True,
              all_tensors_to_one_file=True,
              location=path.stem + ".data", size_threshold=1024)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--onnx", required=True)
    ap.add_argument("--seam", required=True)
    ap.add_argument("--split-at", type=int, required=True)
    ap.add_argument("--layers", type=int, required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--tag", default="prefill", help="basename for the two chunks")
    ap.add_argument("--with-past", action="store_true",
                    help="model carries past_key/value inputs (the decode graph)")
    args = ap.parse_args()

    split, L = args.split_at, args.layers
    out = Path(args.out_dir)

    print(f"loading {args.onnx} ...", flush=True)
    model = onnx.load(args.onnx)
    names_in = {i.name for i in model.graph.input}
    names_out = [o.name for o in model.graph.output]
    assert args.seam in {o for n in model.graph.node for o in n.output}, \
        f"seam {args.seam!r} is not produced by any node"

    head = [n for n in ("inputs_embeds", "attention_mask",
                        "position_ids_cos", "position_ids_sin") if n in names_in]
    deep = sorted(n for n in names_in if n.startswith("deepstack_visual_embed_"))

    def kv_in(lo, hi):
        return [f"past_{k}_{i}_in" for i in range(lo, hi) for k in ("key", "value")
                if f"past_{k}_{i}_in" in names_in]

    def kv_out(lo, hi):
        want = {f"past_{k}_{i}_out" for i in range(lo, hi) for k in ("key", "value")}
        return [o for o in names_out if o in want]

    # chunk 0: model inputs -> seam + its own KV
    c0_in = head + deep + (kv_in(0, split) if args.with_past else [])
    c0_out = [args.seam] + kv_out(0, split)
    # chunk 1: seam + the shared per-step inputs -> logits + its own KV
    c1_in = ([args.seam] + [n for n in head if n != "inputs_embeds"]
             + (kv_in(split, L) if args.with_past else []))
    c1_out = ["logits"] + kv_out(split, L)

    print(f"chunk0: {len(c0_in)} in -> {len(c0_out)} out", flush=True)
    print(f"chunk1: {len(c1_in)} in -> {len(c1_out)} out", flush=True)

    # Extractor resolves every requested tensor against graph.input +
    # graph.output + graph.value_info. AIMET's export carries no value_info for
    # intermediates, so the seam is invisible to it. Running full shape
    # inference over a 15 GB model to learn one shape is not worth it: the seam
    # is a residual-stream tensor, so it is [1, S, H] by construction, and both
    # dims are readable off the graph's own inputs.
    def dims_of(name):
        for vi in model.graph.input:
            if vi.name == name:
                return [d.dim_value for d in vi.type.tensor_type.shape.dim]
        raise SystemExit(f"cannot size the seam: input {name!r} not found")

    mask = dims_of("attention_mask")          # [1, S, total]
    src = "inputs_embeds" if "inputs_embeds" in names_in else head[0]
    hidden = dims_of(src)[-1]
    seam_shape = [1, mask[1], hidden]
    print(f"seam {args.seam} typed as {seam_shape}", flush=True)
    model.graph.value_info.append(
        onnx.helper.make_tensor_value_info(args.seam, onnx.TensorProto.FLOAT, seam_shape))

    ex = Extractor(model)
    del model
    gc.collect()

    for ci, (ins, outs) in enumerate(((c0_in, c0_out), (c1_in, c1_out))):
        print(f"extracting chunk{ci} ...", flush=True)
        sub = ex.extract_model(ins, outs)
        rename_tensor(sub, args.seam, BOUNDARY)
        p = out / f"{args.tag}_{ci}" / f"{args.tag}_{ci}.onnx"
        save(sub, p)
        n_init = len(sub.graph.initializer)
        print(f"  {p}: {len(sub.graph.node)} nodes, {n_init} initializers, "
              f"in={[i.name for i in sub.graph.input][:2]}... "
              f"out={[o.name for o in sub.graph.output][:2]}...", flush=True)
        del sub
        gc.collect()

    print("SPLIT COMPLETE")


if __name__ == "__main__":
    main()
