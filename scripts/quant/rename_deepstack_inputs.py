#!/usr/bin/env python
"""Give the prefill graph its OWN deepstack input names.

Why this exists
---------------
Genie allocates one rpcmem buffer per distinct input tensor NAME across all
graph variants, and sizes the zero-fill from the LAST variant registered for
that name (`nsp-model.cpp:1481`). With prefill (AR=128) and decode (AR=1)
both declaring `deepstack_visual_embed_{0,1,2}`, the shared buffer is memset
for whichever variant registered last -- 1 row, 5120 bytes -- while prefill
reads 128 rows out of it. The remaining 127 rows are uninitialised rpcmem.

That was harmless only for as long as the prefill graph never loaded. The
past-KV rebuild makes prefill selectable, and any prompt short enough to be
served by it then reads the uninitialised tail.

Renaming prefill's three inputs to `deepstack_visual_embed_{k}_p` gives them
their own allocations, each memset at its own full size. It is safe because:

  * these inputs are FLOAT (never quantized), so no encodings entry is
    orphaned by the rename -- asserted here, not assumed; and
  * an input name the runtime does not recognise is explicitly zero-filled by
    `initializeUnconnectedInputs`, which is exactly the behaviour we want for
    the text-only path (deepstack-by-zeros is the defined degradation).

Run AFTER rename_aimet_io.py, on its `model_renamed.onnx`:

  $PY_DEPLOY scripts/quant/rename_deepstack_inputs.py \
      --model  $Q/model_renamed.onnx \
      --encodings $QP/model_filtered_renamed.encodings
"""
import argparse
import json
import os
import sys
from pathlib import Path

import onnx

FLOAT_TYPES = {onnx.TensorProto.FLOAT, onnx.TensorProto.FLOAT16,
               onnx.TensorProto.DOUBLE, onnx.TensorProto.BFLOAT16}


def _dangling_check(g):
    """Every consumed name must be produced somewhere. This is the failure a
    partial rename actually causes, and it needs no tensor payloads."""
    produced = ({i.name for i in g.input} | {t.name for t in g.initializer} |
                {o for n in g.node for o in n.output})
    for n in g.node:
        for name in n.input:
            if name and name not in produced:
                sys.exit(f"FAIL: node {n.name or n.op_type} consumes unproduced "
                         f"tensor {name!r} -- rename left a dangling reference")
    for o in g.output:
        if o.name not in produced:
            sys.exit(f"FAIL: graph output {o.name!r} is not produced")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="model_renamed.onnx to edit")
    ap.add_argument("--encodings", default=None,
                    help="encodings to prove the old names carry none")
    ap.add_argument("--n-deepstack", type=int, default=3)
    ap.add_argument("--suffix", default="_p")
    ap.add_argument("--out", default=None,
                    help="output .onnx (default: <stem>_dsp.onnx beside input)")
    args = ap.parse_args()

    src = Path(args.model)
    old = [f"deepstack_visual_embed_{k}" for k in range(args.n_deepstack)]
    new = [n + args.suffix for n in old]

    # -- 1. the rename must not orphan an encoding ---------------------------
    # A quantized deepstack input would carry an activation encoding keyed by
    # name; renaming would silently detach it and the converter would fall back
    # to a default range. Refuse rather than produce that.
    if args.encodings:
        enc = json.loads(Path(args.encodings).read_text())
        pools = []
        if isinstance(enc, dict):
            for key in ("activation_encodings", "param_encodings"):
                sub = enc.get(key)
                if isinstance(sub, dict):
                    pools.append(sub.keys())
                elif isinstance(sub, list):
                    pools.append([e.get("name") for e in sub])
            if not pools:
                pools.append(enc.keys())
        else:
            pools.append([e.get("name") for e in enc])
        hits = sorted({n for pool in pools for n in pool if n in set(old)})
        if hits:
            sys.exit(f"FAIL: {hits} carry encodings -- these inputs are "
                     "quantized, so renaming would orphan them. Stop.")
        print(f"  OK   no encodings reference {old} (float inputs, as expected)")

    m = onnx.load(str(src), load_external_data=False)
    g = m.graph
    existing = ({i.name for i in g.input} | {o.name for o in g.output} |
                {t.name for t in g.initializer} | {v.name for v in g.value_info})
    clash = sorted(set(new) & existing)
    if clash:
        sys.exit(f"FAIL: target name(s) already present in the graph: {clash}")

    by_name = {i.name: i for i in g.input}
    missing = [n for n in old if n not in by_name]
    if missing:
        sys.exit(f"FAIL: not graph inputs: {missing}")

    for n in old:
        dt = by_name[n].type.tensor_type.elem_type
        if dt not in FLOAT_TYPES:
            sys.exit(f"FAIL: {n} has dtype {onnx.TensorProto.DataType.Name(dt)}, "
                     "not float -- it is quantized; renaming is unsafe.")

    # -- 2. rename inputs and every consumer reference -----------------------
    consumers = {n: 0 for n in old}
    for i, (o, n) in enumerate(zip(old, new)):
        by_name[o].name = n
    for node in g.node:
        for j, name in enumerate(node.input):
            if name in consumers:
                consumers[name] += 1
                node.input[j] = name + args.suffix
    for vi in g.value_info:
        if vi.name in consumers:
            vi.name += args.suffix

    dead = [n for n, c in consumers.items() if c == 0]
    if dead:
        sys.exit(f"FAIL: {dead} had no consumer -- the graph does not use "
                 "them, so this is not the tower we think it is.")

    renamed = [i.name for i in g.input if i.name in set(new)]
    if len(renamed) != len(new):
        sys.exit(f"FAIL: renamed {len(renamed)} of {len(new)} inputs")

    out = Path(args.out) if args.out else src.with_name(src.stem + "_dsp.onnx")
    onnx.save(m, str(out))

    # onnx.checker resolves external data relative to the PROCESS CWD, not to
    # the model file, so check from the model's own directory. A 15 GB tower
    # whose .data sibling is absent (scratch runs) can only be checked
    # structurally -- say so rather than print a clean bill.
    cwd = Path.cwd()
    try:
        os.chdir(out.parent)
        onnx.checker.check_model(out.name)
        print("  OK   onnx.checker clean (full, external data resolved)")
    except onnx.checker.ValidationError as exc:
        if "should be stored in" not in str(exc):
            raise
        # No .data sibling (scratch run). The checker cannot run at all, so do
        # the part that actually matters for a rename: prove no reference was
        # left dangling.
        _dangling_check(g)
        print(f"  WARN external data not beside {out.name}: onnx.checker "
              "skipped; dangling-reference check passed instead")
    finally:
        os.chdir(cwd)

    print(f"  OK   renamed {len(new)} deepstack inputs "
          f"({', '.join(f'{c} consumer(s)' for c in consumers.values())})")
    for o, n in zip(old, new):
        print(f"       {o} -> {n}")
    print(f"  OK   wrote {out}")
    names = [i.name for i in g.input]
    print(f"       graph inputs now: {names[:4]} ... ({len(names)} total)")


if __name__ == "__main__":
    main()
