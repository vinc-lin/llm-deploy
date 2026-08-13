#!/usr/bin/env python
"""Make a decode graph carry the prefill graph's weights, bit for bit.

Why this is needed. AIMET's --export-decode adopts the prefill run's encodings,
but the scales it writes are RECOMPUTED rather than copied: measured on
Qwen3-0.6B and Qwen3-VL-4B, 100/308 and 130/396 param encodings differ, and the
difference is confined to `scale` at ~9.1e-8 relative -- float32 epsilon. That
is 2.3e-5 of one quantization step, so it is numerically irrelevant, but AIMET
bakes those scales into the exported quantize-dequantize weights. A weight
sitting on a rounding boundary then flips by one step in one graph and not the
other, and two consequences follow:

  * prefill and decode compute with slightly different weights, so the model's
    two halves disagree -- invisible without a device;
  * qnn-context-binary-generator cannot dedup the two weight sets, so the
    ctx-bin carries both. Measured at 4B: 7.7 GB instead of ~4.3 GB, with only
    the unquantized norms and lm_head still shared.

Both DLCs already convert against the PREFILL encodings file, so the shipped
quant params are prefill's either way. This makes the weights agree with them.

Only initializers present in both graphs with identical dtype and dims are
touched; anything decode owns alone is left as is.

Usage:
  unify_pair_weights.py --prefill <prefill/model_renamed.onnx> \\
      --decode <decode/model_renamed.onnx> --out <decode_unified.onnx>
"""
import argparse
import gc
import hashlib
from pathlib import Path

import onnx
from onnx import numpy_helper


def digest(t):
    return hashlib.sha256(numpy_helper.to_array(t).tobytes()).hexdigest()[:16]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prefill", required=True)
    ap.add_argument("--decode", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--sample", type=int, default=3,
                    help="tensors to hash before/after as evidence")
    args = ap.parse_args()

    print(f"loading prefill {args.prefill} ...", flush=True)
    pre = onnx.load(args.prefill)
    src = {i.name: i for i in pre.graph.initializer}
    print(f"  {len(src)} initializers", flush=True)

    print(f"loading decode {args.decode} ...", flush=True)
    dec = onnx.load(args.decode)
    print(f"  {len(dec.graph.initializer)} initializers", flush=True)

    shared = [i for i in dec.graph.initializer if i.name in src]
    sample = [t.name for t in shared[: args.sample]]
    before = {n: (digest(src[n]), digest(next(t for t in shared if t.name == n)))
              for n in sample}

    replaced, skipped = 0, []
    for i, init in enumerate(dec.graph.initializer):
        s = src.get(init.name)
        if s is None:
            continue
        if list(s.dims) != list(init.dims) or s.data_type != init.data_type:
            # Shape/dtype mismatch means these are not the same weight; copying
            # would silently corrupt the graph rather than unify it.
            skipped.append((init.name, list(init.dims), list(s.dims)))
            continue
        arr = numpy_helper.to_array(s)
        dec.graph.initializer[i].CopyFrom(numpy_helper.from_array(arr, name=init.name))
        replaced += 1

    del pre
    gc.collect()

    print(f"replaced {replaced} shared initializers; "
          f"{len(dec.graph.initializer) - replaced} left untouched", flush=True)
    if skipped:
        print(f"  SKIPPED {len(skipped)} on shape/dtype mismatch: {skipped[:3]}")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    onnx.save(dec, str(out), save_as_external_data=True,
              all_tensors_to_one_file=True, location=out.stem + ".data",
              size_threshold=1024)
    print(f"wrote {out}", flush=True)

    # Evidence, not assertion-by-assumption: re-read what was written and show
    # the sampled tensors now hash the same as prefill's.
    chk = {i.name: i for i in onnx.load(str(out)).graph.initializer}
    ok = True
    for n in sample:
        sd, dd = before[n]
        nd = digest(chk[n])
        same = (nd == sd)
        ok &= same
        print(f"  {n}\n     prefill={sd}  decode_before={dd}  decode_after={nd}  MATCH={same}")
    if not ok:
        raise SystemExit("unification did not take effect")
    print("UNIFIED")


if __name__ == "__main__":
    main()
