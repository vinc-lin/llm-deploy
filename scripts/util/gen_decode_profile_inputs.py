#!/usr/bin/env python3
"""Generate qnn-net-run input files for profiling the Qwen3-0.6B AR-1 decode graph.

Writes one raw binary per graph input plus the input_list.txt that
`qnn-net-run --input_list` expects, sized for the shipped decode graph
(AR=1, CL=1152, past=1151, 28 layers, 8 KV heads, head_dim 128).

Timing is the point, not numerics: KV is zero-filled by default so the package
compresses to ~1 MB instead of ~132 MB. Pass --random if you suspect
data-dependent timing on HVX/HMX and want to rule it out (same shapes, same
byte count, incompressible).

Usage:
    python3 gen_decode_profile_inputs.py --out ./decode_profile_inputs
    python3 gen_decode_profile_inputs.py --out ./d --random --ar 32 --past 1120
"""
import argparse
import os

import numpy as np

LAYERS = 28
KV_HEADS = 8
HEAD_DIM = 128
ROPE_DIM = 64          # pos-id-dim: the unduplicated half of head_dim
ROPE_THETA = 1.0e6     # Qwen3 (NOT Qwen2's 1e5)
MASK_ALLOW = 0.0       # additive fp16 mask: allow
MASK_DENY = -1000.0    # ... and deny (qualla's value, not -inf)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out", required=True, help="output directory")
    p.add_argument("--ar", type=int, default=1, help="input window (1=decode, 32=verify)")
    p.add_argument("--past", type=int, default=1151, help="past-KV dim (CL - AR)")
    p.add_argument("--n-past", type=int, default=None,
                   help="valid cached positions; default = past (worst-case full context)")
    p.add_argument("--token-id", type=int, default=9707, help="token id to feed")
    p.add_argument("--random", action="store_true",
                   help="fill KV with random fp16 instead of zeros")
    p.add_argument("--seed", type=int, default=0)
    a = p.parse_args()

    cl = a.past + a.ar
    n_past = a.past if a.n_past is None else a.n_past
    if n_past > a.past:
        raise SystemExit(f"--n-past {n_past} exceeds --past {a.past}")

    ins = os.path.join(a.out, "inputs")
    os.makedirs(ins, exist_ok=True)
    rng = np.random.default_rng(a.seed)
    names = []

    def dump(name, arr):
        arr.tofile(os.path.join(ins, name + ".raw"))
        names.append(name)

    # input_ids [1, AR] int32
    dump("input_ids", np.full((1, a.ar), a.token_id, dtype=np.int32))

    # attention_mask [1, AR, CL] fp16, additive. Columns [0, past) are the cache
    # region, [past, CL) the new tokens; row i additionally allows new cols <= i.
    mask = np.full((1, a.ar, cl), MASK_DENY, dtype=np.float16)
    mask[:, :, :n_past] = MASK_ALLOW
    for i in range(a.ar):
        mask[0, i, a.past:a.past + i + 1] = MASK_ALLOW
    dump("attention_mask", mask)

    # RoPE tables [1, AR, 64] fp16 at positions n_past .. n_past+AR-1
    inv_freq = 1.0 / (ROPE_THETA ** (np.arange(0, ROPE_DIM, dtype=np.float64) * 2.0 / HEAD_DIM))
    pos = np.arange(n_past, n_past + a.ar, dtype=np.float64)[:, None]
    ang = pos * inv_freq[None, :]
    dump("position_ids_cos", np.cos(ang)[None].astype(np.float16))
    dump("position_ids_sin", np.sin(ang)[None].astype(np.float16))

    # past KV: keys are TRANSPOSED (sequence last), values are not.
    kshape = (1, KV_HEADS, HEAD_DIM, a.past)
    vshape = (1, KV_HEADS, a.past, HEAD_DIM)
    for i in range(LAYERS):
        for tag, shape in (("key", kshape), ("value", vshape)):
            if a.random:
                arr = rng.normal(0.0, 0.5, size=shape).astype(np.float16)
            else:
                arr = np.zeros(shape, dtype=np.float16)
            dump(f"past_{tag}_{i}_in", arr)

    # qnn-net-run input list: one line per inference, "name:=path" space separated
    with open(os.path.join(a.out, "input_list.txt"), "w") as f:
        f.write(" ".join(f"{n}:=inputs/{n}.raw" for n in names) + "\n")

    total = sum(os.path.getsize(os.path.join(ins, n + ".raw")) for n in names)
    print(f"{len(names)} tensors, {total/2**20:.1f} MiB -> {a.out}")
    print(f"AR={a.ar} CL={cl} past={a.past} n_past={n_past} "
          f"KV={'random' if a.random else 'zeros'}")


if __name__ == "__main__":
    main()
