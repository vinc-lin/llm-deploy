#!/usr/bin/env python
"""Parse and diff Genie's `--save` KV dumps (`kv-cache.primary.qnn-htp`).

WHY THIS EXISTS
---------------
Test N1c dumped Genie's own dialog state after prefill. The device team read the
KV payload as "opaque -- ~35x smaller than expected, maybe compressed", using
36 layers x 2 x **28 heads** x 128 x 2 B = 516,096 B/position.

That formula uses the QUERY head count. This is a GQA model: the cache is sized
by the 8 KV heads (`past_key_0_in [1,8,128,PAST]`, read back from the shipped
bin). 36 x 2 x **8** x 128 x 2 = **147,456 B/position** -- exactly what was
measured. Nothing is compressed and nothing is opaque: it is the full, raw,
uncompressed KV cache for the committed positions.

The layout falls straight out of the two observed file sizes:

    592 B header (16 B file header + 72 tensor descriptors x 8 B)
    then 72 tensors (36 layers x {K,V}), each n_pos x 2048 B
    2048 B = n_kv(8) * head_dim(128) * 2 B, one position of one tensor

    off(t, p) = 592 + t * (n_pos * 2048) + p * 2048

    P20: 592 + 72*21*2048 = 3,097,168  == observed
    P21: 592 + 72*22*2048 = 3,244,624  == observed

THE MEASUREMENT THIS UNLOCKS
----------------------------
Run the two dumps through `--diff`. P20's prompt is 20 tokens and P21's is the
same 20 plus the token '4' (id 19), so:

  * in P20, position 20 holds the KV for '4' as written by the **DECODE** graph
  * in P21, position 20 holds the KV for '4' as written by the **PREFILL** graph

Same token, same position, same model -- written by the two different paths.
Positions 0..19 are prefill-written in both and must be byte-identical.

  positions 0..19 identical, position 20 identical  -> the decode graph's KV
        write is correct, and the defect is downstream of it (mask, position,
        or the embedding fed to the NEXT step)
  positions 0..19 identical, position 20 DIFFERS    -> the decode step writes a
        different cache than prefill does for the same token: the decode-step
        feed is corrupting KV, measured directly for the first time
  positions 0..19 differ                            -> the dumps are not
        comparable (different prompts, or the layout assumption is wrong);
        --verify will have said so first

    parse_genie_kv_dump.py --dump state_p20.bin/kv-cache.primary.qnn-htp
    parse_genie_kv_dump.py --diff  state_p20.bin/kv-cache.primary.qnn-htp \\
                                   state_p21.bin/kv-cache.primary.qnn-htp
"""
import argparse
import struct
from pathlib import Path

import numpy as np

HEADER = 592          # 16 B file header + 72 * 8 B tensor descriptors
N_KV = 8
HEAD_DIM = 128
BYTES_PER_POS = N_KV * HEAD_DIM * 2      # 2048, one position of one tensor


def probe(path: Path, layers: int):
    """Recover n_pos from the file size, and check it divides exactly."""
    size = path.stat().st_size
    n_tensors = layers * 2
    body = size - HEADER
    stride = n_tensors * BYTES_PER_POS
    if body <= 0 or body % stride:
        raise SystemExit(
            f"{path.name}: {size} B does not fit {HEADER} + {n_tensors} tensors "
            f"x n_pos x {BYTES_PER_POS} B. Either --layers is wrong (got {layers}) "
            "or the layout assumption does not hold for this dump -- stop and "
            "report the size rather than trusting anything below.")
    return size, n_tensors, body // stride


def header_fields(raw: bytes):
    n_tensors, magic = struct.unpack_from("<II", raw, 0)
    a, n_kv, head_dim, n_past = struct.unpack_from("<HHHH", raw, 8)
    return {"n_tensors": n_tensors, "magic": hex(magic), "field_8": a,
            "n_kv": n_kv, "head_dim": head_dim, "n_past": n_past}


def load(path: Path, layers: int):
    size, n_tensors, n_pos = probe(path, layers)
    raw = path.read_bytes()
    a = np.frombuffer(raw, dtype=np.uint8, count=n_tensors * n_pos * BYTES_PER_POS,
                      offset=HEADER)
    return raw, a.reshape(n_tensors, n_pos, BYTES_PER_POS), n_pos


def describe(path: Path, layers: int):
    raw, arr, n_pos = load(path, layers)
    h = header_fields(raw)
    print(f"{path}")
    print(f"  size {path.stat().st_size:,} B = {HEADER} + {arr.shape[0]} tensors "
          f"x {n_pos} pos x {BYTES_PER_POS} B")
    print(f"  header: {h}")
    ok = h["n_tensors"] == arr.shape[0] and h["n_kv"] == N_KV and h["head_dim"] == HEAD_DIM
    print(f"  header agrees with the size-derived layout: {ok}")
    if h["n_past"] != n_pos:
        print(f"  NOTE header n_past={h['n_past']} vs size-derived n_pos={n_pos}")
    nz = int(np.count_nonzero(arr.reshape(arr.shape[0], n_pos, -1).any(axis=2)))
    print(f"  non-empty (tensor, position) cells: {nz} / {arr.shape[0] * n_pos}")
    return arr, n_pos


def diff(pa: Path, pb: Path, layers: int):
    a, na = describe(pa, layers)
    print()
    b, nb = describe(pb, layers)
    print()
    if a.shape[0] != b.shape[0]:
        raise SystemExit("different tensor counts -- not comparable")
    shared = min(na, nb)
    print(f"comparing the {shared} positions both dumps hold "
          f"({pa.name}: {na}, {pb.name}: {nb})\n")
    hdr = f"{'position':>9s}  {'tensors differing':>18s}  {'first differing tensor':>22s}"
    print(hdr); print("-" * len(hdr))
    first_div = None
    for p in range(shared):
        d = (a[:, p, :] != b[:, p, :]).any(axis=1)
        n = int(d.sum())
        if n and first_div is None:
            first_div = p
        which = int(np.argmax(d)) if n else -1
        lay = f"layer {which // 2} {'K' if which % 2 == 0 else 'V'}" if n else "--"
        print(f"{p:9d}  {n:18d}  {lay:>22s}")
    print()
    if first_div is None:
        print(f"IDENTICAL across all {shared} shared positions.")
    else:
        print(f"first position that differs: {first_div}")
        if first_div < shared - 1:
            print("  ⚠ this is EARLIER than the last shared position, so the two "
                  "dumps do not share a common prefill prefix -- check that the "
                  "prompts really are prefix-related before reading anything into it.")
        else:
            print("  this is the LAST shared position -- exactly the "
                  "decode-written vs prefill-written comparison (see module docstring).")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dump", type=Path)
    ap.add_argument("--diff", nargs=2, type=Path, metavar=("A", "B"))
    ap.add_argument("--layers", type=int, default=36,
                    help="total layers across ALL shards (4B: 36, 0.6B: 28)")
    args = ap.parse_args()
    if args.diff:
        diff(args.diff[0], args.diff[1], args.layers)
    elif args.dump:
        describe(args.dump, args.layers)
    else:
        ap.error("give --dump or --diff")


if __name__ == "__main__":
    main()
