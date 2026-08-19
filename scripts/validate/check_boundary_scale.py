#!/usr/bin/env python
"""Decide whether the shard-0 -> shard-1 boundary is mis-scaled.

WHY THIS EXISTS
---------------
The 2026-08-15/19 sessions produced an apparently impossible pair: shard 0's
boundary output scored cosine 1.0000 against the host reference, yet feeding
that same file to shard 1 gave argmax 105196 where the host reference gave 374.

Both can be true, because **cosine is scale-invariant**. Measured on the host
against prefill_1.onnx / decode_1.onnx, a UNIFORM boundary scale anywhere in
~[1.25, 3.0] reproduces the device result exactly -- including the row pattern
(row 0 wrong, rows 1-3 right) -- while cosine stays at 1.000000. Correct scale
(0.5-1.1) gives the right answer on every row.

So the boundary must be checked for MAGNITUDE, not direction. This script does
that, on one 5 KB file, with no device and no model.

  check_boundary_scale.py --device <shard0_out.raw> [--ref <reference.raw>]
  check_boundary_scale.py --device-rms 214.4          # if only a number survived

The reference ships in the bundle as
    03_vl4b_v5/decode1tok/decode_1/last_hidden_states.raw
which is exactly the host boundary the isolated run was fed.
"""
import argparse
import sys
from pathlib import Path

import numpy as np

# Host reference, decode1tok, computed fp32 from the source model and written
# fp16 by build_text_probe_kit.py. Hard-coded so the check still works when only
# the device file survives.
REF_RMS = 107.2226
ELEMS = 2560            # decode1tok boundary is [1, 1, 2560]
BYTES = ELEMS * 2       # fp16

TOL = 0.05              # +-5% on the ratio; W8A16 vs fp32 sits far inside this
DEVICE_BAND = (1.25, 3.0)   # the band that reproduces argmax 105196 on the host


def load(p: Path, label: str):
    raw = p.read_bytes()
    if len(raw) != BYTES:
        print(f"  !! {label}: {len(raw)} bytes, expected {BYTES} "
              f"({ELEMS} x fp16). Wrong tensor, wrong case, or truncated pull.")
        if len(raw) % 2:
            sys.exit(2)
    a = np.frombuffer(raw, dtype="<f2").astype(np.float64)
    n_bad = int((~np.isfinite(a)).sum())
    if n_bad:
        print(f"  !! {label}: {n_bad}/{a.size} values are inf/nan. fp16 saturates "
              "at 65504 -- an overflow at the boundary is its own defect, "
              "separate from a scale factor.")
    return a


def rms(a):
    a = a[np.isfinite(a)]
    return float(np.sqrt((a ** 2).mean())) if a.size else float("nan")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", type=Path, help="shard 0's last_hidden_states.raw")
    ap.add_argument("--ref", type=Path, help="host reference (bundle copy)")
    ap.add_argument("--device-rms", type=float, help="if only the number survived")
    args = ap.parse_args()

    if args.device_rms is None and args.device is None:
        ap.error("give --device <file> or --device-rms <number>")

    ref = load(args.ref, "reference") if args.ref else None
    ref_rms = rms(ref) if ref is not None else REF_RMS
    if ref is not None and abs(ref_rms - REF_RMS) / REF_RMS > 0.01:
        print(f"  note: supplied reference RMS {ref_rms:.4f} differs from the "
              f"recorded {REF_RMS:.4f} -- is this the decode1tok boundary?")

    if args.device is not None:
        dev = load(args.device, "device")
        dev_rms = rms(dev)
        if ref is not None and dev.size == ref.size:
            m = np.isfinite(dev) & np.isfinite(ref)
            c = float(dev[m] @ ref[m] /
                      (np.linalg.norm(dev[m]) * np.linalg.norm(ref[m])))
        else:
            c = None
    else:
        dev_rms, c, dev, ref = args.device_rms, None, None, None

    ratio = dev_rms / ref_rms

    print("\n=== shard-0 boundary, decode1tok ===")
    print(f"  reference RMS : {ref_rms:.4f}")
    print(f"  device    RMS : {dev_rms:.4f}")
    print(f"  ratio         : {ratio:.4f}x")
    if c is not None:
        print(f"  cosine        : {c:.6f}   (scale-invariant -- cannot see the ratio)")
    if dev is not None and ref is not None and dev.size == ref.size:
        m = np.isfinite(dev) & np.isfinite(ref) & (np.abs(ref) > 1e-6)
        if m.any():
            per = dev[m] / ref[m]
            print(f"  per-element ratio: median {np.median(per):.4f}  "
                  f"p05 {np.percentile(per, 5):.4f}  p95 {np.percentile(per, 95):.4f}")
            spread = np.percentile(per, 95) - np.percentile(per, 5)
            print(f"  -> {'UNIFORM' if spread < 0.1 * abs(np.median(per)) else 'NON-uniform'}"
                  " across elements")

    print("\n=== verdict ===")
    # Saturation first: with inf/nan present the RMS ratio is computed over the
    # survivors and understates the damage, so it must not be the headline.
    n_bad = int((~np.isfinite(dev)).sum()) if dev is not None else 0
    if n_bad:
        print(f"  BOUNDARY HAS {n_bad} NON-FINITE VALUES -- fp16 overflow at the")
        print("  seam, which is a different defect from a scale factor and takes")
        print("  precedence. The ratio below is computed over the finite values")
        print("  only and understates it. Send the file; do not act on the ratio.")
        return 2
    if not np.isfinite(ratio):
        print("  INCONCLUSIVE: non-finite values dominate. Report the raw file.")
        return 2
    if abs(ratio - 1.0) <= TOL:
        print(f"  Boundary magnitude is CORRECT ({ratio:.4f}x, within +-{TOL:.0%}).")
        print("  The scale hypothesis is WRONG for this build. Do not rebuild on it.")
        if c is not None and c < 0.99:
            print(f"  But cosine is {c:.4f}: the direction is wrong. That is a "
                  "different defect -- report the file.")
        else:
            print("  Shard 0 is then genuinely fine and the fault is downstream:")
            print("    re-check shard1-chained vs shard1-isolated with the CURRENT")
            print("    comparator, which now also prints mag_ratio per row.")
        return 0
    band = DEVICE_BAND[0] <= ratio <= DEVICE_BAND[1]
    print(f"  BOUNDARY IS MIS-SCALED by {ratio:.4f}x.")
    if band:
        print(f"  That is inside [{DEVICE_BAND[0]}, {DEVICE_BAND[1]}], the band that "
              "reproduces the device's wrong argmax (105196) on the host.")
        print("  This is the defect. It is invisible to cosine, which is why every")
        print("  previous check passed it.")
    else:
        print(f"  Outside the [{DEVICE_BAND[0]}, {DEVICE_BAND[1]}] band that reproduces "
              "105196, so the magnitude is wrong but the story is not yet complete.")
    print("\n  Next: this is a ctx-bin property, not a Genie one -- it reproduces")
    print("  under qnn-net-run with no Genie. Send this file plus the ratio; the")
    print("  build side bisects ONNX -> DLC -> ctx-bin from here.")
    return 1


if __name__ == "__main__":
    sys.exit(main())
