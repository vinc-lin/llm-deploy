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


def load(p: Path, label: str, expect_bytes: int = None):
    raw = p.read_bytes()
    if expect_bytes is None:
        expect_bytes = BYTES
    if len(raw) % (ELEMS * 2) == 0 and len(raw) != expect_bytes:
        # A whole number of 2560-wide fp16 rows: a different case (e.g.
        # prefill4tok is 128 rows). Accept it and say so.
        print(f"  note {label}: {len(raw)//(ELEMS*2)} rows of {ELEMS} fp16 "
              "(not the 1-row decode1tok case) -- comparing row-wise.")
    elif len(raw) != expect_bytes:
        print(f"  !! {label}: {len(raw)} bytes, expected {expect_bytes} "
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
    exp = ref.size * 2 if ref is not None else None
    ref_rms = rms(ref) if ref is not None else REF_RMS
    if (ref is not None and ref.size == ELEMS
            and abs(ref_rms - REF_RMS) / REF_RMS > 0.01):
        print(f"  note: supplied reference RMS {ref_rms:.4f} differs from the "
              f"recorded {REF_RMS:.4f} -- is this the decode1tok boundary?")

    if args.device is not None:
        dev = load(args.device, "device", exp)
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

    rows_n = (ref.size // ELEMS) if ref is not None else 1
    case = "decode1tok" if rows_n == 1 else f"{rows_n}-row prefill"
    print(f"\n=== shard-0 boundary, {case} ===")
    print(f"  reference RMS : {ref_rms:.4f}")
    print(f"  device    RMS : {dev_rms:.4f}")
    print(f"  ratio         : {ratio:.4f}x")
    if c is not None:
        print(f"  cosine        : {c:.6f}   (scale-invariant -- cannot see the ratio)")
    if dev is not None and ref is not None and dev.size == ref.size:
        # Uniformity by LEAST SQUARES, not by per-element ratio percentiles.
        # These hidden states are extremely heavy-tailed -- median |x| ~ 1,
        # max ~ 5000 -- so ~78% of elements are near zero and their ratios are
        # pure noise. Percentiles over those said "NON-uniform" for a boundary
        # that is in fact a clean uniform gain (2026-08-15 Test E), sending the
        # hunt toward per-channel dequantization for nothing.
        m = np.isfinite(dev) & np.isfinite(ref)
        g = float(dev[m] @ ref[m] / (ref[m] @ ref[m]))
        resid = dev[m] - g * ref[m]
        rel = float(np.linalg.norm(resid) / np.linalg.norm(dev[m]))
        print(f"  best-fit gain : {g:.5f}x  (least squares)")
        print(f"  residual after removing it: {rel:.4%} of the norm")
        uniform = rel < 0.02
        # Ratios only where the reference is big enough for a ratio to mean
        # anything; state the cut so the number is never read as global.
        floor = max(1e-3, 0.02 * ref_rms)
        big = m & (np.abs(ref) > floor)
        if big.sum() >= 20:
            per = dev[big] / ref[big]
            print(f"  per-element ratio over |ref| > {floor:.3g} "
                  f"({big.sum()}/{ref.size} elems): median {np.median(per):.4f}  "
                  f"p05 {np.percentile(per, 5):.4f}  p95 {np.percentile(per, 95):.4f}")
        # Per-row, when there is more than one row: a gain that differs by row
        # is structural, one that does not is a single scalar fault.
        if ref.size % ELEMS == 0 and ref.size // ELEMS > 1:
            rows = ref.size // ELEMS
            # Only rows carrying real content. A prefill case zero-pads most of
            # its rows (prefill4tok: 4 real of 128), and a gain fitted to a
            # near-zero row is meaningless -- including them would manufacture a
            # huge spread and read as "structural", which is the same mistake
            # the percentile heuristic made.
            row_rms = np.array([np.sqrt((ref[r * ELEMS:(r + 1) * ELEMS] ** 2).mean())
                                for r in range(rows)])
            # 1% of the max cleanly separates real tokens from padding here:
            # measured on prefill4tok the four real rows are 107.2 / 2.26 / 1.26
            # / 1.20 while all 124 padded rows sit at exactly 0.6185. Row 0
            # carries this model's massive-activation channels, which is why it
            # dwarfs the others and why a 5% cut kept only itself.
            live = row_rms > 0.01 * row_rms.max()
            gs, idx = [], []
            for r in range(rows):
                if not live[r]:
                    continue
                rr, dd = ref[r * ELEMS:(r + 1) * ELEMS], dev[r * ELEMS:(r + 1) * ELEMS]
                k = np.isfinite(rr) & np.isfinite(dd)
                if (rr[k] @ rr[k]) > 0:
                    gs.append(float(dd[k] @ rr[k] / (rr[k] @ rr[k]))); idx.append(r)
            if gs:
                gs = np.array(gs)
                shown = "  ".join(f"r{r}={g:.4f}" for r, g in
                                  list(zip(idx, gs))[:8])
                print(f"  per-row gain  : {shown}{'  ...' if len(gs) > 8 else ''}")
                print(f"                  over {len(gs)} content row(s) of {rows}; "
                      f"spread {gs.max()-gs.min():.5f}")
                # A global least-squares fit is dominated by whichever row
                # carries the massive activations (row 0 here, RMS 107 vs ~1),
                # so a row-0-only fault can still show a small global residual.
                # The per-row spread has to be able to overrule it.
                if gs.max() - gs.min() >= 0.02:
                    uniform = False
                    print("  -> VARIES by row -- NOT a single gain, whatever the "
                          "global residual says (that fit is dominated by the "
                          "largest row)")
                else:
                    print("  -> SAME on every row")

    if dev is not None and ref is not None and dev.size == ref.size:
        print(f"  UNIFORMITY: {'a single uniform gain' if uniform else 'NOT a single gain'}")

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
