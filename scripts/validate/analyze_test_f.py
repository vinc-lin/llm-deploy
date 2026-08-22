#!/usr/bin/env python
"""Test F: is the shard-0 row-0 gain caused by the attention-sink CONDITION or
by the ROW INDEX?

Run on the host after pulling text_probe_out/ back from the device:

  $PY_DEPLOY scripts/validate/analyze_test_f.py \
      --kit $LLMDEPLOY_DATA/work/text_probe_f --results ./text_probe_out_f

WHAT THIS DECIDES
-----------------
Test E measured a uniform 1.3896x gain on shard 0's boundary output, and the
per-row follow-up localised it to row 0 alone (rows 1-3 at 0.93/1.00/0.99). The
same 1.3896x appears in decode1tok, a different graph. The two share exactly one
property: the row ATTENDS ONLY TO ITSELF, which is the condition under which
this row carries massive activations (RMS 107.2 vs ~1-2; c4=5244 is 93.4% of its
norm on its own).

Two causes survive, and they need different fixes:

  CONDITION  self-attention / massive activations saturate or clamp somewhere,
             so any such row is amplified wherever it sits in the AR window
  INDEX      something specific to element 0 of the AR window -- a tile edge or
             an offset bug -- and the sink is a coincidence

  f1_row0ctx  keeps the row index, removes the sink  (row 0 attends to all four)
  f2_shift4   keeps the sink, moves the index        (real tokens at rows 4-7)

Those two, read against the two anchors, separate the causes outright.

THE LAYER SCAN
--------------
Shard 0 emits 36 per-layer KV tensors on every run and nobody has ever compared
them, so this part is free. It does NOT localise a scale: both taps sit behind
RMSNorm (v_proj behind input_layernorm, k_proj behind input_layernorm AND
k_norm), and RMSNorm is scale-invariant, so a uniform residual gain is erased
before it reaches either. What the taps do read is the residual DIRECTION and
the health of each RMSNorm denominator -- and a saturating denominator is the
leading mechanism here, since c4^2 = 2.75e7 overflows fp16's 65504 by 420x and
input_layernorm is the first place this row's squares are summed.

  taps clean at all 18 layers -> every denominator is intact; the block maths is
                                 right and only the final magnitude is wrong
  taps dirty from layer k     -> the fault is inside the block maths from layer
                                 k, and the fp16-overflow story is live
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

GAIN_TOL = 0.05          # |g-1| <= this is "clean" (W8A16 vs fp32 sits inside)
BAND = (1.25, 3.0)       # the band that reproduces the device's wrong argmax
RESID_UNIFORM = 0.02     # residual after removing the gain: below = one scalar


def find_out(d: Path, stem: str):
    """qnn-net-run writes <output_dir>/Result_N/<tensor>.raw, but the nesting has
    moved between SDK releases -- search rather than assume."""
    if not d.is_dir():
        return None
    hits = sorted(p for p in d.rglob("*")
                  if p.is_file() and p.stem == stem)
    if not hits:
        hits = sorted(p for p in d.rglob("*")
                      if p.is_file() and p.name.startswith(stem))
    return hits[0] if hits else None


def gain(ref, dev):
    """Least-squares uniform gain and the residual left after removing it.

    Least squares, never per-element ratio percentiles: these hidden states are
    extremely heavy-tailed (median |x| ~ 1, max ~ 5244), so most elements are
    near zero and their ratios are noise. A percentile heuristic reported
    "NON-uniform" for a clean uniform gain once already and sent the device team
    hunting per-channel dequantization for nothing.
    """
    r = np.asarray(ref, np.float64).ravel()
    d = np.asarray(dev, np.float64).ravel()
    m = np.isfinite(r) & np.isfinite(d)
    if not m.any() or (r[m] @ r[m]) == 0:
        return float("nan"), float("nan"), float("nan")
    g = float(d[m] @ r[m] / (r[m] @ r[m]))
    resid = float(np.linalg.norm(d[m] - g * r[m]) /
                  max(np.linalg.norm(d[m]), 1e-30))
    nd, nr = np.linalg.norm(d[m]), np.linalg.norm(r[m])
    c = float(d[m] @ r[m] / (nd * nr)) if nd and nr else float("nan")
    return g, resid, c


def load_rows(p: Path, width: int, rows_wanted, label):
    """Device tensor -> the wanted AR rows as fp32."""
    raw = p.read_bytes()
    for dtype, esz in (("<f2", 2), ("<f4", 4)):
        if len(raw) % (width * esz):
            continue
        n = len(raw) // (width * esz)
        if n <= max(rows_wanted):
            continue
        a = np.frombuffer(raw, dtype=dtype).astype(np.float32).reshape(n, width)
        if esz == 4:
            print(f"    note {label}: file is float32 -- "
                  "--use_native_output_files did not take effect.")
        return a[list(rows_wanted)]
    print(f"    !! {label}: {len(raw)} bytes is not a whole number of "
          f"{width}-wide rows -- truncated pull or wrong tensor")
    return None


def mark(g):
    if not np.isfinite(g):
        return "????"
    if abs(g - 1.0) <= GAIN_TOL:
        return "clean"
    if BAND[0] <= g <= BAND[1]:
        return "AMPL*"      # in the band that reproduces argmax 105196
    return "AMPL "


def boundary_table(kit: Path, results: Path, meta):
    case = meta["case"]
    ref = np.load(kit / case / "ref" / "last_hidden_states.npy")
    dev_p = find_out(results / f"{case}_s0", "last_hidden_states")
    if dev_p is None:
        print(f"  {case:14s} MISSING shard-0 output "
              f"({results / (case + '_s0')})")
        return None
    dev = load_rows(dev_p, ref.shape[1], meta["real_rows"], f"{case} s0")
    if dev is None:
        return None
    out = {}
    cells = []
    for i, r in enumerate(meta["real_rows"]):
        g, resid, c = gain(ref[i], dev[i])
        out[r] = {"gain": g, "resid": resid, "cos": c,
                  "ref_rms": meta["ref_row_rms"][i]}
        cells.append(f"r{r}={g:.4f}[{mark(g)}]")
    print(f"  {case:14s} " + "  ".join(cells))
    worst = max(out.values(), key=lambda v: abs(v["gain"] - 1.0))
    print(f"  {'':14s} worst-row residual after removing the gain "
          f"{worst['resid']:.3%}  cos {worst['cos']:.6f}  "
          f"({'one uniform scale' if worst['resid'] < RESID_UNIFORM else 'NOT a single scale'})")
    return out


def layer_scan(kit: Path, results: Path, meta):
    scan = meta.get("layerscan") or {}
    if not scan:
        return None
    sdir = kit / meta["case"] / "ref" / "layerscan"
    n_kv, hd, ar = meta["n_kv"], meta["head_dim"], meta["ar"]
    rows = meta["real_rows"]
    # The taps hold only the NEW AR-wide slice, so the row index is the AR row
    # itself. key is [1,n_kv,hd,AR], value [1,n_kv,AR,hd] -- on prefill both are
    # [1,8,128,128], so the axes are taken by NAME and never inferred.
    nelem = n_kv * hd * ar
    found, res = 0, {}
    for name in sorted(scan, key=lambda n: (int(n.split("_")[2]),
                                            n.split("_")[1])):
        ref_p = sdir / f"{name}.npy"
        dev_p = find_out(results / f"{meta['case']}_s0", name)
        if dev_p is None or not ref_p.is_file():
            continue
        ref = np.load(ref_p)                       # [n, n_kv, hd]
        raw = dev_p.read_bytes()
        esz = len(raw) // nelem if nelem and len(raw) % nelem == 0 else 0
        if esz not in (2, 4):
            print(f"    !! {name}: {len(raw)} bytes, expected "
                  f"{nelem * 2} (fp16) -- skipped")
            continue
        a = np.frombuffer(raw, dtype="<f2" if esz == 2 else "<f4")
        if name.startswith("past_key_"):
            a = a.reshape(n_kv, hd, ar)
            dev = np.stack([a[:, :, r] for r in rows])              # [n,nkv,hd]
        else:
            a = a.reshape(n_kv, ar, hd)
            dev = np.stack([a[:, r, :] for r in rows])              # [n,nkv,hd]
        found += 1
        layer = int(name.split("_")[2])
        kind = "k" if name.startswith("past_key_") else "v"
        for i, r in enumerate(rows):
            g, resid, c = gain(ref[i], dev[i].astype(np.float32))
            res.setdefault(layer, {})[(kind, r)] = (g, c)
    if not found:
        return None
    return res


def print_scan(meta, res):
    rows = meta["real_rows"]
    sink = rows[0]
    print(f"\n  layer scan ({meta['case']}, sink row {sink}) -- "
          "gain is expected ~1.000 on BOTH taps even if the boundary is "
          "amplified, because both sit behind RMSNorm")
    print(f"    {'layer':>5} {'key gain':>10} {'key cos':>9} "
          f"{'val gain':>10} {'val cos':>9}")
    first_bad = None
    for layer in sorted(res):
        k = res[layer].get(("k", sink))
        v = res[layer].get(("v", sink))
        ks = f"{k[0]:10.4f} {k[1]:9.6f}" if k else f"{'-':>10} {'-':>9}"
        vs = f"{v[0]:10.4f} {v[1]:9.6f}" if v else f"{'-':>10} {'-':>9}"
        bad = any(x and (abs(x[0] - 1) > GAIN_TOL or
                         (np.isfinite(x[1]) and x[1] < 0.99)) for x in (k, v))
        if bad and first_bad is None:
            first_bad = layer
        print(f"    {layer:>5} {ks} {vs}  {'<-- DIVERGES' if bad else ''}")
    return first_bad


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--kit", required=True, type=Path)
    ap.add_argument("--results", required=True, type=Path)
    ap.add_argument("--no-layer-scan", action="store_true")
    args = ap.parse_args()

    cases = json.loads((args.kit / "cases.json").read_text())
    by = {m["case"]: m for m in cases}

    print("=== Test F: shard-0 boundary gain, per row ===")
    print("    clean = within +-5% of 1.0;  AMPL* = inside the "
          f"[{BAND[0]}, {BAND[1]}] band that reproduces argmax 105196\n")
    gains, scans = {}, {}
    for m in cases:
        g = boundary_table(args.kit, args.results, m)
        if g:
            gains[m["case"]] = g
        if not args.no_layer_scan:
            r = layer_scan(args.kit, args.results, m)
            if r:
                scans[m["case"]] = r

    def row_gain(case, row):
        return gains.get(case, {}).get(row, {}).get("gain")

    print("\n=== verdict ===")

    anchor = row_gain("f0_ctrl_dec", 0)
    if anchor is None:
        print("  INCONCLUSIVE: the decode anchor f0_ctrl_dec is missing. "
              "Everything below compares against it; rerun the probe.")
        return 2
    if abs(anchor - 1.0) <= GAIN_TOL:
        print(f"  THE ANCHOR IS CLEAN ({anchor:.4f}x). Test E measured 1.3896x "
              "on this exact case.")
        print("  The device is NOT in the state Test E measured -- different "
              "bins, a rebuilt kit, or the defect is intermittent.")
        print("  Do not read the cases below as a result. Establish the anchor "
              "first.")
        return 2
    print(f"  anchor f0_ctrl_dec row 0: {anchor:.4f}x -- the defect is present "
          "and reproduces.")

    ctrl = row_gain("fp_ctrl_pre", 0)
    ctx = row_gain("f1_row0ctx", 0)
    shift_row = by.get("f2_shift4", {}).get("real_rows", [4])[0]
    shift = row_gain("f2_shift4", shift_row)
    if ctrl is not None:
        print(f"  anchor fp_ctrl_pre row 0: {ctrl:.4f}x")

    if ctx is None or shift is None:
        print("  INCONCLUSIVE: need both f1_row0ctx and f2_shift4 to separate "
              "the causes.")
        return 2

    ctx_amp = abs(ctx - 1.0) > GAIN_TOL
    shift_amp = abs(shift - 1.0) > GAIN_TOL
    sink_removed = None
    fc, f1 = by.get("fp_ctrl_pre"), by.get("f1_row0ctx")
    if fc and f1:
        sink_removed = f1["ref_row_rms"][0] < 0.5 * fc["ref_row_rms"][0]
        print(f"  f1_row0ctx host row-0 RMS {f1['ref_row_rms'][0]:.2f} vs "
              f"{fc['ref_row_rms'][0]:.2f} in the control -- the sink was "
              f"{'REMOVED' if sink_removed else 'NOT removed'}")

    print(f"\n  row 0 with context   (f1_row0ctx)  : {ctx:.4f}x  "
          f"[{mark(ctx)}]")
    print(f"  sink moved to row {shift_row} (f2_shift4)   : {shift:.4f}x  "
          f"[{mark(shift)}]")

    rc = 0
    if shift_amp and not ctx_amp:
        print("\n  -> THE CONDITION, NOT THE INDEX.")
        print("     The gain followed the sink to row "
              f"{shift_row} and vanished when row 0 was given real context.")
        print("     The fault is triggered by a row that attends only to "
              "itself -- i.e. by this row's massive activations -- and not by")
        print("     its position in the AR window.")
        print("     Next: this is a magnitude/saturation fault. Chase the fp16 "
              "range on the sink row, not tile offsets.")
    elif ctx_amp and not shift_amp:
        print("\n  -> THE ROW INDEX, NOT THE CONDITION.")
        print("     Row 0 stayed amplified with real context, and a genuine "
              f"sink at row {shift_row} was clean.")
        print("     The sink is a coincidence. The fault is specific to "
              "element 0 of the AR window -- a tile edge or an offset bug in")
        print("     how the first row of the boundary is written.")
        print("     Next: chase the output write / tiling, not numerics.")
    elif ctx_amp and shift_amp:
        print("\n  -> BOTH are amplified. The two-cause model is incomplete.")
        if sink_removed is False:
            print("     Note f1_row0ctx did NOT actually remove the sink on "
                  "this checkpoint, so it never tested the condition. Read it")
            print("     as 'row 0 is still amplified', nothing more.")
        else:
            print("     Row 0 is amplified even with context AND a sink at "
                  f"row {shift_row} is amplified too, so neither property alone")
            print("     explains it. Send the full table; the model needs "
                  "rebuilding from this data.")
        rc = 1
    else:
        print("\n  -> NEITHER is amplified, yet the anchor is.")
        print("     The gain needs the exact control configuration and neither "
              "variation reproduces it. That is itself a strong constraint;")
        print("     send the full table.")
        rc = 1

    # ---- layer scan ------------------------------------------------------
    if scans:
        for case, res in scans.items():
            fb = print_scan(by[case], res)
            if fb is None:
                print(f"    -> all taps clean. Every RMSNorm denominator in "
                      f"shard 0 is intact for {case}, so the block maths is")
                print("       right and only the final magnitude is wrong: the "
                      "fault is on the residual/output path, NOT an overflow")
                print("       inside the layers.")
            else:
                print(f"    -> taps DIVERGE from layer {fb}. The fault is "
                      f"inside the block maths from layer {fb} onward, which "
                      "is")
                print("       consistent with a saturating sum-of-squares on "
                      "this row's massive activations. That layer is the "
                      "target.")
    else:
        print("\n  layer scan: no per-layer KV outputs found in the results.")
        print("    They are written by the same run -- if the pull excluded "
              "them, re-pull; it needs no new device execution.")
    return rc


if __name__ == "__main__":
    sys.exit(main())
