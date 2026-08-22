#!/usr/bin/env python
"""Does the shard-0 boundary hold on a REALISTIC, chat-templated prompt?

  $PY_DEPLOY scripts/validate/analyze_realistic_probe.py \
      --kit <host_refs> --results ./text_probe_out_r

WHY THIS TEST EXISTS
--------------------
Every probe before it fed bare token ids -- `decode1tok` is token 3838 alone at
position 0, `prefill4tok` is four content words. A production prompt is
chat-templated and begins <|im_start|>, and that difference turned out to be the
whole result: the Test F probe's row 0 lands 1.64x outside its calibrated
activation range and produces the 1.39x boundary gain the device measured, while
realistic windows put row 0 at gain 0.9990 and the model's REAL attention sink
(row 1, RMS 220.3 -- larger than the probe's synthetic 107.2) comes through at
1.0000. So Tests B/C/E/F reproduced a defect the probe itself manufactured.

This kit feeds windows built by vl_calib_build.py: chat-templated turns with real
ViT features spliced onto the image positions, deepstack zero-filled exactly as
the shipped tower runs it.

WHAT THE OUTCOMES MEAN
----------------------
The host predicts a CLEAN boundary (worst |gain-1| ~0.035 across realistic
windows). So:

  device clean  -> the boundary line of enquiry CLOSES. Shard 0 is faithful on
                   production input, and the text garbage is downstream: Genie's
                   feed, the dialog/config path, or shard 1.
  device dirty  -> the fault is in the ctx-bin or the converter. That is the one
                   stage the host clamp simulation CANNOT see -- it models the
                   encodings, not the conversion -- and no test so far has
                   separated "our numbers" from "the toolchain's numbers" on a
                   realistic input.

Both outcomes are decisive, which is the point.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

GAIN_TOL = 0.05      # W8A16 vs fp32 sits far inside this
COS_PASS = 0.99


def find_out(d: Path, stem: str):
    if not d.is_dir():
        return None
    hits = sorted(p for p in d.rglob("*") if p.is_file() and p.stem == stem)
    if not hits:
        hits = sorted(p for p in d.rglob("*")
                      if p.is_file() and p.name.startswith(stem))
    return hits[0] if hits else None


def stats(ref, dev):
    a = np.asarray(ref, np.float64).ravel()
    b = np.asarray(dev, np.float64).ravel()
    m = np.isfinite(a) & np.isfinite(b)
    if not m.any() or (a[m] @ a[m]) == 0:
        return float("nan"), float("nan"), float("nan")
    g = float(b[m] @ a[m] / (a[m] @ a[m]))
    resid = float(np.linalg.norm(b[m] - g * a[m]) /
                  max(np.linalg.norm(b[m]), 1e-30))
    c = float(a[m] @ b[m] /
              (np.linalg.norm(a[m]) * np.linalg.norm(b[m])))
    return g, resid, c


def load_rows(p: Path, width, rows, label):
    raw = p.read_bytes()
    for dt, esz in (("<f2", 2), ("<f4", 4)):
        if len(raw) % (width * esz):
            continue
        n = len(raw) // (width * esz)
        if n <= max(rows):
            continue
        a = np.frombuffer(raw, dt).astype(np.float32).reshape(n, width)
        if esz == 4:
            print(f"    note {label}: fp32 file -- "
                  "--use_native_output_files did not take effect")
        return a[list(rows)]
    print(f"    !! {label}: {len(raw)} bytes is not a whole number of "
          f"{width}-wide rows")
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--kit", required=True, type=Path)
    ap.add_argument("--results", required=True, type=Path)
    args = ap.parse_args()

    cases = json.loads((args.kit / "cases.json").read_text())
    print("=== realistic-prompt boundary check ===")
    print(f"    clean = |gain-1| <= {GAIN_TOL:.0%};  the Test F probe measured "
          "0.390 on device\n")

    worst_all, seen, dirty = 0.0, 0, []
    for m in cases:
        case, rows = m["case"], m["real_rows"]
        # Label by what the case IS, not by whether a key happens to be set:
        # a chunk-0 case has cache_len 0, which is not None, and would print as
        # a decode step under a truthiness-free check.
        cl = m.get("cache_len")
        head = f"  {case}  [{m.get('window_split')}] {m.get('window')}  "
        if m["kind"] == "decode":
            print(head + f"DECODE step, cache_len={cl}, token "
                         f"{m.get('next_token')} -- a real cache, not the empty "
                         f"one every earlier decode probe used")
        elif m.get("mask") == "chunkseq":
            kind = "PARTIAL " if m["n_token_rows"] < m["ar"] else ""
            print(head + f"{kind}PREFILL chunk: {m['n_token_rows']} rows against "
                         f"a {cl}-position cache"
                         + ("  <-- cross-chunk" if cl else "  (anchor, empty cache)"))
        else:
            print(head + f"({m.get('n_token_rows')} real rows)")
        ref = np.load(args.kit / case / "ref" / "last_hidden_states.npy")
        p = find_out(args.results / f"{case}_s0", "last_hidden_states")
        if p is None:
            print("      MISSING shard-0 output")
            continue
        dev = load_rows(p, ref.shape[1], rows, f"{case} s0")
        if dev is None:
            continue
        seen += 1
        for i, r in enumerate(rows):
            g, resid, c = stats(ref[i], dev[i])
            rms = float(np.sqrt((ref[i].astype(np.float64) ** 2).mean()))
            bad = abs(g - 1) > GAIN_TOL or (np.isfinite(c) and c < COS_PASS)
            worst_all = max(worst_all, abs(g - 1))
            if bad:
                dirty.append((case, r, g, c))
            print(f"      row {r:>3d}  refRMS {rms:9.3f}  gain {g:7.4f}  "
                  f"resid {resid:7.3%}  cos {c:9.6f}"
                  f"{'   <== OFF' if bad else ''}")

        # logits: chained (device's own shard-0 output) vs isolated (host
        # reference boundary). The pair is what separates "shard 0 broke it"
        # from "shard 1 is wrong".
        lg = np.load(args.kit / case / "ref" / "logits.npy")
        for tag, sub in (("chained", "s1chain"), ("isolated", "s1iso")):
            q = find_out(args.results / f"{case}_{sub}", "logits")
            if q is None:
                print(f"      logits {tag:9s} MISSING")
                continue
            got = load_rows(q, lg.shape[1], rows, f"{case} {sub}")
            if got is None:
                continue
            agree = sum(int(np.argmax(lg[i])) == int(np.argmax(got[i]))
                        for i in range(len(rows)))
            wc = min(stats(lg[i], got[i])[2] for i in range(len(rows)))
            print(f"      logits {tag:9s} argmax {agree}/{len(rows)}  "
                  f"worst cos {wc:9.6f}")

    print("\n=== verdict ===")
    if not seen:
        print("  INCONCLUSIVE: no shard-0 outputs found. Re-run the probe.")
        return 2
    dec = [m["case"] for m in cases if m["kind"] == "decode" and m.get("cache_len")]
    xch = [m["case"] for m in cases
           if m.get("mask") == "chunkseq" and m.get("cache_len")]
    cov = ["prefill (empty cache)"]
    if xch:
        cov.append(f"cross-chunk prefill ({', '.join(xch)})")
    if dec:
        cov.append(f"decode-with-context ({', '.join(dec)})")
    print("  paths covered: " + "; ".join(cov))
    if not dec and not xch:
        print("  ⚠ this kit covers only single-chunk prefill against an empty "
              "cache -- neither generation's own path nor the cross-chunk path "
              "is exercised by this run")
    print(f"  worst |gain-1| over every reference row: {worst_all:.4f}")
    if not dirty:
        print("\n  BOUNDARY IS CLEAN ON REALISTIC INPUT.")
        print("  Shard 0 is faithful on production-shaped prompts, so the")
        print("  boundary line of enquiry closes: the 1.39x chased through")
        print("  Tests B/C/E/F was an artifact of the earlier probes' bare")
        print("  token ids. Look downstream -- Genie's feed, the dialog/config")
        print("  path, or shard 1 -- not at shard 0's numerics.")
        return 0
    print("\n  BOUNDARY IS OFF ON REALISTIC INPUT:")
    for case, r, g, c in dirty[:12]:
        print(f"    {case} row {r}: gain {g:.4f} cos {c:.6f}")
    print("\n  The host predicts CLEAN here (worst ~0.035 under a full clamp of")
    print("  every calibrated activation range), so this is NOT the activation")
    print("  encodings. It points at the ctx-bin or the converter -- the one")
    print("  stage the host simulation cannot model. Send the results directory.")
    return 1


if __name__ == "__main__":
    sys.exit(main())
