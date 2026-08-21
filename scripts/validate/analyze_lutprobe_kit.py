#!/usr/bin/env python
"""Read a Test L run and say whether the ctx-bin reproduces its own ONNX.

The whole question is one comparison per case: does the argmax of the logits the
DEVICE produced equal the argmax the ONNX produced for the same inputs?

  * all cases match -> the bin is faithful to the graph it was converted from,
    and since parity_lutprobe.py independently gates that graph 3/3 against
    HuggingFace, the bin is right all the way back to HF. A Genie run that
    disagrees is then Genie's fault.
  * a case mismatches -> the bin diverges from its ONNX: a converter defect,
    and a different investigation.

    $PY_DEPLOY scripts/validate/analyze_lutprobe_kit.py --kit <kit dir> --out probe_out_l
"""
import argparse
import json
from pathlib import Path

import numpy as np


def load_logits(d: Path, width: int):
    """qnn-net-run writes Result_N/<output>.raw; find the logits one."""
    hits = sorted(p for p in d.rglob("*.raw") if "logits" in p.name.lower())
    if not hits:
        return None, f"no logits*.raw under {d}"
    raw = hits[0].read_bytes()
    for dt, esz in (("<f2", 2), ("<f4", 4)):
        if len(raw) % (width * esz) == 0:
            a = np.frombuffer(raw, dt).astype(np.float32).reshape(-1, width)
            note = "" if esz == 2 else "  (fp32 file -- --use_native_output_files did not take effect)"
            return a, note
    return None, f"{len(raw)} bytes is not a whole number of {width}-wide rows"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--kit", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    args = ap.parse_args()

    cases = (args.kit / "probe_cases.txt").read_text().split()
    print(f"kit {args.kit}\nrun {args.out}\n")
    hdr = f"{'case':16s} {'kind':8s} {'expect':>8s} {'device':>8s}  result"
    print(hdr); print("-" * len(hdr))

    npass = nfail = nmiss = 0
    prefill_ok = decode_ok = True
    for c in cases:
        meta = json.loads((args.kit / c / "ref" / "meta.json").read_text())
        ref = np.load(args.kit / c / "ref" / "logits.npy")
        want = int(meta["expect_argmax"])
        row_out = int(meta["row_out"]) if meta["kind"] == "prefill" else 0

        d = args.out / c
        if not d.exists():
            print(f"{c:16s} {meta['kind']:8s} {want:8d} {'--':>8s}  NOT RUN")
            nmiss += 1
            continue
        lg, note = load_logits(d, ref.shape[-1])
        if lg is None:
            print(f"{c:16s} {meta['kind']:8s} {want:8d} {'--':>8s}  UNREADABLE: {note}")
            nmiss += 1
            continue
        if row_out >= lg.shape[0]:
            print(f"{c:16s} {meta['kind']:8s} {want:8d} {'--':>8s}  "
                  f"only {lg.shape[0]} rows, need row {row_out}")
            nmiss += 1
            continue

        got = int(np.argmax(lg[row_out]))
        ok = got == want
        npass += ok
        nfail += not ok
        if not ok:
            (prefill_ok, decode_ok) = ((False, decode_ok) if meta["kind"] == "prefill"
                                       else (prefill_ok, False))
        a = ref.reshape(-1) if ref.ndim == 1 else ref[row_out]
        b = lg[row_out]
        cos = float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-30))
        print(f"{c:16s} {meta['kind']:8s} {want:8d} {got:8d}  "
              f"{'MATCH' if ok else 'MISMATCH'}   cos {cos:.6f}"
              f"   {meta['expect_token']!r}{note}")

    print(f"\n{npass} match, {nfail} mismatch, {nmiss} not run / unreadable\n")
    if nmiss and not nfail:
        print("INCONCLUSIVE -- some cases did not run. Fix those before reading "
              "anything into the rest.")
        return
    if nfail == 0 and npass:
        print("PASS: the ctx-bin reproduces its own ONNX on every case.\n"
              "  parity_lutprobe.py independently gates that ONNX 3/3 against HF,\n"
              "  so the bin is correct back to HuggingFace. Genie ran this SAME bin\n"
              "  and produced a different first token => the fault is in GENIE'S\n"
              "  LUT FEED, not in the graph, the LUT, the conversion or the\n"
              "  quantization.")
    elif not prefill_ok:
        print("FAIL at PREFILL: the bin does NOT reproduce its own ONNX.\n"
              "  The ONNX passes 3/3 vs HF, so this is a CONVERTER defect --\n"
              "  a different investigation from Genie's feed. Capture the\n"
              "  converter version and the exact case.")
    else:
        print("FAIL at DECODE only: prefill reproduces the ONNX and decode does\n"
              "  not. If l2a passed and l2b failed, the fault is specifically in\n"
              "  reading back a row the DECODE graph itself wrote.")


if __name__ == "__main__":
    main()
