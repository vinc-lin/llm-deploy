#!/usr/bin/env python
"""Compare the v5 text-probe device outputs against the host fp32 references,
and say which stage is at fault.

Run on the host after pulling text_probe_out/ back from the device:

  $PY_DEPLOY scripts/validate/compare_text_probe.py \
      --kit $LLMDEPLOY_DATA/work/text_probe_v5 --results ./text_probe_out

WHAT THE VERDICT MEANS. Three device runs per case:

  shard0            decode_0 on our inputs            -> last_hidden_states
  shard1-chained    decode_1 on the DEVICE's shard-0  -> logits (what really happens)
  shard1-isolated   decode_1 on the HOST reference    -> logits (shard 1 alone)

The pair of shard-1 runs is what localises the fault, which is why both are
run. A single end-to-end number cannot distinguish "shard 0 corrupted the
boundary" from "shard 1 is broken".

THE BAR IS NOT BIT-EXACTNESS. The ctx-bin is W8A16 and the reference is fp32,
so a healthy graph lands around cos >= 0.99 with the argmax matching. A broken
one does not land near the bar -- quantization noise and a wrong graph are not
close together, which is why a loose threshold is still decisive here. What we
are separating is "approximately right" from "unrelated", not grading precision.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

COS_PASS = 0.99          # healthy W8A16 vs fp32
COS_SUSPECT = 0.90       # below this, not explainable as quantization noise


def cos(a, b):
    a, b = a.reshape(-1).astype(np.float64), b.reshape(-1).astype(np.float64)
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na == 0 or nb == 0:
        return float("nan")
    return float(a @ b / (na * nb))


def find_out(d: Path, stem: str):
    """qnn-net-run writes <output_dir>/Result_0/<tensor>.raw, but the exact
    nesting has moved between SDK releases -- search rather than assume."""
    if not d.is_dir():
        return None
    hits = sorted(p for p in d.rglob("*") if p.is_file() and p.name.startswith(stem))
    return hits[0] if hits else None


def load_rows(p: Path, n_real: int, width: int, label: str):
    """Device tensor -> the first n_real rows, as fp32.

    The device writes ALL rows (AR=128 for prefill), while the reference keeps
    only the real ones -- padding rows are meaningless and would be 78 MB of
    noise. So slice by width rather than assuming the totals match.
    """
    raw = p.read_bytes()
    for dtype, esz in (("<f2", 2), ("<f4", 4)):
        if len(raw) % (width * esz) == 0:
            rows = len(raw) // (width * esz)
            if rows < n_real:
                continue
            a = np.frombuffer(raw, dtype=dtype).astype(np.float32)
            if esz == 4:
                print(f"    note {label}: file is float32, so "
                      "--use_native_output_files did not take effect. "
                      "Comparing anyway.")
            return a.reshape(rows, width)[:n_real]
    print(f"    !! {label}: {len(raw)} bytes is not a whole number of "
          f"{width}-wide rows in fp16 or fp32 -- truncated pull or wrong tensor")
    return None


MAG_TOL = 0.05           # +-5% on RMS ratio; W8A16 vs fp32 sits well inside this


def verdict(tag, ref, got, extra=""):
    """Per-row cosine + argmax agreement. Reported per row because a fault that
    only appears at row>0 -- the cross-token/rope failure mode the prefill case
    exists to catch -- averages away into a healthy-looking single number."""
    if got is None:
        print(f"  {tag:22s} MISSING")
        return None
    cs, mags, agree = [], [], 0
    for r in range(ref.shape[0]):
        c = cos(ref[r], got[r])
        cs.append(c)
        # COSINE IS SCALE-INVARIANT and on its own is not sufficient here.
        # Measured 2026-08-19: a boundary tensor scaled by anything in ~[1.25,3]
        # reproduces the device's wrong argmax (105196) while scoring cosine
        # exactly 1.000000. Test B read that 1.0000 as "shard 0 is perfect" and
        # concluded the ctx-bins were fine. Magnitude has to be checked too.
        rr = float(np.sqrt((ref[r].astype(np.float64) ** 2).mean()))
        gr = float(np.sqrt((got[r].astype(np.float64) ** 2).mean()))
        mags.append(gr / rr if rr else float("nan"))
        if int(np.argmax(ref[r])) == int(np.argmax(got[r])):
            agree += 1
    worst = min(cs)
    worst_mag = max(mags, key=lambda m: abs(np.log(m)) if m and m == m and m > 0 else 0)
    mark = "OK  " if worst >= COS_PASS else ("WEAK" if worst >= COS_SUSPECT else "BAD ")
    per_row = " ".join(f"{c:+.4f}" for c in cs)
    if not (1 - MAG_TOL <= worst_mag <= 1 + MAG_TOL):
        mark = "BAD "
    print(f"  {tag:22s} {mark} worst_cos={worst:+.6f}  mag_ratio={worst_mag:.4f}"
          f"  argmax {agree}/{ref.shape[0]}  {extra}")
    if not (1 - MAG_TOL <= worst_mag <= 1 + MAG_TOL):
        print(f"  {'':22s}      !! MAGNITUDE off by {worst_mag:.3f}x while cosine "
              f"is {worst:.6f} -- direction right, scale wrong. Cosine alone "
              "would have called this perfect.")
    if len(cs) > 1:
        print(f"  {'':22s}      per-row cos: {per_row}")
    return worst


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--kit", required=True, type=Path, help="the built probe kit")
    ap.add_argument("--results", required=True, type=Path, help="pulled text_probe_out/")
    ap.add_argument("--ctxbin-info-0", type=Path,
                    help="shard-0 info.json of the bin that was RUN (dtype cross-check)")
    ap.add_argument("--ctxbin-info-1", type=Path,
                    help="shard-1 info.json of the bin that was RUN (dtype cross-check)")
    args = ap.parse_args()

    cases = json.loads((args.kit / "cases.json").read_text())

    # --- kit/bin encoding cross-check -------------------------------------
    # The kit writes each device input in the tensor's declared native encoding.
    # If the bin that actually ran declares something else, every number below is
    # meaningless -- and meaningless in a way that LOOKS like a broken model:
    # feeding IEEE fp16 into a UFIXED_POINT_16 input is the same byte count, so
    # nothing errors, and cosines collapse to ~0. That is exactly what happened in
    # the 2026-08-15 v5 session and it was read as "the 4B ctx-bins are
    # numerically incorrect". Refuse to render a verdict instead.
    def bin_dtypes(info):
        doc = json.loads(Path(info).read_text())
        out = {}

        def walk(o):
            if isinstance(o, dict):
                for t in o.get("graphInputs", []):
                    ti = t.get("info", t)
                    out[ti["name"]] = ti.get("dataType")
                for v in o.values():
                    walk(v)
            elif isinstance(o, list):
                for v in o:
                    walk(v)

        walk(doc)
        return out

    if args.ctxbin_info_0 or args.ctxbin_info_1:
        d0 = bin_dtypes(args.ctxbin_info_0) if args.ctxbin_info_0 else {}
        d1 = bin_dtypes(args.ctxbin_info_1) if args.ctxbin_info_1 else {}
        # A kit built before this guard existed records nothing, and "nothing to
        # compare" must not read as "verified". That is precisely the kit that
        # produced the bogus 2026-08-15 numbers.
        unrec = [c["case"] for c in cases if not c.get("input_dtypes")]
        if unrec:
            print("STOP: this probe kit predates the encoding cross-check and does")
            print(f"      not record what it wrote ({', '.join(unrec)}).")
            print("      It cannot be trusted against a rebuilt ctx-bin: writing IEEE")
            print("      fp16 into a UFIXED_POINT_16 input is the same byte count, so")
            print("      it fails silently and looks like a broken model.")
            print("      Rebuild it:  build_text_probe_kit.py --ctxbin-info-0 ... --ctxbin-info-1 ...")
            sys.exit(2)
        mism = []
        for c in cases:
            for key, wrote in (c.get("input_dtypes") or {}).items():
                shard1 = key.startswith("s1/")
                nm = key[3:] if shard1 else key
                have = (d1 if shard1 else d0).get(nm)
                if have and wrote and have != wrote:
                    mism.append((c["case"], key, wrote, have))
        if mism:
            print("STOP: the kit and the ctx-bin disagree on input encoding.\n")
            for case, key, wrote, have in mism:
                print(f"  {case}/{key}: kit wrote {wrote}, bin declares {have}")
            print("\nEvery cosine/argmax below would be an artefact of that, not a")
            print("property of the model. Rebuild the kit against THIS bin:")
            print("  build_text_probe_kit.py --ctxbin-info-0 ... --ctxbin-info-1 ...")
            sys.exit(2)
        print(f"kit/bin encoding cross-check: OK "
              f"({len(cases)} case(s), {len(d0)} shard-0 inputs)\n")
    elif any(c.get("input_dtypes") for c in cases):
        print("NOTE: no --ctxbin-info-* given, so the kit/bin encoding cross-check")
        print("      was skipped. A stale kit produces near-zero cosines that look")
        print("      exactly like a broken model. Pass them.\n")
    summary = []
    for meta in cases:
        name, n_real = meta["case"], meta["n_real"]
        print(f"\n=== case {name}  ({meta['kind']}, AR={meta['ar']}, "
              f"{n_real} real row(s)) — {meta['why']}")
        ref_h = np.load(args.kit / name / "ref" / "last_hidden_states.npy")
        ref_l = np.load(args.kit / name / "ref" / "logits.npy")

        # Cross-check that we are scoring the SAME shard-0 file the runner fed
        # to shard 1. If those diverge, "shard 0 is perfect but chaining it
        # fails" is an artefact of file selection, not a property of the model.
        fed = args.results / f"{name}_s0_fed.txt"
        p = find_out(args.results / f"{name}_s0", "last_hidden_states")
        if fed.is_file() and p is not None:
            want = Path(fed.read_text().strip()).name
            if want != p.name:
                print(f"  !! {name}: runner fed {want!r} to shard 1 but this "
                      f"scores {p.name!r} -- the chained result is NOT comparable "
                      "to the shard-0 number below. Fix the probe, not the model.")
        elif not fed.is_file():
            print(f"  note {name}: no _s0_fed.txt (pre-2026-08-19 runner); cannot "
                  "confirm the scored file is the one that was chained.")
        got_h = load_rows(p, n_real, ref_h.shape[1], "shard0") if p else None
        c_h = verdict("shard0", ref_h, got_h)

        p = find_out(args.results / f"{name}_s1chain", "logits")
        got_c = load_rows(p, n_real, ref_l.shape[1], "shard1-chained") if p else None
        c_c = verdict("shard1-chained", ref_l, got_c,
                      f"ref last-row top5={meta['last_row_top10_ids'][:5]}")

        p = find_out(args.results / f"{name}_s1iso", "logits")
        got_i = load_rows(p, n_real, ref_l.shape[1], "shard1-isolated") if p else None
        c_i = verdict("shard1-isolated", ref_l, got_i)

        summary.append({"case": name, "kind": meta["kind"], "shard0": c_h,
                        "chained": c_c, "isolated": c_i})

    print("\n" + "=" * 72)
    print("VERDICT")
    print("=" * 72)
    ok = lambda c: c is not None and c >= COS_PASS          # noqa: E731
    bad = lambda c: c is not None and c < COS_SUSPECT       # noqa: E731

    for s in summary:
        c = s["case"]
        if ok(s["shard0"]) and ok(s["chained"]) and ok(s["isolated"]):
            print(f"  {c}: ctx-bin computes CORRECTLY.")
            print("      => the converter and ctx-bin are exonerated for this "
                  "case. The garbage on device comes from how GENIE FEEDS the "
                  "tower, not from the graph. Next: work probe B's ranked list "
                  "(feed_variants.json); emb_fp32_as_fp16 leads it.")
        elif bad(s["shard0"]):
            print(f"  {c}: SHARD 0 is wrong at the very first graph.")
            print("      => the fault is in the ctx-bin/converter for shard 0. "
                  "A rebuild is justified. Nothing downstream can be trusted, "
                  "so ignore the shard-1 numbers for this case.")
        elif ok(s["shard0"]) and bad(s["isolated"]):
            print(f"  {c}: shard 0 OK, SHARD 1 wrong on a known-good input.")
            print("      => the fault is isolated to shard 1's ctx-bin "
                  "(it owns lm_head). Rebuild shard 1 and re-probe.")
        elif ok(s["shard0"]) and ok(s["isolated"]) and bad(s["chained"]):
            print(f"  {c}: both shards fine alone, CHAIN broken.")
            print("      => the boundary hand-off is corrupting "
                  "last_hidden_states between the two ctx-bins, even though "
                  "both are FLOAT_16. Suspect the concat/layout at the seam.")
        else:
            print(f"  {c}: mixed / borderline — do not guess. "
                  f"shard0={s['shard0']} chained={s['chained']} "
                  f"isolated={s['isolated']}")
            print("      => report the raw numbers; a WEAK band is not a "
                  "verdict and a second opinion is cheaper than a wrong rebuild.")

    dec = next((s for s in summary if s["kind"] == "decode"), None)
    pre = next((s for s in summary if s["kind"] == "prefill"), None)
    if dec and pre:
        if ok(dec["chained"]) and bad(pre["chained"]):
            print("\n  CROSS-CASE: decode is clean but prefill is not.")
            print("      => the fault needs cross-token attention or rope to "
                  "show up -- single-token numerics are fine. Suspect the "
                  "prefill graphs or the rope/mask handling INSIDE them. Note "
                  "the real prompt is three AR=128 prefill calls, so this "
                  "alone would explain the device garbage.")
        elif bad(dec["chained"]) and bad(pre["chained"]):
            print("\n  CROSS-CASE: both decode and prefill are wrong.")
            print("      => the fault is not rope- or attention-specific; it "
                  "is in the shared weight/quantization path. A rebuild is "
                  "justified.")
        elif ok(dec["chained"]) and ok(pre["chained"]):
            print("\n  CROSS-CASE: every graph computes correctly on device.")
            print("      => the ctx-bins are GOOD. Stop looking at the "
                  "converter. Run probe_feed_variants.py and work its ranked "
                  "list of Genie feed mistakes.")


if __name__ == "__main__":
    main()
