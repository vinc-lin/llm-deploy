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


def load_f16(p: Path, n: int, label: str):
    raw = p.read_bytes()
    if len(raw) != n * 2:
        print(f"    !! {label}: {len(raw)} bytes, expected {n*2} "
              f"({n} fp16). Wrong dtype or a truncated pull.")
        if len(raw) == n * 4:
            print("       (that is exactly 4 bytes/element -- the run wrote "
                  "float32, so --use_native_output_files was not in effect)")
            return np.frombuffer(raw, dtype="<f4").astype(np.float32)
        return None
    return np.frombuffer(raw, dtype="<f2").astype(np.float32)


def verdict(tag, ref, got, extra=""):
    if got is None:
        print(f"  {tag:22s} MISSING")
        return None
    c = cos(ref, got)
    ra, ga = int(np.argmax(ref.reshape(-1))), int(np.argmax(got.reshape(-1)))
    mark = "OK  " if c >= COS_PASS else ("WEAK" if c >= COS_SUSPECT else "BAD ")
    agree = "argmax==" if ra == ga else f"argmax {ga} != ref {ra}"
    print(f"  {tag:22s} {mark} cos={c:+.6f}  {agree}  {extra}")
    return c


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--kit", required=True, type=Path, help="the built probe kit")
    ap.add_argument("--results", required=True, type=Path, help="pulled text_probe_out/")
    args = ap.parse_args()

    cases = json.loads((args.kit / "cases.json").read_text())
    summary = []
    for meta in cases:
        name = meta["case"]
        print(f"\n=== case {name}  (token {meta['token']}, position "
              f"{meta['position']}) — {meta['why']}")
        ref_h = np.load(args.kit / name / "ref" / "last_hidden_states.npy")
        ref_l = np.load(args.kit / name / "ref" / "logits.npy")

        p = find_out(args.results / f"{name}_d0", "last_hidden_states")
        got_h = load_f16(p, ref_h.size, "shard0") if p else None
        c_h = verdict("shard0", ref_h, got_h)

        p = find_out(args.results / f"{name}_d1chain", "logits")
        got_c = load_f16(p, ref_l.size, "shard1-chained") if p else None
        c_c = verdict("shard1-chained", ref_l, got_c,
                      f"ref top5={meta['top10_ids'][:5]}")

        p = find_out(args.results / f"{name}_d1iso", "logits")
        got_i = load_f16(p, ref_l.size, "shard1-isolated") if p else None
        c_i = verdict("shard1-isolated", ref_l, got_i)

        summary.append({"case": name, "shard0": c_h,
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
                  "tower, not from the graph. Next: probe B (Genie's own debug "
                  "dump) is the one that matters.")
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

    if len(summary) == 2:
        a, b = summary
        if ok(a["chained"]) and bad(b["chained"]):
            print("\n  CROSS-CASE: pos0 (rope=identity) is fine but pos7 "
                  "(rope active) is not.")
            print("      => the graph mishandles the rope tables. That is an "
                  "in-graph/conversion fault, NOT Genie's table generation, "
                  "because these tables came from us.")


if __name__ == "__main__":
    main()
