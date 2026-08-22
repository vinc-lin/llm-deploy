#!/usr/bin/env python
"""Probe B -- which FEED corruption could explain the device's garbage?

This replaces the original plan of enabling Genie's own debug tensor dump.
That plan is dead and the reason is worth recording: `debug-tensors` /
`debug-path` ARE compiled into the shipped `libGenie.so` (the strings are in
the binary), but `Engine.cpp` validates the public config against a strict
whitelist and throws `Unknown QnnHtp config key` on anything outside it --
`version, spill-fill-bufsize, mmap-budget, use-mmap, pos-id-dim,
shared-engine, cpu-mask, poll, kv-dim, batch-size, kv-update-method,
allow-async-init, rope-theta, enable-graph-switching,
graph-switching-lora-policy, weight-shared-lora, skip-lora-validation`. The
engine level is equally strict. So there is no public route to the dump, and
shipping such a config would only buy a load failure -- the same trap as
correction #31.

What we do instead. Probe A answers "is the ctx-bin right?". If it says yes,
the fault is in how Genie FEEDS the tower -- and this script pre-computes the
shortlist of what that could be, on the host, at zero device cost.

For each hypothesised corruption we run the real split tower and measure how
far the logits move from the correct feed. A corruption that barely moves them
cannot explain garbage output and is eliminated; one that destroys them is a
live candidate. That converts "Genie feeds it wrong somehow" into a ranked list
of specific, checkable mistakes.

PREFILL, FOUR ROWS -- not a single decode step. With one token attending only
to itself, RoPE rotates q and k by the same angle and cancels, so every rope
variant would be a silent no-op and the probe would measure nothing. (Verified:
an earlier probe-A case at position 7 produced byte-identical logits to
position 0.) Four rows through the prefill graphs make rope and cross-token
attention live, and those are the graphs the real 273-token prompt uses.

The variants are not arbitrary -- each is a mistake this project has already
made once, or could make silently:

  mask_multiplicative  0/1 instead of the additive 0/-100 convention
  mask_all_visible     no masking at all (the empty cache left readable)
  mask_row0_only       every row sees only row 0 (a broadcast/stride slip)
  rope_not_applied     cos=1,sin=0 -- tables never engaged
  rope_wrong_theta     theta 1e6 (the 0.6B value) instead of 5e6
  rope_no_offset       every row given position 0 (positions never advanced)
  emb_fp32_as_fp16     the embedding's fp32 bytes read as fp16 -- exactly the
                       class of bug behind the v3 image SIGSEGV, one layer over
  emb_halved           embeddings at half scale (a requantize that no-ops)

Run on tank, against the ONNX that produced the SHIPPED DLCs:

  $PY_DEPLOY scripts/validate/probe_feed_variants.py \
      --onnx-split $LLMDEPLOY_DATA/work/onnx/qwen3vl-4b-gqa-aimet-splitkv \
      --lut        $LLMDEPLOY_DATA/work/lut/qwen3vl-4b --threads 40
"""
import argparse
import gc
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "export"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from modeling_export import MASK_VALUE, rope_tables  # noqa: E402
from build_text_probe_kit import (  # noqa: E402
    lut_row, meta_of, zeros_past, build_mask_rope, TOKENS, ROPE_THETA)

N_REAL = 4
THETA_06B = 1_000_000.0

VARIANTS = ["correct", "mask_multiplicative", "mask_all_visible",
            "mask_row0_only", "rope_not_applied", "rope_wrong_theta",
            "rope_no_offset", "emb_fp32_as_fp16", "emb_halved"]


def corrupt(name, m, mask, cos, sin, rows):
    """Apply one hypothesis to the correct feed. Returns (mask, cos, sin, rows)."""
    import torch
    ar, past, total = m["ar"], m["past"], m["total"]
    mask, cos, sin, rows = mask.copy(), cos.copy(), sin.copy(), rows.copy()

    if name == "correct":
        pass
    elif name == "mask_multiplicative":
        mask = np.zeros((1, ar, total), dtype=np.float32)
        for i in range(N_REAL):
            mask[0, i, past:past + i + 1] = 1.0
    elif name == "mask_all_visible":
        mask = np.zeros((1, ar, total), dtype=np.float32)
    elif name == "mask_row0_only":
        mask = np.full((1, ar, total), MASK_VALUE, dtype=np.float32)
        for i in range(N_REAL):
            mask[0, i, past] = 0.0
    elif name == "rope_not_applied":
        cos = np.ones_like(cos)
        sin = np.zeros_like(sin)
    elif name == "rope_wrong_theta":
        c, s = rope_tables(torch.arange(N_REAL), m["head_dim"], THETA_06B)
        cos[0, :N_REAL] = c.numpy()[0]
        sin[0, :N_REAL] = s.numpy()[0]
    elif name == "rope_no_offset":
        c, s = rope_tables(torch.zeros(N_REAL, dtype=torch.long),
                           m["head_dim"], ROPE_THETA)
        cos[0, :N_REAL] = c.numpy()[0]
        sin[0, :N_REAL] = s.numpy()[0]
    elif name == "emb_fp32_as_fp16":
        for r in range(rows.shape[0]):
            raw = rows[r].astype("<f4").tobytes()
            rows[r] = np.frombuffer(raw, dtype="<f2").astype(np.float32)[:rows.shape[1]]
    elif name == "emb_halved":
        rows = rows * 0.5
    else:
        raise SystemExit(f"unknown variant {name}")
    return mask, cos, sin, rows


def main():
    import onnxruntime as ort
    ap = argparse.ArgumentParser()
    ap.add_argument("--onnx-split", required=True, type=Path)
    ap.add_argument("--lut", required=True, type=Path)
    ap.add_argument("--threads", type=int, default=0)
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()

    rows0 = np.stack([lut_row(args.lut, t) for t in TOKENS[:N_REAL]])
    H = rows0.shape[1]

    so = ort.SessionOptions()
    if args.threads:
        so.intra_op_num_threads = args.threads

    # Both sessions loaded once and reused: reloading a ~2B-param fp32 graph
    # per variant would dominate the runtime by an order of magnitude.
    s0 = ort.InferenceSession(str(args.onnx_split / "prefill_0" / "prefill_0.onnx"),
                              so, providers=["CPUExecutionProvider"])
    m0 = meta_of(s0)
    s1 = ort.InferenceSession(str(args.onnx_split / "prefill_1" / "prefill_1.onnx"),
                              so, providers=["CPUExecutionProvider"])
    m1 = meta_of(s1)
    z0, z1 = zeros_past(m0), zeros_past(m1)
    base_mask, base_cos, base_sin = build_mask_rope(m0, N_REAL)

    results, ref = [], None
    for name in VARIANTS:
        mask, cos, sin, rows = corrupt(name, m0, base_mask, base_cos,
                                       base_sin, rows0)
        emb = np.zeros((1, 1, m0["ar"], H), dtype=np.float32)
        emb[0, 0, :N_REAL] = rows
        f0 = {"inputs_embeds": emb, "attention_mask": mask,
              "position_ids_cos": cos, "position_ids_sin": sin, **z0}
        for dk in m0["deep"]:
            f0[dk] = np.zeros((1, 1, m0["ar"], H), dtype=np.float32)
        hid = s0.run(["last_hidden_states"], f0)[0]

        # The same corruption applies to BOTH shards: Genie feeds each the same
        # mask and rope tables, so corrupting only shard 0 would not model the
        # real failure.
        f1 = {"last_hidden_states": hid.reshape(1, m1["ar"], H),
              "attention_mask": mask, "position_ids_cos": cos,
              "position_ids_sin": sin, **z1}
        lg = s1.run(["logits"], f1)[0].reshape(m1["ar"], -1)[:N_REAL]
        lg = lg.astype(np.float64)

        if name == "correct":
            ref = lg
        per_row = [float(ref[r] @ lg[r] /
                         (np.linalg.norm(ref[r]) * np.linalg.norm(lg[r]) + 1e-30))
                   for r in range(N_REAL)]
        worst = min(per_row)
        am = [int(np.argmax(lg[r])) for r in range(N_REAL)]
        ref_am = [int(np.argmax(ref[r])) for r in range(N_REAL)]
        agree = sum(a == b for a, b in zip(am, ref_am))
        results.append({"variant": name, "worst_cos": worst,
                        "per_row_cos": per_row, "argmax": am,
                        "argmax_agree": agree})
        print(f"  {name:22s} worst_cos={worst:+.6f}  argmax {agree}/{N_REAL}  "
              f"{am}")
        gc.collect()

    print("\n" + "=" * 72)
    print("READING THIS")
    print("=" * 72)
    live = [r for r in results[1:]
            if r["worst_cos"] < 0.9 or r["argmax_agree"] < N_REAL]
    dead = [r for r in results[1:] if r not in live]
    if live:
        print("COULD produce the observed garbage (ranked, worst first):")
        for r in sorted(live, key=lambda x: x["worst_cos"]):
            print(f"  - {r['variant']:22s} worst_cos={r['worst_cos']:+.4f} "
                  f"argmax {r['argmax_agree']}/{N_REAL}")
        print("  -> if probe A exonerates the ctx-bin, check these against what")
        print("     Genie actually feeds, in this order.")
    if dead:
        print("\nELIMINATED -- these barely move the logits, so on their own")
        print("they CANNOT explain garbage output:")
        for r in dead:
            print(f"  - {r['variant']:22s} worst_cos={r['worst_cos']:+.4f}")
    if args.out:
        args.out.write_text(json.dumps(results, indent=1) + "\n")
        print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
