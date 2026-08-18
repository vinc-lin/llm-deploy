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
shipping a config with it would only buy a load failure. (This is the same
trap as correction #31.)

What we do instead. Probe A answers "is the ctx-bin right?". If it says yes,
the fault is in how Genie FEEDS the tower -- and this script pre-computes the
shortlist of what that could be, on the host, with no device time at all.

For each hypothesised corruption we run the real split tower one step and
measure how far the logits move from the correct feed. A corruption that
barely moves them cannot explain garbage output and is eliminated; one that
destroys them is a live candidate. That converts "Genie feeds it wrong
somehow" into a ranked list of specific, checkable mistakes.

The variants are not arbitrary -- each is a mistake this codebase has either
already made once or could make silently:

  mask_multiplicative  the 0/1 convention instead of additive 0/-100
  mask_all_visible     no masking at all (empty cache left unmasked)
  rope_not_applied     cos=1,sin=0 at a real position (tables never engaged)
  rope_wrong_theta     rope-theta 1e6 (the 0.6B value) instead of 5e6
  emb_fp32_as_fp16     the embedding row's fp32 bytes read as fp16 -- the exact
                       class of bug that caused the v3 image SIGSEGV, one
                       layer over
  emb_halved           embeddings at half scale (a requantize that no-ops)

Run on tank (the per-shard ONNX lives there):

  $PY_DEPLOY scripts/validate/probe_feed_variants.py \
      --onnx-split $LLMDEPLOY_DATA/work/onnx/qwen3vl-4b-aimet-split \
      --lut        $LLMDEPLOY_DATA/work/lut/qwen3vl-4b
"""
import argparse
import gc
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "export"))
from modeling_export import MASK_VALUE, rope_tables  # noqa: E402

TOKEN = 3838
POSITION = 7          # nonzero, so rope variants are meaningful
THETA_SHIPPED = 5_000_000.0
THETA_06B = 1_000_000.0
DEEPSTACK = 3


def lut_row(lut_dir: Path, token: int):
    p = json.loads((lut_dir / "embedding_lut_params.json").read_text())
    n, eb = p["size"], p["element-bytes"]
    with (lut_dir / p["lut-path"]).open("rb") as fh:
        fh.seek(token * n * eb)
        return np.frombuffer(fh.read(n * eb), dtype="<f4").astype(np.float32)


def meta_of(sess):
    sh = {i.name: i.shape for i in sess.get_inputs()}
    pk = next(n for n in sh if n.startswith("past_key_") and n.endswith("_in"))
    _, nkv, hd, past = sh[pk]
    layers = sorted(int(n.split("_")[2]) for n in sh
                    if n.startswith("past_key_") and n.endswith("_in"))
    return {"n_kv": nkv, "head_dim": hd, "past": past,
            "total": sh["attention_mask"][-1], "layers": layers}


def base_mask(m):
    a = np.full((1, 1, m["total"]), MASK_VALUE, dtype=np.float32)
    a[0, 0, m["past"]] = 0.0
    return a


def variant_inputs(name, m, emb):
    """-> (mask, cos, sin, emb) for one hypothesis."""
    import torch
    mask = base_mask(m)
    cos, sin = rope_tables(torch.tensor([POSITION]), m["head_dim"], THETA_SHIPPED)
    cos, sin = cos.numpy().astype(np.float32), sin.numpy().astype(np.float32)
    e = emb.copy()

    if name == "correct":
        pass
    elif name == "mask_multiplicative":
        mask = np.zeros((1, 1, m["total"]), dtype=np.float32)
        mask[0, 0, m["past"]] = 1.0
    elif name == "mask_all_visible":
        mask = np.zeros((1, 1, m["total"]), dtype=np.float32)
    elif name == "rope_not_applied":
        cos = np.ones_like(cos)
        sin = np.zeros_like(sin)
    elif name == "rope_wrong_theta":
        c, s = rope_tables(torch.tensor([POSITION]), m["head_dim"], THETA_06B)
        cos, sin = c.numpy().astype(np.float32), s.numpy().astype(np.float32)
    elif name == "emb_fp32_as_fp16":
        # the row's fp32 bytes reinterpreted as fp16, then padded back to width
        raw = e.astype("<f4").tobytes()
        as16 = np.frombuffer(raw, dtype="<f2").astype(np.float32)
        e = as16[: e.shape[0]].copy()
    elif name == "emb_halved":
        e = e * 0.5
    else:
        raise SystemExit(f"unknown variant {name}")
    return mask, cos, sin, e


VARIANTS = ["correct", "mask_multiplicative", "mask_all_visible",
            "rope_not_applied", "rope_wrong_theta", "emb_fp32_as_fp16",
            "emb_halved"]


def main():
    import onnxruntime as ort
    ap = argparse.ArgumentParser()
    ap.add_argument("--onnx-split", required=True, type=Path)
    ap.add_argument("--lut", required=True, type=Path)
    ap.add_argument("--threads", type=int, default=0)
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()

    emb0 = lut_row(args.lut, TOKEN)
    so = ort.SessionOptions()
    if args.threads:
        so.intra_op_num_threads = args.threads

    # shard 0 once per variant, shard 1 once per variant; both sessions are
    # loaded once and reused -- reloading a 2B-param fp32 graph per variant
    # would dominate the runtime.
    s0 = ort.InferenceSession(str(args.onnx_split / "decode_0" / "decode_0.onnx"),
                              so, providers=["CPUExecutionProvider"])
    m0 = meta_of(s0)
    s1 = ort.InferenceSession(str(args.onnx_split / "decode_1" / "decode_1.onnx"),
                              so, providers=["CPUExecutionProvider"])
    m1 = meta_of(s1)

    def zeros(m):
        z = {}
        for i in m["layers"]:
            z[f"past_key_{i}_in"] = np.zeros(
                (1, m["n_kv"], m["head_dim"], m["past"]), dtype=np.float32)
            z[f"past_value_{i}_in"] = np.zeros(
                (1, m["n_kv"], m["past"], m["head_dim"]), dtype=np.float32)
        return z

    z0, z1 = zeros(m0), zeros(m1)
    H = emb0.shape[0]
    results, ref = [], None

    for name in VARIANTS:
        mask, cos, sin, e = variant_inputs(name, m0, emb0)
        f0 = {"inputs_embeds": e.reshape(1, 1, 1, H), "attention_mask": mask,
              "position_ids_cos": cos, "position_ids_sin": sin, **z0}
        for k in range(DEEPSTACK):
            f0[f"deepstack_visual_embed_{k}"] = np.zeros((1, 1, 1, H), np.float32)
        hid = s0.run(["last_hidden_states"], f0)[0]

        # rope/mask corruption applies to BOTH shards -- Genie feeds the same
        # tables and mask to each, so a half-corrupted chain would not model
        # the real failure.
        f1 = {"last_hidden_states": hid.reshape(1, 1, -1),
              "attention_mask": mask, "position_ids_cos": cos,
              "position_ids_sin": sin, **z1}
        lg = s1.run(["logits"], f1)[0].reshape(-1).astype(np.float64)

        if name == "correct":
            ref = lg
        c = float(ref @ lg / (np.linalg.norm(ref) * np.linalg.norm(lg) + 1e-30))
        top = np.argsort(-lg)[:5]
        results.append({"variant": name, "cos_vs_correct": c,
                        "argmax": int(top[0]), "top5": [int(t) for t in top]})
        print(f"  {name:22s} cos={c:+.6f}  argmax={top[0]:6d}  top5={list(top)}")
        gc.collect()

    print("\n" + "=" * 70)
    print("READING THIS")
    print("=" * 70)
    ref_argmax = results[0]["argmax"]
    live = [r for r in results[1:] if r["cos_vs_correct"] < 0.9
            or r["argmax"] != ref_argmax]
    dead = [r for r in results[1:] if r not in live]
    if live:
        print("Corruptions that COULD produce the observed garbage:")
        for r in live:
            print(f"  - {r['variant']:22s} cos={r['cos_vs_correct']:+.4f} "
                  f"argmax {r['argmax']} (correct {ref_argmax})")
        print("  -> if probe A exonerates the ctx-bin, these are the feed "
              "mistakes worth checking, in this order.")
    if dead:
        print("Eliminated -- these barely move the logits, so they CANNOT be "
              "the cause on their own:")
        for r in dead:
            print(f"  - {r['variant']:22s} cos={r['cos_vs_correct']:+.4f}")
    if args.out:
        args.out.write_text(json.dumps(results, indent=1) + "\n")
        print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
