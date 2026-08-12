#!/usr/bin/env python
"""Gate 3: converted DLC executed by qnn-net-run (CPU backend) vs HF.

Earlier gates validate torch (Gate 1) and the ONNX trace (Gate 2). This one
validates what `qairt-converter` produced from that ONNX -- op lowering, weight
layout, the Gemm -> FullyConnected reinterpretations and the Casts the
converter inserts. That is the bug class which would otherwise first surface as
garbage on a device nobody here can debug.

WHAT THIS GATE DOES NOT COVER
-----------------------------
The QNN CPU backend has no FP16 execution path. Pointed at the shipped fp16
DLC it refuses to build the graph at all, before running a single op:

    [QNN_CPU] OpConfig validation failed for FullyConnected
    Exception encountered: Validate OpConfig failed:
        QNN_OP_PACKAGE_ERROR_VALIDATION_FAILURE: Op configuration failed validation
    Failed to successfully compose graph

So this gate is pointed instead at an FP32 DLC converted from the SAME ONNX by
the SAME converter (`qairt-converter` without `--float_bitwidth 16`), built as
a throwaway validation artifact -- it is not shippable and does not belong in
work/dlc/. Because both DLCs come from one ONNX through one converter, this
proves the CONVERSION: topology, weights, op lowering. It proves NOTHING about
fp16 rounding, and nothing about the fp16 DLC's own weight blob. HTP context
binaries cannot execute on x86, so this is the strongest check physically
available without a device; the shipped artifact's fp16 numerics stay
unvalidated until device time, and the model card must say so.

Uses seed=2, distinct from the exporter's seed 0 and Gate 2's seed 1: a graph
that baked its traced input in as a constant would pass those and fail here.

Run:
  $PY_DEPLOY scripts/validate/parity_vit_dlc.py \
      --model $LLMDEPLOY_DATA/models/Qwen3-VL-4B-Instruct \
      --dlc   <scratch>/vit_fp32.dlc
"""
import argparse
import os
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "export"))
from modeling_vit_export import make_pixel_values, output_names  # noqa: E402

TOL = 5e-3
COS_MIN = 0.9995


def sdk_tool(relpath):
    sdk = os.environ.get("QAIRT_SDK")
    assert sdk, "QAIRT_SDK is unset -- `source scripts/env.sh` first"
    p = Path(sdk) / relpath
    assert p.exists(), f"missing {p} -- is $QAIRT_SDK correct?"
    return p


def run_dlc(dlc, pixel_values, workdir):
    """Execute `dlc` on the QNN CPU backend, one inference, and return
    {tensor_name: fp32 ndarray} keyed by the graph's own output names."""
    raw = workdir / "pixel_values.raw"
    # fp32 on the wire: without --use_native_input_files qnn-net-run parses
    # input files as float and converts to the graph's dtype itself.
    pixel_values.numpy().astype(np.float32).tofile(raw)

    listing = workdir / "input_list.txt"
    listing.write_text(f"pixel_values:={raw}\n")

    outdir = workdir / "out"
    cmd = [
        str(sdk_tool("bin/x86_64-linux-clang/qnn-net-run")),
        "--backend", str(sdk_tool("lib/x86_64-linux-clang/libQnnCpu.so")),
        "--dlc_path", str(dlc),
        "--input_list", str(listing),
        "--output_dir", str(outdir),
    ]
    print("  $ " + " ".join(cmd))
    p = subprocess.run(cmd, capture_output=True, text=True)
    if p.returncode != 0:
        # The interesting failures (unsupported dtype, op validation) are
        # backend messages on stdout, so surface both streams verbatim rather
        # than a bare exit code.
        raise SystemExit(
            f"qnn-net-run failed (exit {p.returncode})\n"
            f"--- cmd ---\n{' '.join(cmd)}\n"
            f"--- stdout ---\n{p.stdout}\n--- stderr ---\n{p.stderr}"
        )

    # Result layout varies across QAIRT releases (output_dir/Result_N/<name>.raw
    # today), so locate each tensor by filename rather than hardcoding a path.
    got = {}
    for f in outdir.rglob("*.raw"):
        got.setdefault(f.stem, []).append(f)
    return got, outdir


def read_output(got, outdir, name, shape):
    hits = got.get(name)
    if not hits:
        found = sorted(str(f.relative_to(outdir)) for f in outdir.rglob("*") if f.is_file())
        raise SystemExit(
            f"qnn-net-run produced no '{name}.raw' under {outdir}\n"
            f"files actually present: {found}"
        )
    assert len(hits) == 1, f"{name}: ambiguous outputs {[str(h) for h in hits]}"
    a = np.fromfile(hits[0], dtype=np.float32)
    n = int(np.prod(shape))
    assert a.size == n, f"{name}: {hits[0]} holds {a.size} floats, expected {n}"
    return a.reshape(shape)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--dlc", required=True)
    args = ap.parse_args()

    from transformers import Qwen3VLForConditionalGeneration
    hf = Qwen3VLForConditionalGeneration.from_pretrained(
        args.model, dtype=torch.float32, attn_implementation="eager"
    ).eval()
    vision = hf.model.visual

    pixel_values, grid_thw = make_pixel_values(args.model, seed=2)
    with torch.no_grad():
        ref = vision(pixel_values, grid_thw)

    names = output_names(len(ref.deepstack_features))
    refs = {names[0]: ref.pooler_output.numpy()}
    for i, d in enumerate(ref.deepstack_features):
        refs[names[1 + i]] = d.numpy()
    assert set(refs) == set(names), f"reference keys {sorted(refs)} != {sorted(names)}"

    with tempfile.TemporaryDirectory(prefix="parity_vit_dlc.") as tmp:
        got, outdir = run_dlc(Path(args.dlc), pixel_values, Path(tmp))

        # Pair BY NAME, never by position. The built artifact declares its
        # outputs as deepstack_visual_embed_{0,1,2} then image_features -- i.e.
        # image_features is LAST, not first as output_names() lists it. Zipping
        # positional results against `names` would compare image_features
        # against a deepstack reference and vice versa, which is exactly the
        # silent false-verdict this gate exists to prevent.
        worst_d, worst_cos = 0.0, 1.0
        for name in names:
            b = refs[name]
            a = read_output(got, outdir, name, b.shape)
            # Must fire BEFORE the reduction below: max(0.0, nan) is 0.0 and
            # min(1.0, nan) is 1.0, so a NaN reaching it is silently dropped and
            # the gate exits 0 on a graph emitting garbage.
            assert np.isfinite(a).all(), f"{name}: DLC output has non-finite values"
            assert np.isfinite(b).all(), f"{name}: HF reference has non-finite values"
            d = float(np.abs(a - b).max())
            # float64 for the reduction: in float32 over 655k elements this
            # returns values like 1.00000012, impossible for a cosine.
            af, bf = a.ravel().astype(np.float64), b.ravel().astype(np.float64)
            cos = float(np.dot(af, bf) / (np.linalg.norm(af) * np.linalg.norm(bf)))
            print(f"  {name:28s} max|d|={d:.3e} cos={cos:.8f}")
            worst_d, worst_cos = max(worst_d, d), min(worst_cos, cos)

    assert worst_d < TOL, f"DLC diverges from HF: max|d|={worst_d:.3e} >= {TOL}"
    assert worst_cos > COS_MIN, f"DLC cosine {worst_cos:.8f} <= {COS_MIN}"
    print(f"PASS: DLC matches HF (max|d|={worst_d:.3e}, min cos={worst_cos:.8f})")


if __name__ == "__main__":
    main()
