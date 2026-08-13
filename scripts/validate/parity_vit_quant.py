#!/usr/bin/env python
"""Gate: the shipped W8A16 ViT, driven through its FIXED-POINT I/O, vs HF.

READ THIS FIRST: THE PREMISE THIS GATE WAS COMMISSIONED ON IS FALSE
-------------------------------------------------------------------
The plan for this gate was "a quantized DLC can execute on the QNN CPU backend,
unlike the fp16 one, so run the shipped artifact". It cannot. The CPU backend's
limitation is not FP16 -- it is BITWIDTH. Measured here on QAIRT 2.48.40.260702
with a 4x8 single-Gemm model converted four ways from hand-written encodings:

    activations  weights                 libQnnCpu compose
    uFxp_16      int8 per-tensor         FAILS  (OpConfig validation failed
    uFxp_16      int8 per-channel        FAILS   for FullyConnected)
    uFxp_8       int8 per-tensor         runs
    uFxp_8       int8 per-channel        runs

and converting with --target_backend CPU instead of HTP changes nothing (the
tensors are still uFxp_16). Pointed at the shipped ViT DLC, libQnnCpu rejects
the FIRST op -- patch_embed, whose Input[0] is tensor id 1, pixel_values -- so
this is not about the fp16 attention island either. The other x86 backends do
not help: libQnnGpu refuses to initialise off-target ("TuningMode must be
enabled on x86_64-linux-clang"), and libQnnHtpQemu cannot create a context.
The ctx-bin is not an alternative to the DLC here either -- a context binary is
finalized for one backend, and `--retrieve_context` on the CPU backend fails
with "[QNN_CPU] Context de-serialization failed / Could not create context from
binary" -- so what gets executed below is always a DLC.

So on this SDK there is NO x86 execution path for a W8A16 graph, and the
strongest available evidence is a two-part bracket, which is what this script
measures END TO END in one number:

  * CONVERSION is validated by executing a throwaway FP32 DLC converted from the
    SAME AIMET-exported ONNX by the SAME converter -- op lowering, weight
    layout, the Gemm -> FullyConnected reinterpretations. (parity_vit_dlc.py
    made the same move for Stage 1 and explains it at length.)
  * The FIXED-POINT I/O CONTRACT is validated for real, on the shipped
    artifact's own numbers: pixels are quantized host-side with the ctx-bin's
    pixel_values scale/offset and all four outputs are round-tripped through
    theirs, exactly as Genie's QnnNspImageModel::setupInput<uint16_t> and its
    {UFixed16, Float32} requantize entry will do on device.
  * INTERNAL W8A16 error is NOT covered here and cannot be, without a device.
    quantize_vit_aimet.py --eval measured it on these same held-out images
    (worst min cos 0.997436) and that remains the reference for it.

The script still ATTEMPTS the shipped DLC first, every run, and only falls back
to the surrogate when the backend refuses -- so the day an SDK ships 16-bit CPU
kernels this gate silently upgrades itself to the real thing, and the mode it
ran in is printed in the verdict. Do not delete that attempt to save 50 ms.

WHAT IS FED
-----------
The SAME held-out photographs quantize_vit_aimet.py --eval reports on
(build_image_sets is imported, not reimplemented, so the split cannot drift):
disjoint from the 24 calibration images by construction, and disjoint from the
exporter's traced input, which is make_pixel_values' seed-0 uniform NOISE and
not a photograph at all -- a graph that baked its traced input in as a constant
fails here.

THE BAR
-------
Cosine >= 0.99 on every output. quantsim measured >= 0.9974 on these images, so
this catches a conversion break (op lowering, weight layout, a folded-away
quantization, mispaired outputs, a bogus I/O scale) without tripping on ordinary
requantization noise. max|d| is reported but not asserted -- the fp32 outputs
reach |20|, so an absolute bound would encode the activation scale rather than
correctness. If cosine lands materially below 0.9974, that is a CONVERSION
defect; report it, do not lower the bar.

Run:
  $PY_DEPLOY scripts/validate/parity_vit_quant.py \
      --dlc      $LLMDEPLOY_DATA/work/dlc/qwen3vl-4b-vit-w8a16/vit.dlc \
      --info     $LLMDEPLOY_DATA/work/ctxbin/qwen3vl-4b-vit-w8a16/info.json \
      --fp32-dlc $LLMDEPLOY_DATA/work/scratch/vit-fp32-gate/vit_fp32.dlc
"""
import argparse
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "export"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "quant"))
from modeling_vit_export import output_names  # noqa: E402
from quantize_vit_aimet import GRID, build_image_sets  # noqa: E402

COS_MIN = 0.99
DATA = Path(os.environ.get("LLMDEPLOY_DATA", "/home/vinc/llm-local"))
# The pool the shipped encodings were calibrated from: 40 usable COCO val2017
# photographs (41 on disk, one below --min-edge) plus 4 SDK sample photographs.
POOL_SIZE = 44


def sdk_tool(relpath):
    sdk = os.environ.get("QAIRT_SDK")
    assert sdk, "QAIRT_SDK is unset -- `source scripts/env.sh` first"
    p = Path(sdk) / relpath
    assert p.exists(), f"missing {p} -- is $QAIRT_SDK correct?"
    return p


# --------------------------------------------------------------------------
# the shipped binary's own quantization parameters
# --------------------------------------------------------------------------
def io_quant_params(info_json):
    """{tensor: (scale, offset)} for every graph I/O of the finalized ctx-bin.

    Asserts UFIXED_POINT_16 on the way through. That is the property the whole
    W8A16 ViT exists for -- Genie's setupInputFP16 is an empty stub and its
    requantize table has no Float16 entries, so a float-I/O tower is undrivable
    -- and every number below is computed with these scales, so a float-I/O
    graph must stop the gate rather than quietly skip the round-trip.
    """
    d = json.loads(Path(info_json).read_text())
    graphs = d["info"]["graphs"]
    assert len(graphs) == 1, f"expected 1 graph in {info_json}, found {len(graphs)}"
    g = graphs[0]["info"]
    out = {}
    for t in (g.get("graphInputs") or []) + (g.get("graphOutputs") or []):
        ti = t["info"]
        name, dtype = ti["name"], ti.get("dataType")
        assert dtype == "QNN_DATATYPE_UFIXED_POINT_16", (
            f"{name}: ctx-bin dataType is {dtype}, not QNN_DATATYPE_UFIXED_POINT_16 "
            "-- GeniePipeline cannot drive this tower and this gate's "
            "quantize/dequantize arithmetic does not apply")
        # "scaleOffset" on QAIRT 2.48.40; other releases emitted
        # "scaleOffsetEncoding". Accept either, never neither.
        qpar = ti.get("quantizeParams") or {}
        so = qpar.get("scaleOffset") or qpar.get("scaleOffsetEncoding")
        assert so, f"{name}: no scale/offset in {qpar!r}"
        out[name] = (float(so["scale"]), int(so["offset"]))
    return out


def quantize_u16(x, scale, offset, check_range=False):
    """real -> uint16 code, QNN convention real = scale * (q + offset).

    float64 throughout: at scale 3.05e-05 a float32 divide costs ~2 ulp of the
    quantized code, a needless bias on a tensor this gate measures to 1e-3.

    check_range is for the INPUT only. pixel_values is bounded [-1, 1] by the
    image processor and its encoding covers exactly that, so codes far outside
    uint16 there mean a swapped or mis-scaled encoding -- a bug that would
    otherwise surface only as a mysteriously bad cosine. It must stay OFF for
    outputs: their encodings are per-tensor min/max over the 24 CALIBRATION
    images, a held-out image can legitimately exceed them, and clipping is then
    the correct behaviour -- it is exactly what the device does, and its cost is
    what the cosine below is measuring.
    """
    q = np.rint(np.asarray(x, dtype=np.float64) / scale - offset)
    if check_range:
        # A pixel at exactly +1.0 rounds to code 65536, one past the top of the
        # range, so a little overshoot is normal and gets clipped.
        assert q.min() >= -16.0 and q.max() <= 65551.0, (
            f"codes {q.min()}..{q.max()} fall well outside uint16 -- wrong "
            "scale/offset, or these values are not what the encoding was "
            "calibrated on")
    return np.clip(q, 0, 65535).astype(np.uint16)


def clipped_fraction(x, scale, offset):
    """Share of `x` outside the encoding's representable range."""
    lo, hi = scale * offset, scale * (65535 + offset)
    x = np.asarray(x, dtype=np.float64)
    return float(((x < lo) | (x > hi)).mean())


def dequantize_u16(q, scale, offset):
    return (q.astype(np.float64) + offset) * scale


# --------------------------------------------------------------------------
# execution
# --------------------------------------------------------------------------
def net_run(dlc, backend, listing, outdir, native):
    cmd = [
        str(sdk_tool("bin/x86_64-linux-clang/qnn-net-run")),
        "--backend", str(sdk_tool(f"lib/x86_64-linux-clang/{backend}")),
        "--dlc_path", str(dlc),
        "--input_list", str(listing),
        "--output_dir", str(outdir),
    ]
    if native:
        # native both ways: the host owns the quantization, exactly as Genie's
        # setupInput<uint16_t> does. Without these, qnn-net-run converts float
        # files for us and the gate stops testing the I/O dtype at all.
        cmd += ["--use_native_input_files", "--use_native_output_files"]
    print("  $ " + " ".join(cmd), flush=True)
    return cmd, subprocess.run(cmd, capture_output=True, text=True)


def read_results(outdir, n_expected, dtype):
    results = sorted((d for d in outdir.iterdir() if d.is_dir()),
                     key=lambda d: int(d.name.rsplit("_", 1)[-1]))
    assert len(results) == n_expected, (
        f"qnn-net-run wrote {len(results)} result dirs for {n_expected} inputs: "
        f"{[d.name for d in results]}")
    return [{f.stem: np.fromfile(f, dtype=dtype) for f in d.glob("*.raw")}
            for d in results]


def run_native(dlc, backend, pixel_batch, qp, workdir):
    """The real thing: shipped DLC, uint16 in, uint16 out.

    Returns [{name: dequantized float64 ndarray}] or None if the backend refuses
    to compose the graph (which is what every x86 backend on this SDK does --
    see the module docstring).
    """
    scale, offset = qp["pixel_values"]
    lines = []
    for i, pv in enumerate(pixel_batch):
        raw = workdir / f"pixel_values.{i}.raw"
        quantize_u16(pv.numpy(), scale, offset, check_range=True).tofile(raw)
        lines.append(f"pixel_values:={raw}")
    listing = workdir / "input_list.txt"
    listing.write_text("\n".join(lines) + "\n")

    outdir = workdir / "out_native"
    cmd, p = net_run(dlc, backend, listing, outdir, native=True)
    if p.returncode != 0:
        print("  NATIVE EXECUTION REFUSED by " + backend)
        for line in (p.stdout + p.stderr).splitlines():
            if "ERROR" in line or "error" in line:
                print("    " + line.strip())
        return None
    return [{n: dequantize_u16(v, *qp[n]) for n, v in r.items()}
            for r in read_results(outdir, len(pixel_batch), np.uint16)]


def run_surrogate(fp32_dlc, backend, pixel_batch, qp, workdir):
    """FP32 DLC from the same ONNX, with the SHIPPED encodings applied host-side.

    Pixels are quantized and immediately dequantized with the ctx-bin's
    pixel_values encoding before they enter the graph, and every output is
    round-tripped through its own; so the fixed-point I/O contract is exercised
    with the real scales even though the arithmetic between them is fp32.
    """
    assert fp32_dlc and Path(fp32_dlc).exists(), (
        f"the shipped DLC cannot execute on any x86 backend and the --fp32-dlc "
        f"surrogate ({fp32_dlc}) is missing. Build one (throwaway, do NOT ship "
        "it, do NOT put it in work/dlc/):\n"
        "  $PY_QAIRT $QAIRT_SDK/bin/x86_64-linux-clang/qairt-converter \\\n"
        "      --input_network $LLMDEPLOY_DATA/work/quant/qwen3vl-4b-vit/model.onnx \\\n"
        "      --output_path <scratch>/vit_fp32.dlc --target_backend HTP \\\n"
        "      -d pixel_values 1024,1536")
    scale, offset = qp["pixel_values"]
    lines = []
    for i, pv in enumerate(pixel_batch):
        raw = workdir / f"pixel_values_rt.{i}.raw"
        rt = dequantize_u16(
            quantize_u16(pv.numpy(), scale, offset, check_range=True), scale, offset)
        rt.astype(np.float32).tofile(raw)
        lines.append(f"pixel_values:={raw}")
    listing = workdir / "input_list_fp32.txt"
    listing.write_text("\n".join(lines) + "\n")

    outdir = workdir / "out_surrogate"
    cmd, p = net_run(fp32_dlc, backend, listing, outdir, native=False)
    if p.returncode != 0:
        raise SystemExit(
            f"qnn-net-run failed on the fp32 surrogate (exit {p.returncode})\n"
            f"--- cmd ---\n{' '.join(cmd)}\n"
            f"--- stdout ---\n{p.stdout}\n--- stderr ---\n{p.stderr}")
    out = []
    for i, r in enumerate(read_results(outdir, len(pixel_batch), np.float32)):
        # Round-trip each output through its shipped encoding: this is the loss
        # the device will take moving image features into Genie's embedding
        # accumulator, and it is the half of the I/O contract the input side
        # cannot show. Values beyond the calibrated range clip, exactly as they
        # will on device; the share that clips is printed because a large one is
        # a calibration finding even when the cosine survives it.
        for n, v in r.items():
            frac = clipped_fraction(v, *qp[n])
            if frac:
                print(f"    [{i}] {n}: {frac:.3%} of values clip to the "
                      "calibrated range")
        out.append({n: dequantize_u16(quantize_u16(v, *qp[n]), *qp[n])
                    for n, v in r.items()})
    return out


# --------------------------------------------------------------------------
def compare(name, a, b):
    """max|d| and cosine for one output, reduced in float64.

    Finiteness is asserted BEFORE any reduction: max(0.0, nan) is 0.0 and
    min(1.0, nan) is 1.0, so a NaN reaching the reduction is silently swallowed
    and the gate exits 0 on a graph emitting garbage. Documented past bug.
    """
    assert a.shape == b.shape, f"{name}: shape {a.shape} != {b.shape}"
    assert np.isfinite(a).all(), f"{name}: DLC output has non-finite values"
    assert np.isfinite(b).all(), f"{name}: HF reference has non-finite values"
    assert float(np.abs(a).max()) > 0.0, f"{name}: DLC output is all zeros"
    assert float(a.std()) > 0.0, f"{name}: DLC output is constant"
    assert float(b.std()) > 0.0, f"{name}: HF reference is constant"
    d = float(np.abs(a - b).max())
    af, bf = a.ravel().astype(np.float64), b.ravel().astype(np.float64)
    cos = float(np.dot(af, bf) / (np.linalg.norm(af) * np.linalg.norm(bf)))
    return d, cos, float(np.abs(b).max())


def held_out_images(model_dir, images_dir, n_images):
    """The quantsim eval split: photographs, disjoint from calibration."""
    from transformers import AutoProcessor
    from quantize_vit_aimet import EDGE, list_images

    pool = list_images(images_dir, 256)
    assert len(pool) == POOL_SIZE, (
        f"{len(pool)} usable calibration images on this box, but the shipped "
        f"encodings were drawn from a pool of {POOL_SIZE}. build_image_sets "
        "shuffles the pool, so a different pool means a different split and "
        "these 'held-out' images may be ones the model was calibrated on. "
        "Re-point --images-dir at the pool quantize_vit_aimet.py used.")
    proc = AutoProcessor.from_pretrained(model_dir, min_pixels=EDGE * EDGE,
                                         max_pixels=EDGE * EDGE)
    # Same arguments quantize_vit_aimet.py defaults to, so `held` is the exact
    # set its --eval reported on.
    args = SimpleNamespace(images_dir=images_dir, shuffle_seed=0, n_calib=24,
                           n_eval=6, synthetic=0, min_edge=256)
    calib, held = build_image_sets(args, proc)
    calib_labels = {lb for _, lb in calib}
    for _, lb in held:
        assert lb not in calib_labels, f"held-out image {lb} is also in calibration"
        assert not lb.startswith("synthetic:"), (
            f"{lb} is a synthetic fallback image, not a photograph -- the image "
            "pool is not the one the encodings were calibrated from")
    return held[:n_images]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=str(DATA / "models/Qwen3-VL-4B-Instruct"))
    ap.add_argument("--dlc", default=str(DATA / "work/dlc/qwen3vl-4b-vit-w8a16/vit.dlc"),
                    help="the SHIPPED quantized DLC; tried first, every run")
    ap.add_argument("--info", default=str(
        DATA / "work/ctxbin/qwen3vl-4b-vit-w8a16/info.json"),
        help="finalized ctx-bin info.json -- source of the I/O quant params")
    ap.add_argument("--fp32-dlc", default=str(
        DATA / "work/scratch/vit-fp32-gate/vit_fp32.dlc"),
        help="throwaway fp32 DLC from the same ONNX, used when no x86 backend "
             "can execute the shipped one. Not shippable.")
    ap.add_argument("--images-dir", action="append", default=None,
                    help="calibration image pool (repeatable); default is the "
                         "pool quantize_vit_aimet.py was run with")
    ap.add_argument("--n-images", type=int, default=6,
                    help="how many of the 6 held-out images to run")
    ap.add_argument("--backend", default="libQnnCpu.so",
                    help="QNN backend .so name; only the CPU one runs on x86")
    args = ap.parse_args()

    images_dir = args.images_dir or [
        str(DATA / "work/calib/images-coco-val2017"),
        str(Path(os.environ.get("QAIRT_SDK", "")) / "examples/Models/InceptionV3/data"),
        str(Path(os.environ.get("QAIRT_SDK", "")) / "examples/QAIRT/python/images"),
    ]

    qp = io_quant_params(args.info)
    print(f"== shipped ctx-bin I/O quantization ({args.info}) ==")
    for name, (s, o) in qp.items():
        print(f"  {name:26s} UFIXED_16 scale={s:.12g} offset={o} "
              f"range [{s * o:+.5f}, {s * (65535 + o):+.5f}]")

    print("== held-out images (quantsim --eval split) ==")
    held = held_out_images(args.model, images_dir, args.n_images)
    for _, label in held:
        print(f"  [eval] {label}")
    assert held, "no held-out images"

    print("== loading checkpoint (fp32, eager) ==", flush=True)
    # eager is mandatory: ViT parity was established against the eager path, and
    # an sdpa-loaded model runs a different attention kernel.
    from transformers import Qwen3VLForConditionalGeneration
    hf = Qwen3VLForConditionalGeneration.from_pretrained(
        args.model, dtype=torch.float32, attn_implementation="eager").eval()
    vision = hf.model.visual
    grid_thw = torch.tensor(GRID, dtype=torch.long)
    names = output_names(len(vision.deepstack_visual_indexes))

    refs = []
    for pv, label in held:
        with torch.no_grad():
            r = vision(pv, grid_thw)
        ref = {names[0]: r.pooler_output.numpy().astype(np.float64)}
        for i, d in enumerate(r.deepstack_features):
            ref[names[1 + i]] = d.numpy().astype(np.float64)
        assert set(ref) == set(names), f"reference keys {sorted(ref)} != {sorted(names)}"
        refs.append(ref)
    del hf, vision      # ~17.6 GB of fp32 checkpoint; do not hold it through qnn-net-run

    pixels = [pv for pv, _ in held]
    with tempfile.TemporaryDirectory(prefix="parity_vit_quant.") as tmp:
        print(f"== attempt 1: SHIPPED DLC on {args.backend} ==", flush=True)
        got = run_native(Path(args.dlc), args.backend, pixels, qp, Path(tmp))
        mode = "NATIVE (shipped uFxp_16 DLC)"
        if got is None:
            print("== falling back: fp32 surrogate + shipped I/O encodings ==")
            print("   the shipped W8A16 graph has no x86 execution path on this "
                  "SDK (see this file's docstring); internal quantization error "
                  "stays covered only by quantize_vit_aimet.py --eval.")
            got = run_surrogate(Path(args.fp32_dlc) if args.fp32_dlc else None,
                                args.backend, pixels, qp, Path(tmp))
            mode = "SURROGATE (fp32 graph, shipped fixed-point I/O)"

    stats = {n: {"d": [], "cos": []} for n in names}
    for (pv, label), ref, raw in zip(held, refs, got):
        print(f"[eval] {label}")
        # Pair BY NAME, never by position. The built artifact declares its
        # outputs as deepstack_visual_embed_{0,1,2} then image_features -- i.e.
        # image_features is LAST, not first as output_names() lists it. Zipping
        # positional results against `names` would compare image_features
        # against a deepstack reference and vice versa, which is exactly the
        # silent false-verdict this gate exists to prevent.
        for name in names:
            assert name in raw, f"qnn-net-run produced no {name}.raw; got {sorted(raw)}"
            b = ref[name]
            a = raw[name]
            assert a.size == b.size, (
                f"{name}: {a.size} elements, expected {b.size} -- the graph's "
                "output is not of the expected shape")
            d, cos, absmax = compare(name, a.reshape(b.shape), b)
            stats[name]["d"].append(d)
            stats[name]["cos"].append(cos)
            print(f"    {name:26s} max|d|={d:.4e}  cos={cos:.6f}  "
                  f"(fp32 absmax {absmax:.3f}, rel {d / absmax:.3%})")

    print(f"== summary: {mode}, worst over {len(held)} held-out images ==")
    bad = []
    for name in names:
        s = stats[name]
        worst_d, worst_cos = max(s["d"]), min(s["cos"])
        print(f"  {name:26s} max|d|={worst_d:.4e}  min cos={worst_cos:.6f}  "
              f"mean cos={sum(s['cos']) / len(s['cos']):.6f}")
        if not worst_cos >= COS_MIN:
            bad.append(f"{name}: min cos {worst_cos:.6f} < {COS_MIN}")
    assert not bad, (
        "diverges from HF: " + "; ".join(bad) +
        " -- this is a CONVERSION defect (quantsim measured >= 0.9974 on these "
        "same images); do not lower the bar")
    print(f"PASS [{mode}]: matches HF on all {len(names)} outputs "
          f"(min cos {min(min(stats[n]['cos']) for n in names):.6f} >= {COS_MIN})")


if __name__ == "__main__":
    main()
