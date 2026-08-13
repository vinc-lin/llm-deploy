#!/usr/bin/env python
"""Image file -> the raw UFixed16 pixel_values blob genie-app feeds the ViT.

Genie does NO image preprocessing. `node set image` reads the file as an opaque
blob straight into GenieNode_setData (genie-app/main.cpp:1162-1184; the size is
never checked against the tensor), so these bytes must ALREADY be exactly what
the graph's input tensor expects: [1024, 1536] QNN_DATATYPE_UFIXED_POINT_16 =
3,145,728 bytes.

UFixed16, not fp16, and that is not a style choice: this SDK build cannot drive
an fp16 image encoder at all. QnnNspImageModel::setupInputFP16 is an empty stub
that discards the data and returns success (nsp-image-model.cpp:526-530), while
UFIXED_POINT_16 dispatches to a real setupInput<uint16_t> copy (:541-543).
See docs/NOTES-genie-pipeline.md section D.

The quantization parameters are READ FROM THE GRAPH, never hardcoded -- host
and device must agree bit-for-bit, and the only way to guarantee that is to use
the graph's own encoding. Accepts either the ctx-bin's info.json (preferred:
that is the actual shipped artifact) or AIMET's model.encodings.

The graph is static: 1024 patches = a 32x32 patch grid = a 512x512 image;
merge 2 -> 256 embedding rows. Every image is resized (aspect-distorting, on
purpose -- the goal is "images work at all"), then routed through the real HF
image processor so normalization and patch ordering match the export trace
exactly.

Run:
  $PY_DEPLOY scripts/pipeline/preprocess_image.py \
      --model $LLMDEPLOY_DATA/models/Qwen3-VL-4B-Instruct \
      --image photo.jpg --out sample_image.raw \
      --encodings $LLMDEPLOY_DATA/work/ctxbin/qwen3vl-4b-vit-w8a16/info.json
"""
import argparse
import json
from pathlib import Path

import numpy as np

EDGE = 512
N_PATCH, N_FEAT = 1024, 1536
RAW_BYTES = N_PATCH * N_FEAT * 2          # uint16
TENSOR = "pixel_values"
# Gate on how FAR values fall outside the calibrated range, in LSBs -- never on
# how MANY do. Those are very different failures and only one matters.
#
# Saturated pixels are the common case, not the exception: AIMET calibrated the
# input range from real photographs whose maximum landed a hair under the +1.0
# the processor emits for pure white, so the representable max is +0.99998 and
# EVERY pure-white pixel clips by exactly one LSB -- a 1.5e-05 error. A plain
# white background makes that most of the image (78% on this repo's own test
# card) while being completely harmless. A count-based check fails there and
# teaches you to ignore it.
#
# Real miscalibration -- an image whose statistics the ViT never saw -- shows up
# as values many LSBs outside the range, which is what this bounds.
CLIP_EXCESS_LSB_MAX = 8.0


def load_encoding(path):
    """(scale, offset, bw) for pixel_values, from a ctx-bin info.json or a
    model.encodings. Both are supported because the ctx-bin is the artifact that
    actually ships, but it only exists after the build; model.encodings is
    available earlier."""
    d = json.loads(Path(path).read_text())

    if "info" in d and "graphs" in d.get("info", {}):          # ctx-bin info.json
        for g in d["info"]["graphs"]:
            for t in g["info"]["graphInputs"]:
                c = t["info"]
                if c["name"] != TENSOR:
                    continue
                dt = c["dataType"]
                assert dt == "QNN_DATATYPE_UFIXED_POINT_16", (
                    f"{TENSOR} is {dt}, not UFIXED_POINT_16 -- a stock Genie "
                    "pipeline cannot drive this graph (NOTES-genie-pipeline D)")
                # qnn-context-binary-utility 2.48.40 names this member
                # "scaleOffset"; other releases have emitted
                # "scaleOffsetEncoding". Accept either -- but never fall through
                # to a default, because a silently wrong pixel scale is
                # indistinguishable from a bad image until it reaches silicon.
                qp = c["quantizeParams"]
                q = qp.get("scaleOffset") or qp.get("scaleOffsetEncoding")
                assert q, f"{TENSOR}: no scale/offset in quantizeParams {qp!r}"
                return float(q["scale"]), int(q["offset"]), 16
        raise SystemExit(f"{TENSOR} not found among graph inputs in {path}")

    for e in d["activation_encodings"]:                        # model.encodings
        if e.get("name") == TENSOR:
            assert e["bw"] == 16 and e["dtype"] == "INT", e
            return float(e["scale"][0]), int(e["offset"][0]), int(e["bw"])
    raise SystemExit(f"{TENSOR} not found in {path}")


def preprocess(model_dir, image_path):
    """-> float32 pixel_values [1024, 1536] exactly as the exporter traced it."""
    from PIL import Image
    from transformers import AutoProcessor
    img = Image.open(image_path).convert("RGB").resize((EDGE, EDGE), Image.BICUBIC)
    proc = AutoProcessor.from_pretrained(
        model_dir, min_pixels=EDGE * EDGE, max_pixels=EDGE * EDGE)
    out = proc.image_processor(images=img, return_tensors="np")
    pv, grid = out["pixel_values"], out["image_grid_thw"]
    assert tuple(pv.shape) == (N_PATCH, N_FEAT), (
        f"pixel_values {pv.shape} != {(N_PATCH, N_FEAT)} -- the ViT graph is static")
    assert grid.tolist() == [[1, 32, 32]], (
        f"grid {grid.tolist()} != [[1,32,32]]; vision-param in the image-encoder "
        "config declares 32x32 PATCHES and nothing cross-checks it at runtime")
    assert np.isfinite(pv).all(), "non-finite pixel values from the processor"
    return pv.astype(np.float32), grid


def quantize(pv, scale, offset, bw):
    """real -> stored, per the QNN/AIMET convention real = (q + offset) * scale."""
    qmax = (1 << bw) - 1
    raw = np.rint(pv.astype(np.float64) / scale) - offset
    q = np.clip(raw, 0, qmax).astype(np.uint16)
    # How far outside the representable range anything fell, in LSBs. This is
    # the number that decides whether the image is in-domain; the count is only
    # reported for context.
    excess = float(np.maximum(np.maximum(raw - qmax, -raw), 0.0).max())
    clipped = int(((raw < 0) | (raw > qmax)).sum())
    # Round-trip through the encoding so the reported error is what the DEVICE
    # will actually see, not what a float pipeline would.
    deq = (q.astype(np.float64) + offset) * scale
    return q, clipped, excess, float(np.abs(deq - pv).max())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--image", required=True)
    ap.add_argument("--out", required=True, help="output .raw path")
    ap.add_argument("--encodings", required=True,
                    help="ctx-bin info.json (preferred) or AIMET model.encodings")
    args = ap.parse_args()

    scale, offset, bw = load_encoding(args.encodings)
    lo, hi = offset * scale, ((1 << bw) - 1 + offset) * scale
    print(f"  {TENSOR} encoding: bw={bw} scale={scale:.9e} offset={offset} "
          f"-> representable [{lo:+.5f}, {hi:+.5f}]")

    pv, grid = preprocess(args.model, args.image)
    print(f"  pixel_values fp32 range [{pv.min():+.5f}, {pv.max():+.5f}] "
          f"shape {pv.shape}")

    q, clipped, excess, err = quantize(pv, scale, offset, bw)
    frac = clipped / q.size
    print(f"  clipped {clipped}/{q.size} ({frac:.3%}) by at most {excess:.2f} LSB, "
          f"max round-trip error {err:.3e}")
    assert excess <= CLIP_EXCESS_LSB_MAX, (
        f"pixels fall up to {excess:.1f} LSB outside the graph's calibrated "
        f"input range [{lo:+.5f}, {hi:+.5f}] -- this image's statistics are "
        "outside what the ViT was calibrated on, and the device would clamp "
        "them. Recalibrate with images like this one.")

    out = Path(args.out)
    out.write_bytes(q.tobytes())
    got = out.stat().st_size
    assert got == RAW_BYTES, f"{got} bytes != {RAW_BYTES}"
    out.with_suffix(".json").write_text(json.dumps(
        {"shape": [N_PATCH, N_FEAT], "dtype": "uint16", "bytes": RAW_BYTES,
         "grid_thw": grid.tolist(), "edge": EDGE,
         "encoding": {"scale": scale, "offset": offset, "bitwidth": bw},
         "clipped": clipped, "clip_excess_lsb": excess,
         "max_roundtrip_err": err}, indent=1) + "\n")
    print(f"{args.out}: {got} bytes, grid {grid.tolist()}")


if __name__ == "__main__":
    main()
