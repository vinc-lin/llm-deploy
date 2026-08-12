#!/usr/bin/env python
"""Extract the token-embedding table as an external Genie LUT (ufixed8).

Usage:
  $PY_DEPLOY scripts/export/extract_embed_lut.py \
      --model $LLMDEPLOY_DATA/models/Qwen3-VL-4B-Instruct \
      --out   $LLMDEPLOY_DATA/work/lut/qwen3vl-4b

The text graph takes `inputs_embeds`, not `input_ids`, so that the runtime can
splice visual features into the sequence.  Genie therefore does the token
lookup itself, host-side, from a raw LUT file: qualla::LUT memcpy's
`n_embd * bitWidth/8` bytes starting at `token_id * n_embd * bitWidth/8`
(examples/Genie/Genie/src/qualla/encoders/text-encoders/LUT.cpp).  Hence a flat,
vocab-major, row-contiguous dump with no header of any kind.

Quantization convention -- QNN scale/offset, NOT the ONNX/PyTorch zero_point
convention.  Established from SDK source, not assumed:

  include/QNN/QnnTypes.h:512 (Qnn_ScaleOffset_t)
      float_value = (quantized_value + offset) * scale
  lib/python/.../converters/common/utils/translation_utils.py:220,226
      quantize:   q = rint(clip(x, min, max) / scale - offset)
      dequantize: x = scale * (q + offset)
  lib/python/.../converters/relay/passes/pattern_match/tflite_dequantize.py:235
      "QNN: scale*(q+offset) / offset need to be negative of zero_point here"

So `offset` is a NEGATIVE integer equal to -zero_point, and the representable
range is [scale*offset, scale*(offset + 255)].  This is why the SDK's own
reference VLM config (examples/Genie/configs/glm-4v/glm-4v.json) carries
"offset": -129 -- 129 is the zero-point, and the table spans roughly
[-0.2045, +0.1998].  Reading that -129 as `scale * (q - offset)` would place
the whole table in the positive half-line, which is nonsense for embeddings.

The `size` field of the Genie embedding block is the hidden dim, not the vocab:
Dialog.cpp:190 maps `dialog.embedding.size` -> `context.n-embd`, and LUT.cpp
strides by `_ctx->n_embd()`.  For Qwen3-VL-4B that is 2560.

Reads the tensor straight out of the safetensors shard in row slices; the model
is never instantiated, so peak RSS stays in the low hundreds of MB rather than
the ~18 GB a full load would cost.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from safetensors import safe_open

EMBED_KEY = "model.language_model.embed_tokens.weight"
ROW_CHUNK = 4096
QMAX = 255  # ufixed8: 2**8 - 1 steps, per qnn_backend_base.py:370
ABS_HIST_BINS = 1 << 20


def resolve_key(model_dir):
    """Find the embedding tensor's shard, refusing to guess if the name moved."""
    index_path = model_dir / "model.safetensors.index.json"
    assert index_path.is_file(), f"no safetensors index at {index_path}"
    weight_map = json.loads(index_path.read_text())["weight_map"]
    if EMBED_KEY not in weight_map:
        cand = [k for k in weight_map if "embed_tokens" in k or "embed" in k.lower()]
        raise AssertionError(
            f"{EMBED_KEY!r} not in {index_path}; embedding-like keys present: {cand}"
        )
    return EMBED_KEY, model_dir / weight_map[EMBED_KEY]


def chunks(n_rows):
    for start in range(0, n_rows, ROW_CHUNK):
        yield start, min(start + ROW_CHUNK, n_rows)


def scan_range(sl, n_rows):
    """Pass 1: global min/max/mean/std and per-row max|w|, in fp32."""
    fmin, fmax = np.inf, -np.inf
    total, sq_total = 0.0, 0.0
    row_absmax = np.empty(n_rows, dtype=np.float64)
    for lo, hi in chunks(n_rows):
        w = sl[lo:hi].float().numpy()
        fmin = min(fmin, float(w.min()))
        fmax = max(fmax, float(w.max()))
        w64 = w.astype(np.float64)
        total += float(w64.sum())
        sq_total += float((w64 * w64).sum())
        row_absmax[lo:hi] = np.abs(w64).max(axis=1)
    n = float(n_rows) * sl.get_shape()[1]
    mean = total / n
    std = float(np.sqrt(max(sq_total / n - mean * mean, 0.0)))
    return fmin, fmax, mean, std, row_absmax


def derive_scale_offset(fmin, fmax):
    """Asymmetric ufixed8 scale/offset in the QNN convention (offset <= 0).

    scale = range/255 and offset = -round(-fmin/scale) is the textbook choice,
    but rounding the zero-point to an integer can leave one end of the range
    just outside the representable window, which shows up as clipping error on
    exactly the largest-magnitude values.  Widening scale by the (sub-0.5%)
    amount needed to re-cover both ends keeps max|error| at a clean 0.5*scale.
    """
    assert fmin < 0.0 < fmax, f"unexpected one-sided embedding range [{fmin}, {fmax}]"
    scale = (fmax - fmin) / QMAX
    zero_point = int(min(QMAX, max(0, round(-fmin / scale))))
    if zero_point > 0:
        scale = max(scale, -fmin / zero_point)
    if zero_point < QMAX:
        scale = max(scale, fmax / (QMAX - zero_point))
    return float(scale), -zero_point


def quantize(w, scale, offset):
    """q = rint(w/scale - offset), clipped -- translation_utils.quantize_params."""
    return np.clip(np.rint(w / scale - offset), 0, QMAX)


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--model", required=True, type=Path, help="HF checkpoint dir")
    ap.add_argument("--out", required=True, type=Path, help="output directory")
    args = ap.parse_args()

    key, shard = resolve_key(args.model)
    print(f"embedding tensor: {key}")
    print(f"  shard: {shard}")

    with safe_open(str(shard), framework="pt") as f:
        sl = f.get_slice(key)
        n_rows, n_embd = sl.get_shape()
        print(f"  shape: [{n_rows}, {n_embd}]  dtype: {sl.get_dtype()}")

        cfg = json.loads((args.model / "config.json").read_text())
        text_cfg = cfg.get("text_config", cfg)
        assert n_embd == text_cfg["hidden_size"], (
            f"row stride {n_embd} != config hidden_size {text_cfg['hidden_size']}; "
            "the Genie `size` field is n_embd, so a mismatch means the wrong tensor"
        )
        assert n_rows == text_cfg["vocab_size"], (
            f"{n_rows} rows != config vocab_size {text_cfg['vocab_size']}"
        )

        fmin, fmax, mean, std, row_absmax = scan_range(sl, n_rows)
        scale, offset = derive_scale_offset(fmin, fmax)

        args.out.mkdir(parents=True, exist_ok=True)
        lut_path = args.out / "embedding_int8_lut.bin"

        max_err, abs_err_total = 0.0, 0.0
        min_cos16, max_err16 = 2.0, 0.0
        min_cos_row_scale = 2.0
        zero_rows = 0
        row_cos = np.full(n_rows, np.nan, dtype=np.float64)
        row_norm = np.zeros(n_rows, dtype=np.float64)
        q_hist = np.zeros(QMAX + 1, dtype=np.int64)
        abs_hist = np.zeros(ABS_HIST_BINS, dtype=np.int64)
        abs_hi = float(max(-fmin, fmax))

        # ufixed16 with the same asymmetric fit, for the "is 8 bits enough" call.
        q16max = (1 << 16) - 1
        scale16 = (fmax - fmin) / q16max
        zp16 = int(min(q16max, max(0, round(-fmin / scale16))))
        if zp16 > 0:
            scale16 = max(scale16, -fmin / zp16)
        if zp16 < q16max:
            scale16 = max(scale16, fmax / (q16max - zp16))
        offset16 = -zp16

        with open(lut_path, "wb") as out:
            for lo, hi in chunks(n_rows):
                w = sl[lo:hi].float().numpy().astype(np.float64)
                q = quantize(w, scale, offset)
                out.write(q.astype(np.uint8).tobytes())

                deq = scale * (q + offset)
                assert np.isfinite(deq).all(), f"non-finite dequant in rows {lo}:{hi}"
                err = np.abs(deq - w)
                max_err = max(max_err, float(err.max()))
                abs_err_total += float(err.sum())
                q_hist += np.bincount(
                    q.astype(np.int64).ravel(), minlength=QMAX + 1
                )
                abs_hist += np.histogram(
                    np.abs(w), bins=ABS_HIST_BINS, range=(0.0, abs_hi)
                )[0]

                n_w = np.linalg.norm(w, axis=1)
                n_d = np.linalg.norm(deq, axis=1)
                row_norm[lo:hi] = n_w
                live = n_w > 0
                zero_rows += int((~live).sum())
                if live.any():
                    idx = lo + np.flatnonzero(live)
                    row_cos[idx] = (w * deq).sum(axis=1)[live] / (n_w[live] * n_d[live])

                    q16 = np.clip(np.rint(w / scale16 - offset16), 0, q16max)
                    d16 = scale16 * (q16 + offset16)
                    max_err16 = max(max_err16, float(np.abs(d16 - w).max()))
                    n16 = np.linalg.norm(d16, axis=1)
                    c16 = (w * d16).sum(axis=1)[live] / (n_w[live] * n16[live])
                    min_cos16 = min(min_cos16, float(c16.min()))

                    # Per-row scale/offset (informational: the Genie LUT format
                    # carries one global quant-param, so this is not shippable
                    # as-is -- it only sizes the headroom a change would buy).
                    rmin = w.min(axis=1, keepdims=True)
                    rmax = w.max(axis=1, keepdims=True)
                    rscale = np.maximum((rmax - rmin) / QMAX, 1e-30)
                    rzp = np.clip(np.rint(-rmin / rscale), 0, QMAX)
                    qr = np.clip(np.rint(w / rscale + rzp), 0, QMAX)
                    dr = rscale * (qr - rzp)
                    nr = np.linalg.norm(dr, axis=1)
                    ok = live & (nr > 0)
                    if ok.any():
                        cr = (w * dr).sum(axis=1)[ok] / (n_w[ok] * nr[ok])
                        min_cos_row_scale = min(min_cos_row_scale, float(cr.min()))

    n_vals = n_rows * n_embd
    size_bytes = lut_path.stat().st_size
    assert size_bytes == n_vals, (
        f"{lut_path} is {size_bytes} bytes, expected {n_vals} "
        f"({n_rows} x {n_embd} uint8)"
    )

    deq_min = scale * (0 + offset)
    deq_max = scale * (QMAX + offset)
    assert np.isfinite([deq_min, deq_max, scale]).all(), (
        f"non-finite dequantized table bounds: scale={scale} "
        f"range=[{deq_min}, {deq_max}]"
    )

    params = {
        "version": 1,
        "type": "lut",
        "lut-path": lut_path.name,
        "size": int(n_embd),
        "datatype": "ufixed8",
        "quant-param": {"scale": scale, "offset": int(offset)},
        "n-vocab": int(n_rows),
        "source-key": key,
        "source-dtype": "bfloat16",
        "dequant-formula": "float_value = scale * (q + offset)",
        "bytes": int(size_bytes),
    }
    params_path = args.out / "embedding_lut_params.json"
    params_path.write_text(json.dumps(params, indent=2) + "\n")

    # ---- report -------------------------------------------------------------
    abs_cdf = np.cumsum(abs_hist)

    def pct(p):
        """p-th percentile of |w|, read off the accumulated histogram."""
        i = int(np.searchsorted(abs_cdf, p / 100.0 * n_vals))
        return (i + 1) / ABS_HIST_BINS * abs_hi

    print(f"\nwrote {lut_path} ({size_bytes:,} bytes)")
    print(f"wrote {params_path}")
    print("\nquantization (ufixed8, single global scale/offset)")
    print(f"  scale                    : {scale!r}")
    print(f"  offset                   : {offset}   (= -zero_point)")
    print(f"  representable range      : [{deq_min:.6f}, {deq_max:.6f}]")
    print(f"  max abs dequant error    : {max_err:.6e}   ({max_err / scale:.3f} steps)")
    print(f"  mean abs dequant error   : {abs_err_total / n_vals:.6e}")
    live_cos = row_cos[~np.isnan(row_cos)]
    min_cos = float(live_cos.min())
    min_cos_row = int(np.nanargmin(row_cos))
    print(f"  min per-row cosine sim   : {min_cos:.6f}   (token {min_cos_row})")
    print(f"  all-zero rows (skipped)  : {zero_rows}")

    print("\nper-row cosine distribution (original vs dequantized)")
    for p, lbl in ((0.0, "min"), (0.1, "p0.1"), (1.0, "p1"), (50.0, "median")):
        print(f"  {lbl:<24s} : {np.percentile(live_cos, p):.6f}")
    for thr in (0.9999, 0.999, 0.998):
        n_bad = int((live_cos < thr).sum())
        print(
            f"  rows below {thr:<13g} : {n_bad} ({n_bad / live_cos.size * 100:.3f}%)"
        )
    worst = np.argsort(row_cos)[:5]
    print(f"  worst-5 token ids        : {worst.tolist()}")
    print(f"  their row L2 norms       : {[round(float(row_norm[i]), 4) for i in worst]}")
    print(f"  median row L2 norm       : {float(np.median(row_norm)):.4f}")

    print("\ntable distribution")
    print(f"  min / max                : {fmin:.6f} / {fmax:.6f}")
    print(f"  mean / std               : {mean:.6e} / {std:.6e}")
    for p in (50.0, 90.0, 99.0, 99.9, 99.99, 99.999):
        print(f"  p{p:<8g} of |w|        : {pct(p):.6f}   ({pct(p) / scale:.1f} steps)")

    used = int((q_hist > 0).sum())
    order = np.sort(q_hist)[::-1]
    cum = np.cumsum(order) / n_vals
    n_50 = int(np.searchsorted(cum, 0.50) + 1)
    n_90 = int(np.searchsorted(cum, 0.90) + 1)
    n_99 = int(np.searchsorted(cum, 0.99) + 1)
    print("\nbin occupancy (256 available)")
    print(f"  distinct bins used       : {used}")
    print(f"  bins holding 50% of vals : {n_50}")
    print(f"  bins holding 90% of vals : {n_90}")
    print(f"  bins holding 99% of vals : {n_99}")
    print(f"  vals in top-1 bin        : {order[0] / n_vals * 100:.2f}%")

    top = np.argsort(row_absmax)[::-1]
    print("\noutlier rows (by max |w| in the row)")
    print(f"  top-10 token ids         : {top[:10].tolist()}")
    print(f"  their max |w|            : {[round(float(row_absmax[i]), 4) for i in top[:10]]}")
    for k in (1, 10, 100, 1000):
        capped = float(row_absmax[top[k]])
        print(
            f"  drop top {k:<5d} rows     : |w| cap {capped:.6f} -> "
            f"scale {2 * capped / QMAX:.3e} "
            f"({scale / (2 * capped / QMAX):.2f}x finer)"
        )

    print("\nalternatives (not shipped; sizing the headroom only)")
    print(
        f"  ufixed16 global          : max err {max_err16:.6e}, "
        f"min row cosine {min_cos16:.6f}, {2 * n_vals:,} bytes"
    )
    print(f"  ufixed8 per-row scale    : min row cosine {min_cos_row_scale:.6f}")


if __name__ == "__main__":
    sys.exit(main())
