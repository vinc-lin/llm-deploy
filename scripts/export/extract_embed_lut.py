#!/usr/bin/env python
"""Extract the token-embedding table as an external Genie LUT.

Usage:
  $PY_DEPLOY scripts/export/extract_embed_lut.py \
      --model $LLMDEPLOY_DATA/models/Qwen3-VL-4B-Instruct \
      --out   $LLMDEPLOY_DATA/work/lut/qwen3vl-4b [--datatype ufixed16]

The text graph takes `inputs_embeds`, not `input_ids`, so that the runtime can
splice visual features into the sequence.  Genie therefore does the token
lookup itself, host-side, from a raw LUT file: qualla::LUT memcpy's
`n_embd * bitWidth/8` bytes starting at `token_id * n_embd * bitWidth/8`
(examples/Genie/Genie/src/qualla/encoders/text-encoders/LUT.cpp:93-100).  Hence
a flat, vocab-major, row-contiguous, little-endian dump with no header.

--------------------------------------------------------------------------
WHY float32 IS THE DEFAULT -- a fixed-point LUT SILENTLY NO-OPS against our
graph, which is a worse failure than any precision argument
--------------------------------------------------------------------------
This default used to be ufixed16, justified by "the activation path is all
16-bit".  That argument is superseded: it reasoned about precision, while the
binding constraint turned out to be the runtime's dtype DISPATCH.

Genie converts the LUT into the graph input's encoding at runtime; the raw
copy is only a fast path when the two already agree (dialog.cpp:631/677/844/
1109, all guarded identically):

    requantScale  = lutScale / inputScale                        (dialog.cpp:432)
    requantOffset = requantScale * lutOffset - inputOffset
    if (lutDataType == inputDataType && requantScale == 1 && requantOffset == 0)
        decoderInput = std::move(encoderOutput);   // raw copy
    else
        requantEmbedding(...);                     // convert

So the LUT's scale need NOT match the graph input's -- Genie rescales.  What
Genie cannot do is change dtype CLASS: Dialog::requantEmbedding
(dialog.cpp:485-551) enumerates fixed->fixed pairs only --

    ufixed4  -> {ufixed8, ufixed16}   (anything else: throws)
    ufixed8  -> {ufixed8, ufixed16}
    ufixed16 -> {ufixed8, ufixed16}
    sfixed8  -> {sfixed8, sfixed16, ufixed16}
    sfixed16 -> {sfixed8, sfixed16}

-- and there is NO float16/float32 destination branch.  Outside the ufixed4
case the if/else-if chain does not throw on an unhandled pair: it falls
through and leaves the destination buffer UNTOUCHED.

Our text tower's `inputs_embeds` is FLOAT_16 in the ctx-bin.  AIMET emits no
encoding for it -- the HTP per-channel config leaves the first RMSNorm's INPUT
quantizer disabled, so no graph I/O tensor carries an encoding at all
(verified 2026-08-12) -- and qairt-converter --float_bitwidth 16 therefore
leaves it float.  A ufixed16 LUT against a FLOAT_16 input is exactly the
unhandled pair: no exception, no log, an untouched embedding buffer.

float32 sidesteps the whole dispatch.  dialog.cpp:684 hands the fp32 bytes
straight through, and nsp-model.cpp:3120 quantizeInput() converts per token
into whatever the graph input actually is -- FLOAT_16, UFIXED_8 and UFIXED_16
are all handled there.  It also makes the visual-feature range a non-issue:
with a float32 LUT, Dialog::inputTensorQuantParam (dialog.cpp:444-460) reports
FLOAT_32/1.0/0 to the image pipeline, so ViT features (measured [-5.9, +5.1],
deepstack to +16.0 -- far outside the embedding table's [-0.198, +0.244]) are
written unquantized instead of being squeezed into a table-derived scale.

The cost is 1.56 GB instead of 778 MB, and LUT.cpp mmaps the file
(LUT.cpp:73), so that is page cache, not committed RSS.  Correctness over size
is this project's stated priority.

`--datatype ufixed16` / `ufixed8` are kept as runnable, measured alternatives
for a tight device-memory budget -- the script reports every width on each run
so the A/B is always in front of you.  If you switch to one, you MUST also
give `inputs_embeds` a fixed-point encoding in the ctx-bin, or you get the
silent no-op above; scripts/validate/lint_embedding_dtype.py exists to make
that mistake impossible to ship.

--------------------------------------------------------------------------
Quantization convention -- QNN scale/offset, NOT the ONNX/PyTorch zero_point
convention.  Established from SDK source, not assumed:

  include/QNN/QnnTypes.h:512 (Qnn_ScaleOffset_t)
      float_value = (quantized_value + offset) * scale
  lib/python/.../converters/common/utils/translation_utils.py:220,226
      quantize:   q = rint(clip(x, min, max) / scale - offset)
      dequantize: x = scale * (q + offset)
  lib/python/.../converters/relay/passes/pattern_match/tflite_dequantize.py:235
      "QNN: scale*(q+offset) / offset need to be negative of zero_point here"
  examples/Genie/Genie/src/qualla/dialog.cpp:432 (runtime requant)
      requantScale  = lutScale / inputScale
      requantOffset = requantScale * lutOffset - inputOffset
      -- which is exactly the algebra of lutScale*(q+lutOffset)
         == inputScale*(q_in+inputOffset), and only closes under `+ offset`.

So `offset` is a NEGATIVE integer equal to -zero_point, and the representable
range is [scale*offset, scale*(offset + qmax)].  This is why glm-4v.json
carries "offset": -129 -- 129 is the zero-point, and that table spans roughly
[-0.2045, +0.1998].  Reading that -129 as `scale * (q - offset)` would place
the whole table in the positive half-line, which is nonsense for embeddings.

The `size` field of the Genie embedding block is the hidden dim, not the
vocab: Dialog.cpp:190 maps `dialog.embedding.size` -> `context.n-embd`, and
LUT.cpp strides by `_ctx->n_embd()`.  For Qwen3-VL-4B that is 2560.

Reads the tensor straight out of the safetensors shard in row slices; the
model is never instantiated, so peak RSS stays in the low hundreds of MB
rather than the ~18 GB a full load would cost.
"""
import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
from safetensors import safe_open

EMBED_KEY = "model.language_model.embed_tokens.weight"
ROW_CHUNK = 4096
ABS_HIST_BINS = 1 << 20

# Genie datatype name -> (numpy dtype, bit width).  Little-endian is explicit:
# LUT.cpp memcpy's raw bytes straight to the HTP, which is little-endian, and
# we must not inherit the host's byte order by accident.
DTYPES = {
    "float32": ("<f4", 32),
    "ufixed8": ("<u1", 8),
    "ufixed16": ("<u2", 16),
}


def resolve_key(model_dir):
    """Find the embedding tensor's shard, refusing to guess if the name moved."""
    index_path = model_dir / "model.safetensors.index.json"
    assert index_path.is_file(), f"no safetensors index at {index_path}"
    weight_map = json.loads(index_path.read_text())["weight_map"]
    if EMBED_KEY not in weight_map:
        cand = [k for k in weight_map if "embed" in k.lower()]
        raise AssertionError(
            f"{EMBED_KEY!r} not in {index_path}; embedding-like keys present: {cand}"
        )
    return EMBED_KEY, model_dir / weight_map[EMBED_KEY]


def chunks(n_rows):
    for start in range(0, n_rows, ROW_CHUNK):
        yield start, min(start + ROW_CHUNK, n_rows)


def scan_range(sl, n_rows, n_embd):
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
    n = float(n_rows) * n_embd
    mean = total / n
    std = float(np.sqrt(max(sq_total / n - mean * mean, 0.0)))
    return fmin, fmax, mean, std, row_absmax


def derive_scale_offset(fmin, fmax, qmax):
    """Asymmetric scale/offset in the QNN convention (offset <= 0).

    scale = range/qmax with offset = -round(-fmin/scale) is the textbook
    choice, but rounding the zero-point to an integer can leave one end of the
    range just outside the representable window, which shows up as clipping
    error on exactly the largest-magnitude values.  Widening scale by the
    (sub-1/qmax) amount needed to re-cover both ends keeps max|error| at a
    clean 0.5*scale, i.e. pure rounding with no clipping at all.
    """
    assert fmin < 0.0 < fmax, f"unexpected one-sided embedding range [{fmin}, {fmax}]"
    scale = (fmax - fmin) / qmax
    zero_point = int(min(qmax, max(0, round(-fmin / scale))))
    if zero_point > 0:
        scale = max(scale, -fmin / zero_point)
    if zero_point < qmax:
        scale = max(scale, fmax / (qmax - zero_point))
    return float(scale), -zero_point


def quantize(w, scale, offset, qmax):
    """q = rint(w/scale - offset), clipped -- translation_utils.quantize_params."""
    return np.clip(np.rint(w / scale - offset), 0, qmax)


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--model", required=True, type=Path, help="HF checkpoint dir")
    ap.add_argument("--out", required=True, type=Path, help="output directory")
    ap.add_argument(
        "--datatype",
        default="float32",
        choices=sorted(DTYPES),
        help="LUT element type (default: float32 -- see module docstring; a "
        "fixed-point LUT against our FLOAT_16 `inputs_embeds` is an unhandled "
        "pair in Dialog::requantEmbedding and silently writes nothing)",
    )
    args = ap.parse_args()

    np_dtype, bits = DTYPES[args.datatype]
    is_float = args.datatype == "float32"
    qmax = 0 if is_float else (1 << bits) - 1
    nbytes = bits // 8
    # every OTHER supported width, measured on the same data so the A/B for a
    # tight device-memory budget is never stale
    alts = [n for n in ("ufixed16", "ufixed8") if n != args.datatype]

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

        fmin, fmax, mean, std, row_absmax = scan_range(sl, n_rows, n_embd)
        # float32 ships the values verbatim: scale/offset are identity and are
        # never consulted (dialog.cpp:684 takes the float32 fast path).
        scale, offset = (1.0, 0) if is_float else derive_scale_offset(fmin, fmax, qmax)
        alt_enc = {n: derive_scale_offset(fmin, fmax, (1 << DTYPES[n][1]) - 1)
                   for n in alts}

        args.out.mkdir(parents=True, exist_ok=True)
        lut_path = args.out / f"embedding_{args.datatype}_lut.bin"
        # Write-then-rename: this loop takes minutes and has been killed
        # mid-flight before. A truncated .bin at the real path would sit there
        # looking valid (LUT.cpp mmaps and strides blindly), and a params file
        # describing a different datatype than the .bin beside it is exactly the
        # silent mismatch this pipeline must never ship. os.replace() is atomic
        # within a filesystem, so a kill leaves EITHER the old pair or the new
        # pair -- never a mixture.
        tmp_path = args.out / f".{lut_path.name}.partial"

        max_err, abs_err_total = 0.0, 0.0
        alt_max_err = {n: 0.0 for n in alts}
        alt_min_cos = {n: 2.0 for n in alts}
        min_cos_row_scale = 2.0
        zero_rows = 0
        row_cos = np.full(n_rows, np.nan, dtype=np.float64)
        row_norm = np.zeros(n_rows, dtype=np.float64)
        q_hist = None if is_float else np.zeros(qmax + 1, dtype=np.int64)
        abs_hist = np.zeros(ABS_HIST_BINS, dtype=np.int64)
        abs_hi = float(max(-fmin, fmax))

        with open(tmp_path, "wb") as out:
            for lo, hi in chunks(n_rows):
                if lo % (ROW_CHUNK * 8) == 0:
                    print(f"  rows {lo}/{n_rows} ({lo / n_rows * 100:.0f}%)", flush=True)
                w = sl[lo:hi].float().numpy().astype(np.float64)
                if is_float:
                    # source is bfloat16, so fp32 is exact: deq == w bit for bit
                    stored = w.astype(np_dtype)
                    out.write(stored.tobytes())
                    deq = stored.astype(np.float64)
                else:
                    q = quantize(w, scale, offset, qmax)
                    out.write(q.astype(np_dtype).tobytes())
                    deq = scale * (q + offset)
                    q_hist += np.bincount(q.astype(np.int64).ravel(), minlength=qmax + 1)

                assert np.isfinite(deq).all(), f"non-finite dequant in rows {lo}:{hi}"
                err = np.abs(deq - w)
                max_err = max(max_err, float(err.max()))
                abs_err_total += float(err.sum())
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

                    for n in alts:
                        a_scale, a_offset = alt_enc[n]
                        a_qmax = (1 << DTYPES[n][1]) - 1
                        qa = quantize(w, a_scale, a_offset, a_qmax)
                        da = a_scale * (qa + a_offset)
                        alt_max_err[n] = max(alt_max_err[n], float(np.abs(da - w).max()))
                        na = np.linalg.norm(da, axis=1)
                        ca = (w * da).sum(axis=1)[live] / (n_w[live] * na[live])
                        alt_min_cos[n] = min(alt_min_cos[n], float(ca.min()))

                    # Per-row scale/offset (informational: the Genie embedding
                    # block carries one global quant-param, so this is not
                    # expressible in the format -- it only sizes the headroom).
                    if not is_float:
                        rmin = w.min(axis=1, keepdims=True)
                        rmax = w.max(axis=1, keepdims=True)
                        rscale = np.maximum((rmax - rmin) / qmax, 1e-30)
                        rzp = np.clip(np.rint(-rmin / rscale), 0, qmax)
                        qr = np.clip(np.rint(w / rscale + rzp), 0, qmax)
                        dr = rscale * (qr - rzp)
                        nr = np.linalg.norm(dr, axis=1)
                        ok = live & (nr > 0)
                        if ok.any():
                            cr = (w * dr).sum(axis=1)[ok] / (n_w[ok] * nr[ok])
                            min_cos_row_scale = min(min_cos_row_scale, float(cr.min()))

    n_vals = n_rows * n_embd
    expect_bytes = n_vals * nbytes
    size_bytes = tmp_path.stat().st_size
    assert size_bytes == expect_bytes, (
        f"{tmp_path} is {size_bytes} bytes, expected {expect_bytes} "
        f"({n_rows} x {n_embd} x {nbytes}B {args.datatype}) -- partial file kept "
        "for inspection; nothing was published"
    )

    deq_min, deq_max = (fmin, fmax) if is_float else (scale * offset,
                                                      scale * (qmax + offset))
    assert np.isfinite([deq_min, deq_max, scale]).all(), (
        f"non-finite dequantized table bounds: scale={scale} "
        f"range=[{deq_min}, {deq_max}]"
    )
    if is_float:
        # bfloat16 -> float32 is exact; anything else means we wrote the wrong bytes
        assert max_err == 0.0, f"float32 LUT is not bit-exact: max err {max_err}"

    params = {
        "version": 1,
        "type": "lut",
        "lut-path": lut_path.name,
        "size": int(n_embd),
        "datatype": args.datatype,
        # identity for float32: Genie takes the float32 fast path (dialog.cpp:684)
        # and never consults these, but downstream tooling reads the same schema
        "quant-param": {"scale": scale, "offset": int(offset)},
        "n-vocab": int(n_rows),
        "source-key": key,
        "source-dtype": "bfloat16",
        "dequant-formula": ("float_value = q (raw float32; quant-param is identity "
                            "and unused)" if is_float
                            else "float_value = scale * (q + offset)"),
        "element-bytes": nbytes,
        "byte-order": "little-endian",
        "bytes": int(size_bytes),
    }
    # ---- publish: both files, or neither -----------------------------------
    # .bin first, params second. A kill in the (microsecond) gap leaves the
    # PREVIOUS params still pointing at the PREVIOUS .bin, which is a
    # self-consistent pair; the reverse order would leave params naming a file
    # that does not exist yet.
    params_path = args.out / "embedding_lut_params.json"
    params_tmp = args.out / ".embedding_lut_params.json.partial"
    params_tmp.write_text(json.dumps(params, indent=2) + "\n")
    os.replace(tmp_path, lut_path)
    os.replace(params_tmp, params_path)

    stale = sorted(p.name for p in args.out.glob("embedding_*_lut.bin")
                   if p.name != lut_path.name)
    if stale:
        print(f"\nNOTE: {args.out} also holds {stale}, which "
              f"{'is' if len(stale) == 1 else 'are'} no longer referenced by "
              "embedding_lut_params.json. Delete by hand once you are sure no "
              "bundle needs it.")

    # ---- report -------------------------------------------------------------
    abs_cdf = np.cumsum(abs_hist)

    def pct(p):
        """p-th percentile of |w|, read off the accumulated histogram."""
        i = int(np.searchsorted(abs_cdf, p / 100.0 * n_vals))
        return (i + 1) / ABS_HIST_BINS * abs_hi

    print(f"\nwrote {lut_path} ({size_bytes:,} bytes)")
    print(f"wrote {params_path}")
    if is_float:
        print(f"\nstorage ({args.datatype}, verbatim -- no quantization)")
        print(f"  value range              : [{deq_min:.6f}, {deq_max:.6f}]")
        print(f"  max abs error vs source  : {max_err:.6e}   (bit-exact from bfloat16)")
    else:
        print(f"\nquantization ({args.datatype}, single global scale/offset)")
        print(f"  scale                    : {scale!r}")
        print(f"  offset                   : {offset}   (= -zero_point)")
        print(f"  representable range      : [{deq_min:.6f}, {deq_max:.6f}]")
        print(f"  max abs dequant error    : {max_err:.6e}   ({max_err / scale:.3f} steps)")
        print(f"  mean abs dequant error   : {abs_err_total / n_vals:.6e}")

    live_cos = row_cos[~np.isnan(row_cos)]
    min_cos = float(live_cos.min())
    min_cos_row = int(np.nanargmin(row_cos))
    print(f"  min per-row cosine sim   : {min_cos:.9f}   (token {min_cos_row})")
    print(f"  max per-row (1 - cosine) : {1.0 - min_cos:.6e}")
    print(f"  all-zero rows (skipped)  : {zero_rows}")

    print("\nper-row cosine distribution (original vs dequantized)")
    for p, lbl in ((0.0, "min"), (0.1, "p0.1"), (1.0, "p1"), (50.0, "median")):
        v = float(np.percentile(live_cos, p))
        print(f"  {lbl:<24s} : {v:.9f}   (1-cos {1.0 - v:.3e})")
    for thr in (0.9999, 0.999, 0.998):
        n_bad = int((live_cos < thr).sum())
        print(f"  rows below {thr:<13g} : {n_bad} ({n_bad / live_cos.size * 100:.3f}%)")
    worst = np.argsort(row_cos)[:5]
    print(f"  worst-5 token ids        : {worst.tolist()}")
    print(f"  their row L2 norms       : {[round(float(row_norm[i]), 4) for i in worst]}")
    print(f"  median row L2 norm       : {float(np.median(row_norm)):.4f}")

    print("\ntable distribution")
    print(f"  min / max                : {fmin:.6f} / {fmax:.6f}")
    print(f"  mean / std               : {mean:.6e} / {std:.6e}")
    for p in (50.0, 90.0, 99.0, 99.9, 99.99, 99.999):
        steps = "" if is_float else f"   ({pct(p) / scale:.1f} steps)"
        print(f"  p{p:<8g} of |w|        : {pct(p):.6f}{steps}")

    if not is_float:
        used = int((q_hist > 0).sum())
        order = np.sort(q_hist)[::-1]
        cum = np.cumsum(order) / n_vals
        print(f"\nbin occupancy ({qmax + 1} available)")
        print(f"  distinct bins used       : {used} ({used / (qmax + 1) * 100:.1f}%)")
        for frac in (0.50, 0.90, 0.99):
            print(
                f"  bins holding {frac * 100:.0f}% of vals: "
                f"{int(np.searchsorted(cum, frac) + 1)}"
            )
        print(f"  vals in top-1 bin        : {order[0] / n_vals * 100:.2f}%")

    top = np.argsort(row_absmax)[::-1]
    print("\noutlier rows (by max |w| in the row)")
    print(f"  top-10 token ids         : {top[:10].tolist()}")
    print(f"  their max |w|            : {[round(float(row_absmax[i]), 4) for i in top[:10]]}")
    if not is_float:
        for k in (1, 10, 100, 1000):
            capped = float(row_absmax[top[k]])
            print(
                f"  drop top {k:<5d} rows     : |w| cap {capped:.6f} -> "
                f"scale {2 * capped / qmax:.3e} "
                f"({scale / (2 * capped / qmax):.2f}x finer)"
            )

    print("\nalternative widths (not shipped by this run; --datatype to switch)")
    for n in alts:
        print(
            f"  {n:<24s} : max err {alt_max_err[n]:.6e}, "
            f"min row cosine {alt_min_cos[n]:.9f}, "
            f"{n_vals * DTYPES[n][1] // 8:,} bytes"
        )
    if not is_float:
        print(f"  per-row scale @ {args.datatype:<9s}: min row cosine {min_cos_row_scale:.9f}")
    else:
        print("  NOTE: switching to a fixed-point LUT also requires a fixed-point")
        print("        `inputs_embeds` encoding in the ctx-bin -- see the module")
        print("        docstring and scripts/validate/lint_embedding_dtype.py")


if __name__ == "__main__":
    sys.exit(main())
