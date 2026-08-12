#!/usr/bin/env python
"""Verify the embedding LUT byte-for-byte at the offsets Genie actually reads.

qualla::LUT does no parsing: it mmaps the file (LUT.cpp:73) and memcpy's
`n_embd * bitWidth/8` bytes from `token_id * n_embd * bitWidth/8`
(LUT.cpp:93-100). There is no header, no shape, no dtype tag -- a file with the
right size and the wrong stride, byte order, row order or element type is
indistinguishable from a correct one until the model talks nonsense on device.

So this gate reads the file the way the runtime does -- raw offsets computed
from the params JSON, not from numpy's idea of the array -- and compares each
row against the checkpoint. It checks the tokens whose corruption would be
hardest to notice as well as the obvious ends of the table: the image/vision
markers (a wrong row there breaks only multimodal prompts), the pad token
(wrong -> every padded prefill window is subtly poisoned), token 0, and the
last row (catches an off-by-one in the vocab dimension).

Run:
  $PY_DEPLOY scripts/validate/parity_embed_lut.py \
      --model $LLMDEPLOY_DATA/models/Qwen3-VL-4B-Instruct \
      --lut   $LLMDEPLOY_DATA/work/lut/qwen3vl-4b
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
from safetensors import safe_open

EMBED_KEY = "model.language_model.embed_tokens.weight"
NP_DTYPE = {"float32": "<f4", "ufixed8": "<u1", "ufixed16": "<u2"}
# tokens whose corruption is quietest, plus the table's ends
NAMED = {0: "token 0", 151643: "bos", 151645: "eos/pad (qualla pad-token)",
         151652: "vision_start", 151653: "vision_end", 151655: "image_pad",
         151656: "video_pad"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, type=Path)
    ap.add_argument("--lut", required=True, type=Path, help="dir with the params JSON")
    ap.add_argument("--samples", type=int, default=64, help="extra random rows")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    params = json.loads((args.lut / "embedding_lut_params.json").read_text())
    dt = params["datatype"]
    n_embd, n_vocab = params["size"], params["n-vocab"]
    ebytes = params["element-bytes"]
    scale = params["quant-param"]["scale"]
    offset = params["quant-param"]["offset"]
    lut_path = args.lut / params["lut-path"]
    stride = n_embd * ebytes

    print(f"params : {dt}, n_embd={n_embd}, n_vocab={n_vocab}, "
          f"element-bytes={ebytes}")
    print(f"stride : {stride} bytes/row  (LUT.cpp: token_id * n_embd * bitWidth/8)")
    print(f"file   : {lut_path}")

    size = lut_path.stat().st_size
    expect = n_vocab * stride
    assert size == expect, f"{lut_path} is {size} bytes, expected {expect}"
    assert size == params["bytes"], (
        f"params says {params['bytes']} bytes, file is {size} -- params and .bin "
        "disagree, which is exactly the mismatch the atomic write exists to prevent")
    print(f"size   : {size:,} bytes == {n_vocab} x {stride} OK")

    idx = json.loads((args.model / "model.safetensors.index.json").read_text())
    shard = args.model / idx["weight_map"][EMBED_KEY]

    rng = np.random.default_rng(args.seed)
    ids = sorted(set(list(NAMED) + [n_vocab - 1, n_vocab - 2]
                     + rng.integers(0, n_vocab, args.samples).tolist()))

    blob = np.memmap(lut_path, dtype=np.uint8, mode="r")
    worst, worst_tok = 0.0, -1
    with safe_open(str(shard), framework="pt") as f:
        sl = f.get_slice(EMBED_KEY)
        for tok in ids:
            # read exactly the bytes the runtime would memcpy for this token
            raw = np.asarray(blob[tok * stride:(tok + 1) * stride])
            row = raw.view(NP_DTYPE[dt])
            assert row.shape == (n_embd,), f"token {tok}: got {row.shape}"
            deq = row.astype(np.float64)
            if dt != "float32":
                deq = scale * (deq + offset)
            ref = sl[tok:tok + 1].float().numpy()[0].astype(np.float64)
            assert np.isfinite(deq).all(), f"token {tok}: non-finite LUT row"
            err = float(np.abs(deq - ref).max())
            if err > worst:
                worst, worst_tok = err, tok
            tol = 0.0 if dt == "float32" else 0.5 * scale * (1 + 1e-6)
            assert err <= tol, (
                f"token {tok} ({NAMED.get(tok, 'random')}): max|LUT - checkpoint| "
                f"= {err:.3e} > {tol:.3e}")

    print(f"rows   : {len(ids)} checked ({len(NAMED) + 2} named + random)")
    for tok in sorted(NAMED):
        print(f"         {tok:>6d}  {NAMED[tok]}")
    kind = "bit-exact" if dt == "float32" else f"<= 0.5*scale ({0.5 * scale:.3e})"
    print(f"worst  : {worst:.3e} at token {worst_tok}  [{kind}]")
    print("PASS: LUT rows match the checkpoint at the runtime's own byte offsets")


if __name__ == "__main__":
    sys.exit(main())
