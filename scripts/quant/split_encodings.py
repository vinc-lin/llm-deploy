#!/usr/bin/env python
"""Split a whole-tower AIMET encodings file into per-chunk files.

Why this is sound rather than a shortcut: the chunked export is *bit-identical*
to the unsplit tower (scripts/validate/parity_vl_text_split.py -- logits and all
72 KV outputs at max|d| 0.000e+00). Splitting the graph moves no arithmetic, so
every tensor's observed range is unchanged and the calibrated encodings remain
exactly as valid. Re-quantizing per chunk would additionally require chaining
calibration (chunk 1 must be fed chunk 0's real hidden states) and would then
produce a DIFFERENT range for the boundary tensor on each side.

That last point is the real motivation. Genie treats identically-named tensors
across splits as the same buffer and requires identical quantization params
(validateModel check 4). Deriving both chunks from ONE calibration run makes
that automatic instead of something to reconcile afterwards.

Naming: the shipped path cuts AIMET's ONNX with onnx.utils.extract_model, which
PRESERVES the original global names, so entries keep their layer indices. Pass
--renumber only for a chunk graph that was re-exported with its own module list
(local 0..n-1); getting this backwards matches 0/198 params and silently
converts the whole chunk as FP16.

Usage:
  split_encodings.py --encodings <whole.encodings> --split-at 18 --layers 36 \
      --out-dir <dir>
writes chunk0.encodings and chunk1.encodings.
"""
import argparse
import json
import re
from pathlib import Path

PARAM_RE = re.compile(r"^layers\.(\d+)\.")
# Activations come in two spellings -- '/layers/layers.3/attn/...' and
# '/layers.3/mlp/...' -- and this pattern deliberately matches the '/layers.N/'
# that is common to both. Matching only the doubled form silently drops one
# entry per layer into the "layer-agnostic" bucket, which lands every layer's
# mlp output in the last chunk: the first chunk then converts those activations
# as FP16 and the last chunk carries names its graph does not have. Both
# failures are invisible until measured on hardware.
ACT_RE = re.compile(r"/layers\.(\d+)/")


def layer_of(name):
    """Global layer index a named tensor belongs to, or None if it belongs to
    no layer (final norm, lm_head, graph I/O)."""
    if not isinstance(name, str):
        return None
    m = PARAM_RE.search(name) or ACT_RE.search(name)
    return int(m.group(1)) if m else None


def renumber(name, delta):
    name = PARAM_RE.sub(lambda m: f"layers.{int(m.group(1)) + delta}.", name)
    return ACT_RE.sub(lambda m: f"/layers/layers.{int(m.group(1)) + delta}/", name)


def name_of(entry):
    return entry.get("name") if isinstance(entry, dict) else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--encodings", required=True)
    ap.add_argument("--split-at", type=int, required=True)
    ap.add_argument("--layers", type=int, required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--renumber", action="store_true",
                    help="renumber a later chunk's layers to local 0..n-1. Only "
                         "correct when the chunk graph was RE-EXPORTED with its "
                         "own module list. The shipped path cuts AIMET's ONNX "
                         "with onnx.utils.extract_model, which preserves the "
                         "original global names, so leave this off there.")
    args = ap.parse_args()

    split, L = args.split_at, args.layers
    if not 0 < split < L:
        raise SystemExit(f"--split-at {split} must be inside (0, {L})")

    data = json.loads(Path(args.encodings).read_text())
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    chunks = [{}, {}]
    for c in chunks:
        for k, v in data.items():
            if k not in ("activation_encodings", "param_encodings"):
                c[k] = v

    counts = [[0, 0], [0, 0]]
    unassigned = []
    for si, section in enumerate(("activation_encodings", "param_encodings")):
        entries = data.get(section, [])
        if not isinstance(entries, list):
            raise SystemExit(f"{section} is {type(entries).__name__}, expected list")
        buckets = [[], []]
        for e in entries:
            n = name_of(e)
            li = layer_of(n)
            if li is None:
                # Final norm / lm_head / anything layer-agnostic belongs to the
                # LAST chunk, which is the chunk that owns them.
                unassigned.append(n)
                buckets[1].append(e)
                continue
            if li >= L:
                raise SystemExit(f"layer index {li} >= --layers {L} in {n!r}")
            if li < split:
                buckets[0].append(e)
            else:
                if args.renumber:
                    e = dict(e)
                    e["name"] = renumber(n, -split)
                buckets[1].append(e)
        for ci in (0, 1):
            chunks[ci][section] = buckets[ci]
            counts[ci][si] = len(buckets[ci])

    # Every entry must land in exactly one chunk: a dropped weight encoding
    # silently becomes FP16 in the DLC, which is indistinguishable from a
    # successful build until it is measured on hardware nobody has.
    for si, section in enumerate(("activation_encodings", "param_encodings")):
        total = len(data.get(section, []))
        got = counts[0][si] + counts[1][si]
        if got != total:
            raise SystemExit(f"{section}: {got} entries after split, expected {total}")

    for ci in (0, 1):
        p = out / f"chunk{ci}.encodings"
        p.write_text(json.dumps(chunks[ci]))
        print(f"  {p.name}: {counts[ci][0]} activation, {counts[ci][1]} param")

    # Renumbering must produce a contiguous local 0..n-1 in each chunk, or the
    # names will not match the chunk's ONNX and the encodings silently no-op.
    for ci in (0, 1):
        seen = {layer_of(name_of(e))
                for e in chunks[ci]["param_encodings"]} - {None}
        want = (set(range(split)) if ci == 0
                else set(range(L - split)) if args.renumber
                else set(range(split, L)))
        if seen != want:
            raise SystemExit(f"chunk{ci} layers {sorted(seen)[:5]}... != "
                             f"expected {min(want)}..{max(want)}")

    if unassigned:
        print(f"  layer-agnostic entries -> chunk1: {len(unassigned)} "
              f"({unassigned[:3]})")
    print(f"PASS: split {L} layers at {split} "
          f"({counts[0][1]} + {counts[1][1]} param encodings)")


if __name__ == "__main__":
    main()
