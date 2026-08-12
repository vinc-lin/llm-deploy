#!/usr/bin/env python
"""Export the Qwen3-VL vision tower to a static-shape ONNX graph.

Usage:
  $PY_DEPLOY scripts/export/export_qwen3vl_vit.py \
      --model $LLMDEPLOY_DATA/models/Qwen3-VL-4B-Instruct \
      --out   $LLMDEPLOY_DATA/work/onnx/qwen3vl-4b-vit [--parity-check]

Emits vit.onnx: pixel_values [1024,1536] -> image_features [256,2560]
plus deepstack_visual_embed_{0,1,2} [256,2560].

FP16 happens later at qairt-converter (--float_bitwidth 16); this graph is
all-FP32, matching the text pipeline's convention.

eager attention is mandatory: torch parity for the wrapper was established
against the eager path, and an sdpa-loaded model traces a different attention
subgraph than the one validated.

The graph MUST have exactly one input. The attention path derives per-window
lengths from cu_seqlens via a host-side .tolist(); at trace time that folds to
a single no-op torch.split. Any extra graph input means the fold did not
happen -- hence the hard assertions below rather than eyeballed prints.
"""
import argparse
import sys
from pathlib import Path

import onnx
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from modeling_vit_export import (  # noqa: E402
    ExportQwen3VLViT,
    make_pixel_values,
    output_names,
)

OPSET = 17


def dims_of(t):
    """Dim list; symbolic dims come back as their dim_param string, unset as 0.

    Returns None for a tensor carrying no shape at all (unknown rank), which is
    a strictly worse form of non-static than an unset dim and must not be
    confused with the empty dim list of a rank-0 scalar.
    """
    if not t.type.tensor_type.HasField("shape"):
        return None
    out = []
    for d in t.type.tensor_type.shape.dim:
        out.append(d.dim_param if d.WhichOneof("value") == "dim_param" else d.dim_value)
    return out


def is_static(dims):
    """True only if rank is known and every dim is a positive literal.

    `all([])` is True, so unknown rank has to be rejected explicitly or it
    sails through the very check meant to catch it. A rank-0 scalar also has
    an empty dim list but IS static, hence None rather than [] as the sentinel.
    """
    return dims is not None and all(isinstance(x, int) and x > 0 for x in dims)


def fmt_dims(dims):
    return "<unknown rank>" if dims is None else str(dims)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--parity-check", action="store_true",
                    help="compare wrapper outputs vs HF before exporting")
    args = ap.parse_args()

    from transformers import Qwen3VLForConditionalGeneration
    hf = Qwen3VLForConditionalGeneration.from_pretrained(
        args.model, dtype=torch.float32, attn_implementation="eager"
    ).eval()
    vision = hf.model.visual

    pixel_values, grid_thw = make_pixel_values(args.model, seed=0)
    print(f"grid_thw={grid_thw.tolist()} pixel_values={tuple(pixel_values.shape)}")

    model = ExportQwen3VLViT(vision, grid_thw).eval()

    if args.parity_check:
        with torch.no_grad():
            ref = vision(pixel_values, grid_thw)
            ours = model(pixel_values)
        d = (ours[0] - ref.pooler_output).abs().max().item()
        print(f"wrapper-vs-HF max|d image_features| = {d:.3e}")
        assert d < 1e-4, "wrapper does not match HF vision forward"

    names_out = output_names(len(vision.deepstack_visual_indexes))
    out_path = Path(args.out) / "vit.onnx"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with torch.no_grad():
        torch.onnx.export(
            model,
            (pixel_values,),
            str(out_path),
            input_names=["pixel_values"],
            output_names=names_out,
            opset_version=OPSET,
            do_constant_folding=True,
            dynamo=False,
        )

    onnx.checker.check_model(str(out_path))
    m = onnx.load(str(out_path), load_external_data=False)
    print(f"  {out_path}: {len(m.graph.node)} nodes, opset {OPSET}")

    ins = [(t.name, dims_of(t)) for t in m.graph.input]
    outs = [(t.name, dims_of(t)) for t in m.graph.output]
    for name, dims in ins:
        print(f"  IN  {name}: {fmt_dims(dims)}")
    for name, dims in outs:
        print(f"  OUT {name}: {fmt_dims(dims)}")

    assert [n for n, _ in ins] == ["pixel_values"], (
        f"graph must have exactly one input 'pixel_values', got {[n for n, _ in ins]} "
        "-- constant folding of the cu_seqlens/position path did not happen"
    )
    assert [n for n, _ in outs] == names_out, (
        f"unexpected outputs {[n for n, _ in outs]}, expected {names_out}"
    )
    bad = [(n, fmt_dims(d)) for n, d in ins + outs if not is_static(d)]
    assert not bad, (
        f"non-static shapes {bad} -- a constant did not fold; the graph is "
        "dynamic and unusable for the static-shape QNN pipeline"
    )


if __name__ == "__main__":
    main()
