#!/usr/bin/env python
"""Gate 4: static contract lint of the ViT ctx-bin against Genie's expectations.

Genie's image model (nsp-image-model.hpp) keys off exact tensor NAMES:
  input  "pixel_values"   -> LayerType::INPUT
  output "image_features" -> LayerType::OUTPUT (the built-in default; the runtime
                             only overrides it for outputs literally named
                             "vision_embedding" or "cross_attention_states")
A name mismatch is not a load error -- it is silent wrong behaviour on device,
so it is checked here rather than discovered later.

Everything is keyed by NAME, never by position: the ctx-bin declares
image_features LAST among the four outputs, and that order is not part of the
contract.

Also asserts FP16 for every tensor. This build exists to ship FP16; a silent
FP32 fallback would double the artifact and change device behaviour, and no
other gate looks at the ctx-bin's dtypes.

Run:
  $PY_DEPLOY scripts/validate/lint_vit_contract.py \
      --info $LLMDEPLOY_DATA/work/ctxbin/qwen3vl-4b-vit-fp16/info.json
"""
import argparse
import json
import sys
from pathlib import Path

DTYPE = "QNN_DATATYPE_FLOAT_16"

EXPECT_IN = {"pixel_values": [1024, 1536]}
EXPECT_OUT = {
    "image_features": [256, 2560],
    "deepstack_visual_embed_0": [256, 2560],
    "deepstack_visual_embed_1": [256, 2560],
    "deepstack_visual_embed_2": [256, 2560],
}


def unwrap(node, what):
    """qnn-context-binary-utility wraps every record as {"version":..,"info":{..}}.

    Tolerate an already-unwrapped record so the lint fails on the contract, not
    on a utility version bump.
    """
    if not isinstance(node, dict):
        raise SystemExit(f"FAIL: malformed info.json: {what} is not an object")
    return node["info"] if isinstance(node.get("info"), dict) else node


def tensors(graph, key, count_key, problems, kind):
    """name -> tensor info dict, flagging duplicates and a lying count field."""
    raw = graph.get(key) or []
    declared = graph.get(count_key)
    if declared is not None and declared != len(raw):
        problems.append(
            f"{count_key}={declared} but {key} lists {len(raw)} tensor(s)")
    out = {}
    for i, t in enumerate(raw):
        info = unwrap(t, f"{key}[{i}]")
        name = info.get("name")
        if name is None:
            problems.append(f"{kind} #{i} has no name")
            continue
        if name in out:
            problems.append(f"duplicate {kind} name {name!r}")
        out[name] = info
    return out


def report(kind, got):
    for name in sorted(got):
        info = got[name]
        print(f"  {kind:6s} {name:28s} {info.get('dimensions')} "
              f"{info.get('dataType')} {info.get('type')}")


def check(kind, got, expect, problems):
    for name in sorted(set(expect) - set(got)):
        problems.append(f"missing {kind}: {name!r} (Genie binds by exact name)")
    for name in sorted(set(got) - set(expect)):
        problems.append(f"unexpected extra {kind}: {name!r}")
    for name in sorted(set(got) & set(expect)):
        info = got[name]
        dims = info.get("dimensions")
        if dims != expect[name]:
            problems.append(f"{kind} {name!r}: dims {dims} != {expect[name]}")
        rank = info.get("rank")
        if isinstance(dims, list) and rank is not None and rank != len(dims):
            problems.append(f"{kind} {name!r}: rank {rank} != len(dimensions) {len(dims)}")
        dtype = info.get("dataType")
        if dtype != DTYPE:
            problems.append(f"{kind} {name!r}: dataType {dtype} != {DTYPE}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--info", required=True,
                    help="info.json from qnn-context-binary-utility")
    args = ap.parse_args()

    doc = json.loads(Path(args.info).read_text())
    info = unwrap(doc, "top level")

    problems = []
    graphs = info.get("graphs") or []
    num = info.get("numGraphs")
    if num is not None and num != len(graphs):
        problems.append(f"numGraphs={num} but graphs lists {len(graphs)}")
    if len(graphs) != 1:
        print(f"FAIL: expected exactly 1 graph, found {len(graphs)}")
        for p in problems:
            print(f"FAIL: {p}")
        sys.exit(1)

    graph = unwrap(graphs[0], "graphs[0]")
    print(f"graph: {graph.get('graphName')!r}")

    ins = tensors(graph, "graphInputs", "numGraphInputs", problems, "input")
    outs = tensors(graph, "graphOutputs", "numGraphOutputs", problems, "output")
    report("in", ins)
    report("out", outs)

    check("input", ins, EXPECT_IN, problems)
    check("output", outs, EXPECT_OUT, problems)

    if problems:
        print(f"\n{len(problems)} contract violation(s):")
        for p in problems:
            print(f"  FAIL: {p}")
        sys.exit(1)

    print(f"PASS: ctx-bin matches the Genie image-encoder contract "
          f"({len(ins)} input, {len(outs)} outputs, all {DTYPE})")


if __name__ == "__main__":
    main()
