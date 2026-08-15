#!/usr/bin/env python
"""Export the Qwen3-VL text tower to prefill.onnx + decode.onnx (QAIRT/Genie).

Usage:
  $PY_DEPLOY scripts/export/export_qwen3vl_text.py \
      --model $LLMDEPLOY_DATA/models/Qwen3-VL-4B-Instruct \
      --out   $LLMDEPLOY_DATA/work/onnx/qwen3vl-4b-text [--parity-check]

Same prefill/decode pair as export_qwen3.py -- same KV contract, same additive
rank-3 mask, same cos/sin inputs -- with the two VL-specific differences:

- embeddings-in: the first input is `inputs_embeds`, not input_ids. The NAME is
  load-bearing (qualla nsp-model.cpp:668 matches it literally and thereby sets
  InputType::EMBEDDINGS) and the rank-4 [1, 1, AR, H] shape is Genie's
  documented convention (nsp-graph.cpp:92). The runtime owns the token lookup
  because it must splice visual features in before the tower runs.
- deepstack: n extra [1, 1, AR, H] inputs added to the residual stream after
  decoder layers 0..n-1. The host zero-pads them to full width (a static graph
  cannot scatter onto visual positions only); all-zeros is exactly the
  text-only prompt.

position_ids_cos/sin are unchanged from the text pipeline: Genie computes MRoPE
host-side and feeds ordinary half-width cos/sin tables, so Qwen3-VL's
interleaved MRoPE is a config concern, not a graph one.

Prefill MUST emit all-position logits [1, AR, V] (logits_last_only=False for
BOTH graphs): qualla samples logits row n_process-1 of an all-position tensor
(nsp-model.cpp:3295), so a [1, 1, V] head passes load validation silently and
then reads past the end of a one-row buffer -- root cause of the 2026-08-11
device garbage.

Each graph is written to its OWN subdirectory. This tower is ~4.0 B fp32 params
(~16 GB per graph), far past the 2 GiB protobuf limit, so the TorchScript
exporter spills every initializer to a separate external-data file beside the
.onnx, named from an auto-generated `onnx::MatMul_<n>` counter. Those names are
not stable across graphs, so two graphs sharing one directory can silently
overwrite each other's weights.
"""
import argparse
import gc
import sys
from pathlib import Path

import onnx
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from modeling_export import ExportQwen3, causal_mask, rope_tables, rope_theta_of  # noqa: E402

OPSET = 17
PARITY_S = 16
PARITY_TOL = 5e-3


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


BOUNDARY = "last_hidden_states"


def io_spec(cfg, n_deepstack, seq, past_len, layer_range=None):
    """(name, shape) for every graph I/O, inputs in ExportQwen3.forward order.

    Single source of truth: the dummy trace inputs, the declared input_names
    and the post-export shape guard are all derived from this, so the three can
    never drift apart.

    `layer_range=(start, end)` describes one chunk of a split export. Two rules
    from docs/NOTES-genie-splits.md are load-bearing here:

    * KV tensors keep GLOBAL layer indices. Renumbering a later chunk's caches
      to 0..n-1 would collide with the first chunk's names, and Genie treats
      same-named tensors across splits as THE SAME BUFFER -- the two chunks'
      caches would silently alias.
    * The boundary tensor is rank-3 [1, S, H] on BOTH sides. `inputs_embeds` is
      rank-4 by Genie's convention, but the module returns rank 3, and Genie
      does not check that split boundary shapes agree -- so the two sides are
      made identical here rather than left to match by element count.
    """
    L, n_kv, D, H = (cfg.num_hidden_layers, cfg.num_key_value_heads,
                     cfg.head_dim, cfg.hidden_size)
    start, end = (0, L) if layer_range is None else layer_range
    is_first, is_last = start == 0, end == L

    ins = [
        ("inputs_embeds", [1, 1, seq, H]) if is_first else (BOUNDARY, [1, seq, H]),
        ("attention_mask", [1, seq, past_len + seq]),
        ("position_ids_cos", [1, seq, D // 2]),
        ("position_ids_sin", [1, seq, D // 2]),
    ]
    ins += [(f"deepstack_visual_embed_{i}", [1, 1, seq, H]) for i in range(n_deepstack)]
    if past_len:
        for i in range(start, end):
            # Genie contract: keys transposed [1, n_kv, D, P], values [1, n_kv, P, D]
            ins += [(f"past_key_{i}_in", [1, n_kv, D, past_len]),
                    (f"past_value_{i}_in", [1, n_kv, past_len, D])]
    outs = [("logits", [1, seq, cfg.vocab_size]) if is_last else (BOUNDARY, [1, seq, H])]
    for i in range(start, end):
        outs += [(f"past_key_{i}_out", [1, n_kv, D, seq]),
                 (f"past_value_{i}_out", [1, n_kv, seq, D])]
    return ins, outs


def export_graph(model, cfg, out_path, seq, past_len, n_deepstack, layer_range=None):
    ins, outs = io_spec(cfg, n_deepstack, seq, past_len, layer_range)
    # use_past only gates whether the past-KV arguments are read, so the same
    # weights serve both graphs; see the note in main() on why that matters.
    model.use_past = past_len > 0

    L = cfg.num_hidden_layers
    start, end = (0, L) if layer_range is None else layer_range

    cos, sin = rope_tables(torch.arange(past_len, past_len + seq),
                           cfg.head_dim, rope_theta_of(cfg))
    head = (torch.zeros(1, 1, seq, cfg.hidden_size) if start == 0
            else torch.zeros(1, seq, cfg.hidden_size))
    args = [head, causal_mask(seq, past_len + seq), cos, sin]
    args += [torch.zeros(1, 1, seq, cfg.hidden_size) for _ in range(n_deepstack)]
    if past_len:
        for _ in range(end - start):
            args += [torch.zeros(1, cfg.num_key_value_heads, cfg.head_dim, past_len),
                     torch.zeros(1, cfg.num_key_value_heads, past_len, cfg.head_dim)]
    assert [list(a.shape) for a in args] == [s for _, s in ins], (
        "trace inputs disagree with io_spec: "
        f"{[list(a.shape) for a in args]} != {[s for _, s in ins]}"
    )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with torch.no_grad():
        torch.onnx.export(
            model,
            tuple(args),
            str(out_path),
            input_names=[n for n, _ in ins],
            output_names=[n for n, _ in outs],
            opset_version=OPSET,
            do_constant_folding=True,
            dynamo=False,
        )


def verify_graph(out_path, ins, outs):
    """Static-shape guard: names, ranks and dims must be exactly as declared."""
    onnx.checker.check_model(str(out_path))
    m = onnx.load(str(out_path), load_external_data=False)
    got_in = [(t.name, dims_of(t)) for t in m.graph.input]
    got_out = [(t.name, dims_of(t)) for t in m.graph.output]

    print(f"{out_path}: {len(m.graph.node)} nodes, opset {OPSET}, "
          f"inputs={len(got_in)}, outputs={len(got_out)}")
    for name, dims in got_in:
        print(f"  IN  {name}: {fmt_dims(dims)}")
    for name, dims in got_out:
        print(f"  OUT {name}: {fmt_dims(dims)}")

    assert [n for n, _ in got_in] == [n for n, _ in ins], (
        f"input names/order differ from declared: {[n for n, _ in got_in]}"
    )
    assert [n for n, _ in got_out] == [n for n, _ in outs], (
        f"output names/order differ from declared: {[n for n, _ in got_out]}"
    )
    # unknown rank must be rejected explicitly -- is_static() handles the
    # `all([]) is True` hole, this loop must not reintroduce it
    bad = [(n, fmt_dims(d)) for n, d in got_in + got_out if not is_static(d)]
    assert not bad, (
        f"non-static shapes {bad} -- the graph is dynamic and unusable for the "
        "static-shape QNN pipeline"
    )
    wrong = [(n, fmt_dims(d), want) for (n, d), (_, want) in
             zip(got_in + got_out, ins + outs) if d != want]
    assert not wrong, f"shape mismatch (got, want): {wrong}"

    files = sorted(p for p in out_path.parent.iterdir() if p != out_path)
    total = out_path.stat().st_size + sum(p.stat().st_size for p in files)
    print(f"  {out_path.name} {out_path.stat().st_size:,} B + {len(files)} external "
          f"data files, {total:,} B total")


def parity_check(model, hf, cfg, n_deepstack):
    """Wrapper vs HF text tower on random embeds, rank-4 as exported."""
    model.use_past = False
    S, H = PARITY_S, cfg.hidden_size
    torch.manual_seed(0)
    embeds = torch.randn(1, 1, S, H) * 0.02
    deep = [torch.randn(1, 1, S, H) * 0.02 for _ in range(n_deepstack)]
    mask = causal_mask(S, S)
    cos, sin = rope_tables(torch.arange(S), cfg.head_dim, rope_theta_of(cfg))
    with torch.no_grad():
        ours = model(embeds, mask, cos, sin, *deep)[0]
        ref = hf.model.language_model(
            inputs_embeds=embeds[0],
            attention_mask=torch.ones(1, S, dtype=torch.long),
            deepstack_visual_embeds=[d[0, 0] for d in deep],
            visual_pos_masks=torch.ones(1, S, dtype=torch.bool),
        ).last_hidden_state
        ref = hf.lm_head(ref)
    assert torch.isfinite(ours).all(), "wrapper produced non-finite logits"
    assert ours.shape == ref.shape, f"{tuple(ours.shape)} != {tuple(ref.shape)}"
    d = (ours - ref).abs().max().item()
    print(f"wrapper-vs-HF max|dlogits| = {d:.3e}  shape={tuple(ours.shape)}")
    assert d < PARITY_TOL, "wrapper does not match HF forward"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--cl-prefill", type=int, default=128)
    ap.add_argument("--ctx", type=int, default=2048)
    ap.add_argument("--n-deepstack", type=int, default=3)
    ap.add_argument("--prefill-past", type=int, default=0,
                    help="past-KV length for the prefill graph (0 = the AR==CL "
                         "bertcache prefill; use --ctx for the past-KV prefill, "
                         "the only shape a split tower can load). Exports as "
                         "'prefillkv' so it sits beside the bertcache pair.")
    ap.add_argument("--split-at", type=int, default=0,
                    help="split the layer stack at this GLOBAL layer index, "
                         "emitting prefill_0/decode_0 (layers [0,N)) and "
                         "prefill_1/decode_1 (layers [N,L)). 0 = no split. "
                         "Required for 4B: one ctx-bin cannot hold the tower "
                         "(3.5 GiB per-graph serialization limit).")
    ap.add_argument("--parity-check", action="store_true",
                    help="compare wrapper logits vs HF forward before export")
    args = ap.parse_args()

    from transformers import Qwen3VLForConditionalGeneration
    hf = Qwen3VLForConditionalGeneration.from_pretrained(
        args.model, dtype=torch.float32, attn_implementation="eager"
    ).eval()
    cfg = hf.config.text_config
    out = Path(args.out)

    L = cfg.num_hidden_layers
    split = args.split_at
    if split and not 0 < split < L:
        raise SystemExit(f"--split-at {split} must be inside (0, {L})")
    chunks = [(0, split), (split, L)] if split else [(0, L)]

    # decode past length: window is ctx + cl_prefill - 1 so that after one
    # decode step total == ctx + cl_prefill (== ctx-bin max CL, e.g. 2048+128)
    past_len = args.ctx + args.cl_prefill - 1

    if args.parity_check and split:
        raise SystemExit(
            "--parity-check compares a whole-tower forward and does not apply to a "
            "split export; scripts/validate/parity_vl_text_split.py proves chunk "
            "equivalence against the unsplit tower instead")

    # Graph NAMES are the split wiring: Genie assigns split index by sorted name
    # within each (AR, CL) group (nsp-model.cpp:365-372), and a graph's name is
    # baked in at conversion time from the DLC basename. prefill_0 < prefill_1
    # and decode_0 < decode_1 sort into chunk order. See
    # docs/NOTES-genie-splits.md.
    # A prefill with past-KV. --prefill-past 0 keeps the AR==CL bertcache
    # prefill; --prefill-past <ctx> gives mask [1,AR,ctx+AR] and past <ctx>,
    # which is the ONLY prefill shape a SPLIT tower can load (REFERENCE.md
    # 3.6). This is the fp32 counterpart of the quantized rebuild, used by the
    # e2e gate's chain0b.
    p_past = args.prefill_past
    if p_past and p_past + args.cl_prefill != past_len + 1:
        raise SystemExit(
            f"--prefill-past {p_past} + --cl-prefill {args.cl_prefill} = "
            f"{p_past + args.cl_prefill}, but decode's window is {past_len + 1}. "
            "Both AR variants must land in the SAME cache group or Genie sees "
            "two context sizes.")

    todo = []
    for ci, (s, e) in enumerate(chunks):
        sfx = f"_{ci}" if split else ""
        # deepstack lands on GLOBAL layers 0..n-1, so only the first chunk takes it
        nd = args.n_deepstack if s == 0 else 0
        tag = "prefillkv" if p_past else "prefill"
        todo.append((ci, (s, e), nd, f"{tag}{sfx}", args.cl_prefill, p_past))
        todo.append((ci, (s, e), nd, f"decode{sfx}", 1, past_len))

    specs = []
    for ci, (s, e) in enumerate(chunks):
        # One wrapper per chunk, serving both its AR variants. At fp32 the HF
        # model is ~18 GB and a full wrapper another ~16 GB, so wrappers are
        # built and freed one at a time; use_past only gates reading the past-KV
        # args, so export_graph flips it per graph.
        print(f"building chunk {ci}: layers [{s}, {e})", flush=True)
        model = ExportQwen3.from_hf_vl_text(
            hf, use_past=True, logits_last_only=False,
            n_deepstack=args.n_deepstack if s == 0 else 0,
            layer_range=(s, e) if split else None)
        if args.parity_check:
            parity_check(model, hf, cfg, args.n_deepstack)
        for c, rng, nd, tag, seq, past in todo:
            if c != ci:
                continue
            print(f"exporting {tag} (S={seq}, past={past}, layers {rng})...", flush=True)
            # own subdirectory per graph: external data filenames are not
            # graph-stable and collide between graphs
            path = out / tag / f"{tag}.onnx"
            export_graph(model, cfg, path, seq, past, nd,
                         layer_range=rng if split else None)
            specs.append((tag, path, io_spec(cfg, nd, seq, past,
                                             rng if split else None)))
        del model
        gc.collect()

    # free the weights before onnx.checker: it reloads the external data
    del hf
    gc.collect()

    for tag, path, (ins, outs) in specs:
        verify_graph(path, ins, outs)


if __name__ == "__main__":
    main()
