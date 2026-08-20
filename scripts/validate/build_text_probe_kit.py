#!/usr/bin/env python
"""Build the v5 text-graph probe: known inputs + fp32 references for running the
shipped text ctx-bins under `qnn-net-run`, with no Genie involved.

WHY THIS EXISTS. Every numerical gate in this repo validates ONNX; none of them
executes the shipped `.bin`. After the 2026-08-18 session exactly one hop was
left untested -- ONNX -> DLC -> ctx-bin -> Genie's feed -- and the text tower
produces garbage on device while scoring 20/20 against HF on the host. This
probe splits that hop in two:

  * correct logits from qnn-net-run  => the ctx-bin and the converter are
    exonerated, and the fault is in how **Genie feeds** the tower;
  * wrong logits                     => the **converter / ctx-bin** is at
    fault, and a rebuild is justified -- aimed at something specific.

TWO CASES, AND THE SECOND ONE IS NOT OPTIONAL.

  decode1tok   one token, empty cache, through decode_0 -> decode_1.
               Deliberately the simplest thing that can fail. Note this case
               says NOTHING about rope: with a single token attending only to
               itself, RoPE rotates q and k by the same angle and the dot
               product is rotation-invariant, so the logits are identical at
               every position. (Measured, not assumed -- an earlier version of
               this script had a "position 7" case and it produced byte-equal
               output to position 0, which is exactly right and exactly
               useless.) So this case is a clean read on pure numerics.

  prefill4tok  four real tokens through prefill_0 -> prefill_1 with an empty
               cache. Here row i attends to rows 0..i at *different* positions,
               so rope and cross-token attention are genuinely exercised -- and
               these are the graphs the real 273-token prompt actually uses
               (three AR=128 calls), which the decode case never touches.

decode clean + prefill broken therefore points at rope/attention or the prefill
graphs specifically, and the pair localises far better than either alone.

In both cases the KV cache is empty, so all 72 past-KV inputs are zeros AND
fully masked -- their contents cannot affect the result, which is why the
device generates one shared zero file instead of us shipping 320 MB.

THE REFERENCE COMES FROM THE SAME ONNX THE DLCs WERE CONVERTED FROM
(`work/onnx/qwen3vl-4b-aimet-split`, cut at the layer seam by
`split_aimet_onnx.py`), never from HF and never from a re-export. Anything else
compares two different graphs and blames the wrong stage.

The feed contracts (mask layout, deepstack `_p` rename, rope tables) are copied
from parity_e2e_vl.PrefillKV/Decoder rather than re-derived -- a probe that
feeds the graph differently from the gate would produce a mismatch that means
nothing.

Because the per-shard ONNX lives on tank, so does this script:

  ssh tank
  cd ~/llm-deploy && source scripts/env.sh
  $PY_DEPLOY scripts/validate/build_text_probe_kit.py \
      --onnx-split $LLMDEPLOY_DATA/work/onnx/qwen3vl-4b-aimet-split \
      --lut        $LLMDEPLOY_DATA/work/lut/qwen3vl-4b \
      --out        $LLMDEPLOY_DATA/work/text_probe_v5

Expect ~10 GB RSS per shard (fp32 weights) and a few minutes per case.
"""
import argparse
import gc
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "export"))
from modeling_export import MASK_VALUE, rope_tables  # noqa: E402

ROPE_THETA = 5_000_000.0
DEEPSTACK = 3
# Ordinary in-vocab ids. Ids, not words: the comparison is numeric, so nothing
# here needs a tokenizer to be present on tank.
TOKENS = [3838, 374, 264, 1273]

CASES_V5 = [
    {"name": "decode1tok", "kind": "decode", "n_real": 1, "mask": "causal",
     "why": "1 token, empty cache -- rope cancels in self-attention, so this "
            "is a clean read on pure numerics"},
    {"name": "prefill4tok", "kind": "prefill", "n_real": 4, "mask": "causal",
     "why": "4 tokens, empty cache -- rope and cross-token attention are live, "
            "and these are the graphs the real prompt uses"},
]

# --- Test F ---------------------------------------------------------------
# Test E measured a UNIFORM 1.3896x gain on shard 0's boundary output, and the
# per-row follow-up localised it to ROW 0 ALONE (rows 1-3 sit at 0.93/1.00/0.99).
# The same 1.3896x appears in decode1tok, a different graph. What those two share
# is not the graph and not the position value: it is that the row ATTENDS ONLY TO
# ITSELF -- the attention-sink condition, under which this row carries massive
# activations (RMS 107.2 vs ~1-2 elsewhere; c4=5244 alone is 93.4% of its norm).
#
# Two candidate causes remain and they need different fixes:
#   (a) the CONDITION  -- self-attention / massive activations overflow or clamp
#                         somewhere, so any such row is amplified wherever it sits;
#   (b) the ROW INDEX  -- something specific to element 0 of the AR window
#                         (a tile edge, an offset bug), and the sink is incidental.
#
# These four cases separate them by changing ONLY the mask and where the real
# tokens sit in the AR window -- both already shipped graph inputs, so no rebuild,
# no new bytes, and the host reference is computed from the identical feed.
#
#            sink at row 0     no sink at row 0
#   row 0 :  fp_ctrl  (known 1.39x)   f1_row0ctx
#   row 4 :  f2_shift4                 --
#
# f2_shift4 is the sharp one: its row 4 performs a computation numerically
# IDENTICAL to fp_ctrl's row 0 (same token, same self-only mask, same rope
# position 0), differing only in which row of the tensor it occupies. The builder
# asserts that host-side equality, so any device difference between them is the
# index effect and nothing else.
CASES_F = [
    {"name": "f0_ctrl_dec", "kind": "decode", "n_real": 1, "mask": "causal",
     "why": "decode anchor -- must reproduce the measured 1.3896x, otherwise the "
            "device is not in the state Test E measured and nothing below is "
            "comparable",
     "asks": "is the device still in the Test E state?"},
    {"name": "fp_ctrl_pre", "kind": "prefill", "n_real": 4, "mask": "causal",
     "why": "prefill anchor -- the known pattern: row 0 at ~1.39x, rows 1-3 clean",
     "asks": "is the prefill row pattern still row-0-only?"},
    {"name": "f1_row0ctx", "kind": "prefill", "n_real": 4, "mask": "row0_full",
     "why": "row 0 is made NON-causal: it attends to all four tokens instead of "
            "itself alone. Row index still 0, sink condition removed",
     "asks": "does the gain survive when row 0 has real context?"},
    {"name": "f2_shift4", "kind": "prefill", "n_real": 4, "mask": "causal",
     "row_offset": 4,
     "why": "the same four tokens moved to rows 4-7 with rope positions 0-3; rows "
            "0-3 are masked padding. Row 4 now does exactly what fp_ctrl's row 0 "
            "does. Sink condition kept, row index moved",
     "asks": "does the gain follow the sink to row 4, or stay at row 0?"},
]

# --- realistic suite -------------------------------------------------------
# WHY THIS SUITE EXISTS. Every probe before it fed BARE TOKEN IDS -- decode1tok
# is token 3838 alone at position 0, prefill4tok is four content words. A
# production prompt never looks like that: it is chat-templated and begins
# <|im_start|>. Measured 2026-08-20, that difference is the whole result. The
# Test F probe's row 0 lands 1.64x outside its calibrated activation range and
# produces the 1.39x boundary gain the device measured; six realistic
# chat-templated windows put row 0 at gain 0.9990, worst 0.035 over rows 0-3,
# and the model's REAL attention sink turns out to sit at row 1 with RMS 220.3
# -- larger than the probe's synthetic row-0 sink at 107.2 -- and comes through
# at 1.0000. So the earlier probes reproduced a defect they had manufactured.
#
# These cases are built from the multimodal calibration/eval windows
# (vl_calib_build.py), which are chat-templated turns with real ViT features
# spliced onto the image-token positions -- i.e. exactly what qualla feeds. The
# EVAL split is held out from calibration and is preferred for that reason.
#
# DEEPSTACK IS ZEROED, deliberately, even though the windows carry real values.
# The shipped model runs with the deepstack inputs zero-filled
# (initializeUnconnectedInputs; the tower is built HF-minus-deepstack), so
# feeding the real features would make the probe LESS production-faithful, not
# more -- the opposite of this suite's whole purpose.
CASES_R = [
    {"name": "r0_text", "kind": "prefill", "mask": "window",
     "match": "The capital of France", "prefer_split": "eval",
     "why": "held-out chat-templated TEXT turn: row 0 is <|im_start|>, not a "
            "content word",
     "asks": "does the boundary hold on a realistic text prompt?"},
    {"name": "r1_image", "kind": "prefill", "mask": "window",
     "match": "img100", "prefer_split": "eval",
     "why": "held-out chat-templated IMAGE+text turn with real ViT features "
            "spliced in -- the actual production path",
     "asks": "does the boundary hold on a realistic image prompt?"},
    {"name": "r2_chunk0", "kind": "prefill", "mask": "window",
     "match": "chunk0[0:128]", "prefer_split": "calib",
     "why": "a FULL 128-row first chunk of a long image turn -- no padding at "
            "all, the densest realistic input available",
     "asks": "does the boundary hold with the AR window completely full?"},
]

SUITES = {"v5": CASES_V5, "f": CASES_F, "r": CASES_R}


def lut_row(lut_dir: Path, token: int) -> np.ndarray:
    """One embedding row, read the way the runtime reads it (raw byte offset),
    matching parity_embed_lut.py rather than numpy's view of the array."""
    p = json.loads((lut_dir / "embedding_lut_params.json").read_text())
    n, eb = p["size"], p["element-bytes"]
    assert p["datatype"] == "float32" and eb == 4, p
    with (lut_dir / p["lut-path"]).open("rb") as fh:
        fh.seek(token * n * eb)
        raw = fh.read(n * eb)
    assert len(raw) == n * eb, f"short read at token {token}"
    return np.frombuffer(raw, dtype="<f4").astype(np.float32)


def meta_of(sess):
    sh = {i.name: i.shape for i in sess.get_inputs()}
    names = set(sh)
    pk = next(n for n in sh if n.startswith("past_key_") and n.endswith("_in"))
    _, nkv, hd, past = sh[pk]
    idx = sorted(int(n.split("_")[2]) for n in sh
                 if n.startswith("past_key_") and n.endswith("_in"))
    total = sh["attention_mask"][-1]
    ar = sh["attention_mask"][-2] if len(sh["attention_mask"]) == 3 else 1
    deep = []
    for k in range(DEEPSTACK):
        c = sorted(x for x in names if x.startswith(f"deepstack_visual_embed_{k}"))
        if c:
            assert len(c) == 1, f"deepstack {k}: ambiguous {c}"
            deep.append(c[0])
    assert total == past + ar, f"mask {total} != past {past} + AR {ar}"
    return {"n_kv": nkv, "head_dim": hd, "past": past, "total": total,
            "ar": ar, "layer_idx": idx, "deep": deep, "shapes": sh}


def zeros_past(m):
    f = {}
    for i in m["layer_idx"]:
        f[f"past_key_{i}_in"] = np.zeros(
            (1, m["n_kv"], m["head_dim"], m["past"]), dtype=np.float32)
        f[f"past_value_{i}_in"] = np.zeros(
            (1, m["n_kv"], m["past"], m["head_dim"]), dtype=np.float32)
    return f


def build_mask_rope(m, case):
    """Mask + rope tables for an EMPTY cache, per parity_e2e_vl.

    decode  (AR==1): additive MASK_VALUE, 0.0 only at index PAST -- the new
                     token sits at the END in the concat layout, not at 0.
    prefill (AR>1) : row i sees the valid past span [0, nv) -- empty here, so
                     nothing -- plus the causal new span [PAST, PAST+i].
    Rows past n_real stay fully masked; their KV is never committed.

    Test F varies two things and nothing else:

      row_offset   where the real tokens sit in the AR window. The real rows
                   become [off, off+n_real); rope position i is assigned to row
                   off+i, so row off computes EXACTLY what row 0 computes in the
                   unshifted case -- same token, same self-only attention, same
                   rope angle -- at a different tensor row.
      mask=row0_full
                   the first real row additionally attends to every other real
                   row (non-causal). Not a valid LM step, but a valid graph
                   execution, and the host reference is computed from the same
                   mask -- which is the only thing that has to be true.

    Fully-masked rows are well defined here: MASK_VALUE is -100.0, not -inf, so
    softmax over an all-masked row is uniform rather than NaN.

    Returns (mask, cos, sin, real_rows) where real_rows are the AR indices that
    carry a token, in token order.
    """
    import torch
    ar, total, past, half = m["ar"], m["total"], m["past"], m["head_dim"] // 2
    if case.get("mask") == "window":
        # A prebuilt calibration/eval window. Its mask is the UNSPLIT causal
        # [1, AR, AR] the quantizer calibrated against; this graph is the past-KV
        # prefill and wants [1, AR, PAST+AR] with the past span fully masked.
        # Re-laying it here rather than regenerating keeps the probe feeding
        # exactly the window the encodings were fitted on.
        w = case["_win"]
        if ar != w["ar"]:
            raise SystemExit(f"{case['name']}: window AR {w['ar']} != graph AR {ar}")
        mask = np.full((1, ar, total), MASK_VALUE, dtype=np.float32)
        mask[0, :, past:] = np.where(w["mask"][0] >= 0, 0.0, MASK_VALUE)
        return (mask, w["cos"].astype(np.float32), w["sin"].astype(np.float32),
                list(range(w["n_valid"])))
    n_real = case["n_real"]
    if ar == 1:
        if case.get("row_offset") or case.get("mask", "causal") != "causal":
            raise SystemExit(f"{case['name']}: a decode graph has one row; "
                             "row_offset/mask variants need kind=prefill")
        mask = np.full((1, 1, total), MASK_VALUE, dtype=np.float32)
        mask[0, 0, past] = 0.0
        cos, sin = rope_tables(torch.arange(1), m["head_dim"], ROPE_THETA)
        return (mask, cos.numpy().astype(np.float32),
                sin.numpy().astype(np.float32), [0])

    off = case.get("row_offset", 0)
    kind = case.get("mask", "causal")
    if off + n_real > ar:
        raise SystemExit(f"{case['name']}: rows {off}..{off + n_real - 1} "
                         f"do not fit in AR={ar}")
    real_rows = [off + i for i in range(n_real)]

    mask = np.full((1, ar, total), MASK_VALUE, dtype=np.float32)
    for i in range(n_real):
        # causal over the real span only; no valid past in an empty cache
        mask[0, off + i, past + off:past + off + i + 1] = 0.0
    if kind == "row0_full":
        mask[0, off, past + off:past + off + n_real] = 0.0
    elif kind != "causal":
        raise SystemExit(f"{case['name']}: unknown mask {kind!r}")

    c, s = rope_tables(torch.arange(n_real), m["head_dim"], ROPE_THETA)
    cos = np.zeros((1, ar, half), dtype=np.float32)
    sin = np.zeros((1, ar, half), dtype=np.float32)
    cos[0, off:off + n_real] = c.numpy()[0]
    sin[0, off:off + n_real] = s.numpy()[0]
    return mask, cos, sin, real_rows


def run(onnx, feeds, out_name, threads, extra_outputs=()):
    """Run one shard. `extra_outputs` additionally fetches named graph outputs
    (the per-layer KV taps) and returns them in a dict."""
    import onnxruntime as ort
    so = ort.SessionOptions()
    if threads:
        so.intra_op_num_threads = threads
    sess = ort.InferenceSession(str(onnx), so, providers=["CPUExecutionProvider"])
    m = meta_of(sess)
    f = feeds(m)
    wanted = [out_name] + list(extra_outputs)
    have = {o.name for o in sess.get_outputs()}
    missing = [w for w in wanted if w not in have]
    if missing:
        raise SystemExit(f"{onnx}: no such graph output(s) {missing}")
    res = sess.run(wanted, f)
    out = np.asarray(res[0], dtype=np.float32)
    extra = {n: np.asarray(v, dtype=np.float32)
             for n, v in zip(wanted[1:], res[1:])}
    del sess, res
    gc.collect()
    return out, m, f, extra


# --- device input encoding -------------------------------------------------
# qnn-net-run is invoked with --use_native_input_files, so every .raw must hold
# the tensor's NATIVE bytes. Writing IEEE fp16 unconditionally was correct only
# while every input was FLOAT_16; once inputs_embeds became UFIXED_POINT_16 the
# same 2 bytes/element meant NO size error and NO warning, while the graph
# decoded fp16 bit patterns as quantized integers. Measured: cosine(intended,
# received) = -0.72, and a true 0.0 arrives as -11.65. That silently produced
# "the 4B ctx-bins are numerically wrong" in the 2026-08-15 v5 session.
#
# So the encoding is read from the ctx-bin that will actually execute, never
# assumed, and an unhandled dtype is a hard error rather than a default.
def tensor_specs(info_json: Path) -> dict:
    """{tensor name: (dtype, scale, offset)} for every graph input in a bin."""
    doc = json.loads(info_json.read_text())
    specs = {}

    def walk(o):
        if isinstance(o, dict):
            if "graphInputs" in o:
                for t in o["graphInputs"]:
                    ti = t.get("info", t)
                    so = (ti.get("quantizeParams") or {}).get("scaleOffset") or {}
                    specs[ti["name"]] = (ti.get("dataType"),
                                         so.get("scale"), so.get("offset"))
            for v in o.values():
                walk(v)
        elif isinstance(o, list):
            for v in o:
                walk(v)

    walk(doc)
    if not specs:
        raise SystemExit(f"no graphInputs found in {info_json}")
    return specs


def write_native(p: Path, a, name: str, specs: dict):
    """Write `a` as the bytes the graph declares for input `name`."""
    if name not in specs:
        raise SystemExit(f"{name!r} is not an input of the ctx-bin "
                         f"(have: {sorted(specs)[:8]}...) -- kit and bin disagree")
    dtype, scale, offset = specs[name]
    a = np.ascontiguousarray(a, dtype=np.float32)
    if dtype == "QNN_DATATYPE_FLOAT_16":
        raw = a.astype("<f2")
    elif dtype == "QNN_DATATYPE_FLOAT_32":
        raw = a.astype("<f4")
    elif dtype in ("QNN_DATATYPE_UFIXED_POINT_16", "QNN_DATATYPE_UFIXED_POINT_8"):
        if scale is None:
            raise SystemExit(f"{name}: {dtype} with no scaleOffset in the bin")
        qmax = 65535 if dtype.endswith("16") else 255
        # value = (q + offset) * scale  ->  q = value/scale - offset
        q = np.rint(a / scale) - offset
        lo, hi = float(q.min()), float(q.max())
        if lo < 0 or hi > qmax:
            clipped = int((q < 0).sum() + (q > qmax).sum())
            print(f"      WARN {name}: {clipped} value(s) outside the bin's "
                  f"encoding range [q {lo:.0f}..{hi:.0f}] vs [0..{qmax}] -- clipped")
        raw = np.clip(q, 0, qmax).astype("<u2" if qmax == 65535 else "<u1")
    else:
        raise SystemExit(f"{name}: unhandled input dtype {dtype!r}. Add a case "
                         "here rather than letting it fall through to fp16.")
    p.write_bytes(raw.tobytes())
    return dtype


def build_case(case, onnx_split: Path, lut: Path, out: Path, threads: int,
               specs0: dict, specs1: dict, layer_scan: bool = False):
    name, kind = case["name"], case["kind"]
    win = case.get("_win")
    n_real = case["n_real"] = case.get("n_real", win["n_valid"] if win else None)
    g0, g1 = f"{kind}_0", f"{kind}_1"
    cdir = out / name
    for sub in (g0, g1, "ref"):
        (cdir / sub).mkdir(parents=True, exist_ok=True)

    if win is None:
        toks = case.get("tokens", TOKENS[:n_real])
        rows = np.stack([lut_row(lut, t) for t in toks])           # [n, H]
        H = rows.shape[1]
    else:
        toks, rows = [], None
        H = win["embeds"].shape[-1]
    real_rows_box = {}

    def feeds0(m):
        mask, cos, sin, real_rows = build_mask_rope(m, case)
        real_rows_box["r"] = real_rows
        if win is None:
            emb = np.zeros((1, 1, m["ar"], H), dtype=np.float32)
            emb[0, 0, real_rows] = rows
        else:
            emb = win["embeds"].astype(np.float32).reshape(1, 1, m["ar"], H)
        f = {"inputs_embeds": emb, "attention_mask": mask,
             "position_ids_cos": cos, "position_ids_sin": sin, **zeros_past(m)}
        # Zero on BOTH paths. The shipped tower runs deepstack zero-filled, so a
        # realistic-input probe that fed the window's real deepstack features
        # would be less production-faithful, not more.
        for dk in m["deep"]:
            f[dk] = np.zeros((1, 1, m["ar"], H), dtype=np.float32)
        return f

    # The per-layer KV taps are the layer scan: shard 0 already writes all 36 of
    # these tensors on every run and they have never once been compared, so this
    # costs no extra device time at all.
    #
    # WHAT THEY CAN AND CANNOT SEE -- verified against the graph, not assumed.
    # Tracing decode_0 back from `past_value_0_out` shows v_proj consuming
    # `input_layernorm`'s output, and `past_key_0_out` consuming k_proj ->
    # k_norm, i.e. TWO RMSNorms. RMSNorm is scale-invariant, so:
    #
    #   * a uniform gain on the residual is normalised away before it reaches
    #     either tap. The scan therefore CANNOT tell us which layer a pure scale
    #     enters at -- the first draft of this code claimed it could, and that
    #     was wrong.
    #   * what the taps do read is the residual DIRECTION at every layer, and
    #     the correctness of each RMSNorm denominator.
    #
    # That second property is the sharp one. The leading mechanism for a row-0-
    # only fault is fp16 saturation on this row's massive activations: c4=5244
    # gives c4^2 = 2.75e7 against an fp16 max of 65504, a 420x overflow, and the
    # first place a sum of squares is taken is precisely input_layernorm. If that
    # denominator saturates, RMSNorm's output is wrong, and v_proj sees it -- so
    # the taps WOULD be dirty, at the layer where it starts. Hence:
    #
    #   taps clean at all 18 layers  -> every RMSNorm denominator is intact; the
    #                                   computation is fine and only the final
    #                                   magnitude is wrong (output/residual path)
    #   taps dirty from layer k      -> the fault is inside the block maths from
    #                                   layer k, and the overflow story is live
    #
    # Both outcomes are informative, which is what makes the scan worth pulling.
    scan_names = []
    if layer_scan:
        import onnx as _onnx
        g = _onnx.load(str(onnx_split / g0 / f"{g0}.onnx"),
                       load_external_data=False)
        outs = {o.name for o in g.graph.output}
        scan_names = sorted(
            (n for n in outs
             if n.startswith(("past_key_", "past_value_")) and n.endswith("_out")),
            key=lambda n: (int(n.split("_")[2]), n.split("_")[1]))
        del g

    hidden, m0, f0, taps = run(onnx_split / g0 / f"{g0}.onnx", feeds0,
                               "last_hidden_states", threads, scan_names)
    token_rows = real_rows_box["r"]
    # Rows we keep REFERENCES for. Identical to the token rows for the small
    # synthetic cases, but a realistic window carries up to 128 of them and full
    # logits are 151936 wide -- 78 MB per case in fp32, for rows nobody reads.
    # Keep the head (where the sink and the boundary defect live) and the last
    # real row (the one that actually produces the next token).
    if len(token_rows) <= 8:
        real_rows = token_rows
    else:
        real_rows = sorted(set(token_rows[:4] + [token_rows[-1]]))

    def feeds1(m):
        mask, cos, sin, _ = build_mask_rope(m, case)
        return {"last_hidden_states": hidden.reshape(1, m["ar"], H),
                "attention_mask": mask, "position_ids_cos": cos,
                "position_ids_sin": sin, **zeros_past(m)}

    logits, m1, f1, _ = run(onnx_split / g1 / f"{g1}.onnx", feeds1, "logits",
                            threads)

    # ---- device inputs, in each tensor's DECLARED native encoding ---------
    written = {}
    for t in ("inputs_embeds", "attention_mask", "position_ids_cos",
              "position_ids_sin"):
        written[t] = write_native(cdir / g0 / f"{t}.raw", f0[t], t, specs0)
    for dk in m0["deep"]:
        written[dk] = write_native(cdir / g0 / f"{dk}.raw", f0[dk], dk, specs0)
    for t in ("attention_mask", "position_ids_cos", "position_ids_sin"):
        written[f"s1/{t}"] = write_native(cdir / g1 / f"{t}.raw", f1[t], t, specs1)
    # shard 1 fed the HOST reference boundary: the isolation run, which answers
    # "is shard 1 right given good input?" independently of shard 0. Its inputs
    # are unaffected by shard 0's encoding, which is what makes it the one run
    # still valid if the kit and shard 0 ever fall out of sync again.
    written["s1/last_hidden_states"] = write_native(
        cdir / g1 / "last_hidden_states.raw", f1["last_hidden_states"],
        "last_hidden_states", specs1)

    # ---- references. Only the real rows: padding rows are meaningless, and
    # full prefill logits would be 78 MB of mostly-padding per case. ---------
    hid_r = hidden.reshape(m0["ar"], H)[real_rows]
    lg_r = logits.reshape(m1["ar"], -1)[real_rows]
    np.save(cdir / "ref" / "last_hidden_states.npy", hid_r)
    np.save(cdir / "ref" / "logits.npy", lg_r)

    # ---- layer-scan references: the real rows' slice of every per-layer KV tap.
    #
    # These graphs emit only the NEW AR-wide slice, not the concatenated cache:
    # key  is [1, n_kv, head_dim, AR] and value [1, n_kv, AR, head_dim]. So the
    # row index is the AR row r, NOT past+r -- indexing at past+r raised
    # IndexError on the decode graph, whose taps are [.., 1], and would have
    # silently read the wrong token on prefill, where AR=128 happens to equal
    # head_dim and every wrong index is still in bounds. The axis carrying AR
    # also differs between key and value, so both are taken by NAME, never by
    # shape -- on prefill the two tensors are both [1,8,128,128] and shape alone
    # cannot tell them apart.
    scan_meta = {}
    if scan_names:
        sdir = cdir / "ref" / "layerscan"
        sdir.mkdir(parents=True, exist_ok=True)
        for n, t in taps.items():
            is_key = n.startswith("past_key_")
            ar_axis = 3 if is_key else 2
            if t.shape[ar_axis] != m0["ar"]:
                raise SystemExit(
                    f"{n}: axis {ar_axis} is {t.shape[ar_axis]}, expected "
                    f"AR={m0['ar']} -- the tap layout is not what this assumes")
            # Drop the batch axis FIRST. `t[0, :, :, real_rows]` looks like the
            # obvious spelling and is a trap: numpy counts the leading integer 0
            # as an advanced index too, so with the row list it forms TWO
            # advanced indices separated by slices and the broadcast dimension
            # is moved to the FRONT. The result is already [n, n_kv, head_dim]
            # and the "corrective" transpose then scrambles it -- measured, the
            # key refs came out [128, n, 8]. Indexing a plain 3-D view has one
            # advanced index and no such rule.
            tt = t[0]
            if is_key:                                   # [n_kv, head_dim, AR]
                sl = np.transpose(tt[:, :, real_rows], (2, 0, 1))
            else:                                        # [n_kv, AR, head_dim]
                sl = np.transpose(tt[:, real_rows, :], (1, 0, 2))
            if sl.shape != (len(real_rows), m0["n_kv"], m0["head_dim"]):
                raise SystemExit(f"{n}: sliced to {sl.shape}, expected "
                                 f"{(len(real_rows), m0['n_kv'], m0['head_dim'])}")
            np.save(sdir / f"{n}.npy", np.ascontiguousarray(sl))
            scan_meta[n] = {"shape": list(t.shape), "ar_axis": ar_axis}
        del taps
        gc.collect()

    # ---- a spec the POSIX-sh runner can `.` source -------------------------
    # PAST_BYTES is per-case, not a constant: the past-KV prefill carries
    # past=2048 while decode carries past=2175 (both plus AR make 2176). One
    # shared zero file across both would feed the wrong number of bytes.
    graph_idx = 0 if kind == "prefill" else 1      # bins hold [prefill_N, decode_N]
    past_bytes = m0["n_kv"] * m0["head_dim"] * m0["past"] * 2
    assert past_bytes == m1["n_kv"] * m1["head_dim"] * m1["past"] * 2, \
        "shards disagree on past-KV size"
    env = [f"KIND={kind}", f"G0={g0}", f"G1={g1}", f"GRAPH_IDX={graph_idx}",
           f"AR={m0['ar']}", f"N_REAL={n_real}",
           f"REAL_ROWS='{' '.join(str(r) for r in real_rows)}'",
           f"TOKEN_ROWS={len(token_rows)}",
           f"LAYER_BASE_0={m0['layer_idx'][0]}", f"LAYER_N_0={len(m0['layer_idx'])}",
           f"LAYER_BASE_1={m1['layer_idx'][0]}", f"LAYER_N_1={len(m1['layer_idx'])}",
           f"DEEP='{' '.join(m0['deep'])}'",
           f"PAST_BYTES={past_bytes}",
           f"HIDDEN_BYTES={m0['ar'] * H * 2}"]
    (cdir / "case.env").write_text("\n".join(env) + "\n")

    top = np.argsort(-lg_r[-1])[:10]
    # Per-row RMS of the host boundary. This is the case's OWN evidence that it
    # achieved the condition it was built for: f1_row0ctx is only informative if
    # giving row 0 real context actually collapses its magnitude, and that is an
    # empirical question about this checkpoint, not something to assume. Printed
    # at build time so a case that failed its premise is caught here rather than
    # after a device session.
    row_rms = [float(np.sqrt((hid_r[i].astype(np.float64) ** 2).mean()))
               for i in range(hid_r.shape[0])]
    meta = {"case": name, "kind": kind, "n_real": n_real, "why": case["why"],
            "asks": case.get("asks"), "tokens": [int(t) for t in toks],
            "graph_idx": graph_idx, "mask": case.get("mask", "causal"),
            "row_offset": case.get("row_offset", 0),
            "real_rows": [int(r) for r in real_rows],
            "n_token_rows": len(token_rows),
            "window": (case.get("_win") or {}).get("desc"),
            "window_split": (case.get("_win") or {}).get("split"),
            "ar": m0["ar"], "hidden_rows": list(hid_r.shape),
            "logits_rows": list(lg_r.shape),
            "ref_row_rms": row_rms,
            "last_row_top10_ids": [int(t) for t in top],
            "last_row_argmax": int(top[0]),
            "mask_value": MASK_VALUE, "rope_theta": ROPE_THETA,
            "deep_names": m0["deep"], "past": m0["past"],
            "n_kv": m0["n_kv"], "head_dim": m0["head_dim"],
            "layerscan": scan_meta,
            # What the kit encoded each input AS. compare_text_probe.py checks
            # this against the bin it was actually run on and refuses to give a
            # verdict if they differ -- a stale kit must never look like a defect.
            "input_dtypes": written}
    (cdir / "ref" / "meta.json").write_text(json.dumps(meta, indent=1) + "\n")
    rms_s = " ".join(f"r{r}={v:.3f}" for r, v in zip(real_rows, row_rms))
    print(f"  {name:12s} AR={m0['ar']:<4} rows={real_rows}  argmax="
          f"{meta['last_row_argmax']:<7} host row RMS: {rms_s}"
          f"{'  +layerscan' if scan_meta else ''}")
    gc.collect()
    return meta


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--onnx-split", required=True, type=Path)
    ap.add_argument("--lut", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--threads", type=int, default=0)
    ap.add_argument("--ctxbin-info-0", required=True, type=Path,
                    help="info.json of the shard-0 ctx-bin that will execute")
    ap.add_argument("--ctxbin-info-1", required=True, type=Path,
                    help="info.json of the shard-1 ctx-bin that will execute")
    ap.add_argument("--suite", default="v5", choices=sorted(SUITES),
                    help="v5 = the original two cases; f = the Test F matrix")
    ap.add_argument("--windows", type=Path,
                    help="--suite r: the multimodal calibration/eval windows "
                         "(vl_calib_build.py .npz). These are chat-templated "
                         "turns with real ViT features, i.e. production-shaped "
                         "input, unlike the bare token ids the other suites use")
    ap.add_argument("--layer-scan", action="store_true",
                    help="also save per-layer KV sink-row references, so the "
                         "gain can be localised to a layer from outputs the "
                         "device already writes (default on for --suite f)")
    args = ap.parse_args()

    for g in ("decode_0", "decode_1", "prefill_0", "prefill_1"):
        p = args.onnx_split / g / f"{g}.onnx"
        if not p.is_file():
            raise SystemExit(f"missing {p} -- must be the SAME ONNX the DLCs "
                             "were converted from, not a re-export")
    specs0 = tensor_specs(args.ctxbin_info_0)
    specs1 = tensor_specs(args.ctxbin_info_1)
    print("encoding device inputs to match the ctx-bins:")
    for n in ("inputs_embeds", "attention_mask"):
        if n in specs0:
            print(f"  shard0 {n}: {specs0[n][0]} scale={specs0[n][1]} offset={specs0[n][2]}")
    if "last_hidden_states" in specs1:
        print(f"  shard1 last_hidden_states: {specs1['last_hidden_states'][0]}")
    cases = SUITES[args.suite]
    if args.suite == "r":
        if not args.windows:
            raise SystemExit("--suite r needs --windows <calib .npz>")
        cases = attach_windows(cases, args.windows)
    scan = args.layer_scan or args.suite in ("f", "r")
    args.out.mkdir(parents=True, exist_ok=True)
    print(f"building text probe suite {args.suite!r} -> {args.out}")
    metas = [build_case(c, args.onnx_split, args.lut, args.out, args.threads,
                        specs0, specs1, scan)
             for c in cases]
    (args.out / "cases.json").write_text(json.dumps(metas, indent=1) + "\n")
    (args.out / "probe_cases.txt").write_text(
        "\n".join(m["case"] for m in metas) + "\n")
    if args.suite == "f":
        check_f_premises(metas, args.out)
    print(f"\nwrote {len(metas)} case(s) to {args.out}")


def attach_windows(cases, npz: Path):
    """Bind each realistic case to one calibration/eval window.

    Selection is by DESCRIPTOR SUBSTRING and is required to be unambiguous: a
    silently different window would change what the probe measures without
    changing anything visible, and this project has already lost days to a probe
    that was not what it claimed to be.
    """
    with np.load(npz) as f:
        d = {k: f[k] for k in f.files}
    desc = [str(x) for x in d["desc"]]
    split = [str(x) for x in d["split"]]
    ar = int(d["ar"])
    out = []
    for c in cases:
        hits = [i for i, s in enumerate(desc) if c["match"] in s]
        pref = [i for i in hits if split[i] == c.get("prefer_split")]
        chosen = pref or hits
        if not chosen:
            raise SystemExit(
                f"{c['name']}: no window matching {c['match']!r} in {npz}. "
                f"Available: {desc}")
        if len(chosen) > 1:
            raise SystemExit(
                f"{c['name']}: {c['match']!r} matches {len(chosen)} windows "
                f"{[desc[i] for i in chosen]} -- ambiguous, refuse to guess")
        i = chosen[0]
        c = dict(c)
        c["_win"] = {"embeds": d["embeds"][i], "mask": d["mask"][i],
                     "cos": d["cos"][i], "sin": d["sin"][i],
                     "n_valid": int(d["n_valid"][i]), "ar": ar,
                     "desc": desc[i], "split": split[i]}
        out.append(c)
        print(f"  {c['name']:12s} <- [{split[i]}] {desc[i]} "
              f"({int(d['n_valid'][i])} real rows of {ar})")
    return out


def check_f_premises(metas, out: Path):
    """Test F only means anything if its cases really are what they claim.

    Both checks are host-side and free, and both have caught nothing so far --
    which is the point: they are here so that a case whose premise silently
    failed is rejected at build time instead of consuming a device session and
    then being interpreted as a result.
    """
    by = {m["case"]: m for m in metas}
    print("\n=== Test F premises (host side) ===")
    ok = True

    # 1. f2_shift4's row 4 must compute EXACTLY what fp_ctrl's row 0 computes.
    #    Same token, same self-only mask, same rope position 0 -- only the row
    #    index differs. If the host disagrees, the shift changed the computation
    #    and the case cannot isolate the index.
    a, b = by.get("fp_ctrl_pre"), by.get("f2_shift4")
    if a and b:
        ha = np.load(out / a["case"] / "ref" / "last_hidden_states.npy")[0]
        hb = np.load(out / b["case"] / "ref" / "last_hidden_states.npy")[0]
        d = float(np.abs(ha - hb).max())
        rel = d / max(float(np.abs(ha).max()), 1e-9)
        good = rel < 1e-4
        ok &= good
        print(f"  {'OK ' if good else 'FAIL'} f2_shift4 row4 == fp_ctrl row0 on "
              f"the host: max|diff|={d:.3e} (rel {rel:.2e})")
        if not good:
            print("       -> the shift changed the computation; f2 cannot "
                  "isolate the row index. Do not ship this case.")

    # 2. f1_row0ctx must actually remove the sink, i.e. collapse row 0's
    #    magnitude. If giving row 0 context leaves it at ~107 RMS, the case
    #    tests nothing about massive activations and must be read accordingly.
    c = by.get("f1_row0ctx")
    if c and a:
        r0_ctrl, r0_ctx = a["ref_row_rms"][0], c["ref_row_rms"][0]
        drop = r0_ctx / r0_ctrl if r0_ctrl else float("nan")
        good = drop < 0.5
        print(f"  {'OK ' if good else 'WARN'} f1_row0ctx removes the sink: row-0 "
              f"RMS {r0_ctrl:.2f} -> {r0_ctx:.2f} ({drop:.2f}x)")
        if not good:
            print("       -> context did NOT collapse row 0's magnitude on this "
                  "checkpoint. The case still tests 'attends only to itself', "
                  "but it is NOT a magnitude test. Say so when reporting.")
    if not ok:
        raise SystemExit("Test F premise check failed -- see above")


if __name__ == "__main__":
    main()
