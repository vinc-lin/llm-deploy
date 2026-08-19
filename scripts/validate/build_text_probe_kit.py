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

CASES = [
    {"name": "decode1tok", "kind": "decode", "n_real": 1,
     "why": "1 token, empty cache -- rope cancels in self-attention, so this "
            "is a clean read on pure numerics"},
    {"name": "prefill4tok", "kind": "prefill", "n_real": 4,
     "why": "4 tokens, empty cache -- rope and cross-token attention are live, "
            "and these are the graphs the real prompt uses"},
]


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


def build_mask_rope(m, n_real):
    """Mask + rope tables for an EMPTY cache, per parity_e2e_vl.

    decode  (AR==1): additive MASK_VALUE, 0.0 only at index PAST -- the new
                     token sits at the END in the concat layout, not at 0.
    prefill (AR>1) : row i sees the valid past span [0, nv) -- empty here, so
                     nothing -- plus the causal new span [PAST, PAST+i].
    Rows past n_real stay fully masked; their KV is never committed.
    """
    import torch
    ar, total, past, half = m["ar"], m["total"], m["past"], m["head_dim"] // 2
    if ar == 1:
        mask = np.full((1, 1, total), MASK_VALUE, dtype=np.float32)
        mask[0, 0, past] = 0.0
        cos, sin = rope_tables(torch.arange(1), m["head_dim"], ROPE_THETA)
        return mask, cos.numpy().astype(np.float32), sin.numpy().astype(np.float32)

    mask = np.full((1, ar, total), MASK_VALUE, dtype=np.float32)
    for i in range(n_real):
        mask[0, i, past:past + i + 1] = 0.0     # causal new span; no valid past
    c, s = rope_tables(torch.arange(n_real), m["head_dim"], ROPE_THETA)
    cos = np.zeros((1, ar, half), dtype=np.float32)
    sin = np.zeros((1, ar, half), dtype=np.float32)
    cos[0, :n_real] = c.numpy()[0]
    sin[0, :n_real] = s.numpy()[0]
    return mask, cos, sin


def run(onnx, feeds, out_name, threads):
    import onnxruntime as ort
    so = ort.SessionOptions()
    if threads:
        so.intra_op_num_threads = threads
    sess = ort.InferenceSession(str(onnx), so, providers=["CPUExecutionProvider"])
    m = meta_of(sess)
    f = feeds(m)
    out = np.asarray(sess.run([out_name], f)[0], dtype=np.float32)
    del sess
    gc.collect()
    return out, m, f


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
               specs0: dict, specs1: dict):
    name, kind, n_real = case["name"], case["kind"], case["n_real"]
    g0, g1 = f"{kind}_0", f"{kind}_1"
    cdir = out / name
    for sub in (g0, g1, "ref"):
        (cdir / sub).mkdir(parents=True, exist_ok=True)

    rows = np.stack([lut_row(lut, t) for t in TOKENS[:n_real]])   # [n, H]
    H = rows.shape[1]

    def feeds0(m):
        mask, cos, sin = build_mask_rope(m, n_real)
        emb = np.zeros((1, 1, m["ar"], H), dtype=np.float32)
        emb[0, 0, :n_real] = rows
        f = {"inputs_embeds": emb, "attention_mask": mask,
             "position_ids_cos": cos, "position_ids_sin": sin, **zeros_past(m)}
        for dk in m["deep"]:
            f[dk] = np.zeros((1, 1, m["ar"], H), dtype=np.float32)
        return f

    hidden, m0, f0 = run(onnx_split / g0 / f"{g0}.onnx", feeds0,
                         "last_hidden_states", threads)

    def feeds1(m):
        mask, cos, sin = build_mask_rope(m, n_real)
        return {"last_hidden_states": hidden.reshape(1, m["ar"], H),
                "attention_mask": mask, "position_ids_cos": cos,
                "position_ids_sin": sin, **zeros_past(m)}

    logits, m1, f1 = run(onnx_split / g1 / f"{g1}.onnx", feeds1, "logits", threads)

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
    hid_r = hidden.reshape(m0["ar"], H)[:n_real]
    lg_r = logits.reshape(m1["ar"], -1)[:n_real]
    np.save(cdir / "ref" / "last_hidden_states.npy", hid_r)
    np.save(cdir / "ref" / "logits.npy", lg_r)

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
           f"LAYER_BASE_0={m0['layer_idx'][0]}", f"LAYER_N_0={len(m0['layer_idx'])}",
           f"LAYER_BASE_1={m1['layer_idx'][0]}", f"LAYER_N_1={len(m1['layer_idx'])}",
           f"DEEP='{' '.join(m0['deep'])}'",
           f"PAST_BYTES={past_bytes}",
           f"HIDDEN_BYTES={m0['ar'] * H * 2}"]
    (cdir / "case.env").write_text("\n".join(env) + "\n")

    top = np.argsort(-lg_r[-1])[:10]
    meta = {"case": name, "kind": kind, "n_real": n_real, "why": case["why"],
            "tokens": TOKENS[:n_real], "graph_idx": graph_idx,
            "ar": m0["ar"], "hidden_rows": list(hid_r.shape),
            "logits_rows": list(lg_r.shape),
            "last_row_top10_ids": [int(t) for t in top],
            "last_row_argmax": int(top[0]),
            "mask_value": MASK_VALUE, "rope_theta": ROPE_THETA,
            "deep_names": m0["deep"],
            # What the kit encoded each input AS. compare_text_probe.py checks
            # this against the bin it was actually run on and refuses to give a
            # verdict if they differ -- a stale kit must never look like a defect.
            "input_dtypes": written}
    (cdir / "ref" / "meta.json").write_text(json.dumps(meta, indent=1) + "\n")
    print(f"  {name:12s} AR={m0['ar']:<4} rows={n_real}  last-row argmax="
          f"{meta['last_row_argmax']}  top5={meta['last_row_top10_ids'][:5]}")
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
    args.out.mkdir(parents=True, exist_ok=True)
    print(f"building text probe -> {args.out}")
    metas = [build_case(c, args.onnx_split, args.lut, args.out, args.threads,
                        specs0, specs1)
             for c in CASES]
    (args.out / "cases.json").write_text(json.dumps(metas, indent=1) + "\n")
    (args.out / "probe_cases.txt").write_text(
        "\n".join(m["case"] for m in metas) + "\n")
    print(f"\nwrote {len(metas)} case(s) to {args.out}")


if __name__ == "__main__":
    main()
