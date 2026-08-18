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

It is deliberately a SINGLE-TOKEN decode from an EMPTY cache, because that
removes every moving part we are not testing:

  * empty cache -> the 72 past-KV inputs are zeros AND fully masked out, so
    their contents cannot change the result. The device generates them as one
    reused zero file instead of us shipping 320 MB.
  * case `pos0` -> position 0 means cos=1, sin=0: rope is the identity. A
    failure there is pure matmul/quantization with rope removed as a variable.
  * case `pos7` -> rope is active. Comparing the two cases separates an
    in-graph rope fault from a numerics fault, which no single case can do.

THE REFERENCE COMES FROM THE SAME ONNX THE DLC WAS CONVERTED FROM
(`work/onnx/qwen3vl-4b-aimet-split`, cut at the layer seam by
`split_aimet_onnx.py`), never from HF and never from a re-export. Anything else
would compare two different graphs and blame the wrong stage.

Because those per-shard ONNX files live on tank, so does this script:

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

# The probe tokens. Ids, not words: the comparison is numeric, so nothing here
# depends on a tokenizer being present. 3838 and 40 are ordinary in-vocab ids.
CASES = [
    {"name": "pos0", "token": 3838, "position": 0,
     "why": "rope is the identity (cos=1,sin=0) -- isolates pure numerics"},
    {"name": "pos7", "token": 3838, "position": 7,
     "why": "rope active -- differs from pos0 only by the rope tables"},
]
DEEPSTACK = 3           # deepstack_visual_embed_{0,1,2}, zero-fed by contract
ROPE_THETA = 5_000_000.0


def load_lut_row(lut_dir: Path, token: int) -> np.ndarray:
    """One embedding row, read the way the runtime reads it (raw byte offset).

    Deliberately NOT via numpy's view of the whole array: this is the same
    access pattern parity_embed_lut.py gates, so a stride bug would show up
    here identically rather than being papered over by a reshape.
    """
    params = json.loads((lut_dir / "embedding_lut_params.json").read_text())
    n_embd = params["size"]
    ebytes = params["element-bytes"]
    assert params["datatype"] == "float32" and ebytes == 4, params
    path = lut_dir / params["lut-path"]
    off = token * n_embd * ebytes
    with path.open("rb") as fh:
        fh.seek(off)
        raw = fh.read(n_embd * ebytes)
    assert len(raw) == n_embd * ebytes, f"short read at token {token}"
    return np.frombuffer(raw, dtype="<f4").astype(np.float32)


def graph_meta(sess):
    shapes = {i.name: i.shape for i in sess.get_inputs()}
    outs = [o.name for o in sess.get_outputs()]
    layers = sum(1 for n in shapes if n.startswith("past_key_") and n.endswith("_in"))
    pk = next(n for n in shapes if n.startswith("past_key_") and n.endswith("_in"))
    _, nkv, head_dim, past = shapes[pk]
    total = shapes["attention_mask"][-1]
    assert total == past + 1, f"mask {total} != past {past} + 1"
    return {"layers": layers, "n_kv": nkv, "head_dim": head_dim,
            "past": past, "total": total, "outs": outs, "shapes": shapes}


def decode_mask(meta):
    """Empty-cache decode mask -- the contract from parity_e2e_vl.Decoder.step.

    Additive MASK_VALUE everywhere; 0.0 only at index PAST, because the new
    token sits at the END in the concat layout, not at index 0. Getting this
    backwards produces a model that attends to nothing but zeros and still
    runs, which is exactly the failure this probe is trying to localise -- so
    it is copied, not re-derived.
    """
    m = np.full((1, 1, meta["total"]), MASK_VALUE, dtype=np.float32)
    m[0, 0, meta["past"]] = 0.0
    return m


def run_shard(onnx_path, feeds_builder, out_name, threads):
    import onnxruntime as ort
    so = ort.SessionOptions()
    if threads:
        so.intra_op_num_threads = threads
    sess = ort.InferenceSession(str(onnx_path), so,
                                providers=["CPUExecutionProvider"])
    meta = graph_meta(sess)
    feeds = feeds_builder(meta)
    out = sess.run([out_name], feeds)[0]
    del sess
    gc.collect()
    return np.asarray(out, dtype=np.float32), meta


def build_case(case, onnx_split: Path, lut: Path, out: Path, threads: int):
    name = case["name"]
    cdir = out / name
    (cdir / "decode_0").mkdir(parents=True, exist_ok=True)
    (cdir / "decode_1").mkdir(parents=True, exist_ok=True)
    (cdir / "ref").mkdir(parents=True, exist_ok=True)

    pos = np.array([case["position"]])
    emb = load_lut_row(lut, case["token"])

    # ---- shard 0: inputs_embeds -> last_hidden_states -----------------------
    def feeds0(meta):
        import torch
        H = emb.shape[0]
        cos, sin = rope_tables(torch.tensor(pos), meta["head_dim"], ROPE_THETA)
        f = {
            "inputs_embeds": emb.reshape(1, 1, 1, H),
            "attention_mask": decode_mask(meta),
            "position_ids_cos": np.ascontiguousarray(cos.numpy(), dtype=np.float32),
            "position_ids_sin": np.ascontiguousarray(sin.numpy(), dtype=np.float32),
        }
        for k in range(DEEPSTACK):
            f[f"deepstack_visual_embed_{k}"] = np.zeros((1, 1, 1, H), dtype=np.float32)
        L, NKV, D, P = meta["layers"], meta["n_kv"], meta["head_dim"], meta["past"]
        for i in range(L):
            f[f"past_key_{i}_in"] = np.zeros((1, NKV, D, P), dtype=np.float32)
            f[f"past_value_{i}_in"] = np.zeros((1, NKV, P, D), dtype=np.float32)
        build_case._f0 = f
        return f

    hidden, m0 = run_shard(onnx_split / "decode_0" / "decode_0.onnx",
                           feeds0, "last_hidden_states", threads)

    # ---- shard 1: last_hidden_states -> logits ------------------------------
    def feeds1(meta):
        import torch
        cos, sin = rope_tables(torch.tensor(pos), meta["head_dim"], ROPE_THETA)
        f = {
            "last_hidden_states": hidden.reshape(1, 1, -1),
            "attention_mask": decode_mask(meta),
            "position_ids_cos": np.ascontiguousarray(cos.numpy(), dtype=np.float32),
            "position_ids_sin": np.ascontiguousarray(sin.numpy(), dtype=np.float32),
        }
        L, NKV, D, P = meta["layers"], meta["n_kv"], meta["head_dim"], meta["past"]
        base = min(int(n.split("_")[2]) for n in meta["shapes"]
                   if n.startswith("past_key_") and n.endswith("_in"))
        for i in range(base, base + L):
            f[f"past_key_{i}_in"] = np.zeros((1, NKV, D, P), dtype=np.float32)
            f[f"past_value_{i}_in"] = np.zeros((1, NKV, P, D), dtype=np.float32)
        build_case._f1 = f
        return f

    logits, m1 = run_shard(onnx_split / "decode_1" / "decode_1.onnx",
                           feeds1, "logits", threads)

    # ---- write device inputs, native fp16 (what the graphs declare) ---------
    def w16(p, a):
        Path(p).write_bytes(np.ascontiguousarray(a, dtype=np.float32)
                            .astype("<f2").tobytes())

    f0, f1 = build_case._f0, build_case._f1
    for n in ("inputs_embeds", "attention_mask", "position_ids_cos",
              "position_ids_sin"):
        w16(cdir / "decode_0" / f"{n}.raw", f0[n])
    for k in range(DEEPSTACK):
        w16(cdir / "decode_0" / f"deepstack_visual_embed_{k}.raw",
            f0[f"deepstack_visual_embed_{k}"])
    for n in ("attention_mask", "position_ids_cos", "position_ids_sin"):
        w16(cdir / "decode_1" / f"{n}.raw", f1[n])
    # shard 1 fed with the HOST reference boundary -- this is the isolation
    # run: it answers "is shard 1 alone correct?" independently of shard 0.
    w16(cdir / "decode_1" / "last_hidden_states.raw", f1["last_hidden_states"])

    # ---- references --------------------------------------------------------
    np.save(cdir / "ref" / "last_hidden_states.npy", hidden)
    np.save(cdir / "ref" / "logits.npy", logits)
    lg = logits.reshape(-1)
    top = np.argsort(-lg)[:10]
    meta = {
        "case": name, "token": case["token"], "position": case["position"],
        "why": case["why"],
        "shard0": {"layers": m0["layers"], "past": m0["past"], "total": m0["total"]},
        "shard1": {"layers": m1["layers"], "past": m1["past"], "total": m1["total"]},
        "hidden_shape": list(hidden.shape), "logits_shape": list(logits.shape),
        "top10_ids": [int(t) for t in top],
        "top10_logits": [float(lg[t]) for t in top],
        "argmax": int(top[0]),
        "mask_value": MASK_VALUE, "rope_theta": ROPE_THETA,
    }
    (cdir / "ref" / "meta.json").write_text(json.dumps(meta, indent=1) + "\n")
    print(f"  {name}: argmax={meta['argmax']} top5={meta['top10_ids'][:5]} "
          f"hidden|{hidden.shape}| logits|{logits.shape}|")
    del build_case._f0, build_case._f1
    gc.collect()
    return meta


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--onnx-split", required=True, type=Path,
                    help="work/onnx/qwen3vl-4b-aimet-split (per-shard ONNX)")
    ap.add_argument("--lut", required=True, type=Path, help="dir with the LUT + params")
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--threads", type=int, default=0)
    args = ap.parse_args()

    for g in ("decode_0", "decode_1"):
        p = args.onnx_split / g / f"{g}.onnx"
        if not p.is_file():
            raise SystemExit(f"missing {p} -- this must be the SAME ONNX the DLC "
                             "was converted from, not a re-export")
    args.out.mkdir(parents=True, exist_ok=True)

    print(f"building text probe -> {args.out}")
    metas = [build_case(c, args.onnx_split, args.lut, args.out, args.threads)
             for c in CASES]
    (args.out / "cases.json").write_text(json.dumps(metas, indent=1) + "\n")
    print(f"\nwrote {len(metas)} case(s) to {args.out}")
    print("the device runner (configs/run_text_probe.sh) is static -- it "
          "discovers cases from cases.json and generates the zero past-KV "
          "files on device, so nothing large ships.")


if __name__ == "__main__":
    main()
