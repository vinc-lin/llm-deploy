#!/usr/bin/env python
"""Build the Test L kit: run the 0.6B LUT probe's ctx-bin under qnn-net-run.

WHY THIS EXISTS
---------------
Test K ran the LUT probe through Genie on device and it was wrong from the very
first token -- `</think>` (151668) where the reference is `' '` (220). But
`parity_lutprobe.py` passes 3/3 against HuggingFace on the SAME graph and the
SAME LUT, on the host. So one of exactly two things is true:

  * the ctx-bin reproduces its ONNX, and the divergence is entirely in what
    GENIE feeds it            -> Genie's LUT feed is the defect
  * the ctx-bin does NOT reproduce its ONNX
                              -> a converter defect, a different investigation

`qnn-net-run` settles it, because it takes the inputs from files instead of
from Genie. This script writes those files.

WHAT IT BUILDS
--------------
Three prefill cases (the same prompts `parity_lutprobe.py` gates on, so their
host answers are already known-good) and two decode cases chained off the first:

    l1a_2plus2      prefill 12 tok   "What is 2+2? Answer with one number."
    l1b_paris       prefill  6 tok   "The capital of France is"
    l1c_boils       prefill  7 tok   "Water boils at a temperature of"
    l2a_decode_s1   decode, cache = l1a's prefill KV
    l2b_decode_s2   decode, cache ALSO holds a row the DECODE graph wrote

`l2b` is the recurrence -- the same thing Test J's `j2` tested for the 4B, and
the only part of a decode path that a single step never exercises.

The expected argmax for every case is taken from the ONNX itself: the question
is whether the BIN reproduces the graph it was converted from. That graph is
independently gated against HF by `parity_lutprobe.py`, so a match here chains
all the way back to HuggingFace.

    $PY_DEPLOY scripts/validate/build_lutprobe_kit.py \
        --onnx-prefill $LLMDEPLOY_DATA/work/quant/qwen3-0.6b-w8a16-lutprobe-prefillkv128/model_renamed.onnx \
        --onnx-decode  $LLMDEPLOY_DATA/work/quant/qwen3-0.6b-w8a16-lutprobe-decode/model_renamed.onnx \
        --lut          $LLMDEPLOY_DATA/work/lut/qwen3-0.6b \
        --model        $LLMDEPLOY_DATA/models/Qwen3-0.6B \
        --info         $LLMDEPLOY_DATA/work/ctxbin/qwen3-0.6b-w8a16-lutprobe-ladekv/qwen3-0.6b-w8a16-lutprobe-ladekv_ctx.info.json \
        --out          $LLMDEPLOY_DATA/work/lutprobe_kit
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "export"))
from modeling_export import MASK_VALUE, rope_tables, rope_theta_of  # noqa: E402

PROMPTS = [
    ("l1a_2plus2", "What is 2+2? Answer with one number."),
    ("l1b_paris", "The capital of France is"),
    ("l1c_boils", "Water boils at a temperature of"),
]
# Chain two decode steps off l1a. HF fp32 continues ' 2+2=4' = [220,17,10,17,28,19].
DECODE_FROM = "l1a_2plus2"
DECODE_STEPS = 2


def graph_specs(info_json: Path) -> dict:
    """{graph name: {tensor: (dtype, scale, offset)}} -- per graph, not merged.

    The 4B kit could merge, because each shard is its own bin. This bin holds
    prefill AND decode, and they share input names with different shapes and
    potentially different encodings, so merging would silently quantize one
    graph's tensors with the other's scale.
    """
    doc = json.loads(info_json.read_text())
    out = {}
    for g in doc["info"]["graphs"]:
        gi = g["info"]
        specs = {}
        for t in gi["graphInputs"]:
            ti = t.get("info", t)
            so = (ti.get("quantizeParams") or {}).get("scaleOffset") or {}
            specs[ti["name"]] = (ti.get("dataType"), so.get("scale"), so.get("offset"))
        out[gi["graphName"]] = specs
    return out


def write_native(p: Path, a, name: str, specs: dict) -> str:
    """Write `a` as the bytes the graph declares for input `name`."""
    if name not in specs:
        raise SystemExit(f"{name!r} is not an input of this graph "
                         f"(have {sorted(specs)[:6]}...) -- kit and bin disagree")
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
        q = np.rint(a / scale) - offset
        lo, hi = float(q.min()), float(q.max())
        if lo < 0 or hi > qmax:
            n = int((q < 0).sum() + (q > qmax).sum())
            print(f"      WARN {name}: {n} value(s) outside the bin's encoding "
                  f"range [q {lo:.0f}..{hi:.0f}] vs [0..{qmax}] -- clipped")
        raw = np.clip(q, 0, qmax).astype("<u2" if qmax == 65535 else "<u1")
    else:
        raise SystemExit(f"{name}: unhandled dtype {dtype!r} -- add a case here "
                         "rather than letting it fall through to fp16")
    p.write_bytes(raw.tobytes())
    return dtype


def lut_rows(lut_dir: Path, ids):
    """Rows read the way qualla::LUT reads them: raw byte offsets, no reshape."""
    p = json.loads((lut_dir / "embedding_lut_params.json").read_text())
    n, eb = p["size"], p["element-bytes"]
    assert p["datatype"] == "float32" and eb == 4, p
    out = np.empty((len(ids), n), dtype=np.float32)
    with (lut_dir / p["lut-path"]).open("rb") as fh:
        for k, t in enumerate(ids):
            fh.seek(int(t) * n * eb)
            raw = fh.read(n * eb)
            assert len(raw) == n * eb, f"short read at token {t}"
            out[k] = np.frombuffer(raw, dtype="<f4")
    return out


def main():
    import onnxruntime as ort
    import torch
    from transformers import AutoConfig, AutoTokenizer

    ap = argparse.ArgumentParser()
    for f in ("onnx-prefill", "onnx-decode", "lut", "model", "info", "out"):
        ap.add_argument(f"--{f}", required=True, type=Path)
    ap.add_argument("--threads", type=int, default=0)
    args = ap.parse_args()

    specs = graph_specs(args.info)
    if set(specs) != {"prefill", "decode"}:
        raise SystemExit(f"expected graphs prefill+decode, bin has {sorted(specs)}")
    print(f"bin graphs: {sorted(specs)}")
    print(f"  prefill inputs_embeds {specs['prefill']['inputs_embeds'][0]}")
    print(f"  decode  inputs_embeds {specs['decode']['inputs_embeds'][0]}")

    so = ort.SessionOptions()
    if args.threads:
        so.intra_op_num_threads = args.threads
    sp = ort.InferenceSession(str(args.onnx_prefill), so,
                              providers=["CPUExecutionProvider"])
    sd = ort.InferenceSession(str(args.onnx_decode), so,
                              providers=["CPUExecutionProvider"])

    shp = {i.name: i.shape for i in sp.get_inputs()}
    shd = {i.name: i.shape for i in sd.get_inputs()}
    if "inputs_embeds" not in shp:
        raise SystemExit("prefill graph's first input is not 'inputs_embeds' -- "
                         "qualla selects InputType::EMBEDDINGS by that literal "
                         "name (nsp-model.cpp:668)")
    AR = shp["attention_mask"][-2]
    TOTAL = shp["attention_mask"][-1]
    H = shp["inputs_embeds"][-1]
    PAST_P = TOTAL - AR
    PAST_D = shd["attention_mask"][-1] - 1
    past_p = sorted(n for n in shp if n.startswith("past_") and n.endswith("_in"))
    past_d = sorted(n for n in shd if n.startswith("past_") and n.endswith("_in"))
    NKV, D = shd["past_key_0_in"][1], shd["past_key_0_in"][2]
    kv_out_p = [f"past_{s}_{i}_out" for i in range(len(past_p) // 2) for s in ("key", "value")]
    kv_out_d = [f"past_{s}_{i}_out" for i in range(len(past_d) // 2) for s in ("key", "value")]

    print(f"prefill: AR={AR} TOTAL={TOTAL} PAST={PAST_P} H={H}, {len(past_p)} past inputs")
    print(f"decode : AR=1 TOTAL={TOTAL} PAST={PAST_D}, n_kv={NKV} head_dim={D}")

    cfg = AutoConfig.from_pretrained(str(args.model))
    tok = AutoTokenizer.from_pretrained(str(args.model))
    theta = rope_theta_of(cfg)
    HD = cfg.head_dim

    def rope(pos):
        c, s = rope_tables(torch.tensor(np.asarray(pos)), HD, theta)
        return c.numpy()[0], s.numpy()[0]

    args.out.mkdir(parents=True, exist_ok=True)
    metas, decode_seed = [], None

    # ---------------- prefill cases ----------------
    for name, prompt in PROMPTS:
        ids = tok(prompt, return_tensors="np").input_ids[0].tolist()[:AR]
        n = len(ids)
        emb = np.zeros((1, 1, AR, H), np.float32)
        emb[0, 0, :n] = lut_rows(args.lut, ids)
        mask = np.full((1, AR, TOTAL), MASK_VALUE, np.float32)
        for r in range(n):
            mask[0, r, PAST_P:PAST_P + r + 1] = 0.0
        cos = np.zeros((1, AR, HD // 2), np.float32)
        sin = np.zeros((1, AR, HD // 2), np.float32)
        c, s = rope(np.arange(n))
        cos[0, :n], sin[0, :n] = c, s

        feeds = {"inputs_embeds": emb, "attention_mask": mask,
                 "position_ids_cos": cos, "position_ids_sin": sin}
        for nm in past_p:
            feeds[nm] = np.zeros([d if isinstance(d, int) else 1 for d in shp[nm]],
                                 np.float32)
        res = dict(zip(["logits"] + kv_out_p, sp.run(["logits"] + kv_out_p, feeds)))
        argmax = int(np.argmax(res["logits"][0, n - 1]))

        cdir = args.out / name
        (cdir / "prefill").mkdir(parents=True, exist_ok=True)
        (cdir / "ref").mkdir(parents=True, exist_ok=True)
        written = {t: write_native(cdir / "prefill" / f"{t}.raw", feeds[t], t,
                                   specs["prefill"])
                   for t in ("inputs_embeds", "attention_mask",
                             "position_ids_cos", "position_ids_sin")}
        np.save(cdir / "ref" / "logits.npy", res["logits"][0, n - 1])
        past_bytes = NKV * D * PAST_P * 2
        (cdir / "case.env").write_text("\n".join([
            "KIND=prefill", "GRAPH_IDX=0", f"AR={AR}", f"N_REAL={n}",
            f"ROW_OUT={n - 1}", f"N_PAST_PAIRS={len(past_p) // 2}",
            "PAST_MODE=zero", f"PAST_BYTES={past_bytes}",
            f"EXPECT_ARGMAX={argmax}"]) + "\n")
        (cdir / "ref" / "meta.json").write_text(json.dumps({
            "case": name, "kind": "prefill", "prompt": prompt, "tokens": ids,
            "n_real": n, "row_out": n - 1, "expect_argmax": argmax,
            "expect_token": tok.decode([argmax]),
            "top5": [int(t) for t in np.argsort(-res["logits"][0, n - 1])[:5]],
            "input_dtypes": written, "ar": AR, "total": TOTAL, "past": PAST_P,
            "mask_value": MASK_VALUE, "rope_theta": theta,
        }, indent=1) + "\n")
        print(f"  {name:14s} n={n:<3} argmax={argmax:<7} {tok.decode([argmax])!r}")
        metas.append(name)

        if name == DECODE_FROM:
            decode_seed = (ids, argmax, res)

    # ---------------- decode cases, chained ----------------
    ids, tok_next, res = decode_seed
    nv = len(ids)
    cache = {}
    for i in range(len(past_d) // 2):
        k = np.zeros((1, NKV, D, PAST_D), np.float32)
        v = np.zeros((1, NKV, PAST_D, D), np.float32)
        k[:, :, :, :nv] = res[f"past_key_{i}_out"][:, :, :, :nv]
        v[:, :, :nv, :] = res[f"past_value_{i}_out"][:, :, :nv, :]
        cache[f"past_key_{i}_in"], cache[f"past_value_{i}_in"] = k, v

    for step in range(1, DECODE_STEPS + 1):
        name = f"l2{'ab'[step - 1]}_decode_s{step}"
        emb = lut_rows(args.lut, [tok_next]).reshape(1, 1, 1, H)
        mask = np.full((1, 1, TOTAL), MASK_VALUE, np.float32)
        mask[0, 0, :nv] = 0.0            # valid past
        mask[0, 0, PAST_D] = 0.0         # this step's own slot
        c, s = rope([nv])
        feeds = {"inputs_embeds": emb, "attention_mask": mask,
                 "position_ids_cos": c.reshape(1, 1, HD // 2),
                 "position_ids_sin": s.reshape(1, 1, HD // 2), **cache}
        r = dict(zip(["logits"] + kv_out_d, sd.run(["logits"] + kv_out_d, feeds)))
        argmax = int(np.argmax(r["logits"].reshape(-1)))

        cdir = args.out / name
        (cdir / "decode").mkdir(parents=True, exist_ok=True)
        (cdir / "ref").mkdir(parents=True, exist_ok=True)
        written = {t: write_native(cdir / "decode" / f"{t}.raw", feeds[t], t,
                                   specs["decode"])
                   for t in ("inputs_embeds", "attention_mask",
                             "position_ids_cos", "position_ids_sin")}
        for nm in past_d:
            write_native(cdir / "decode" / f"{nm}.raw", cache[nm], nm, specs["decode"])
        np.save(cdir / "ref" / "logits.npy", r["logits"].reshape(-1))
        (cdir / "case.env").write_text("\n".join([
            "KIND=decode", "GRAPH_IDX=1", "AR=1", "N_REAL=1", "ROW_OUT=0",
            f"N_PAST_PAIRS={len(past_d) // 2}", "PAST_MODE=files",
            f"PAST_BYTES={NKV * D * PAST_D * 2}", f"EXPECT_ARGMAX={argmax}"]) + "\n")
        (cdir / "ref" / "meta.json").write_text(json.dumps({
            "case": name, "kind": "decode", "step": step,
            "fed_token": tok_next, "fed_token_str": tok.decode([tok_next]),
            "cache_len": nv, "position": nv,
            "cache_contains_decode_written_row": step > 1,
            "expect_argmax": argmax, "expect_token": tok.decode([argmax]),
            "top5": [int(t) for t in np.argsort(-r["logits"].reshape(-1))[:5]],
            "input_dtypes": written, "total": TOTAL, "past": PAST_D,
        }, indent=1) + "\n")
        print(f"  {name:14s} fed={tok_next:<7} nv={nv:<3} argmax={argmax:<7} "
              f"{tok.decode([argmax])!r}"
              + ("   <- cache holds a DECODE-written row" if step > 1 else ""))
        metas.append(name)

        # Commit this step's KV so the NEXT case's cache holds a decode-written
        # row. The two caches are laid out differently and mixing them up is
        # silent: key is [1,NKV,D,PAST] (position is the LAST axis) and value is
        # [1,NKV,PAST,D] (position is axis 2). Assert rather than trust.
        for i in range(len(past_d) // 2):
            ko, vo = r[f"past_key_{i}_out"], r[f"past_value_{i}_out"]
            assert ko.shape[-1] == 1 and vo.shape[2] == 1, (
                f"expected a 1-wide new slice, got key {ko.shape} value {vo.shape}")
            cache[f"past_key_{i}_in"][:, :, :, nv] = ko[:, :, :, 0]
            cache[f"past_value_{i}_in"][:, :, nv, :] = vo[:, :, 0, :]
        nv += 1
        tok_next = argmax

    (args.out / "probe_cases.txt").write_text("\n".join(metas) + "\n")
    print(f"\nkit -> {args.out}   cases: {' '.join(metas)}")


if __name__ == "__main__":
    main()
