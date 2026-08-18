#!/usr/bin/env python
"""Gate the 0.6B LUT probe on the host before it ever reaches a device.

Without this the probe is worthless: if the embeddings-in export is itself
wrong, garbage on device proves nothing about Genie's LUT feed -- it just
reproduces our own bug. So this feeds the probe graph exactly what the runtime
will feed it (rows read from the LUT file at the runtime's own byte offsets)
and checks the logits against HuggingFace.

Passing here means: "the graph and the LUT are correct together on the host."
Only then does a device failure implicate the runtime.

  $PY_DEPLOY scripts/validate/parity_lutprobe.py \
      --onnx  $LLMDEPLOY_DATA/work/quant/qwen3-0.6b-w8a16-lutprobe-prefill/model_renamed.onnx \
      --lut   $LLMDEPLOY_DATA/work/lut/qwen3-0.6b \
      --model $LLMDEPLOY_DATA/models/Qwen3-0.6B
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "export"))
from modeling_export import MASK_VALUE, rope_tables, rope_theta_of  # noqa: E402

PROMPTS = [
    "What is 2+2? Answer with one number.",
    "The capital of France is",
    "Water boils at a temperature of",
]
AGREE_MIN = 0.75      # same bar quantize_aimet.py --eval uses (3/4)


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
    return out, n


def main():
    import onnxruntime as ort
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    ap = argparse.ArgumentParser()
    ap.add_argument("--onnx", required=True, type=Path)
    ap.add_argument("--lut", required=True, type=Path)
    ap.add_argument("--model", required=True, type=Path)
    ap.add_argument("--threads", type=int, default=0)
    args = ap.parse_args()

    so = ort.SessionOptions()
    if args.threads:
        so.intra_op_num_threads = args.threads
    sess = ort.InferenceSession(str(args.onnx), so, providers=["CPUExecutionProvider"])
    shapes = {i.name: i.shape for i in sess.get_inputs()}
    assert "inputs_embeds" in shapes, (
        f"first input is {list(shapes)[0]!r}, not 'inputs_embeds' -- qualla "
        "selects InputType::EMBEDDINGS by matching that literal name "
        "(nsp-model.cpp:668), so this build would be driven as token ids")
    AR = shapes["attention_mask"][-2]
    TOTAL = shapes["attention_mask"][-1]
    H = shapes["inputs_embeds"][-1]
    PAST = TOTAL - AR                 # 0 for a bertcache prefill
    past_names = sorted(n for n in shapes
                        if n.startswith("past_") and n.endswith("_in"))
    kind = "past-KV" if PAST else "bertcache"
    print(f"graph  : inputs_embeds {shapes['inputs_embeds']} AR={AR} "
          f"TOTAL={TOTAL} PAST={PAST} ({kind} prefill, "
          f"{len(past_names)} past inputs)")

    tok = AutoTokenizer.from_pretrained(str(args.model))
    hf = AutoModelForCausalLM.from_pretrained(str(args.model),
                                              dtype=torch.float32).eval()
    cfg = hf.config
    theta = rope_theta_of(cfg)

    agree = 0
    for prompt in PROMPTS:
        ids = tok(prompt, return_tensors="pt").input_ids[0].tolist()[:AR]
        n = len(ids)
        rows, n_embd = lut_rows(args.lut, ids)
        assert n_embd == H, f"LUT width {n_embd} != graph hidden {H}"

        emb = np.zeros((1, 1, AR, H), dtype=np.float32)
        mask = np.full((1, AR, TOTAL), MASK_VALUE, dtype=np.float32)
        if PAST:
            # Past-KV prefill with an EMPTY cache, per parity_e2e_vl.PrefillKV:
            # content LEFT-aligned, row i sees the valid past span (nothing here)
            # plus the causal new span [PAST, PAST+i]. Rows past n stay masked.
            emb[0, 0, :n] = rows
            for r in range(n):
                mask[0, r, PAST:PAST + r + 1] = 0.0
            pos = np.arange(n)
            row_out = n - 1
        else:
            # Bertcache prefill: right-aligned window, pad columns masked out,
            # positions restarted at the first real token -- exactly how
            # text_batches builds calibration windows.
            emb[0, 0, -n:] = rows
            for r in range(AR - n, AR):
                mask[0, r, AR - n:r + 1] = 0.0
            pos = np.concatenate([np.zeros(AR - n, dtype=np.int64), np.arange(n)])
            row_out = AR - 1
        c, s = rope_tables(torch.tensor(pos), cfg.head_dim, theta)
        cos = np.zeros((1, AR, cfg.head_dim // 2), dtype=np.float32)
        sin = np.zeros((1, AR, cfg.head_dim // 2), dtype=np.float32)
        if PAST:
            cos[0, :n] = c.numpy()[0]
            sin[0, :n] = s.numpy()[0]
        else:
            cos[:] = c.numpy()
            sin[:] = s.numpy()
        feeds = {"inputs_embeds": emb, "attention_mask": mask,
                 "position_ids_cos": cos, "position_ids_sin": sin}
        for nm in past_names:
            feeds[nm] = np.zeros([d if isinstance(d, int) else 1
                                  for d in shapes[nm]], dtype=np.float32)
        got = sess.run(["logits"], feeds)[0][0, row_out]

        with torch.no_grad():
            ref = hf(torch.tensor([ids])).logits[0, -1].numpy()

        a, b = int(np.argmax(got)), int(np.argmax(ref))
        ok = a == b
        agree += ok
        print(f"  {'OK ' if ok else 'MISS'} argmax graph={a:6d} hf={b:6d}  "
              f"{tok.decode([a])!r} vs {tok.decode([b])!r}   <- {prompt!r}")

    frac = agree / len(PROMPTS)
    print(f"\nlast-token argmax agreement: {agree}/{len(PROMPTS)} = {frac:.0%}")
    if frac < AGREE_MIN:
        print("FAIL: the embeddings-in build does not reproduce HF on the host.")
        print("      Do NOT ship this probe -- a garbage device result would be")
        print("      our own bug, not evidence about Genie's LUT feed.")
        return 1
    print("PASS: graph + LUT agree with HF on the host, so a device failure")
    print("      would implicate the runtime rather than this build.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
