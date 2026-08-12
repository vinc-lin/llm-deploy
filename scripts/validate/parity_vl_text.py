#!/usr/bin/env python
"""Gate 4 (Stage 2): Qwen3-VL text tower ONNX under qualla's REAL feed/read pattern.

A plain "ONNX matches HF" check does not catch the class of bug that shipped
five garbage bundles on 2026-08-11: the graph was numerically perfect, the
*feed and read pattern* was wrong. qualla's basic dialog

  - writes the prompt LEFT-ALIGNED into the AR window, slots 0..n-1, pad after
    (nsp-model.cpp:1994-2016, setupInput non-ENCODER path),
  - then samples logits at ROW n_process-1 of an ALL-POSITION logits tensor
    (nsp-model.cpp:3294-3295, "assumes right-padded input"),

so a [1, 1, V] last-token-only head passes Genie's load-time shape validation
and then reads past the end of a one-row buffer -> zeros -> argmax = token 0 =
"!". This script emulates exactly that feed and that read, for the Qwen3-VL
text tower (embeddings-in + deepstack), and is the analogue of
parity_qualla_read.py / parity_ladekv_read.py for the VL pipeline.

What it checks
  1. prefill logits are all-position [1, AR, V]  (structural)
  2. LEFT-aligned feed, read row n_process-1: argmax == HF greedy next token
  3. rows 0..n-1 agree with HF per-position logits (max|dlogits| reported)
  4. >= 8 decode steps through decode.onnx with the KV cache threaded per the
     Genie contract (outputs carry ONLY the new slice, keys transposed), greedy
     ARGMAX equality against HF at every step -- argmax, not a float tolerance
  5. deepstack fed non-zero over a CONTIGUOUS span and zero elsewhere, mirroring
     a real visual span; the HF reference uses the matching visual_pos_masks. A
     zero-deepstack control proves the inputs are not ignored.
  6. finiteness asserted on both sides BEFORE any reduction (Stage 1 lesson:
     max(0.0, nan) == 0.0, so a naive worst-case reduction swallows NaN)

Memory: the fp32 HF model (~18 GB) and a 15 GB ORT session do not comfortably
coexist, so the run is staged -- ALL HF references first, then the model is
freed (only the 1.5 GB embedding table is kept, because the runtime owns the
token lookup and we need it to embed whatever token the graph itself predicts),
then prefill.onnx, then decode.onnx. The two sessions are never live together.

Run:
  $PY_DEPLOY scripts/validate/parity_vl_text.py \
      --model $LLMDEPLOY_DATA/models/Qwen3-VL-4B-Instruct \
      --onnx  $LLMDEPLOY_DATA/work/onnx/qwen3vl-4b-text
"""
import argparse
import gc
import sys
import time
from pathlib import Path

import numpy as np
import onnxruntime as ort
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "export"))
from modeling_export import MASK_VALUE, causal_mask, rope_tables, rope_theta_of  # noqa: E402

PAD_TOKEN = 151645  # qualla pad-token defaults to the first eos token (context.cpp:51)
LOGIT_TOL = 5e-3    # reported/checked for the valid rows; the BAR is argmax equality

PROMPTS = [
    "You are a careful visual assistant. Study the photograph attached below, "
    "then answer in one short sentence. Question: what is the single most "
    "prominent object in the picture? Answer: the most prominent object is a",
    "Describe what you see in the image. In this picture there is",
]


def finite(name, arr):
    """Assert finiteness BEFORE any reduction -- max(0.0, nan) returns 0.0."""
    a = arr.detach().numpy() if isinstance(arr, torch.Tensor) else arr
    bad = ~np.isfinite(a)
    assert not bad.any(), f"{name}: {int(bad.sum())}/{a.size} non-finite values"


def visual_span(n):
    """A contiguous visual span strictly inside the prompt, never touching n-1.

    Real layout is <text><image tokens><text><generate here>, so the sampled row
    must sit outside the span: that way the span only reaches the read row
    through attention, which is what a real visual prompt does.
    """
    v0 = max(1, n // 4)
    v1 = min(n - 2, v0 + max(4, n // 4))
    assert v0 < v1 < n - 1, f"prompt too short (n={n}) to host a visual span"
    return v0, v1


def make_deepstack(n, hidden, n_deep, seed):
    """[1, n, H] per level: non-zero over the span, exactly zero elsewhere."""
    v0, v1 = visual_span(n)
    g = torch.Generator().manual_seed(seed)
    deep = []
    for _ in range(n_deep):
        d = torch.zeros(1, n, hidden)
        d[0, v0:v1] = torch.randn(v1 - v0, hidden, generator=g) * 0.02
        deep.append(d)
    return deep, v0, v1


# --------------------------------------------------------------------------- #
# phase A: every HF reference, computed while the model is alive
# --------------------------------------------------------------------------- #
def hf_references(model_dir, prompts, ar, steps):
    from transformers import AutoTokenizer, Qwen3VLForConditionalGeneration

    tok = AutoTokenizer.from_pretrained(model_dir)
    hf = Qwen3VLForConditionalGeneration.from_pretrained(
        model_dir, dtype=torch.float32, attn_implementation="eager"
    ).eval()
    cfg = hf.config.text_config
    lm = hf.model.language_model
    n_deep = 3

    refs = []
    for pi, text in enumerate(prompts):
        ids = tok(text, return_tensors="pt").input_ids[0, :ar]
        n = int(ids.shape[0])
        assert n < ar, (
            f"prompt fills the AR={ar} window (n={n}); row n-1 would coincide "
            "with the last row and the gate could not tell the two reads apart")
        embeds = lm.embed_tokens(ids[None].to(torch.long)).detach()   # [1, n, H]
        deep, v0, v1 = make_deepstack(n, cfg.hidden_size, n_deep, seed=pi)
        vmask = torch.zeros(1, n, dtype=torch.bool)
        vmask[0, v0:v1] = True

        with torch.no_grad():
            out = lm(
                inputs_embeds=embeds,
                attention_mask=torch.ones(1, n, dtype=torch.long),
                visual_pos_masks=vmask,
                deepstack_visual_embeds=[d[0, v0:v1] for d in deep],
                use_cache=True,
            )
            prefill_logits = hf.lm_head(out.last_hidden_state)[0]      # [n, V]
        finite(f"HF prefill logits (prompt {pi})", prefill_logits)
        toks = [int(prefill_logits[n - 1].argmax())]

        # decode reference: no visual positions exist past the prompt, so HF
        # gets no deepstack at all -- which is exactly the all-zero tensor the
        # graph is fed (adding 0 is a no-op).
        cache = out.past_key_values
        if pi == 0:
            for j in range(steps):
                with torch.no_grad():
                    o = lm(
                        inputs_embeds=lm.embed_tokens(
                            torch.tensor([[toks[-1]]], dtype=torch.long)).detach(),
                        attention_mask=torch.ones(1, n + j + 1, dtype=torch.long),
                        past_key_values=cache,
                        use_cache=True,
                    )
                    lg = hf.lm_head(o.last_hidden_state)[0, 0]
                finite(f"HF decode logits (step {j})", lg)
                cache = o.past_key_values
                toks.append(int(lg.argmax()))
        del cache, out
        gc.collect()

        refs.append(dict(text=text, ids=ids, n=n, embeds=embeds, deep=deep,
                         v0=v0, v1=v1, prefill_logits=prefill_logits, toks=toks))

    # the runtime owns the token LUT; keep it so we can embed whatever token the
    # GRAPH predicts (which is the whole point of an autoregressive check)
    lut = lm.embed_tokens.weight.detach().clone().numpy().astype(np.float32)
    theta, head_dim = rope_theta_of(cfg), cfg.head_dim
    meta = dict(hidden=cfg.hidden_size, head_dim=head_dim, theta=theta,
                layers=cfg.num_hidden_layers, n_kv=cfg.num_key_value_heads,
                vocab=cfg.vocab_size, n_deep=n_deep)
    del hf, lm, cfg
    gc.collect()
    return tok, lut, meta, refs


# --------------------------------------------------------------------------- #
# phase B: prefill.onnx, qualla feed + qualla read
# --------------------------------------------------------------------------- #
def qualla_prefill_feeds(ref, lut, meta, ar, zero_deepstack=False, right_align=False):
    """Build the exact window qualla would hand the graph.

    left-aligned (the real semantic): tokens at slots 0..n-1, pad after.
    right_align is offered ONLY so the mutation test can flip it.
    """
    n, H = ref["n"], meta["hidden"]
    off = ar - n if right_align else 0

    emb = np.tile(lut[PAD_TOKEN], (1, 1, ar, 1)).astype(np.float32)
    emb[0, 0, off:off + n] = ref["embeds"][0].numpy()

    mask = causal_mask(ar, ar).clone()          # [1, AR, AR] additive, rank-3
    if right_align:
        mask[:, :off, :] = MASK_VALUE           # pad rows (before) fully masked
        mask[:, off:, :off] = MASK_VALUE        # nobody attends the pad prefix
    else:
        mask[:, n:, :] = MASK_VALUE             # pad rows (after) fully masked

    pos = torch.zeros(ar, dtype=torch.long)
    pos[off:off + n] = torch.arange(n)
    cos, sin = rope_tables(pos, meta["head_dim"], meta["theta"])

    feeds = {
        "inputs_embeds": emb,
        "attention_mask": mask.numpy(),
        "position_ids_cos": cos.numpy(),
        "position_ids_sin": sin.numpy(),
    }
    for i, d in enumerate(ref["deep"]):
        w = np.zeros((1, 1, ar, H), dtype=np.float32)
        if not zero_deepstack:
            w[0, 0, off:off + n] = d[0].numpy()
        feeds[f"deepstack_visual_embed_{i}"] = w
    return feeds


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--onnx", required=True,
                    help="dir holding prefill/prefill.onnx and decode/decode.onnx")
    ap.add_argument("--ar", type=int, default=128, help="prefill window (AR)")
    ap.add_argument("--steps", type=int, default=8, help="decode steps (>= 8)")
    ap.add_argument("--threads", type=int, default=0, help="0 = onnxruntime default")
    args = ap.parse_args()
    assert args.steps >= 8, "the gate must exercise at least 8 decode steps"

    onnx_dir = Path(args.onnx)
    prefill_path = onnx_dir / "prefill" / "prefill.onnx"
    decode_path = onnx_dir / "decode" / "decode.onnx"
    for p in (prefill_path, decode_path):
        assert p.exists(), f"missing {p}"

    so = ort.SessionOptions()
    if args.threads:
        so.intra_op_num_threads = args.threads
    t0 = time.time()

    print("== phase A: HF references (model alive, no ORT session) ==", flush=True)
    tok, lut, meta, refs = hf_references(args.model, PROMPTS, args.ar, args.steps)
    AR, L, NKV, D = args.ar, meta["layers"], meta["n_kv"], meta["head_dim"]
    for pi, r in enumerate(refs):
        print(f"  prompt{pi}: n={r['n']:3d} visual span [{r['v0']},{r['v1']}) "
              f"read row {r['n'] - 1} (window last row is {AR - 1})  {r['text'][:40]!r}")
    print(f"  HF refs done, t={time.time() - t0:.0f}s", flush=True)

    failures = 0

    print("== phase B: prefill.onnx (qualla LEFT-aligned feed, read row n-1) ==",
          flush=True)
    sess = ort.InferenceSession(str(prefill_path), so,
                                providers=["CPUExecutionProvider"])
    lshape = {o.name: o.shape for o in sess.get_outputs()}["logits"]
    print(f"  prefill logits shape {lshape}")
    assert lshape[1] == AR, (
        f"logits covers {lshape[1]} position(s), need all {AR} -- a "
        "last-token-only head is the 2026-08-11 device-garbage root cause")

    kv_names = [f"past_{k}_{i}_out" for i in range(L) for k in ("key", "value")]
    cache_k = [None] * L
    cache_v = [None] * L
    seed_tok = None

    for pi, r in enumerate(refs):
        n = r["n"]
        outs = dict(zip(["logits"] + kv_names,
                        sess.run(["logits"] + kv_names,
                                 qualla_prefill_feeds(r, lut, meta, AR))))
        logits = outs["logits"][0]                       # [AR, V]
        finite(f"ONNX prefill logits (prompt {pi})", logits)

        row = torch.from_numpy(logits[n - 1])            # THE qualla read
        ref_row = r["prefill_logits"][n - 1]
        am_dev, am_ref = int(row.argmax()), r["toks"][0]
        d_row = float((row - ref_row).abs().max())
        d_all = float((torch.from_numpy(logits[:n]) - r["prefill_logits"]).abs().max())
        ok = am_dev == am_ref
        failures += 0 if ok else 1
        print(f"  [{'OK ' if ok else 'FAIL'}] prompt{pi} n={n} row{n - 1} argmax "
              f"dev={am_dev} ref={am_ref} ({tok.decode([am_dev])!r} vs "
              f"{tok.decode([am_ref])!r})  max|d row{n - 1}|={d_row:.2e}  "
              f"max|d rows0..{n - 1}|={d_all:.2e}")
        # argmax is the bar, but the valid rows are also checked numerically:
        # mutation testing showed that dropping the visual span from the feed
        # can leave the argmax intact while moving the row by ~6e-01, so
        # argmax alone is not a sufficient net for a wrong-feed bug.
        if d_all >= LOGIT_TOL:
            failures += 1
            print(f"  [FAIL] prompt{pi} valid-row logits drift {d_all:.2e} "
                  f">= {LOGIT_TOL:.0e}")

        # deepstack must matter: same feed, zeroed visual span
        zl = sess.run(["logits"], qualla_prefill_feeds(
            r, lut, meta, AR, zero_deepstack=True))[0][0]
        finite(f"ONNX prefill logits, zero deepstack (prompt {pi})", zl)
        dz = float(np.abs(zl[n - 1] - logits[n - 1]).max())
        print(f"        zero-deepstack delta at row{n - 1} = {dz:.3e} (must be >> 0)")
        assert dz > 1e-3, "deepstack inputs are ignored by the graph"

        if pi == 0:
            # Genie appends ONLY the valid slice into its cache (kvmanager.cpp
            # :430-465); the pad slots' KV is never retained.
            for i in range(L):
                cache_k[i] = outs[f"past_key_{i}_out"][:, :, :, :n].copy()
                cache_v[i] = outs[f"past_value_{i}_out"][:, :, :n, :].copy()
            seed_tok = am_dev
        del outs, logits, zl
        gc.collect()

    del sess
    gc.collect()
    print(f"  prefill done, t={time.time() - t0:.0f}s", flush=True)

    print(f"== phase C: decode.onnx ({args.steps} steps, KV threaded by hand) ==",
          flush=True)
    sess = ort.InferenceSession(str(decode_path), so,
                                providers=["CPUExecutionProvider"])
    shapes = {i.name: i.shape for i in sess.get_inputs()}
    PAST = shapes["past_key_0_in"][-1]
    TOTAL = shapes["attention_mask"][-1]
    assert shapes["past_key_0_in"] == [1, NKV, D, PAST], shapes["past_key_0_in"]
    assert shapes["past_value_0_in"] == [1, NKV, PAST, D], shapes["past_value_0_in"]
    assert TOTAL == PAST + 1, f"decode mask {TOTAL} != past {PAST} + 1"
    print(f"  past={PAST} total={TOTAL} keys[1,{NKV},{D},{PAST}] "
          f"values[1,{NKV},{PAST},{D}]")

    r0 = refs[0]
    n = r0["n"]
    # grow the prefill slice into the full decode-sized ring
    for i in range(L):
        k = np.zeros((1, NKV, D, PAST), dtype=np.float32)
        v = np.zeros((1, NKV, PAST, D), dtype=np.float32)
        k[:, :, :, :n] = cache_k[i]
        v[:, :, :n, :] = cache_v[i]
        cache_k[i], cache_v[i] = k, v
    gc.collect()

    zero_deep = {f"deepstack_visual_embed_{i}":
                 np.zeros((1, 1, 1, meta["hidden"]), dtype=np.float32)
                 for i in range(meta["n_deep"])}
    out_names = ["logits"] + [f"past_{k}_{i}_out"
                              for i in range(L) for k in ("key", "value")]

    cache_len = n
    cur = seed_tok
    agree = []
    dev_toks = [seed_tok]
    for j in range(args.steps):
        assert cache_len < PAST, f"decode ran past the {PAST}-slot cache"
        mask = np.full((1, 1, TOTAL), MASK_VALUE, dtype=np.float32)
        mask[0, 0, :cache_len] = 0.0     # everything committed to the cache
        mask[0, 0, PAST] = 0.0           # the new token itself (concat layout)
        cos, sin = rope_tables(torch.tensor([cache_len]), D, meta["theta"])

        feeds = {
            "inputs_embeds": lut[cur].reshape(1, 1, 1, -1).copy(),
            "attention_mask": mask,
            "position_ids_cos": cos.numpy(),
            "position_ids_sin": sin.numpy(),
            **zero_deep,
        }
        for i in range(L):
            feeds[f"past_key_{i}_in"] = cache_k[i]
            feeds[f"past_value_{i}_in"] = cache_v[i]

        outs = dict(zip(out_names, sess.run(out_names, feeds)))
        lg = outs["logits"][0, 0]
        finite(f"ONNX decode logits (step {j})", lg)
        ref_lg_tok = r0["toks"][j + 1]
        am = int(lg.argmax())
        ok = am == ref_lg_tok
        agree.append(ok)
        failures += 0 if ok else 1
        print(f"  [{'OK ' if ok else 'FAIL'}] step {j} pos={cache_len} fed={cur} "
              f"({tok.decode([cur])!r}) -> dev={am} ref={ref_lg_tok} "
              f"({tok.decode([am])!r} vs {tok.decode([ref_lg_tok])!r})", flush=True)

        # Genie contract: outputs are the NEW SLICE only, keys transposed
        for i in range(L):
            cache_k[i][:, :, :, cache_len:cache_len + 1] = outs[f"past_key_{i}_out"]
            cache_v[i][:, :, cache_len:cache_len + 1, :] = outs[f"past_value_{i}_out"]
        cache_len += 1
        cur = am           # true autoregression: feed what the GRAPH decided
        dev_toks.append(am)
        del outs
    del sess
    gc.collect()

    print(f"  decode argmax agreement {sum(agree)}/{len(agree)} steps "
          f"({100.0 * sum(agree) / len(agree):.0f}%)")
    print(f"  graph continuation: {tok.decode(dev_toks)!r}")
    print(f"  HF    continuation: {tok.decode(r0['toks'])!r}")
    print(f"wall {time.time() - t0:.0f}s")

    if failures:
        sys.exit(f"FAIL: {failures} qualla feed/read check(s) failed")
    print(f"PASS: qualla-pattern parity for the Qwen3-VL text tower "
          f"({len(refs)} prefill reads at row n-1 + {args.steps} decode steps)")


if __name__ == "__main__":
    main()
