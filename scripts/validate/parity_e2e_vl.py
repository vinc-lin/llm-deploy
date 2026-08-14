#!/usr/bin/env python
"""Gate 1 (Stage 3): full-path device-free parity -- image+text -> description.

HTP context binaries cannot execute on x86, so this ONNX-level gate is the ONLY
numerical proof that the ASSEMBLED pipeline is correct before it reaches a
device. It emulates the deployed host loop end to end: LUT lookup, image
features spliced at the <|image_pad|> rows, deepstack rows placed there and
zeroed everywhere else (the graph adds them at FULL WIDTH unconditionally, so
the host owns the zero-padding contract -- nothing in the graph enforces it),
interleaved MRoPE cos/sin, the text tower under qualla's real feed, then
free-running greedy generation with the Genie KV contract (outputs carry only
the NEW slice, keys transposed). Reference: hf.generate(greedy), token for token.

Four chains, all sharing one prompt/image:

  chain0-alldecode  HF visual + real deepstack, EVERY prompt row through
                    decode.onnx from an empty cache -- no prefill call at all.
                    BAR: token-for-token identical to hf.generate.
                    This is the path the DEVICE actually takes: qualla derives a
                    graph's ctx_size from the attention_mask trailing dim
                    (nsp-graph.cpp:146-155), our prefill is [1,128,128] so its
                    ctx_size == AR == 128, and prepareInferenceStrategy
                    (kvmanager.cpp:411-416) skips that bucket outright for a
                    ~290-token prompt. See docs/NOTES-genie-pipeline.md probe C.
  chain1-hf-vit     HF visual + real deepstack, prefill.onnx for rows 0..AR-1
                    then decode.onnx for the tail. BAR: token-for-token.
                    Validates the prefill graph, which still matters if prefill
                    is later re-exported with past-KV (the probe-C fix).
  chain2-onnx-vit   ONNX ViT features + real deepstack; the fully-ONNX path,
                    carrying the ViT's own fp32 trace delta. BAR: >= 75% step
                    agreement, text printed for human review.
  tierA-zero-deep   HF visual + ZERO deepstack. NOT gated. This is the text the
                    device will actually produce under the deepstack-zeros
                    degradation (no deepstack path exists in a stock Genie
                    pipeline); it is printed for the bundle README.

Mutations (chain1's prefill feed, row AR-1 logits) prove the gate is not
vacuous: zeroing deepstack must move the logits, and feeding flat rope instead
of interleaved MRoPE must move them.

Memory is staged exactly like parity_vl_text.py -- HF fp32 (~18 GB) first, then
freed (LUT + references kept, because the runtime owns the token lookup); ViT
ORT; prefill ORT; decode ORT. Two sessions are never live at once.

Run:
  $PY_DEPLOY scripts/validate/parity_e2e_vl.py \
      --model $LLMDEPLOY_DATA/models/Qwen3-VL-4B-Instruct \
      --vit-onnx $LLMDEPLOY_DATA/work/onnx/qwen3vl-4b-vit/vit.onnx \
      --text-onnx $LLMDEPLOY_DATA/work/onnx/qwen3vl-4b-text
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
from modeling_export import (  # noqa: E402
    MASK_VALUE, causal_mask, mrope_tables, rope_tables, rope_theta_of)

AR = 128
STEPS = 24
PROMPT = "Describe this image in one sentence."
CHAIN2_MIN_AGREE = 0.75
# same bar as parity_vl_text.py: one graph call vs one HF call, no accumulation
PREFILL_LOGIT_TOL = 5e-3
MUTATION_MIN = 1e-3

ALL_CHAINS = ("chain0-alldecode", "chain1-hf-vit", "chain2-onnx-vit",
              "tierA-zero-deep")
GATED_EXACT = ("chain0-alldecode", "chain1-hf-vit")


def finite(name, arr):
    """Assert finiteness BEFORE any reduction -- max(0.0, nan) returns 0.0."""
    a = arr.detach().numpy() if isinstance(arr, torch.Tensor) else arr
    bad = ~np.isfinite(a)
    assert not bad.any(), f"{name}: {int(bad.sum())}/{a.size} non-finite values"


def make_image():
    """Deterministic, semantically legible: red circle + blue square."""
    from PIL import Image, ImageDraw
    img = Image.new("RGB", (512, 512), "white")
    d = ImageDraw.Draw(img)
    d.ellipse((96, 96, 288, 288), fill=(220, 30, 30))
    d.rectangle((300, 300, 460, 460), fill=(30, 60, 220))
    return img


# --------------------------------------------------------------------------- #
# phase A: every HF reference, computed while the 18 GB fp32 model is alive
# --------------------------------------------------------------------------- #
def hf_phase(model_dir, steps):
    from transformers import AutoProcessor, Qwen3VLForConditionalGeneration

    proc = AutoProcessor.from_pretrained(
        model_dir, min_pixels=512 * 512, max_pixels=512 * 512)
    hf = Qwen3VLForConditionalGeneration.from_pretrained(
        model_dir, dtype=torch.float32, attn_implementation="eager").eval()
    cfg = hf.config.text_config

    messages = [{"role": "user", "content": [
        {"type": "image"}, {"type": "text", "text": PROMPT}]}]
    text = proc.apply_chat_template(messages, tokenize=False,
                                    add_generation_prompt=True)
    inputs = proc(text=[text], images=[make_image()], return_tensors="pt")
    ids = inputs.input_ids[0]
    pv, grid = inputs.pixel_values, inputs.image_grid_thw
    assert tuple(pv.shape) == (1024, 1536), tuple(pv.shape)
    assert grid.tolist() == [[1, 32, 32]], grid.tolist()
    n = int(ids.numel())
    assert AR < n, f"n={n} <= AR={AR}: the gate must exercise the decode tail"

    img_tok = hf.config.image_token_id
    img_pos = (ids == img_tok).nonzero()[:, 0]
    assert img_pos.numel() == 256, f"{img_pos.numel()} image_pad rows != 256"
    assert int(img_pos[-1] - img_pos[0]) == 255, "image span not contiguous"
    assert int(img_pos[0]) < AR, (
        "the image span starts past the prefill window; the prefill call would "
        "then see only flat text positions and prove nothing about MRoPE")

    # transformers 5.14 requires mm_token_type_ids (text=0/image=1/video=2),
    # supplied by the processor (see parity_mrope.py).
    pos3, deltas = hf.model.get_rope_index(
        ids[None], mm_token_type_ids=inputs.mm_token_type_ids,
        image_grid_thw=grid, attention_mask=torch.ones(1, n, dtype=torch.long))
    assert pos3.shape == (3, 1, n), pos3.shape
    pos3 = pos3[:, 0]                                        # [3, n]
    assert not torch.equal(pos3[0], pos3[1]), "no image span in positions?"
    mx = int(pos3.max())
    # HF's rope_deltas is max+1-n and the prompt's last row carries the max on
    # all three axes (trailing text advances all of t/h/w together), so a
    # generated token at sequence index n+j lands at position mx+1+j.
    assert int(deltas[0, 0]) == mx + 1 - n, (deltas.tolist(), mx, n)
    assert int(pos3[:, n - 1].min()) == mx == int(pos3[:, n - 1].max()), \
        "prompt tail does not carry the max position on all three axes"

    with torch.no_grad():
        vis = hf.model.visual(pv, grid)
    feats_hf = vis.pooler_output.detach().numpy().astype(np.float32)
    deep_hf = [d.detach().numpy().astype(np.float32)
               for d in vis.deepstack_features]
    for k, d in enumerate([feats_hf] + deep_hf):
        finite(f"HF visual[{k}]", d)
    assert feats_hf.shape == (256, cfg.hidden_size), feats_hf.shape

    # per-position reference logits for the whole prompt: lets every chain be
    # checked row by row, not just at the sampled row.
    with torch.no_grad():
        ref_logits = hf(**inputs).logits[0].detach().numpy().astype(np.float32)
    finite("HF prompt logits", ref_logits)
    assert ref_logits.shape[0] == n, ref_logits.shape

    # Record the position_ids HF itself uses at every generation step, so the
    # host-side formula below is proved against HF rather than assumed.
    seen = []
    hook = hf.model.language_model.rotary_emb.register_forward_pre_hook(
        lambda m, a, kw: seen.append(
            (kw.get("position_ids") if "position_ids" in kw else a[1]).detach()
            .clone()),
        with_kwargs=True)
    try:
        with torch.no_grad():
            gen = hf.generate(**inputs, do_sample=False, max_new_tokens=steps)
    finally:
        hook.remove()
    ref_new = gen[0, n:].tolist()
    assert len(ref_new) >= 8, f"HF stopped after {len(ref_new)} tokens"
    # the two HF paths must agree with each other before either is used as a bar
    assert int(ref_logits[n - 1].argmax()) == ref_new[0], (
        f"HF forward row {n - 1} argmax {int(ref_logits[n - 1].argmax())} != "
        f"hf.generate's first token {ref_new[0]}")

    assert len(seen) >= 2, f"only {len(seen)} rotary calls captured"
    assert tuple(seen[0].shape) == (3, 1, n), tuple(seen[0].shape)
    assert torch.equal(seen[0][:, 0], pos3), "HF prefill positions != pos3"
    for j in range(1, len(seen)):
        got = seen[j]
        assert tuple(got.shape) == (3, 1, 1), (j, tuple(got.shape))
        # HF feeds generated token j-1 (sequence index n+j-1) on rotary call j
        assert int(got.min()) == int(got.max()) == mx + j, (
            f"HF gen position at call {j} is {int(got.min())}, "
            f"host formula says {mx + j}")
    print(f"  position formula verified against HF for {len(seen) - 1} "
          f"generated tokens: index n+j -> position mx+1+j (mx={mx})")

    tok = proc.tokenizer
    print(f"  HF greedy ({len(ref_new)} toks): {tok.decode(ref_new)!r}",
          flush=True)

    eos = hf.generation_config.eos_token_id
    eos = [eos] if isinstance(eos, int) else list(eos or [])
    lut = hf.model.language_model.embed_tokens.weight.detach().numpy().astype(
        np.float32)
    meta = dict(hidden=cfg.hidden_size, head_dim=cfg.head_dim,
                theta=rope_theta_of(cfg), layers=cfg.num_hidden_layers,
                n_kv=cfg.num_key_value_heads, n_deep=len(deep_hf), mx=mx)
    ref = dict(tok=tok, lut=lut, meta=meta, ids=ids.numpy(), n=n,
               img_pos=img_pos.numpy(), pos3=pos3, pv=pv.numpy().copy(),
               feats_hf=feats_hf, deep_hf=deep_hf, ref_new=ref_new,
               ref_logits=ref_logits, eos=eos)
    del hf, vis, gen, inputs, proc, seen
    gc.collect()
    return ref


# --------------------------------------------------------------------------- #
# host-side feed construction (this is the code the device runtime must mirror)
# --------------------------------------------------------------------------- #
def spliced_inputs(ref, feats, deep):
    """Full-prompt embeds + FULL-WIDTH deepstack, zero outside the image span.

    HF scatter-adds deepstack at visual positions only; the graph adds it at
    full width unconditionally, so zeroing the non-visual rows here is what
    makes the two equivalent. Nothing downstream enforces it.
    """
    H = ref["meta"]["hidden"]
    n, img_pos = ref["n"], ref["img_pos"]
    emb = ref["lut"][ref["ids"]].copy()                      # [n, H]
    emb[img_pos] = feats
    deep_all = []
    for d in deep:
        w = np.zeros((n, H), dtype=np.float32)
        w[img_pos] = d
        deep_all.append(w)
    return emb, deep_all


def prefill_feeds(emb, deep_all, cos, sin, meta,
                  zero_deepstack=False, rope=None):
    H = meta["hidden"]
    c = cos[:, :AR].numpy().copy() if rope is None else rope[0]
    s = sin[:, :AR].numpy().copy() if rope is None else rope[1]
    feeds = {
        "inputs_embeds": emb[:AR].reshape(1, 1, AR, H).copy(),
        "attention_mask": causal_mask(AR, AR).numpy(),
        "position_ids_cos": c,
        "position_ids_sin": s,
    }
    for k, w in enumerate(deep_all):
        z = np.zeros((1, 1, AR, H), dtype=np.float32)
        if not zero_deepstack:
            z[0, 0] = w[:AR]
        feeds[f"deepstack_visual_embed_{k}"] = z
    return feeds


class Decoder:
    """decode.onnx driven with the Genie KV contract, one token at a time."""

    def __init__(self, sess, meta):
        self.sess, self.meta = sess, meta
        shapes = {i.name: i.shape for i in sess.get_inputs()}
        L, NKV, D = meta["layers"], meta["n_kv"], meta["head_dim"]
        self.PAST = shapes["past_key_0_in"][-1]
        self.TOTAL = shapes["attention_mask"][-1]
        assert shapes["past_key_0_in"] == [1, NKV, D, self.PAST], \
            shapes["past_key_0_in"]
        assert shapes["past_value_0_in"] == [1, NKV, self.PAST, D], \
            shapes["past_value_0_in"]
        assert self.TOTAL == self.PAST + 1, \
            f"decode mask {self.TOTAL} != past {self.PAST} + 1"
        self.out_names = ["logits"] + [f"past_{s}_{i}_out"
                                       for i in range(L) for s in ("key", "value")]
        self.ck = self.cv = None
        self.cache_len = 0

    def reset(self, ck=None, cv=None, cache_len=0):
        """Allocate a full-width ring, optionally seeded with a prefill slice."""
        L, NKV, D = self.meta["layers"], self.meta["n_kv"], self.meta["head_dim"]
        self.ck = [np.zeros((1, NKV, D, self.PAST), dtype=np.float32)
                   for _ in range(L)]
        self.cv = [np.zeros((1, NKV, self.PAST, D), dtype=np.float32)
                   for _ in range(L)]
        if ck is not None:
            for i in range(L):
                self.ck[i][:, :, :, :cache_len] = ck[i]
                self.cv[i][:, :, :cache_len, :] = cv[i]
        self.cache_len = cache_len
        gc.collect()

    def step(self, emb_row, deep_rows, cos1, sin1):
        H, L = self.meta["hidden"], self.meta["layers"]
        assert self.cache_len < self.PAST, \
            f"decode ran past the {self.PAST}-slot cache"
        mask = np.full((1, 1, self.TOTAL), MASK_VALUE, dtype=np.float32)
        mask[0, 0, :self.cache_len] = 0.0    # everything committed to the cache
        mask[0, 0, self.PAST] = 0.0          # the new token itself (concat layout)
        feeds = {
            "inputs_embeds": emb_row.reshape(1, 1, 1, H).astype(np.float32),
            "attention_mask": mask,
            "position_ids_cos": np.ascontiguousarray(cos1, dtype=np.float32),
            "position_ids_sin": np.ascontiguousarray(sin1, dtype=np.float32),
        }
        for k in range(self.meta["n_deep"]):
            feeds[f"deepstack_visual_embed_{k}"] = \
                deep_rows[k].reshape(1, 1, 1, H).astype(np.float32)
        for i in range(L):
            feeds[f"past_key_{i}_in"] = self.ck[i]
            feeds[f"past_value_{i}_in"] = self.cv[i]

        outs = dict(zip(self.out_names, self.sess.run(self.out_names, feeds)))
        # Genie contract: the outputs are the NEW SLICE only, keys transposed
        for i in range(L):
            self.ck[i][:, :, :, self.cache_len:self.cache_len + 1] = \
                outs[f"past_key_{i}_out"]
            self.cv[i][:, :, self.cache_len:self.cache_len + 1, :] = \
                outs[f"past_value_{i}_out"]
        self.cache_len += 1
        return outs["logits"][0, 0]


def run_tail(label, dec, ref, emb, deep_all, cos, sin, start_row):
    """Prompt rows start_row..n-1 teacher-forced, then free-running generation."""
    meta, tok = ref["meta"], ref["tok"]
    n, H, D = ref["n"], meta["hidden"], meta["head_dim"]
    ref_logits, ref_new, mx = ref["ref_logits"], ref["ref_new"], meta["mx"]

    worst, mism, lg = 0.0, [], None
    t0 = time.time()
    for r in range(start_row, n):
        lg = dec.step(emb[r], [d[r] for d in deep_all],
                      cos[:, r:r + 1].numpy(), sin[:, r:r + 1].numpy())
        finite(f"{label} prompt row {r} logits", lg)
        worst = max(worst, float(np.abs(lg - ref_logits[r]).max()))
        if int(lg.argmax()) != int(ref_logits[r].argmax()):
            mism.append(r)
        if (r - start_row) % 64 == 63:
            print(f"    {label} prompt row {r}/{n - 1} "
                  f"({time.time() - t0:.0f}s)", flush=True)
    assert lg is not None, f"{label}: no prompt rows fed"
    print(f"  [{label}] prompt rows {start_row}..{n - 1}: max|dlogits| vs HF "
          f"= {worst:.3e}, argmax mismatches {len(mism)}"
          + (f" at rows {mism[:8]}" if mism else ""), flush=True)

    zero_row = [np.zeros(H, dtype=np.float32)] * meta["n_deep"]
    got, agree = [int(lg.argmax())], []
    agree.append(got[0] == ref_new[0])
    for j in range(1, len(ref_new)):
        # generated token j-1 sits at sequence index n+j-1; HF continues all
        # three axes from the prompt max, so its position is mx+j (verified
        # against HF's own rotary calls in phase A).
        c, s = mrope_tables(torch.full((3, 1), mx + j), D, meta["theta"])
        lg = dec.step(ref["lut"][got[j - 1]], zero_row, c.numpy(), s.numpy())
        finite(f"{label} gen step {j} logits", lg)
        got.append(int(lg.argmax()))
        agree.append(got[j] == ref_new[j])
    first_bad = next((j for j, a in enumerate(agree) if not a), None)
    print(f"  [{label}] generated {sum(agree)}/{len(agree)} match HF"
          + (f" (first divergence at step {first_bad})" if first_bad is not None
             else " (token-for-token identical)"))
    print(f"  [{label}] text: {tok.decode(got)!r}", flush=True)
    return got, agree


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--vit-onnx", required=True)
    ap.add_argument("--text-onnx", required=True,
                    help="dir holding prefill/prefill.onnx and decode/decode.onnx")
    ap.add_argument("--steps", type=int, default=STEPS)
    ap.add_argument("--chains", default=",".join(ALL_CHAINS),
                    help="comma-separated subset of " + ",".join(ALL_CHAINS))
    ap.add_argument("--threads", type=int, default=0, help="0 = ORT default")
    args = ap.parse_args()
    chains = [c.strip() for c in args.chains.split(",") if c.strip()]
    for c in chains:
        assert c in ALL_CHAINS, f"unknown chain {c!r}"

    text_dir = Path(args.text_onnx)
    prefill_path = text_dir / "prefill" / "prefill.onnx"
    decode_path = text_dir / "decode" / "decode.onnx"
    for p in (Path(args.vit_onnx), prefill_path, decode_path):
        assert p.exists(), f"missing {p}"
    so = ort.SessionOptions()
    if args.threads:
        so.intra_op_num_threads = args.threads
    t0 = time.time()
    failures = []

    print("== phase A: HF reference (model alive, no ORT session) ==", flush=True)
    ref = hf_phase(args.model, args.steps)
    meta, n, L = ref["meta"], ref["n"], ref["meta"]["layers"]
    print(f"  n={n} image rows [{ref['img_pos'][0]},{ref['img_pos'][-1]}] "
          f"layers={L} hidden={meta['hidden']}  t={time.time() - t0:.0f}s",
          flush=True)

    print("== phase B: ViT ONNX ==", flush=True)
    sess = ort.InferenceSession(str(args.vit_onnx), so,
                                providers=["CPUExecutionProvider"])
    vit = dict(zip([o.name for o in sess.get_outputs()],
                   sess.run(None, {"pixel_values": ref["pv"]})))
    del sess
    gc.collect()
    feats_ort = vit["image_features"]
    deep_ort = [vit[f"deepstack_visual_embed_{k}"] for k in range(meta["n_deep"])]
    for name, a, b in zip(["image_features"]
                          + [f"deepstack_visual_embed_{k}"
                             for k in range(meta["n_deep"])],
                          [feats_ort] + deep_ort,
                          [ref["feats_hf"]] + ref["deep_hf"]):
        finite(f"ViT ONNX {name}", a)
        assert a.shape == b.shape, f"{name}: {a.shape} != {b.shape}"
        print(f"  {name:26s} max|d| vs HF = {np.abs(a - b).max():.3e}")
    print(f"  ViT done, t={time.time() - t0:.0f}s", flush=True)

    variants = {
        "chain0-alldecode": (ref["feats_hf"], ref["deep_hf"]),
        "chain1-hf-vit": (ref["feats_hf"], ref["deep_hf"]),
        "chain2-onnx-vit": (feats_ort, deep_ort),
        "tierA-zero-deep": (ref["feats_hf"],
                            [np.zeros_like(d) for d in ref["deep_hf"]]),
    }
    cos, sin = mrope_tables(ref["pos3"], meta["head_dim"], meta["theta"])
    finite("host mrope cos", cos)
    finite("host mrope sin", sin)
    feeds_cache = {c: spliced_inputs(ref, *variants[c]) for c in chains}

    # ----------------------------------------------------------------- prefill
    need_prefill = [c for c in chains if c != "chain0-alldecode"]
    seeds = {}
    if need_prefill:
        print("== phase C: prefill.onnx (rows 0..%d) ==" % (AR - 1), flush=True)
        sess = ort.InferenceSession(str(prefill_path), so,
                                    providers=["CPUExecutionProvider"])
        lshape = {o.name: o.shape for o in sess.get_outputs()}["logits"]
        print(f"  prefill logits shape {lshape}")
        assert lshape[1] == AR, (
            f"logits covers {lshape[1]} position(s), need all {AR} -- a "
            "last-token-only head is the 2026-08-11 device-garbage root cause")
        kv_names = [f"past_{s}_{i}_out" for i in range(L) for s in ("key", "value")]

        for c in need_prefill:
            emb, deep_all = feeds_cache[c]
            outs = dict(zip(["logits"] + kv_names,
                            sess.run(["logits"] + kv_names,
                                     prefill_feeds(emb, deep_all, cos, sin, meta))))
            lg = outs["logits"][0]
            finite(f"{c} prefill logits", lg)
            d = float(np.abs(lg - ref["ref_logits"][:AR]).max())
            mism = int((lg.argmax(-1) != ref["ref_logits"][:AR].argmax(-1)).sum())
            print(f"  [{c}] prefill rows 0..{AR - 1}: max|dlogits| vs HF = "
                  f"{d:.3e}, argmax mismatches {mism}", flush=True)
            if c == "chain1-hf-vit" and d >= PREFILL_LOGIT_TOL:
                failures.append(f"{c}: prefill logit drift {d:.3e} "
                                f">= {PREFILL_LOGIT_TOL:.0e}")
            seeds[c] = ([outs[f"past_key_{i}_out"].copy() for i in range(L)],
                        [outs[f"past_value_{i}_out"].copy() for i in range(L)])
            del outs, lg
            gc.collect()

        # mutations: chain1's feed, row AR-1 logits must MOVE
        emb, deep_all = feeds_cache.get("chain1-hf-vit") or \
            spliced_inputs(ref, *variants["chain1-hf-vit"])
        base = sess.run(["logits"],
                        prefill_feeds(emb, deep_all, cos, sin, meta))[0][0][AR - 1]
        zd = sess.run(["logits"],
                      prefill_feeds(emb, deep_all, cos, sin, meta,
                                    zero_deepstack=True))[0][0][AR - 1]
        fc, fs = rope_tables(torch.arange(AR), meta["head_dim"], meta["theta"])
        fr = sess.run(["logits"],
                      prefill_feeds(emb, deep_all, cos, sin, meta,
                                    rope=(fc.numpy(), fs.numpy())))[0][0][AR - 1]
        for nm, x in (("base", base), ("zero-deepstack", zd), ("flat-rope", fr)):
            finite(f"mutation {nm}", x)
        dz = float(np.abs(zd - base).max())
        dm = float(np.abs(fr - base).max())
        print(f"  mutation zero-deepstack row{AR - 1} delta = {dz:.3e} "
              f"(must be > {MUTATION_MIN:.0e})")
        print(f"  mutation flat-rope      row{AR - 1} delta = {dm:.3e} "
              f"(must be > {MUTATION_MIN:.0e})")
        assert dz > MUTATION_MIN, "deepstack inputs are ignored -- gate is vacuous"
        assert dm > MUTATION_MIN, "mrope tables are ignored -- gate is vacuous"
        del sess, base, zd, fr
        gc.collect()
        print(f"  prefill done, t={time.time() - t0:.0f}s", flush=True)

    # ------------------------------------------------------------------ decode
    print("== phase D: decode.onnx (prompt tail + free-running generation) ==",
          flush=True)
    sess = ort.InferenceSession(str(decode_path), so,
                                providers=["CPUExecutionProvider"])
    dec = Decoder(sess, meta)
    print(f"  past={dec.PAST} total={dec.TOTAL} "
          f"keys[1,{meta['n_kv']},{meta['head_dim']},{dec.PAST}] "
          f"values[1,{meta['n_kv']},{dec.PAST},{meta['head_dim']}]", flush=True)
    assert n < dec.PAST, f"prompt n={n} does not fit the {dec.PAST}-slot cache"

    results = {}
    for c in chains:
        print(f"  -- {c} (t={time.time() - t0:.0f}s) --", flush=True)
        emb, deep_all = feeds_cache[c]
        if c == "chain0-alldecode":
            dec.reset()
            start = 0
        else:
            ck, cv = seeds.pop(c)
            dec.reset(ck, cv, AR)
            del ck, cv
            gc.collect()
            start = AR
        results[c] = run_tail(c, dec, ref, emb, deep_all, cos, sin, start)
    dec.ck = dec.cv = None
    del sess, dec
    gc.collect()

    # ------------------------------------------------------------------ verdict
    print("== verdict ==")
    tok, ref_new = ref["tok"], ref["ref_new"]
    print(f"  HF reference     : {tok.decode(ref_new)!r}")
    for c in chains:
        got, agree = results[c]
        rate = sum(agree) / len(agree)
        print(f"  {c:17s}: {sum(agree)}/{len(agree)} ({100 * rate:.0f}%) "
              f"{tok.decode(got)!r}")
        if c in GATED_EXACT and sum(agree) != len(agree):
            failures.append(f"{c}: {len(agree) - sum(agree)} generated token(s) "
                            "differ from hf.generate")
        if c == "chain2-onnx-vit" and rate < CHAIN2_MIN_AGREE:
            failures.append(f"{c}: agreement {rate:.2f} < {CHAIN2_MIN_AGREE}")
    if "tierA-zero-deep" in results:
        # every chain runs a FIXED len(ref_new) steps so the comparison stays
        # aligned; a real runtime stops at eos, so cut there for the README.
        raw = results["tierA-zero-deep"][0]
        cut = next((i for i, t in enumerate(raw) if t in ref["eos"]), len(raw))
        print()
        print("  >>> TIER A (deepstack zeroed) -- the text the DEVICE pipeline "
              "will produce; goes verbatim into the bundle README:")
        print(f"  >>> {tok.decode(raw[:cut])!r}")
        print(f"  >>> (raw, past eos: {tok.decode(raw)!r})")
        print()
    print(f"wall {time.time() - t0:.0f}s")

    if failures:
        for f in failures:
            print(f"FAIL: {f}")
        sys.exit(f"FAIL: {len(failures)} check(s) failed")
    print(f"PASS: full-path device-free parity ({len(chains)} chains, "
          f"n={n} prompt rows, {len(ref_new)} generated tokens)")


if __name__ == "__main__":
    main()
