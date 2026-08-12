#!/usr/bin/env python
"""Build the MULTIMODAL calibration/eval windows for the Qwen3-VL text tower.

W8A16 activation encodings are whatever calibration says they are, so the
calibration set has to look like production: a text tower that will be fed
spliced ViT features must be calibrated with spliced ViT features, not with
text-only prompts. Everything here is real:

- image features come from the Stage 1 ViT ONNX itself (fp32, the shipped
  graph), fed by the real HF image processor via make_pixel_values
- text embeddings come from the real embedding table, read straight out of the
  checkpoint's safetensors (the tower does NOT contain the table -- the runtime
  owns the lookup -- so no 18 GB model load is needed here)
- prompts go through the model's chat template, so the image tokens sit where
  the processor puts them, and the visual span is found by matching the image
  token id (151655) rather than by construction

Each output sample is ONE prefill window exactly as qualla feeds it
(nsp-model.cpp:1994-2016): content LEFT-aligned into the AR slots, remaining
slots carrying the pad-token embedding, pad rows fully masked. Three window
shapes are produced, matching how a real prompt is chunked AR tokens at a time
(kvmanager.cpp:430-465) -- one 512x512 image alone is 256 tokens, i.e. two full
AR=128 windows:

  chunk-0  text preamble + the head of the image span
  chunk-1  nothing but image features (the window a mid-image chunk sees)
  chunk-2  image tail + the question + the assistant generation prompt
  compact  a self-contained window: preamble + a PREFIX of the image span +
           question, so the whole multimodal turn fits in AR slots and the read
           row at n_valid-1 sees the image through attention. Feature rows are
           raster order over 2x2-merged patches, so a prefix is the top of the
           image -- real ViT output either way, which is what encodings see.
  text     chat-templated text-only turns (the tower still serves those)

THE ZERO-PADDING CONTRACT: deepstack_visual_embed_i must be exactly zero at
every non-visual position. The graph adds it to the residual stream
unconditionally (a static graph cannot scatter), so a non-zero value on a text
row silently adds visual features to a text token. Asserted per sample here,
and again at load time in quantize_aimet.vl_batches.

Run:
  $PY_DEPLOY scripts/quant/vl_calib_build.py \
      --model $LLMDEPLOY_DATA/models/Qwen3-VL-4B-Instruct \
      --vit-onnx $LLMDEPLOY_DATA/work/onnx/qwen3vl-4b-vit/vit.onnx \
      --out $LLMDEPLOY_DATA/work/quant/qwen3vl-4b-calib.npz --ar 128
"""
import argparse
import gc
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "export"))
from modeling_export import causal_mask, rope_tables  # noqa: E402
from modeling_vit_export import make_pixel_values  # noqa: E402

IMAGE_TOKEN_ID = 151655   # <|image_pad|>; config.json image_token_id
PAD_TOKEN = 151645        # qualla pad-token = first eos (context.cpp:51)
EMBED_KEY = "model.language_model.embed_tokens.weight"

# questions asked ABOUT the image (calibration images are noise, so the answer
# is meaningless; what matters is that the token mix around the span is real)
IMAGE_QUESTIONS = [
    "What is the single most prominent object in this picture?",
    "描述一下这张图片里的主要内容。",
    "List every colour you can see in the image, most common first.",
    "Is this photo taken indoors or outdoors? Explain in one sentence.",
]
IMAGE_QUESTIONS_EVAL = [
    "What is happening in this image?",
    "这张图片里有几个人?",
]
# text-only turns: the tower serves plain chat as well, and the text-only
# activation ranges must stay covered (deepstack all-zero is exactly that case)
TEXT_PROMPTS = [
    "The quick brown fox jumps over the lazy dog. Explain the sentence.",
    "def quicksort(arr):\n    if len(arr) <= 1:\n        return arr",
    "解释一下什么是注意力机制,它在Transformer中起什么作用?",
    "求解方程 x^2 - 5x + 6 = 0 的两个根。",
    "SELECT name, COUNT(*) FROM users GROUP BY name;",
    "深度学习模型的量化是指将浮点权重转换为低比特整数表示。",
]
TEXT_PROMPTS_EVAL = [
    "The capital of France is",
    "1+2+3+...+100 =",
]


def load_embed_table(model_dir):
    """The embedding table only, straight from safetensors (fp32)."""
    from safetensors import safe_open
    idx = json.loads((Path(model_dir) / "model.safetensors.index.json").read_text())
    shard = idx["weight_map"][EMBED_KEY]
    with safe_open(str(Path(model_dir) / shard), framework="pt") as f:
        w = f.get_tensor(EMBED_KEY)
    print(f"embed table {tuple(w.shape)} {w.dtype} from {shard}")
    return w.to(torch.float32)


def vit_features(vit_onnx, model_dir, seeds, n_deepstack):
    """Real ViT outputs per image: (image_features, [deepstack_i]) as [T, H]."""
    import onnxruntime as ort
    sess = ort.InferenceSession(str(vit_onnx), providers=["CPUExecutionProvider"])
    names = ["image_features"] + [f"deepstack_visual_embed_{i}" for i in range(n_deepstack)]
    out = {}
    for s in seeds:
        pv, grid = make_pixel_values(model_dir, seed=s, edge=512)
        got = sess.run(names, {"pixel_values": pv.numpy()})
        for n, a in zip(names, got):
            bad = ~np.isfinite(a)
            assert not bad.any(), f"seed {s} {n}: {int(bad.sum())} non-finite values"
        out[s] = (torch.from_numpy(got[0]), [torch.from_numpy(a) for a in got[1:]])
        print(f"  seed {s}: grid {grid.tolist()} -> {tuple(got[0].shape)} features; "
              + " ".join(f"{n} [{a.min():+.3f},{a.max():+.3f}]" for n, a in zip(names, got)))
    del sess
    gc.collect()
    return out


def build_sequence(proc, lut, feats, deeps, question, n_deepstack):
    """Full multimodal token sequence with real features spliced in.

    Returns (embeds [T,H], deep [n_deepstack][T,H], vis [T] bool, ids [T]).
    The visual span is located by matching IMAGE_TOKEN_ID, so it is wherever
    the processor's chat template actually put it.
    """
    msgs = [{"role": "user", "content": [{"type": "image"},
                                         {"type": "text", "text": question}]}]
    text = proc.apply_chat_template(msgs, add_generation_prompt=True, tokenize=False)
    # the processor expands one <|image_pad|> into grid//merge**2 copies; feed it
    # a dummy image of the same grid so the count matches the features we have
    from PIL import Image
    dummy = Image.fromarray(np.zeros((512, 512, 3), dtype=np.uint8))
    enc = proc(text=[text], images=[dummy], return_tensors="pt")
    ids = enc["input_ids"][0]
    vis = ids == IMAGE_TOKEN_ID
    n_vis = int(vis.sum())
    assert n_vis == feats.shape[0], (
        f"{n_vis} image tokens but {feats.shape[0]} feature rows -- grid mismatch")

    embeds = lut[ids.to(torch.long)].clone()
    embeds[vis] = feats
    deep = []
    for i in range(n_deepstack):
        d = torch.zeros_like(embeds)
        d[vis] = deeps[i]
        deep.append(d)
    return embeds, deep, vis, ids


def text_sequence(proc, tok, lut, prompt, n_deepstack):
    """Chat-templated text-only turn: no visual positions, deepstack all zero."""
    msgs = [{"role": "user", "content": [{"type": "text", "text": prompt}]}]
    text = proc.apply_chat_template(msgs, add_generation_prompt=True, tokenize=False)
    ids = tok(text, return_tensors="pt").input_ids[0]
    embeds = lut[ids.to(torch.long)].clone()
    vis = torch.zeros(ids.shape[0], dtype=torch.bool)
    deep = [torch.zeros_like(embeds) for _ in range(n_deepstack)]
    return embeds, deep, vis, ids


def window(embeds, deep, vis, start, end, ar, lut, head_dim, theta, desc):
    """One AR-slot prefill window, fed the way qualla feeds it.

    Content occupies slots 0..n-1 (LEFT-aligned, nsp-model.cpp:1994-2016), the
    rest carry the pad-token embedding and their rows are fully masked. rope
    positions are the ABSOLUTE sequence positions of the chunk.
    """
    n = end - start
    assert 0 < n <= ar, f"window [{start},{end}) does not fit AR={ar}"
    H = embeds.shape[1]

    emb = lut[PAD_TOKEN].repeat(ar, 1).clone()
    emb[:n] = embeds[start:end]
    w_vis = torch.zeros(ar, dtype=torch.bool)
    w_vis[:n] = vis[start:end]
    w_deep = []
    for d in deep:
        t = torch.zeros(ar, H)
        t[:n] = d[start:end]
        w_deep.append(t)

    mask = causal_mask(ar, ar).clone()      # [1, AR, AR] additive rank-3
    mask[:, n:, :] = -100.0                 # pad rows attend nothing

    pos = torch.zeros(ar, dtype=torch.long)
    pos[:n] = torch.arange(start, end)
    cos, sin = rope_tables(pos, head_dim, theta)

    # THE contract: deepstack is zero everywhere the window is not visual
    for j, t in enumerate(w_deep):
        off = t[~w_vis]
        assert not off.any(), (
            f"{desc}: deepstack level {j} has {int((off != 0).sum())} non-zero "
            "elements outside the visual span")
    assert not w_vis[n:].any(), f"{desc}: pad slots marked visual"

    return dict(embeds=emb[None, None].numpy(), mask=mask.numpy(),
                cos=cos.numpy(), sin=sin.numpy(),
                deep=np.stack([t[None, None].numpy() for t in w_deep]),
                vis_mask=w_vis.numpy(), n_valid=n, desc=desc)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--vit-onnx", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--ar", type=int, default=128)
    ap.add_argument("--n-deepstack", type=int, default=3)
    ap.add_argument("--images", type=int, default=4,
                    help="calibration images (seeds 0..N-1); eval uses seeds 100+")
    ap.add_argument("--compact-vis", type=int, default=96,
                    help="feature rows spliced into a self-contained window")
    args = ap.parse_args()

    from transformers import AutoProcessor, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(args.model)
    proc = AutoProcessor.from_pretrained(args.model, min_pixels=512 * 512,
                                         max_pixels=512 * 512)
    cfg = json.loads((Path(args.model) / "config.json").read_text())["text_config"]
    head_dim, theta = cfg["head_dim"], cfg["rope_theta"]
    AR = args.ar

    print("== ViT features (real Stage 1 graph) ==", flush=True)
    calib_seeds = list(range(args.images))
    eval_seeds = [100 + i for i in range(len(IMAGE_QUESTIONS_EVAL))]
    feats = vit_features(args.vit_onnx, args.model, calib_seeds + eval_seeds,
                         args.n_deepstack)

    lut = load_embed_table(args.model)
    print(f"embed table range [{float(lut.min()):+.4f}, {float(lut.max()):+.4f}]")

    samples, splits = [], []

    def add(s, split):
        samples.append(s)
        splits.append(split)

    print("== windows ==", flush=True)
    for k, seed in enumerate(calib_seeds):
        f, d = feats[seed]
        q = IMAGE_QUESTIONS[k % len(IMAGE_QUESTIONS)]
        embeds, deep, vis, ids = build_sequence(proc, lut, f, d, q, args.n_deepstack)
        T = int(ids.shape[0])
        # chunk the whole turn AR at a time, exactly like a real long prompt
        for c, start in enumerate(range(0, T, AR)):
            end = min(start + AR, T)
            add(window(embeds, deep, vis, start, end, AR, lut, head_dim, theta,
                       f"img{seed}-chunk{c}[{start}:{end}]"), "calib")
        # and one self-contained window: preamble + prefix of the span + question
        keep = torch.ones(T, dtype=torch.bool)
        span = vis.nonzero().flatten()
        keep[span[args.compact_vis:]] = False
        c_embeds, c_vis = embeds[keep], vis[keep]
        c_deep = [x[keep] for x in deep]
        add(window(c_embeds, c_deep, c_vis, 0, int(keep.sum()), AR, lut, head_dim,
                   theta, f"img{seed}-compact[{int(keep.sum())} tok, "
                          f"{args.compact_vis} visual]"), "calib")

    for p in TEXT_PROMPTS:
        embeds, deep, vis, ids = text_sequence(proc, tok, lut, p, args.n_deepstack)
        T = min(int(ids.shape[0]), AR)
        add(window(embeds, deep, vis, 0, T, AR, lut, head_dim, theta,
                   f"text[{T} tok] {p[:28]!r}"), "calib")

    # ---- held-out eval windows: self-contained, so the read row at n_valid-1
    # sees the whole turn (the prefill graph has no past KV to fall back on)
    for k, seed in enumerate(eval_seeds):
        f, d = feats[seed]
        q = IMAGE_QUESTIONS_EVAL[k]
        embeds, deep, vis, ids = build_sequence(proc, lut, f, d, q, args.n_deepstack)
        T = int(ids.shape[0])
        keep = torch.ones(T, dtype=torch.bool)
        span = vis.nonzero().flatten()
        keep[span[args.compact_vis:]] = False
        add(window(embeds[keep], [x[keep] for x in deep], vis[keep], 0,
                   int(keep.sum()), AR, lut, head_dim, theta,
                   f"EVAL img{seed} {q[:24]!r}"), "eval")
    for p in TEXT_PROMPTS_EVAL:
        embeds, deep, vis, ids = text_sequence(proc, tok, lut, p, args.n_deepstack)
        T = min(int(ids.shape[0]), AR)
        add(window(embeds, deep, vis, 0, T, AR, lut, head_dim, theta,
                   f"EVAL text {p[:28]!r}"), "eval")

    for s, sp in zip(samples, splits):
        print(f"  [{sp:5s}] n_valid={s['n_valid']:3d} visual={int(s['vis_mask'].sum()):3d} "
              f"{s['desc']}")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        out,
        embeds=np.concatenate([s["embeds"][None] for s in samples]),
        mask=np.concatenate([s["mask"][None] for s in samples]),
        cos=np.concatenate([s["cos"][None] for s in samples]),
        sin=np.concatenate([s["sin"][None] for s in samples]),
        deep=np.concatenate([s["deep"][None] for s in samples]),
        vis_mask=np.stack([s["vis_mask"] for s in samples]),
        n_valid=np.array([s["n_valid"] for s in samples]),
        desc=np.array([s["desc"] for s in samples]),
        split=np.array(splits),
        hidden=np.int64(samples[0]["embeds"].shape[-1]),
        ar=np.int64(AR),
        n_deepstack=np.int64(args.n_deepstack),
    )
    n_calib = splits.count("calib")
    print(f"wrote {out} ({out.stat().st_size / 1e6:.0f} MB): "
          f"{n_calib} calib + {len(splits) - n_calib} eval windows")


if __name__ == "__main__":
    main()
