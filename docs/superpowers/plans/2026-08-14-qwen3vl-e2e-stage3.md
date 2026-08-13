# Qwen3-VL-4B End-to-End Pipeline (Stage 3) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Wire the shipped vision tower and text tower into a working
image+text → description pipeline for the SA8797P, prove full-path numerical
parity against HF device-free, and upload the assembled pipeline bundle to
`vinccniv/sa8797p-qwen3vl-4b-bundles`.

**Architecture:** Tier A — a stock `genie-app` `GeniePipeline`
(imageEncoder → lutEncoder → textGenerator), modeled byte-for-byte on the
SDK's own GLM-4v example, with the three deepstack tensors fed zeros
(= exactly HF-minus-deepstack; the graph was built so zeros are a no-op add).
Full-fidelity deepstack has no route through a stock pipeline
(`ImageEncoder.cpp` exposes ONE output) and is **explicitly out of scope**
until Tier A has device results — see "Scope" below. Device-free validation
is a new full-path gate: real image → ViT ONNX → splice → text ONNX under
qualla's exact feed → token-for-token vs HF `generate`.

**Tech Stack:** PyTorch / transformers 5.x (`qwen3-deploy` env), ONNX Runtime,
QAIRT 2.48.40.260702 examples (`genie-app`, qualla source), HF Hub.

---

## Scope — what "fully functional" means here

The user's standing goal is **"images work at all"** (capability, not speed or
peak quality). This plan delivers:

- **Ships and runs on device (untested here):** stock `genie-app` pipeline,
  image + text in → description out, deepstack = zeros (HF-minus-deepstack).
- **Proven device-free:** the *full-fidelity* path (real deepstack) is
  numerically token-exact vs HF through the ONNX chain, so the only quality
  gap on device is the documented deepstack-zeros deviation plus W8A16.
- **Out of scope (deferred, recorded in README):** Tier B — a custom QNN
  driver reading all four ViT outputs to feed real deepstack on device. Do
  not start it in this plan; it is pointless before first device results.

**The Stage 3 device gate cannot close on this machine.** HTP ctx-bins do not
execute on x86. "Validated" below means: full-path numerical parity + static
contract checks + a device test script for the user to run. Say this plainly
in every README this plan touches.

## Context an implementer needs

Read first: `docs/superpowers/specs/2026-08-12-qwen3-vl-4b-sa8797p-design.md`
(§6 Stage 3, §7-RESOLVED), `docs/NOTES-genie-io.md`, `docs/NOTES-genie-splits.md`,
`CLAUDE.md` (HF proxy/watchdog gotchas, disk_guard).

**Always `source scripts/env.sh` first.** Use `$PY_DEPLOY`, never bare
`python`. No pytest — gates are standalone argparse scripts in
`scripts/validate/` that assert and exit non-zero. Mutation-test every new
gate (flip one contract knob, watch it fail) before trusting a PASS.

**RAM:** the E2E gate stages an 18 GB fp32 HF model and one ~15 GB ORT
session at a time (never two sessions live at once — same discipline as
`parity_vl_text.py`). Fits local WSL (63 GB); `tank` is the fallback.
`disk_guard` before any multi-GB write.

### Existing artifacts (all already built and gated)

| Artifact | Where |
|---|---|
| ViT ctx-bin (FP16, graph `vit`) | `$LLMDEPLOY_DATA/work/ctxbin/qwen3vl-4b-vit-fp16/qwen3vl-4b-vit-fp16_ctx.bin` + `info.json` |
| Text ctx-bins (2-split W8A16) | `$LLMDEPLOY_DATA/work/ctxbin/qwen3vl-4b-w8a16-split/{1,2}_of_2/` + `info.json` |
| Text ONNX (prefill/decode, unsplit) | `$LLMDEPLOY_DATA/work/onnx/qwen3vl-4b-text/{prefill,decode}/` |
| ViT ONNX | `$LLMDEPLOY_DATA/work/onnx/qwen3vl-4b-vit/vit.onnx` |
| Embedding LUT (float32) | `$LLMDEPLOY_DATA/work/lut/qwen3vl-4b/embedding_float32_lut.bin` (+params json) |
| Assembled device bundles | `$LLMDEPLOY_DATA/bundles/qwen3vl_4b_text_w8a16/`, `.../qwen3vl_4b_vit_fp16/` |
| HF model | `$LLMDEPLOY_DATA/models/Qwen3-VL-4B-Instruct` |

### Contracts this plan builds on (verified, with sources)

1. **ViT ctx-bin IO** (read from `info.json`): input `pixel_values [1024,1536]`
   **QNN_DATATYPE_FLOAT_16**; outputs `image_features [256,2560]` +
   `deepstack_visual_embed_{0,1,2} [256,2560]`. `node set image` feeds an
   **opaque blob** (no preprocessing in Genie — spec §7-RESOLVED), so the
   `.raw` must be exactly 1024·1536·2 = **3,145,728 bytes of fp16**.
2. **Static grid:** 1024 patches = 32×32 grid of 16-px patches = a 512×512
   image; merge 2 → 256 embedding rows. Host must resize (distort) every
   image to 512×512.
3. **Stock `ImageEncoder` node exposes ONE output**
   (`GENIE_NODE_IMAGE_ENCODER_EMBEDDING_OUTPUT`;
   `examples/Genie/Genie/src/pipeline/ImageEncoder.cpp`). Deepstack has no
   pipeline route → Tier A zeros.
4. **`genie-app` is the driver** (`-s <script>`; `main.cpp:3791`). The GLM-4v
   example (`examples/Genie/genie-app/scripts/glm-4v`,
   `examples/Genie/configs/glm-4v/*.json`) is the schema precedent for all
   three node configs and the pipeline script.
5. **Text prefill graph has NO past-KV** (`export_qwen3vl_text.py` exports
   prefill with `past=0`). An image prompt is ~290 tokens > AR=128, so the
   tail of the prompt must go through the **decode** graph one token at a
   time. The E2E gate emulates exactly that (prefill first 128 rows, decode
   the rest). Probe C confirms Genie does the same.
6. **HF interleaved MRoPE** (`transformers/models/qwen3_vl/modeling_qwen3_vl.py`,
   `apply_interleaved_mrope`): half-dim 64; t/h/w interleave as
   `[THWTHW...TT]` — h owns dims 1,4,…,58; w owns 2,5,…,59; t owns the rest
   incl. the 60–63 tail (`mrope_section=[24,20,20]`).
7. **KV cache advances by `n_process`** (`kvmanager.cpp:454`); decode outputs
   carry only the new slice; keys transposed `[1,8,128,P]`.

### File structure

| File | Action | Responsibility |
|---|---|---|
| `scripts/export/modeling_export.py` | modify | add `mrope_tables()` next to `rope_tables()` |
| `scripts/validate/parity_mrope.py` | create | Gate 0: `mrope_tables` vs HF rotary, exact |
| `scripts/pipeline/preprocess_image.py` | create | image file → fp16 `.raw` + meta json |
| `scripts/validate/parity_e2e_vl.py` | create | **Gate 1: full-path image→tokens vs HF `generate`** |
| `configs/genie_text_generator_qwen3vl_4b.json` | create | textGenerator node config (pipeline twin of the dialog config) |
| `configs/genie_text_encoder_qwen3vl.json` | create | lutEncoder node config (float32 LUT) |
| `configs/genie_pipeline_qwen3vl.script` | create | genie-app pipeline script |
| `scripts/build/vl_pipeline_bundle.sh` | create | assemble the flat e2e device bundle |
| `scripts/validate/lint_pipeline_bundle.py` | create | Gate 2: static bundle contract checks |
| `docs/DEVICE_TEST_qwen3vl_e2e.md` | create | device run + triage handoff |
| `docs/NOTES-genie-io.md` | modify | append probe findings |

---

### Task 0: Branch

**Files:** none (git only)

- [ ] **Step 0.1:** `cd /mnt/x/code/llm-deploy && git fetch origin && git checkout main && git pull && git checkout -b qwen3vl-4b-stage3`
- [ ] **Step 0.2:** `source scripts/env.sh && disk_guard 6` — confirm both succeed before anything else.

---

## Phase 0 — Source probes (read-only, ~1h, do these FIRST)

Each probe answers a question the configs depend on. Findings go into
`docs/NOTES-genie-io.md` under a new `## Stage 3 probes (2026-08-14)` heading,
each with a file:line citation — same style as the existing notes. If a probe
contradicts an assumption below, STOP and revisit the affected task before
writing code.

Source root for all probes:
`Q=/home/vinc/llm-local/sdk/qairt/2.48.40.260702/examples/Genie/Genie/src`

### Task 1: Probe A — do the deepstack graph inputs read zeros under stock Genie?

Tier A's correctness rests on this: qualla only *fills* tensors it knows by
name; the deepstack inputs are unknown to it. Buffers are allocated
generically (`QnnApi::allocateAll`, `qualla/engines/qnn-api/QnnApi.cpp:2901` →
`IOTensor::allocateBuffers`). The question is whether those buffers are
guaranteed zero.

- [ ] **Step 1.1:** Find the allocator:
  `grep -rn "allocateBuffers\|rpcmem_alloc\|std::calloc\|memset" $Q/qualla/engines/qnn-api/ | head -30`
  and read the implementation (likely `IOTensor.cpp` / `IBufferAlloc`).
- [ ] **Step 1.2:** Determine: fresh ION/dmabuf or heap pages zero-filled by
  allocator or kernel? Any explicit memset? Also confirm model load does not
  reject unknown input names: re-read `QnnNspModel::validateModel`
  (`$Q/qualla/engines/qnn-htp/nsp-model.cpp:619` on) — it looks up known
  names, never enumerates graph inputs demanding recognition.
- [ ] **Step 1.3:** Record verdict in NOTES with citations. Decision matrix:
  - **Zero-guaranteed** → Tier A ships with existing ctx-bins. Proceed.
  - **Not guaranteed / load rejects** → execute Contingency 1b before Phase 2.
- [ ] **Step 1.4:** Commit: `git add docs/NOTES-genie-io.md && git commit -m "docs: probe A — deepstack input buffer initialization under stock Genie"`

**Contingency 1b (ONLY if Probe A fails):** rebuild split 1 without deepstack
inputs: `export_qwen3vl_text.py` already threads `n_deepstack` through
`io_spec`/`export_graph` — export chunk 0 with `nd=0` (zeros-add is exactly
absent-add, so encodings stay valid), re-run `split_aimet_onnx.py` /
`split_encodings.py` / `unify_pair_weights.py` / `vl_text_ctxbin_split.sh` for
chunk 0 only, re-verify with `parity_vl_text_split.py`. This is a ~half-day
tank job; do not start it speculatively.

### Task 2: Probe B — Genie's `qwen3vl-mrope`: does it match HF, and where does the grid come from?

Our dialog/text-generator configs declare
`"rope-scaling": {"rope-type": "qwen3vl-mrope", "mrope-section": [24,20,20], "spatial-merge-size": 2, "time-step": 2}` —
a schema written from config-key grepping, never yet parsed by a running Genie.

- [ ] **Step 2.1:** `grep -rn "QWEN3VL_MROPE\|qwen3vl-mrope\|mrope" $Q --include=*.cpp --include=*.hpp -l` then read the hits
  (`Dialog.cpp`, `Engine.cpp`, `pipeline/TextGenerator.cpp`,
  `qualla/engines/qnn-htp/nsp-model.cpp`, `nsp-utils/nsp-params.cpp`).
- [ ] **Step 2.2:** Answer, with citations: (a) config keys actually parsed
  (exact spelling; fix our configs if they differ); (b) is the cos/sin
  layout the HF **interleaved** `[THWTHW...TT]` or the older blocked
  Qwen2-VL layout; (c) how image rows get t/h/w — presumably the ImageEncoder
  marks its accumulator rows and Genie derives an h×w grid from row count +
  `spatial-merge-size` (256 rows → 16×16); note any square-grid assumption;
  (d) do text positions after an image continue at `max+1` (HF semantics).
- [ ] **Step 2.3:** Record in NOTES. Decision matrix:
  - Layout matches HF → proceed.
  - Layout is blocked/other → the device will diverge from HF numerics for
    image spans; record precisely, add it to the bundle README limitations,
    and (if the delta is a layout permutation) note the future option of
    baking the permutation into export. Do NOT silently change the gate to
    match Genie — the gate's reference stays HF.
- [ ] **Step 2.4:** Commit: `git commit -am "docs: probe B — qualla qwen3vl-mrope semantics vs HF"`

### Task 3: Probe C — long-prompt path with a no-past prefill graph

- [ ] **Step 3.1:** Read the prefill/`n_process` loop: `grep -n "n_process\|selectVariant\|bestFit\|DECODER_PREFILL" $Q/qualla/engines/qnn-htp/nsp-model.cpp | head -30`, plus `$Q/qualla/engines/qnn-htp/KVCache/kvmanager.cpp:400-470`.
- [ ] **Step 3.2:** Confirm: with graphs {prefill (AR=128, no past), decode
  (AR=1, past 2175)}, a 290-token prompt processes as prefill(128) then
  162 decode steps — i.e. Genie picks by (AR, CL) best-fit per remaining
  chunk and never requires a past-KV prefill. Also confirm the sampled row
  after a decode-graph prompt chunk is row 0 (`n_process-1` with
  n_process=1).
- [ ] **Step 3.3:** Record in NOTES (this also documents expected device
  prefill latency: ~162 extra decode steps ≈ 15–20 s at 10 tok/s — an
  acceptable "works at all" cost; note ladekv-style past-KV prefill as the
  known future fix). If Genie instead REJECTS prompts > prefill CL, STOP:
  Phase 2 needs a past-KV prefill rebuild first — surface to the user before
  proceeding.
- [ ] **Step 3.4:** Commit: `git commit -am "docs: probe C — chunked prefill via decode graph with no-past prefill"`

### Task 4: Probe D — ImageEncoder output → accumulator dtype/scale

- [ ] **Step 4.1:** Read `$Q/qualla/engines/qnn-htp/nsp-image-model.cpp`
  (esp. around :199) and `pipeline/ImageEncoder.cpp` +
  `pipeline/TextGenerator.cpp:150-220`: how is the fp16 `image_features`
  output converted when appended to the embedding accumulator the LUT
  (float32) also feeds? Any dtype/width config key on the image-encoder node
  we must set?
- [ ] **Step 4.2:** Confirm `node set image` size-checks the blob against the
  graph input (find the check in `genie-app/main.cpp:1171-1210`); record the
  exact expected byte count (3,145,728).
- [ ] **Step 4.3:** Record in NOTES; fix `configs/genie_image_encoder_qwen3vl.json`
  if a key is missing. Commit: `git commit -am "docs: probe D — image embedding dtype flow into the accumulator"`

---

## Phase 1 — Host reference implementation + the device-free E2E gate

### Task 5: `mrope_tables()` + Gate 0 (`parity_mrope.py`)

**Files:**
- Modify: `scripts/export/modeling_export.py` (add below `rope_tables`, line ~359)
- Create: `scripts/validate/parity_mrope.py`

- [ ] **Step 5.1: Write the gate first** (`scripts/validate/parity_mrope.py`):

```python
#!/usr/bin/env python
"""Gate 0 (Stage 3): host-side interleaved MRoPE tables vs HF, exact.

The device graphs take precomputed cos/sin [1, S, 64]; for image prompts the
host must build them from 3-D (t,h,w) positions with Qwen3-VL's INTERLEAVED
mrope layout. A blocked (Qwen2-VL-style) layout is numerically wrong for
image spans yet identical for pure text -- exactly the kind of bug that only
shows up on device. This gate pins our tables to HF's own rotary embedding on
a real templated image prompt, and mutation-tests the interleave.

Run:
  $PY_DEPLOY scripts/validate/parity_mrope.py \
      --model $LLMDEPLOY_DATA/models/Qwen3-VL-4B-Instruct
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "export"))
from modeling_export import mrope_tables, rope_tables, rope_theta_of  # noqa: E402

TOL = 1e-5  # same math, different op order; fp32 gives ~1e-7


def make_image():
    from PIL import Image, ImageDraw
    img = Image.new("RGB", (512, 512), "white")
    d = ImageDraw.Draw(img)
    d.ellipse((96, 96, 288, 288), fill=(220, 30, 30))
    return img


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    args = ap.parse_args()

    from transformers import AutoProcessor, Qwen3VLForConditionalGeneration
    proc = AutoProcessor.from_pretrained(
        args.model, min_pixels=512 * 512, max_pixels=512 * 512)
    hf = Qwen3VLForConditionalGeneration.from_pretrained(
        args.model, dtype=torch.float32, attn_implementation="eager").eval()
    cfg = hf.config.text_config

    messages = [{"role": "user", "content": [
        {"type": "image"}, {"type": "text", "text": "Describe this image."}]}]
    text = proc.apply_chat_template(messages, tokenize=False,
                                    add_generation_prompt=True)
    inputs = proc(text=[text], images=[make_image()], return_tensors="pt")
    ids, grid = inputs.input_ids, inputs.image_grid_thw
    assert grid.tolist() == [[1, 32, 32]], grid.tolist()
    n = ids.shape[1]

    pos3, _ = hf.model.get_rope_index(
        ids, image_grid_thw=grid, attention_mask=torch.ones_like(ids))
    assert pos3.shape == (3, 1, n), pos3.shape
    pos3 = pos3[:, 0]                                    # [3, n]
    # the prompt must actually exercise 3-D positions or the gate is vacuous
    assert not torch.equal(pos3[0], pos3[1]), "no image span in positions?"

    x = torch.zeros(1, n, cfg.hidden_size)
    ref_cos, ref_sin = hf.model.language_model.rotary_emb(x, pos3[:, None])
    half = cfg.head_dim // 2                             # 64
    ref_cos, ref_sin = ref_cos[0, :, :half], ref_sin[0, :, :half]  # [n, 64]

    cos, sin = mrope_tables(pos3, cfg.head_dim, rope_theta_of(cfg))
    d = max(float((cos[0] - ref_cos).abs().max()),
            float((sin[0] - ref_sin).abs().max()))
    print(f"  interleaved tables vs HF rotary: max|d|={d:.3e} (n={n})")
    assert d < TOL, f"mrope_tables diverges from HF: {d:.3e}"

    # degenerate case: flat positions must reproduce rope_tables exactly
    flat = torch.arange(n)
    c1, s1 = mrope_tables(flat[None].expand(3, -1), cfg.head_dim,
                          rope_theta_of(cfg))
    c2, s2 = rope_tables(flat, cfg.head_dim, rope_theta_of(cfg))
    assert torch.equal(c1, c2) and torch.equal(s1, s2), \
        "text-only mrope != rope_tables"

    # mutation: a BLOCKED layout must disagree on this prompt
    sec = (24, 20, 20)
    ang = pos3.to(torch.float32)[:, :, None] * (
        1.0 / (rope_theta_of(cfg) ** (torch.arange(0, half).float() / half)))
    blocked = torch.cat([ang[i, :, sum(sec[:i]):sum(sec[:i + 1])]
                         for i in range(3)], dim=-1)
    db = float((blocked.cos() - ref_cos).abs().max())
    print(f"  blocked-layout control: max|d|={db:.3e} (must be >> {TOL})")
    assert db > 1e-2, "blocked layout matches HF?! gate is vacuous"

    print("PASS: mrope_tables matches HF interleaved MRoPE")


if __name__ == "__main__":
    main()
```

- [ ] **Step 5.2: Run it — must FAIL** with
  `ImportError: cannot import name 'mrope_tables'`:
  `source scripts/env.sh && $PY_DEPLOY scripts/validate/parity_mrope.py --model $LLMDEPLOY_DATA/models/Qwen3-VL-4B-Instruct`
- [ ] **Step 5.3: Implement** in `scripts/export/modeling_export.py`, directly
  below `rope_tables`:

```python
def mrope_tables(pos3, head_dim, theta, mrope_section=(24, 20, 20)):
    """Qwen3-VL interleaved MRoPE half-tables: pos3 [3, S] (t,h,w) ->
    ([1, S, D/2], [1, S, D/2]).

    Layout per HF Qwen3VLTextRotaryEmbedding.apply_interleaved_mrope
    (modeling_qwen3_vl.py): half-dim k takes t/h/w as [THWTHW...TT] -- h owns
    dims 1,4,..,3*sec[1]-2, w owns 2,5,..,3*sec[2]-1, t owns the rest
    including the tail. All three rows equal -> identical to rope_tables().
    """
    half = head_dim // 2
    inv_freq = 1.0 / (theta ** (torch.arange(0, half, dtype=torch.float32) / half))
    ang = pos3.to(torch.float32)[:, :, None] * inv_freq[None, None, :]  # [3,S,half]
    out = ang[0].clone()
    for dim, off in ((1, 1), (2, 2)):
        idx = slice(off, mrope_section[dim] * 3, 3)
        out[:, idx] = ang[dim][:, idx]
    return out.cos()[None], out.sin()[None]
```

- [ ] **Step 5.4: Run the gate — must PASS** (same command as 5.2). If the
  interleaved compare fails but the blocked control "passes", transformers'
  layout differs from the version this plan read — re-read
  `apply_interleaved_mrope` in the installed package and fix `mrope_tables`,
  never the gate.
- [ ] **Step 5.5: Commit:**
  `git add scripts/export/modeling_export.py scripts/validate/parity_mrope.py && git commit -m "feat: interleaved mrope_tables + parity gate vs HF rotary"`

### Task 6: `preprocess_image.py`

**Files:** Create: `scripts/pipeline/preprocess_image.py`

- [ ] **Step 6.1: Write it:**

```python
#!/usr/bin/env python
"""Image file -> the raw fp16 pixel_values blob genie-app feeds the ViT.

Genie does NO image preprocessing: `node set image` reads the file as an
opaque blob into GenieNode_setData (spec 7-RESOLVED), so these bytes must be
EXACTLY the ctx-bin's input tensor: [1024, 1536] QNN_DATATYPE_FLOAT_16
(work/ctxbin/qwen3vl-4b-vit-fp16/info.json) = 3,145,728 bytes.

The graph is static: 1024 patches = 32x32 grid = a 512x512 image. Every image
is first resized (aspect-distorting, deliberately -- "images work at all") to
512x512, then run through the real HF processor so normalization and patch
ordering match the export trace bit-for-bit.

Run:
  $PY_DEPLOY scripts/pipeline/preprocess_image.py \
      --model $LLMDEPLOY_DATA/models/Qwen3-VL-4B-Instruct \
      --image photo.jpg --out sample_image.raw
"""
import argparse
import json
from pathlib import Path

import numpy as np

EDGE = 512
N_PATCH, N_FEAT = 1024, 1536
RAW_BYTES = N_PATCH * N_FEAT * 2          # fp16


def preprocess(model_dir, image_path):
    from PIL import Image
    from transformers import AutoProcessor
    img = Image.open(image_path).convert("RGB").resize(
        (EDGE, EDGE), Image.BICUBIC)
    proc = AutoProcessor.from_pretrained(
        model_dir, min_pixels=EDGE * EDGE, max_pixels=EDGE * EDGE)
    out = proc.image_processor(images=img, return_tensors="np")
    pv, grid = out["pixel_values"], out["image_grid_thw"]
    assert tuple(pv.shape) == (N_PATCH, N_FEAT), \
        f"pixel_values {pv.shape} != {(N_PATCH, N_FEAT)} -- ViT graph is static"
    assert grid.tolist() == [[1, 32, 32]], f"grid {grid.tolist()} != [[1,32,32]]"
    blob = pv.astype(np.float16)
    assert np.isfinite(blob).all(), "non-finite pixel values after fp16 cast"
    return blob, grid


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--image", required=True)
    ap.add_argument("--out", required=True, help="output .raw path")
    args = ap.parse_args()
    blob, grid = preprocess(args.model, args.image)
    Path(args.out).write_bytes(blob.tobytes())
    Path(args.out).with_suffix(".json").write_text(json.dumps(
        {"shape": [N_PATCH, N_FEAT], "dtype": "float16",
         "bytes": RAW_BYTES, "grid_thw": grid.tolist(), "edge": EDGE},
        indent=1) + "\n")
    got = Path(args.out).stat().st_size
    assert got == RAW_BYTES, f"{got} bytes != {RAW_BYTES}"
    print(f"{args.out}: {got} bytes, grid {grid.tolist()}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 6.2: Test it** on a generated image:

```bash
source scripts/env.sh
$PY_DEPLOY - <<'EOF'
from PIL import Image, ImageDraw
img = Image.new("RGB", (640, 480), "white")   # deliberately NOT 512x512
d = ImageDraw.Draw(img)
d.ellipse((120, 90, 360, 330), fill=(220, 30, 30))
img.save("/tmp/claude-1000/-mnt-x-code-llm-deploy/0db36e5c-7da7-4885-9c25-a3950f13c354/scratchpad/test_circle.png")
EOF
$PY_DEPLOY scripts/pipeline/preprocess_image.py \
  --model $LLMDEPLOY_DATA/models/Qwen3-VL-4B-Instruct \
  --image /tmp/claude-1000/-mnt-x-code-llm-deploy/0db36e5c-7da7-4885-9c25-a3950f13c354/scratchpad/test_circle.png \
  --out   /tmp/claude-1000/-mnt-x-code-llm-deploy/0db36e5c-7da7-4885-9c25-a3950f13c354/scratchpad/test_circle.raw
```
  Expected: `test_circle.raw: 3145728 bytes, grid [[1, 32, 32]]`.
- [ ] **Step 6.3: Commit:**
  `git add scripts/pipeline/preprocess_image.py && git commit -m "feat: host-side image preprocessing to the ViT ctx-bin raw contract"`

### Task 7: Gate 1 — `parity_e2e_vl.py` (the Stage 3 device-free gate)

**Files:** Create: `scripts/validate/parity_e2e_vl.py`

The centerpiece. Emulates the deployed pipeline end-to-end on x86:
real image → (HF-visual AND ViT-ONNX) features → LUT splice → text-tower ONNX
under qualla's exact feed (no-past prefill for rows 0–127, decode graph for
every remaining prompt row, then free-running generation) → token-for-token
vs `hf.generate(greedy)`.

Two chains, one bar each:
- **Chain 1 (HF visual features, real deepstack):** MUST match HF `generate`
  token-for-token. Proves splice, MRoPE, chunked feed, and KV threading are
  exact — any device divergence is then quantization or Genie-side, not our math.
- **Chain 2 (ONNX ViT features, real deepstack):** the fully-ONNX path;
  carries ViT's own ~1e-3 fp32-trace delta. Bar: ≥ 75% step agreement, text
  printed for human review (expected: identical or near-identical).
- **Tier-A preview (zero deepstack), reported not gated:** the text the
  device pipeline will actually produce; printed for the README.

Mutations (run while the prefill session is live, before trusting PASS):
zero-deepstack must move row-127 logits (> 1e-3); flat-rope-instead-of-mrope
must move them (> 1e-3).

- [ ] **Step 7.1: Write the gate:**

```python
#!/usr/bin/env python
"""Gate 1 (Stage 3): full-path device-free parity -- image+text -> description.

Emulates the DEPLOYED pipeline: LUT lookup, image features spliced at the
<|image_pad|> rows, deepstack rows placed there and zero elsewhere (the graph
adds full-width, so the host owns the zero-padding contract), interleaved
MRoPE cos/sin, no-past prefill for the first AR rows, decode graph for every
remaining prompt row (the prefill graph has no past-KV -- probe C), then
free-running greedy generation with the KV contract (new-slice outputs, keys
transposed). Reference: hf.generate(greedy), token-for-token.

Memory is staged exactly like parity_vl_text.py: HF fp32 (~18 GB) first, then
freed (LUT + refs kept); ViT ORT; prefill ORT; decode ORT -- never two
sessions live at once.

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
    MASK_VALUE, causal_mask, mrope_tables, rope_theta_of)

AR = 128
STEPS = 24
PROMPT = "Describe this image in one sentence."
CHAIN2_MIN_AGREE = 0.75


def finite(name, arr):
    a = arr.detach().numpy() if isinstance(arr, torch.Tensor) else arr
    bad = ~np.isfinite(a)
    assert not bad.any(), f"{name}: {int(bad.sum())}/{a.size} non-finite"


def make_image():
    """Deterministic, semantically legible: red circle + blue square."""
    from PIL import Image, ImageDraw
    img = Image.new("RGB", (512, 512), "white")
    d = ImageDraw.Draw(img)
    d.ellipse((96, 96, 288, 288), fill=(220, 30, 30))
    d.rectangle((300, 300, 460, 460), fill=(30, 60, 220))
    return img


# ------------------------------------------------------------------ phase A
def hf_phase(model_dir):
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
    assert tuple(pv.shape) == (1024, 1536) and grid.tolist() == [[1, 32, 32]]
    n = int(ids.numel())
    assert AR < n < 512, f"n={n}: gate assumes AR < n < decode window"

    img_tok = hf.config.image_token_id
    img_pos = (ids == img_tok).nonzero()[:, 0]
    assert img_pos.numel() == 256, f"{img_pos.numel()} image_pad rows != 256"
    assert int(img_pos[-1] - img_pos[0]) == 255, "image span not contiguous"

    pos3, _ = hf.model.get_rope_index(
        ids[None], image_grid_thw=grid,
        attention_mask=torch.ones(1, n, dtype=torch.long))
    pos3 = pos3[:, 0]                                        # [3, n]

    with torch.no_grad():
        vis = hf.model.visual(pv, grid)
    feats_hf = vis.pooler_output.detach().numpy()            # [256, 2560]
    deep_hf = [d.detach().numpy() for d in vis.deepstack_features]
    for k, d in enumerate([feats_hf] + deep_hf):
        finite(f"HF visual[{k}]", d)

    with torch.no_grad():
        gen = hf.generate(**inputs, do_sample=False, max_new_tokens=STEPS)
    ref_new = gen[0, n:].tolist()
    assert len(ref_new) >= 8, f"HF stopped after {len(ref_new)} tokens"
    tok = proc.tokenizer
    print(f"  HF greedy ({len(ref_new)} toks): {tok.decode(ref_new)!r}")

    lut = hf.model.language_model.embed_tokens.weight.detach().numpy().astype(
        np.float32)
    meta = dict(hidden=cfg.hidden_size, head_dim=cfg.head_dim,
                theta=rope_theta_of(cfg), layers=cfg.num_hidden_layers,
                n_kv=cfg.num_key_value_heads, n_deep=len(deep_hf))
    del hf, vis, gen, inputs
    gc.collect()
    return (tok, lut, meta, ids.numpy(), n, img_pos.numpy(), pos3,
            pv.numpy(), feats_hf, deep_hf, ref_new)


# ------------------------------------------------------------ feed builders
def spliced_inputs(ids, n, img_pos, lut, feats, deep, meta):
    """Full-prompt embeds + full-width deepstack (zero outside the span)."""
    H = meta["hidden"]
    emb = lut[ids].copy()                                    # [n, H]
    emb[img_pos] = feats
    deep_all = []
    for d in deep:
        w = np.zeros((n, H), dtype=np.float32)
        w[img_pos] = d
        deep_all.append(w)
    return emb, deep_all


def prefill_feeds(emb, deep_all, cos, sin, meta, zero_deepstack=False,
                  flat_rope=None):
    H = meta["hidden"]
    feeds = {
        "inputs_embeds": emb[:AR].reshape(1, 1, AR, H).copy(),
        "attention_mask": causal_mask(AR, AR).numpy(),
        "position_ids_cos": cos[:, :AR].numpy().copy(),
        "position_ids_sin": sin[:, :AR].numpy().copy(),
    }
    if flat_rope is not None:
        feeds["position_ids_cos"], feeds["position_ids_sin"] = flat_rope
    for k, w in enumerate(deep_all):
        z = np.zeros((1, 1, AR, H), dtype=np.float32)
        if not zero_deepstack:
            z[0, 0] = w[:AR]
        feeds[f"deepstack_visual_embed_{k}"] = z
    return feeds


# ------------------------------------------------------------- text chain
def run_chain(label, feats, deep, ids, n, img_pos, pos3, lut, meta,
              prefill_sess, kv_names):
    """Prefill rows 0..AR-1; returns (cache_k, cache_v, valid_len)."""
    L = meta["layers"]
    emb, deep_all = spliced_inputs(ids, n, img_pos, lut, feats, deep, meta)
    cos, sin = mrope_tables(pos3, meta["head_dim"], meta["theta"])
    outs = dict(zip(["logits"] + kv_names,
                    prefill_sess.run(["logits"] + kv_names,
                                     prefill_feeds(emb, deep_all, cos, sin,
                                                   meta))))
    finite(f"{label} prefill logits", outs["logits"])
    ck = [outs[f"past_key_{i}_out"][:, :, :, :AR].copy() for i in range(L)]
    cv = [outs[f"past_value_{i}_out"][:, :, :AR, :].copy() for i in range(L)]
    return emb, deep_all, cos, sin, ck, cv


def decode_all(label, sess, meta, emb, deep_all, cos, sin, ids, n, pos3,
               lut, ck, cv, ref_new, tok):
    """Prompt rows AR..n-1 through the decode graph, then greedy generation."""
    L, H, D = meta["layers"], meta["hidden"], meta["head_dim"]
    shapes = {i.name: i.shape for i in sess.get_inputs()}
    PAST = shapes["past_key_0_in"][-1]
    TOTAL = shapes["attention_mask"][-1]
    assert TOTAL == PAST + 1
    for i in range(L):
        k = np.zeros((1, meta["n_kv"], D, PAST), dtype=np.float32)
        v = np.zeros((1, meta["n_kv"], PAST, D), dtype=np.float32)
        k[:, :, :, :AR], v[:, :, :AR, :] = ck[i], cv[i]
        ck[i], cv[i] = k, v

    out_names = ["logits"] + [f"past_{s}_{i}_out"
                              for i in range(L) for s in ("key", "value")]
    mx = int(pos3.max())

    def step(row_emb, row_deep, c, s, cache_len):
        mask = np.full((1, 1, TOTAL), MASK_VALUE, dtype=np.float32)
        mask[0, 0, :cache_len] = 0.0
        mask[0, 0, PAST] = 0.0
        feeds = {"inputs_embeds": row_emb.reshape(1, 1, 1, H).copy(),
                 "attention_mask": mask,
                 "position_ids_cos": c.numpy().copy(),
                 "position_ids_sin": s.numpy().copy()}
        for k in range(meta["n_deep"]):
            feeds[f"deepstack_visual_embed_{k}"] = \
                row_deep[k].reshape(1, 1, 1, H).copy()
        for i in range(L):
            feeds[f"past_key_{i}_in"] = ck[i]
            feeds[f"past_value_{i}_in"] = cv[i]
        outs = dict(zip(out_names, sess.run(out_names, feeds)))
        for i in range(L):
            ck[i][:, :, :, cache_len:cache_len + 1] = outs[f"past_key_{i}_out"]
            cv[i][:, :, cache_len:cache_len + 1, :] = outs[f"past_value_{i}_out"]
        return outs["logits"][0, 0]

    cache_len = AR
    for r in range(AR, n):                      # prompt tail, teacher-forced
        assert cache_len < PAST
        lg = step(emb[r], [d[r] for d in deep_all],
                  cos[:, r:r + 1], sin[:, r:r + 1], cache_len)
        finite(f"{label} prompt row {r} logits", lg)
        cache_len += 1
    first = int(lg.argmax())                    # logits of row n-1

    got, agree = [first], [first == ref_new[0]]
    cur = first
    zero_row = [np.zeros(H, dtype=np.float32)] * meta["n_deep"]
    for j in range(1, len(ref_new)):
        # generated token j-1 sits at sequence index n-1+j; HF continues all
        # three axes from the prompt max: position = mx + j (rope_deltas)
        p = torch.full((3, 1), mx + j)
        c, s = mrope_tables(p, D, meta["theta"])
        lg = step(lut[cur], zero_row, c, s, cache_len)
        finite(f"{label} gen step {j} logits", lg)
        cache_len += 1
        cur = int(lg.argmax())
        got.append(cur)
        agree.append(cur == ref_new[j])
    print(f"  [{label}] agree {sum(agree)}/{len(agree)}: {tok.decode(got)!r}")
    return got, agree


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--vit-onnx", required=True)
    ap.add_argument("--text-onnx", required=True)
    args = ap.parse_args()
    t0 = time.time()

    print("== phase A: HF reference (model alive) ==", flush=True)
    (tok, lut, meta, ids, n, img_pos, pos3, pv, feats_hf, deep_hf,
     ref_new) = hf_phase(args.model)
    L = meta["layers"]
    print(f"  n={n} image rows [{img_pos[0]},{img_pos[-1]}] "
          f"t={time.time() - t0:.0f}s", flush=True)

    print("== phase B: ViT ONNX ==", flush=True)
    sess = ort.InferenceSession(str(args.vit_onnx),
                                providers=["CPUExecutionProvider"])
    got = sess.run(None, {"pixel_values": pv})
    names = [o.name for o in sess.get_outputs()]
    vit = dict(zip(names, got))
    feats_ort = vit["image_features"]
    deep_ort = [vit[f"deepstack_visual_embed_{k}"] for k in range(3)]
    for k, (a, b) in enumerate(zip([feats_ort] + deep_ort,
                                   [feats_hf] + deep_hf)):
        finite(f"ViT ONNX out[{k}]", a)
        print(f"  vit out[{k}] max|d| vs HF = {np.abs(a - b).max():.3e}")
    del sess
    gc.collect()

    print("== phase C: prefill.onnx (both chains + mutations) ==", flush=True)
    prefill_sess = ort.InferenceSession(
        str(Path(args.text_onnx) / "prefill" / "prefill.onnx"),
        providers=["CPUExecutionProvider"])
    kv_names = [f"past_{s}_{i}_out" for i in range(L) for s in ("key", "value")]
    chains = {}
    for label, f, d in (("chain1-hf-vit", feats_hf, deep_hf),
                        ("chain2-onnx-vit", feats_ort, deep_ort),
                        ("tierA-zero-deep", feats_hf,
                         [np.zeros_like(x) for x in deep_hf])):
        chains[label] = run_chain(label, f, d, ids, n, img_pos, pos3, lut,
                                  meta, prefill_sess, kv_names)

    # mutations on chain1's feed, row AR-1 logits must MOVE
    emb, deep_all, cos, sin, _, _ = chains["chain1-hf-vit"]
    base = prefill_sess.run(["logits"],
                            prefill_feeds(emb, deep_all, cos, sin, meta))[0][0]
    zd = prefill_sess.run(["logits"],
                          prefill_feeds(emb, deep_all, cos, sin, meta,
                                        zero_deepstack=True))[0][0]
    dz = float(np.abs(zd[AR - 1] - base[AR - 1]).max())
    from modeling_export import rope_tables
    fc, fs = rope_tables(torch.arange(AR), meta["head_dim"], meta["theta"])
    fr = prefill_sess.run(["logits"], prefill_feeds(
        emb, deep_all, cos, sin, meta,
        flat_rope=(fc.numpy(), fs.numpy())))[0][0]
    dm = float(np.abs(fr[AR - 1] - base[AR - 1]).max())
    print(f"  mutation zero-deepstack row{AR - 1} delta = {dz:.3e}")
    print(f"  mutation flat-rope      row{AR - 1} delta = {dm:.3e}")
    assert dz > 1e-3, "deepstack inputs ignored -- gate is vacuous"
    assert dm > 1e-3, "mrope tables ignored -- gate is vacuous"
    del prefill_sess
    gc.collect()

    print("== phase D: decode.onnx (prompt tail + generation) ==", flush=True)
    sess = ort.InferenceSession(
        str(Path(args.text_onnx) / "decode" / "decode.onnx"),
        providers=["CPUExecutionProvider"])
    results = {}
    for label in ("chain1-hf-vit", "chain2-onnx-vit", "tierA-zero-deep"):
        emb, deep_all, cos, sin, ck, cv = chains[label]
        results[label] = decode_all(label, sess, meta, emb, deep_all, cos,
                                    sin, ids, n, pos3, lut, ck, cv, ref_new,
                                    tok)
        gc.collect()
    del sess
    gc.collect()

    print(f"  HF reference: {tok.decode(ref_new)!r}")
    print(f"wall {time.time() - t0:.0f}s")

    got1, agree1 = results["chain1-hf-vit"]
    assert all(agree1), (
        f"chain1 (HF visual, real deepstack) diverged from hf.generate at "
        f"step {agree1.index(False)}: our math is wrong somewhere")
    got2, agree2 = results["chain2-onnx-vit"]
    frac = sum(agree2) / len(agree2)
    assert frac >= CHAIN2_MIN_AGREE, (
        f"chain2 (ONNX ViT) agreement {frac:.0%} < {CHAIN2_MIN_AGREE:.0%}")
    print(f"PASS: chain1 token-exact vs HF over {len(agree1)} steps; "
          f"chain2 {frac:.0%}; tierA preview printed above")


if __name__ == "__main__":
    main()
```

- [ ] **Step 7.2: Run — expect import/attr failures first** (this file is the
  most likely place for HF API drift: `get_rope_index` kwargs,
  `visual(...)` return fields, `image_token_id`). Fix against the installed
  transformers source, never by weakening an assert:
  `$PY_DEPLOY scripts/validate/parity_e2e_vl.py --model $LLMDEPLOY_DATA/models/Qwen3-VL-4B-Instruct --vit-onnx $LLMDEPLOY_DATA/work/onnx/qwen3vl-4b-vit/vit.onnx --text-onnx $LLMDEPLOY_DATA/work/onnx/qwen3vl-4b-text`
  (~30–60 min wall; run under `nohup ... &` and watch the log. If OOM, run
  on tank — the artifacts and model already exist there.)
- [ ] **Step 7.3:** Must reach **PASS** with chain1 token-exact. Two known
  failure classes and their meaning: chain1 first-token wrong → splice or
  mrope bug (check `pos3` continuation and the `mx+1+j` line — the position
  of generated token `n-1+j` must equal HF's `rope_deltas` continuation);
  chain1 diverges only after step ~10 → KV threading (compare against
  `parity_vl_text.py` phase C, which is the proven pattern).
- [ ] **Step 7.4:** Record the three generated strings (chain1 == HF, chain2,
  tierA) — the tierA string goes verbatim into the bundle README as the
  expected device behavior.
- [ ] **Step 7.5: Commit:**
  `git add scripts/validate/parity_e2e_vl.py && git commit -m "feat: Stage 3 full-path device-free gate (image->ViT->splice->text->tokens vs HF)"`

---

## ⚠ PHASE 2 REVISED (2026-08-14, after the Phase 0 probes)

**The original Phase 2 below is SUPERSEDED.** Probe D established, and manual
re-verification confirmed, that a stock `GeniePipeline` cannot drive an **FP16**
vision tower on this SDK build:

- `QnnNspImageModel::setupInputFP16` is an empty stub that discards the pixel
  blob and returns success (`nsp-image-model.cpp:526-530`).
- `getConverterMap()` has no `Float16` entries in either direction, so the
  ImageEncoder→accumulator append throws (`Quantization.cpp:163-192`).

See `docs/NOTES-genie-pipeline.md` §D for the full citation trail.

**Decision (user, 2026-08-14): re-quantize the vision tower to fixed-point IO.**
Rejected alternatives: a `qnn-net-run` + raw-embedding two-step (fastest, but
skipping the ImageEncoder node means `setVisionParam` is never called, so MRoPE
never engages for image rows — `ImageEncoder.cpp:136-138`, and there is no C API
to set it otherwise); and a custom QNN driver (full fidelity including
deepstack, but untestable without a device).

Target: **W8A16** — weights int8, activations uint16 — matching the text tower.
`QNN_DATATYPE_UFIXED_POINT_16` dispatches to the real `setupInput<uint16_t>`
copy path (`nsp-image-model.cpp:541-543`), and `{UFixed16, Float32}` is present
in the converter map, so both gaps close.

**Unexpected upside:** the QNN CPU backend has no FP16 execution path, which is
why `parity_vit_dlc.py` today validates via a throwaway FP32 DLC and explicitly
disclaims that fp16 numerics stay unvalidated until device time. A *quantized*
DLC **can** execute on the CPU backend, so the shipped artifact itself becomes
directly gateable device-free — strictly stronger evidence than Stage 1 had.

Revised task list replacing Tasks 8-11:

- **8a** `scripts/quant/quantize_vit_aimet.py` — AIMET W8A16 quantsim over
  `ExportQwen3VLViT`, calibrated on a diverse image set, exporting
  `model.onnx` + `model.encodings`. Gate: quantsim-vs-fp32 cosine on
  `image_features` and all three deepstack outputs.
- **8b** `scripts/build/vit_build_quant.sh` — convert with
  `--quantization_overrides`, **assert the converted IO dtypes are actually
  UFIXED_16** (not silently folded back to float), build the ctx-bin, verify
  graph name and HTP config binding by reading the finalised binary back.
- **8c** Gate: execute the shipped quantized DLC under `qnn-net-run` on the CPU
  backend vs HF — the check Stage 1 could not run.
- **8d** `preprocess_image.py` emits UFixed16 quantized with the graph's own
  input encoding (scale/offset read from the DLC), not float.
- **8e** `configs/genie_image_encoder_qwen3vl.json` gains
  `engine.model.vision-param: {height: 32, width: 32}` — **patch units,
  pre-merge** (`nsp-image-model.cpp:373`), mandatory for MRoPE to engage at all
  (`ImageEncoder.cpp:46-47`).

Tasks 9-13 (pipeline script, lint, bundle, runbook, upload) carry over with the
image-encoder config and ctx-bin swapped for the quantized ones.

---

## Phase 2 — Device pipeline assembly (Tier A) — SUPERSEDED, see above

### Task 8: Pipeline node configs

**Files:**
- Create: `configs/genie_text_generator_qwen3vl_4b.json`
- Create: `configs/genie_text_encoder_qwen3vl.json`
- Verify: `configs/genie_image_encoder_qwen3vl.json` (exists; fix only if Probe D demands)

Schema precedent: `examples/Genie/configs/glm-4v/{glm-4v,text-encoder,siglip}.json`.
Apply any key-spelling corrections Probe B found.

- [ ] **Step 8.1:** Write `configs/genie_text_generator_qwen3vl_4b.json` — the
  pipeline twin of `genie_dialog_qwen3vl_4b.json` (same context/sampler/
  tokenizer/embedding/engine blocks, top-level key `text-generator` with
  `"type": "basic"` and an accumulator):

```json
{
  "text-generator": {
    "version": 1,
    "type": "basic",
    "accumulator-size": 64000000,
    "context": {
      "version": 1,
      "size": 2048,
      "n-vocab": 151936,
      "bos-token": 151643,
      "eos-token": [151645, 151643]
    },
    "sampler": {
      "version": 1,
      "seed": 42,
      "temp": 0.0,
      "top-k": 1,
      "top-p": 1.0,
      "greedy": true
    },
    "tokenizer": { "version": 1, "path": "tokenizer.json" },
    "embedding": {
      "version": 1,
      "type": "lut",
      "lut-path": "embedding_float32_lut.bin",
      "size": 2560,
      "datatype": "float32"
    },
    "engine": {
      "version": 1,
      "n-threads": 3,
      "backend": {
        "version": 1,
        "type": "QnnHtp",
        "QnnHtp": {
          "version": 1,
          "spill-fill-bufsize": 0,
          "use-mmap": true,
          "mmap-budget": 0,
          "poll": false,
          "cpu-mask": "0xe0",
          "pos-id-dim": 64,
          "kv-dim": 128,
          "allow-async-init": false
        },
        "extensions": "htp_backend_ext_config_vltext.json"
      },
      "model": {
        "version": 1,
        "type": "binary",
        "binary": {
          "version": 1,
          "ctx-bins": [
            "qwen3vl-4b-w8a16_1_of_2.bin",
            "qwen3vl-4b-w8a16_2_of_2.bin"
          ]
        },
        "positional-encoding": {
          "type": "rope",
          "rope-dim": 64,
          "rope-theta": 5000000.0,
          "rope-scaling": {
            "rope-type": "qwen3vl-mrope",
            "mrope-section": [24, 20, 20],
            "spatial-merge-size": 2,
            "time-step": 2
          }
        }
      }
    }
  }
}
```

- [ ] **Step 8.2:** Write `configs/genie_text_encoder_qwen3vl.json`:

```json
{
  "text-encoder": {
    "version": 1,
    "type": "lut",
    "lut": {
      "version": 1,
      "lut-path": "embedding_float32_lut.bin",
      "size": 2560,
      "datatype": "float32"
    },
    "tokenizer": { "version": 1, "path": "tokenizer.json" }
  }
}
```

- [ ] **Step 8.3:** Sanity: `$PY_DEPLOY -c "import json; [json.load(open(f)) for f in ('configs/genie_text_generator_qwen3vl_4b.json','configs/genie_text_encoder_qwen3vl.json','configs/genie_image_encoder_qwen3vl.json')]" && echo OK`
- [ ] **Step 8.4: Commit:**
  `git add configs/genie_text_*_qwen3vl*.json && git commit -m "feat: pipeline node configs for the qwen3vl e2e path"`

### Task 9: genie-app pipeline script

**Files:** Create: `configs/genie_pipeline_qwen3vl.script`

The text segments MUST reproduce the tokenizer's own chat template exactly —
the accumulator sequence (text-tokens, 256 image rows, text-tokens) must
equal Gate 1's `input_ids` with the `<|image_pad|>` rows replaced by image
embeddings. Derive the segments, don't hand-write them:

- [ ] **Step 9.1: Print the authoritative segments:**

```bash
source scripts/env.sh
$PY_DEPLOY - <<'EOF'
from transformers import AutoProcessor
import os
proc = AutoProcessor.from_pretrained(
    os.path.expandvars("$LLMDEPLOY_DATA/models/Qwen3-VL-4B-Instruct"))
messages = [{"role": "user", "content": [
    {"type": "image"},
    {"type": "text", "text": "Describe this image in one sentence."}]}]
text = proc.apply_chat_template(messages, tokenize=False,
                                add_generation_prompt=True)
pad = "<|image_pad|>"
i = text.index(pad)
j = text.rindex(pad) + len(pad)
seg1, seg2 = text[:i], text[j:]
print("SEG1:", repr(seg1))
print("SEG2:", repr(seg2))
EOF
```

- [ ] **Step 9.2: Write `configs/genie_pipeline_qwen3vl.script`** (modeled on
  `examples/Genie/genie-app/scripts/glm-4v`; substitute the two printed
  segments, with `\n` escaped as `\\n` exactly as the glm-4v script does):

```text
version
pipeline config create pipelineConfig
pipeline create GeniePipeline pipelineConfig

# Vision tower (FP16 ctx-bin, single stock output = image_features).
# The three deepstack outputs have NO route through a stock ImageEncoder
# node (ImageEncoder.cpp exposes one IO); the text graphs read zeros there
# (probe A) => exactly HF-minus-deepstack. Tier B (custom driver) pending.
node config create imageEncoderConfig genie_image_encoder_qwen3vl.json
node create imageEncoder imageEncoderConfig

# LUT text encoder (float32 -- a fixed-point LUT silently no-ops against
# the FP16 inputs_embeds input; see extract_embed_lut.py)
node config create lutEncoderConfig genie_text_encoder_qwen3vl.json
node create lutEncoder lutEncoderConfig

# Text generator: 2-split W8A16 ctx-bins, qwen3vl-mrope host-side
node config create textGeneratorConfig genie_text_generator_qwen3vl_4b.json
node create textGenerator textGeneratorConfig
node set textCallback textGenerator GENIE_NODE_TEXT_GENERATOR_TEXT_OUTPUT

pipeline add GeniePipeline imageEncoder
pipeline add GeniePipeline lutEncoder
pipeline add GeniePipeline textGenerator

pipeline connect GeniePipeline imageEncoder GENIE_NODE_IMAGE_ENCODER_EMBEDDING_OUTPUT textGenerator GENIE_NODE_TEXT_GENERATOR_EMBEDDING_INPUT
pipeline connect GeniePipeline lutEncoder GENIE_NODE_TEXT_ENCODER_EMBEDDING_OUTPUT textGenerator GENIE_NODE_TEXT_GENERATOR_EMBEDDING_INPUT

# Segment order builds the accumulator; the tokenizer's chat template is
# reproduced EXACTLY (lint_pipeline_bundle.py re-derives and diffs it).
# <SEG1/SEG2 from Step 9.1 -- placeholders below MUST be replaced>
node set text lutEncoder GENIE_NODE_TEXT_ENCODER_TEXT_INPUT "SEG1"
node set image imageEncoder GENIE_NODE_IMAGE_ENCODER_IMAGE_INPUT sample_image.raw
node set text lutEncoder GENIE_NODE_TEXT_ENCODER_TEXT_INPUT "SEG2"

pipeline execute GeniePipeline

node free imageEncoder
node free lutEncoder
node free textGenerator
pipeline free GeniePipeline
```

  Expected SEG values (verify against Step 9.1 output, do not trust this
  plan's memory of the template):
  `SEG1 = "<|im_start|>user\\n<|vision_start|>"`,
  `SEG2 = "<|vision_end|>Describe this image in one sentence.<|im_end|>\\n<|im_start|>assistant\\n"`.
- [ ] **Step 9.3: Commit:**
  `git add configs/genie_pipeline_qwen3vl.script && git commit -m "feat: genie-app pipeline script for qwen3vl e2e"`

### Task 10: Gate 2 — `lint_pipeline_bundle.py`

**Files:** Create: `scripts/validate/lint_pipeline_bundle.py`

Static cross-checks over an ASSEMBLED bundle dir. Every failure class here is
a silent device failure (missing file → load error at best; unbound graph →
HTP defaults; template drift → subtly wrong prompt).

- [ ] **Step 10.1: Write it:**

```python
#!/usr/bin/env python
"""Gate 2 (Stage 3): static contract lint of the assembled e2e bundle.

Checks (each one maps to a known silent-failure class):
  1. every file referenced by the pipeline script + all three node configs
     exists in the bundle (missing ref = runtime load error)
  2. every graph in every ctx-bin is bound by its node's htp extensions
     graph_names (unbound = O=0/4MB VTCM defaults, or lade-class SIGSEGV)
  3. LUT: bytes == 151936*2560*4, config datatype float32/size 2560
  4. sample_image.raw == 3,145,728 bytes and its meta json grid == [1,32,32]
  5. the script's text segments + 256*<|image_pad|> re-tokenize EXACTLY to
     the HF chat template of the same prompt (accumulator == Gate 1 input)

Run:
  $PY_DEPLOY scripts/validate/lint_pipeline_bundle.py \
      --bundle $LLMDEPLOY_DATA/bundles/qwen3vl_4b_e2e_pipeline \
      --model  $LLMDEPLOY_DATA/models/Qwen3-VL-4B-Instruct \
      --ctx-info $LLMDEPLOY_DATA/work/ctxbin/qwen3vl-4b-vit-fp16/info.json \
      --ctx-info $LLMDEPLOY_DATA/work/ctxbin/qwen3vl-4b-w8a16-split/1_of_2/info.json \
      --ctx-info $LLMDEPLOY_DATA/work/ctxbin/qwen3vl-4b-w8a16-split/2_of_2/info.json
"""
import argparse
import json
import re
import sys
from pathlib import Path

RAW_BYTES = 1024 * 1536 * 2
LUT_BYTES = 151936 * 2560 * 4
PROMPT = "Describe this image in one sentence."
IMG_ROWS = 256


def fail(msg):
    print(f"  LINT FAIL: {msg}")
    return 1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bundle", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--ctx-info", action="append", required=True)
    args = ap.parse_args()
    b = Path(args.bundle)
    bad = 0

    script = (b / "genie_pipeline_qwen3vl.script").read_text()

    # -- 1. reference closure ------------------------------------------------
    cfg_files = re.findall(r"node config create \S+ (\S+)", script)
    raw_files = re.findall(r"IMAGE_INPUT (\S+)", script)
    refs = set(cfg_files) | set(raw_files)
    for cf in cfg_files:
        node = json.load(open(b / cf))
        (key,) = node.keys()
        eng = node[key].get("engine", {})
        if "backend" in eng:
            refs.add(eng["backend"]["extensions"])
            refs.update(eng["model"]["binary"]["ctx-bins"])
        tk = node[key].get("tokenizer", {})
        if tk:
            refs.add(tk["path"])
        emb = node[key].get("embedding") or node[key].get("lut")
        if emb:
            refs.add(emb["lut-path"])
    for r in sorted(refs):
        ok = (b / r).exists()
        print(f"  {'OK  ' if ok else 'MISS'} {r}")
        if not ok:
            bad += fail(f"referenced file missing from bundle: {r}")

    # -- 2. graph binding ----------------------------------------------------
    have = set()
    for ci in args.ctx_info:
        info = json.load(open(ci))
        have |= {g["info"]["graphName"] for g in info["info"]["graphs"]}
    listed = set()
    for ext in b.glob("htp_backend_ext_config*.json"):
        e = json.load(open(ext))
        listed |= {n for g in e.get("graphs", []) for n in g["graph_names"]}
    print(f"  graphs {sorted(have)} / bound {sorted(listed)}")
    if have - listed:
        bad += fail(f"unbound graphs (HTP defaults!): {sorted(have - listed)}")

    # -- 3. LUT --------------------------------------------------------------
    lut = b / "embedding_float32_lut.bin"
    if lut.stat().st_size != LUT_BYTES:
        bad += fail(f"LUT {lut.stat().st_size} bytes != {LUT_BYTES}")

    # -- 4. sample image -----------------------------------------------------
    raw = b / raw_files[0]
    if raw.stat().st_size != RAW_BYTES:
        bad += fail(f"{raw.name} {raw.stat().st_size} bytes != {RAW_BYTES}")
    meta = json.load(open(raw.with_suffix(".json")))
    if meta["grid_thw"] != [[1, 32, 32]]:
        bad += fail(f"sample grid {meta['grid_thw']} != [[1,32,32]]")

    # -- 5. chat template equivalence ---------------------------------------
    segs = re.findall(r'TEXT_INPUT "((?:[^"\\]|\\.)*)"', script)
    if len(segs) != 2:
        bad += fail(f"expected 2 text segments in script, got {len(segs)}")
    else:
        seg1, seg2 = (s.replace("\\n", "\n") for s in segs)
        from transformers import AutoProcessor
        proc = AutoProcessor.from_pretrained(args.model)
        messages = [{"role": "user", "content": [
            {"type": "image"}, {"type": "text", "text": PROMPT}]}]
        want = proc.apply_chat_template(messages, tokenize=False,
                                        add_generation_prompt=True)
        got = seg1 + "<|image_pad|>" * IMG_ROWS + seg2
        if got != want:
            bad += fail("script text segments != tokenizer chat template:\n"
                        f"    script: {got[:80]!r}...\n    hf    : {want[:80]!r}...")

    if bad:
        sys.exit(f"FAIL: {bad} bundle lint error(s)")
    print("PASS: pipeline bundle contract clean")


if __name__ == "__main__":
    main()
```

- [ ] **Step 10.2: Mutation-test it** once the bundle exists (Task 11): (a)
  temporarily rename a ctx-bin in a COPY of the bundle → check 1 fails; (b)
  edit a copied ext config to drop `vit` from `graph_names` → check 2 fails;
  (c) edit one script text segment → check 5 fails. All three must fail
  loudly. (Run against a copy; never mutate the real bundle.)
- [ ] **Step 10.3: Commit:**
  `git add scripts/validate/lint_pipeline_bundle.py && git commit -m "feat: static lint gate for the e2e pipeline bundle"`

### Task 11: `vl_pipeline_bundle.sh`

**Files:** Create: `scripts/build/vl_pipeline_bundle.sh`

One flat dir (~7.1 GB): both towers + pipeline glue + runtime. Flat because
Genie resolves `.so` files and every config-referenced path from the bundle
root. Reuses the two existing gated bundles as sources so bytes are the
already-verified ones.

- [ ] **Step 11.1: Write it:**

```bash
#!/usr/bin/env bash
# Assemble the flat, push-ready END-TO-END Qwen3-VL-4B pipeline bundle:
# ViT ctx-bin + 2-split text ctx-bins + LUT + node configs + genie-app script
# + runtime .so set + a preprocessed sample image. Sources are the two
# already-gated single-tower bundles, so every binary byte here is one that
# passed its own gates.
#
# Usage: vl_pipeline_bundle.sh [bundlename]
set -euo pipefail
source "$(dirname "$0")/../env.sh"

BUNDLE=${1:-qwen3vl_4b_e2e_pipeline}
TEXT=$LLMDEPLOY_DATA/bundles/qwen3vl_4b_text_w8a16
VIT=$LLMDEPLOY_DATA/bundles/qwen3vl_4b_vit_fp16
MODEL=${MODEL:-$LLMDEPLOY_DATA/models/Qwen3-VL-4B-Instruct}
OUT=$LLMDEPLOY_DATA/bundles/$BUNDLE
SCRATCH=${SCRATCH:-$LLMDEPLOY_DATA/work/pipeline-scratch}

for d in "$TEXT" "$VIT"; do
    [ -d "$d" ] || { echo "missing source bundle $d"; exit 1; }
done

disk_guard 16
rm -rf "$OUT"; mkdir -p "$OUT" "$SCRATCH"

# text tower: ctx-bins, LUT, tokenizer, runtime (.so + genie binaries)
for f in qwen3vl-4b-w8a16_1_of_2.bin qwen3vl-4b-w8a16_2_of_2.bin \
         embedding_float32_lut.bin embedding_lut_params.json tokenizer.json \
         htp_backend_ext_config_vltext.json genie-app genie-t2t-run \
         libGenie.so libQnnHtp.so libQnnSystem.so libQnnHtpPrepare.so \
         libQnnHtpNetRunExtensions.so libQnnHtpV81Stub.so libQnnHtpV81Skel.so; do
    cp "$TEXT/$f" "$OUT/"
done
# vision tower: ctx-bin + its htp ext config (its .so set is the same files)
cp "$VIT/qwen3vl-4b-vit-fp16_ctx.bin" "$OUT/"
cp "$VIT/htp_backend_ext_config_vit.json" "$OUT/"

# pipeline glue
cp "$LLMDEPLOY_ROOT/configs/genie_image_encoder_qwen3vl.json" "$OUT/"
cp "$LLMDEPLOY_ROOT/configs/genie_text_encoder_qwen3vl.json" "$OUT/"
cp "$LLMDEPLOY_ROOT/configs/genie_text_generator_qwen3vl_4b.json" "$OUT/"
cp "$LLMDEPLOY_ROOT/configs/genie_pipeline_qwen3vl.script" "$OUT/"

# deterministic sample image (red circle + blue square -- same scene as the
# E2E gate, so device output is directly comparable to the gate's tierA text)
$PY_DEPLOY - <<'PYEOF'
from PIL import Image, ImageDraw
import os
img = Image.new("RGB", (512, 512), "white")
d = ImageDraw.Draw(img)
d.ellipse((96, 96, 288, 288), fill=(220, 30, 30))
d.rectangle((300, 300, 460, 460), fill=(30, 60, 220))
img.save(os.path.join(os.environ["SCRATCH"], "sample_image.png"))
PYEOF
$PY_DEPLOY "$LLMDEPLOY_ROOT/scripts/pipeline/preprocess_image.py" \
    --model "$MODEL" --image "$SCRATCH/sample_image.png" \
    --out "$OUT/sample_image.raw"
cp "$SCRATCH/sample_image.png" "$OUT/"

# Gate 2: static contract lint (fails the build on any violation)
$PY_DEPLOY "$LLMDEPLOY_ROOT/scripts/validate/lint_pipeline_bundle.py" \
    --bundle "$OUT" --model "$MODEL" \
    --ctx-info "$LLMDEPLOY_DATA/work/ctxbin/qwen3vl-4b-vit-fp16/info.json" \
    --ctx-info "$LLMDEPLOY_DATA/work/ctxbin/qwen3vl-4b-w8a16-split/1_of_2/info.json" \
    --ctx-info "$LLMDEPLOY_DATA/work/ctxbin/qwen3vl-4b-w8a16-split/2_of_2/info.json"

tar -C "$LLMDEPLOY_DATA/bundles" -czf "$OUT.tar.gz" "$BUNDLE"
du -sh "$OUT" "$OUT.tar.gz"
echo "BUNDLE COMPLETE: $OUT"
```

  NOTE: verify the exact split-ctx-bin dir name before running
  (`ls $LLMDEPLOY_DATA/work/ctxbin/ | grep split` — this plan writes
  `qwen3vl-4b-w8a16-split`; `vl_text_bundle.sh:24` is the authority).
- [ ] **Step 11.2:** `chmod +x scripts/build/vl_pipeline_bundle.sh && scripts/build/vl_pipeline_bundle.sh`
  Expected: lint `PASS`, `BUNDLE COMPLETE`, dir ≈ 7.1 GB.
- [ ] **Step 11.3:** Run the Task 10.2 lint mutations now, against a copy.
- [ ] **Step 11.4: Commit:**
  `git add scripts/build/vl_pipeline_bundle.sh && git commit -m "feat: e2e pipeline bundle assembly with lint gate"`

---

## Phase 3 — Handoff docs + HF upload

### Task 12: Device test doc

**Files:** Create: `docs/DEVICE_TEST_qwen3vl_e2e.md`

- [ ] **Step 12.1: Write it** — must contain, concretely: (a) push+run:

```bash
adb push qwen3vl_4b_e2e_pipeline /data/local/tmp/qwen3vl
adb shell
cd /data/local/tmp/qwen3vl && chmod +x genie-app
LD_LIBRARY_PATH=. ./genie-app -s genie_pipeline_qwen3vl.script
```

  (b) the expected output: the tierA string recorded in Step 7.4, with the
  caveat that W8A16 + fp16 ViT may reword it — "a red circle and a blue
  square" content is the success bar, not byte equality; (c) expected
  prefill latency ~15–25 s (162 decode-graph prompt steps — probe C); (d) a
  triage table:

| Symptom | First suspect | Check |
|---|---|---|
| load error mentioning tensor/quant mismatch | split encodings lineage | `docs/NOTES-genie-splits.md` |
| SIGSEGV at first token | unbound graph_names | `qnn-context-binary-utility --json_file`, compare both htp ext configs |
| output is `!!!...` (token 0) | logits read past one-row buffer | prefill all-position contract, `NOTES-genie-io.md` |
| fluent text ignoring the image | image embeddings not spliced | pipeline connect lines; accumulator order |
| gibberish only when image present | mrope grid derivation on device | probe B notes; try text-only prompt through same script |
| `node set image` size error | raw blob dtype/size | must be 3,145,728 bytes fp16 |

  (e) what to capture on failure: full stdout, `logcat | grep -i genie`, and
  the exact script used.
- [ ] **Step 12.2: Commit:**
  `git add docs/DEVICE_TEST_qwen3vl_e2e.md && git commit -m "docs: device test + triage runbook for the e2e pipeline"`

### Task 13: Bundle README + HF upload

**Files:** Create (scratch, not repo): `README.md` for the new HF folder; update the HF root `README.md`.

- [ ] **Step 13.1:** Write `qwen3vl_4b_e2e_pipeline/README.md` covering:
  what it is (Tier A stock pipeline, deepstack zeros = HF-minus-deepstack,
  by construction); the four validation gates and their results (Gate 0/1/2
  outputs, verbatim numbers from this run); **never executed on device**;
  the Step 7.4 tierA text as expected output; prefill-latency note;
  deepstack Tier B deferred; third-party Qualcomm binaries notice (copy the
  wording from the existing text README). Copy it into
  `$LLMDEPLOY_DATA/bundles/qwen3vl_4b_e2e_pipeline/README.md`.
- [ ] **Step 13.2:** Update the HF **root** README: three-folder table (add
  `qwen3vl_4b_e2e_pipeline/`, ~7.1 GB), and REWRITE the "They are not yet
  wired together" section to: wired via stock pipeline with deepstack zeros;
  full-fidelity deepstack remains open; still never run on device.
- [ ] **Step 13.3: Upload.** Rules from CLAUDE.md apply verbatim: proxy
  `http://127.0.0.1:17890`, `scripts/util/hf_upload_watchdog.sh` with
  `SOCKET_CHECKS=999999`, mind the 128 commits/hour cap. The three big
  files (2 text ctx-bins, LUT) are byte-identical to the already-uploaded
  copies → xet/LFS dedup makes the upload cheap; the genuinely new bytes are
  the ViT ctx-bin copy (~0.9 GB) + glue.

```bash
source scripts/env.sh
SOCKET_CHECKS=999999 scripts/util/hf_upload_watchdog.sh \
    vinccniv/sa8797p-qwen3vl-4b-bundles \
    "$LLMDEPLOY_DATA/bundles/qwen3vl_4b_e2e_pipeline" \
    qwen3vl_4b_e2e_pipeline
```

  (Check the watchdog's actual argument order in the script header before
  running; upload the updated root README as a separate single-file commit.)
- [ ] **Step 13.4: Verify:** list the repo
  (`HfApi().list_repo_files`) — expect 31 + ~17 files; spot-check one big
  file's size/hash via `HfApi().get_paths_info`. **Read and REPORT repo
  visibility (`repo_info(r).private`); never change it** (four incidents on
  record — CLAUDE.md).
- [ ] **Step 13.5:** Merge + push the branch:
  `git checkout main && git merge --no-ff qwen3vl-4b-stage3 -m "feat: Stage 3 e2e pipeline (Tier A) + full-path device-free gate" && git push origin main`
  (public repo: re-run the pre-push secret/size scan exactly as done for the
  Stage 2 push).

---

## Execution notes

- **Task order is the dependency order.** Probes (1–4) first — they are
  cheap and every later task consumes their findings. Gate 0 (Task 5) before
  Gate 1 (Task 7). Bundle (11) needs configs (8), script (9), lint (10),
  preprocess (6).
- **STOP conditions:** Probe A negative → Contingency 1b before Phase 2.
  Probe C shows Genie rejects >AR prompts → stop, surface to user (device
  path needs a past-KV prefill rebuild; the E2E gate is still worth
  finishing). Gate 1 chain1 not token-exact → do not proceed to Phase 2
  until it is; that failure means our splice/mrope/KV math is wrong and the
  device would be garbage.
- **Nothing here executes on the SA8797P.** The deliverable ends at: gated
  bundle + runbook on HF, waiting for a device.

## Self-review (done at planning time)

- Spec §6 Stage 3 coverage: pipeline configs (Task 8–9) ✓, driver =
  genie-app (§7-RESOLVED) ✓, full-path parity gate (Task 7) ✓, deepstack
  mitigation-by-zeros carried as Tier A ✓, Tier B explicitly deferred ✓.
- Known API-drift risks are flagged where they live (Step 7.2, Step 9.1,
  Step 13.3 watchdog args, Step 11.1 ctx dir name) rather than assumed.
- Type/name consistency: `mrope_tables` signature identical in Tasks 5/7;
  bundle filenames in Tasks 8/9/10/11 all match `vl_text_bundle.sh`'s
  shipped names; `PROMPT` string identical in Gate 1, Step 9.1, and lint
  check 5 (drift there is caught by lint, by design).
