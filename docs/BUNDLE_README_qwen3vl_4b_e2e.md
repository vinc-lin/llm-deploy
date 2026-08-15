# Qwen3-VL-4B-Instruct end-to-end pipeline v2 — SA8797P (QNN/Genie)

Image + text in, description out, as one flat `genie-app` bundle for the
Qualcomm SA8797P (Hexagon v81 HTP, Android GVM), built against QAIRT
2.48.40.260702 / libGenie 1.19.

**v2 rebuilds the text tower with a past-KV prefill.** The v1 bundle
(2026-08-14) never loaded: in a split tower, shard 0's prefill has no `logits`,
so it classifies `DECODER_PREFILL`, its expected CL is rewritten to the
cache-group maximum, and the AR==CL "bertcache" mask can never satisfy that —
two `ShapeError`s and a SIGSEGV before a single token. v2 ships
`attention_mask [1,128,2176]` with `past_key_N_in [1,8,128,2048]`, which is what
the validator demanded, and a load-simulation gate that reproduces the v1
failure before it will certify anything.

> **This bundle has never run on an SA8797P.** It was built and validated
> entirely without device access. Read *Validation* for exactly what that
> covers — and `DEVICE_TEST.md` before the first run.

## What it does

```bash
adb push qwen3vl_4b_e2e_pipeline /data/local/tmp/qwen3vl
adb shell 'cd /data/local/tmp/qwen3vl && chmod +x genie-app && \
           LD_LIBRARY_PATH=. ./genie-app -s genie_pipeline_qwen3vl.script'
```

Three Genie nodes — an `ImageEncoder` (the W8A16 ViT), a `lut` text encoder, and
a `TextGenerator` (the 2-split W8A16 text tower) — wired into one
`GeniePipeline`. The bundled sample is a red circle and a blue square on white,
with the prompt *"Describe this image in one sentence."*

**Expected output**, recorded from the parity gate's `tierB` chain — the past-KV
prefill feed with deepstack zeroed, i.e. exactly the configuration shipped here:

> A red circle and a blue square are positioned side by side on a white background.

Wording will drift on device (both towers are W8A16; that string came from an
fp32 reference run). The success bar is semantic: both shapes, both colours, the
white background.

### Weather / road test kit

Six real photographs covering rain, fog, snow, clear, overcast and traffic, each
with its own script (`wx_*.script`) and its own expected caption. See
`TEST_IMAGES.md`. All six sit inside the ViT's calibrated input range (≤ 1.00 LSB
clip), so none of them is testing an out-of-domain input.

### Decode-only fallback

If the primary somehow still fails at load, `genie_pipeline_qwen3vl_decodeonly.script`
swaps in a text-generator config carrying `load-select-graphs` /
`execute-select-graphs`, filtering the prefill graphs out before they become
variants. It costs all prefill. It is **untested on device** and carries one
unverified risk (the same name list goes to both contexts). `DEVICE_TEST.md` has
the procedure and the failure signature.

## Validation

Everything below is numerical parity against HuggingFace plus static contract
checks on the finalised binaries. No gate here executed the Hexagon backend.

| Gate | Result |
|---|---|
| **Full path**: image → ViT → splice → text tower → generated tokens, vs `hf.generate` | **20/20 token-identical on 4 chains** |
| ↳ `chain0b-prefillkv` — **the device path**: past-KV prefill under qualla's real chunk plan (128/128/17) | 20/20 |
| ↳ `chain0-alldecode` — worst case, every prompt row through decode | 20/20 |
| ↳ `chain1` — bertcache prefill + decode tail, HF visual features | 20/20 |
| ↳ `chain2` — fully-ONNX, features from the exported ViT graph | 20/20 |
| Per-chunk prompt logits vs HF (past-KV prefill) | 4.6e-03 / 1.3e-03 / 3.3e-04, **0 argmax mismatches** in all 273 rows |
| **Genie load simulation** (`validateModel` replay on the shipped `info.json`s) | **PASS** — and it reproduces the v1 failure on the v1 binaries, both shards, with the device's exact message |
| Interleaved MRoPE tables vs HF rotary | **bit-exact**, 0.000e+00 |
| ViT W8A16 quantsim vs fp32, 6 held-out photos | `image_features` cos **0.9975**; deepstack 0.9998 / 0.9986 / 0.9977 |
| ViT ctx-bin I/O contract | `pixel_values` + all 4 outputs `UFIXED_POINT_16`, scales byte-equal to the encodings |
| Text tower split vs unsplit | bit-identical — logits and all 72 KV outputs at 0.000e+00 |
| Weight sharing in the rebuilt ctx-bins | 1.70 GB / 2.42 GB shared — the two AR variants in each shard share one weight set |
| Bundle contract lint | 11 checks: reference closure, graph binding, schema clashes, vision-param, ViT dtype, LUT, sample image, chat template, **load simulation, fallback integrity, kit closure** |

The gate is non-vacuous six ways: zeroing deepstack moves row-127 logits by
1.4e+01, flat rope instead of MRoPE moves them by 1.2e+01, zeroing the carried
past moves chunk-1 logits by 2.3e+01, dropping the position offset moves them by
1.2e+01, `tierA`/`tierB` are live controls that flip 108/128 prefill-row
argmaxes while the gated chains stay exact, and the load simulation is only
accepted once it reproduces the real v1 device failure.

Calibration was real throughout: **24 photographs** (COCO val2017 + SDK samples)
for the vision tower, and 22 multimodal windows with real ViT features spliced
at the image-token positions for the text tower.

## Known limitations

**1. Never executed on target hardware.** HTP context binaries cannot run on
x86, and this SDK has no x86 path for W8A16 either — `libQnnCpu` ships no
16-bit fixed-point kernels, verified with a single-Gemm probe. So the internal
W8A16 error is covered only at quantsim level. Combined footprint is ~6.3 GB
before KV; the device memory budget is unverified.

**2. Deepstack is fed zeros.** Genie's stock `ImageEncoder` publishes exactly one
output and throws on any other IO name, so the ViT's three
`deepstack_visual_embed` tensors have no route into the text tower. Qwen3-VL
injects those into the first three decoder layers. Those graph inputs are
explicitly zeroed at load by qualla's `initializeUnconnectedInputs`, so this is a
defined degradation — exactly HF-minus-deepstack — not undefined behaviour.

Measured cost: numerically large (0/20 token agreement with HF), qualitatively
small. The caption above names both shapes, both colours and the background
correctly. It costs phrasing, not image understanding. Every expected caption
shipped here was generated under the *same* degradation, so they are directly
comparable to what the device produces.

**3. ~~A short prompt would read uninitialised memory.~~ CLOSED in v2.** The
zeroing memset is sized from the *last* registered variant for a tensor NAME, so
a shared `deepstack_visual_embed_k` was memset at decode's 1 row while prefill
read 128. v1 was safe only because its prefill never loaded — which stops being
true the moment prefill works. v2 renames the prefill graph's three deepstack
inputs to `..._p`, giving them their own allocations, each zero-filled at its own
full size.

**4. ~~First token takes ~30 s.~~ ADDRESSED in v2.** v1's `[1,128,128]` prefill
made `ctx_size == AR`, so every prompt token went through the AR=1 decode graph
(and, in the split tower, the bundle did not load at all). v2's past-KV prefill
is selected: a 273-token prompt runs as **three AR=128 calls** instead of 273
sequential decode steps. **No TTFT number is quoted** — nothing has measured this
model on this device, and extrapolating the 0.6B's 155 ms/step across ~4.5× the
weight traffic would be inventing one. The structural change is the claim;
measuring it is what the device run is for.

**5. Single image, first turn.** qualla's rope-delta continuation resets its base
on a second image, and `visionPos` is batch-local while the rope table is indexed
by absolute KV position. A second image, or an image in turn ≥2, lands at wrong
offsets.

**6. Fixed 512×512 input.** The graph is static: 1024 patches on a 32×32 grid →
256 embedding rows. Every image is resized to 512×512, distorting aspect ratio.

**7. The attention body runs fp16, not W8A16** — aimet-torch attaches quantizers
to module outputs, and Qwen3-VL's attention is functional code. Not a
correctness issue; unexamined for performance.

## Third-party components

Embeds Qualcomm QAIRT 2.48.40.260702 runtime binaries (7 aarch64 `.so`,
`genie-app`, `genie-t2t-run`) which the repository's licence tag does **not**
cover. Redistributed under Qualcomm's SDK licence terms.

The test-kit photographs are freely licensed works from Wikimedia Commons and
COCO val2017; per-image licence and author are recorded in `TEST_IMAGES.md` and
in each `wx_*.json` sidecar.
