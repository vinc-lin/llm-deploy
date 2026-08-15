# Qwen3-VL-4B on SA8797P — deployment and test guide

Image + text in, description out, as one flat `genie-app` bundle for the
Qualcomm **SA8797P** (Hexagon v81 HTP, Android GVM), built against **QAIRT
2.48.40.260702 / libGenie 1.19**.

This document is self-contained: file manifest, deployment, how to run and
verify, the test plan, the metrics to collect, and what the report must
contain. Read §7 before running anything — it defines the metrics precisely,
because a unit mismatch in this project once produced a phantom "+134%
regression".

> **Nothing here has ever executed on an SA8797P.** HTP context binaries cannot
> run on x86, and this SDK has no x86 W8A16 path either (`libQnnCpu` ships no
> 16-bit fixed-point kernels). Everything claimed below is numerical parity
> against HuggingFace at the ONNX level plus static contract checks on the
> finalised binaries. **This device run is the real test.**

---

## 1. TL;DR

```bash
adb push qwen3vl_4b_e2e_pipeline_v2 /data/local/tmp/qwen3vl
adb shell
cd /data/local/tmp/qwen3vl && chmod +x genie-app
LD_LIBRARY_PATH=. ./genie-app -s genie_pipeline_qwen3vl.script
```

Expected (semantically — wording will differ, see §5):

> A red circle and a blue square are positioned side by side on a white background.

---

## 2. What changed since v1 (why there is a v2)

The v1 bundle **never loaded** (device attempt, 2026-08-14). Node creation died
with two `ShapeError`s on `attention_mask` (`Expected [1,128,2176]`, `Found
[1,128,128]`) and a SIGSEGV, before a single token.

Cause: in a **split** tower, shard 0's prefill graph has no `logits`, so
libGenie classifies it `DECODER_PREFILL` and rewrites its expected context
length to the cache-group maximum. An `AR == CL` "bertcache" prefill mask can
never satisfy that.

v2 rebuilds the text tower with a **past-KV prefill** — `attention_mask
[1,128,2176]`, `past_key_N_in [1,8,128,2048]` — which is byte-for-byte what the
validator demanded, and the same recipe as the device-proven 0.6B `ladekv`
build. A **load simulation** now replays libGenie's own `validateModel` against
the shipped `info.json`s as a build gate; it was accepted as a gate only after
it reproduced the v1 failure, on the v1 binaries, with the device's exact
message.

Consequence for you: a 273-token prompt now runs as **three AR=128 prefill
calls** instead of 273 sequential AR=1 decode steps.

---

## 3. File manifest — what is required and why

Everything is **flat on purpose**: Genie's loader resolves the `.so` files and
every config-referenced path from the bundle root, not from a `lib/`
subdirectory. Do not reorganise into subfolders.

### 3.1 Required to run (do not remove)

| File | Size | Purpose |
|---|---:|---|
| `genie-app` | 1.0 MB | The driver. The only prebuilt binary exposing `GenieNode_*` / `GeniePipeline_*` and the image-encoder roles |
| `libGenie.so` | 9.8 MB | Genie runtime |
| `libQnnHtp.so` | 3.6 MB | QNN HTP backend |
| `libQnnHtpPrepare.so` | 84 MB | HTP graph preparation |
| `libQnnHtpV81Stub.so` | 0.7 MB | v81 stub (CPU side) |
| `libQnnHtpV81Skel.so` | 12 MB | v81 skeleton (DSP side) |
| `libQnnHtpNetRunExtensions.so` | 1.3 MB | Backend-extension loader (binds the HTP tuning blocks) |
| `libQnnSystem.so` | 3.9 MB | QNN system interface |
| `qwen3vl-4b-vit-w8a16_ctx.bin` | 413 MB | **Vision tower** ctx-bin, W8A16, `UFIXED_POINT_16` I/O |
| `qwen3vl-4b-w8a16_1_of_2.bin` | 1.78 GB | **Text tower shard 1** — graphs `prefill_0`, `decode_0` (layers 0–17) |
| `qwen3vl-4b-w8a16_2_of_2.bin` | 2.52 GB | **Text tower shard 2** — graphs `prefill_1`, `decode_1` (layers 18–35) + `lm_head` |
| `embedding_float32_lut.bin` | 1.48 GB | Token embedding table, float32. A fixed-point LUT silently no-ops against this graph |
| `tokenizer.json` | 6.7 MB | Tokenizer |
| `genie_image_encoder_qwen3vl.json` | <1 KB | ImageEncoder node config (carries `vision-param` in **patch** units) |
| `genie_text_encoder_qwen3vl.json` | <1 KB | LUT text-encoder node config |
| `genie_text_generator_qwen3vl_4b.json` | <1 KB | TextGenerator node config (context size, sampler, MRoPE) |
| `genie_pipeline_qwen3vl.script` | 4 KB | **The primary script.** Wires the three nodes and feeds the prompt |
| `htp_backend_ext_config_vit.json` | <1 KB | HTP tuning for the ViT graph |
| `htp_backend_ext_config_vltext.json` | <1 KB | HTP tuning for the four text graphs |
| `embedding_lut_params.json` | <1 KB | LUT dtype/size declaration |
| `prompt_seg1.txt` / `prompt_seg2.txt` | <1 KB | The chat-template halves around the image. **Byte-exact, no trailing newline** |
| `sample_image.raw` | 3.0 MB | Smoke-test image, pre-quantized to the ViT's own input encoding |

Total required: **~6.2 GB** on device, before KV cache.

> **Why `.raw` and not `.png`:** Genie does **no** preprocessing. `node set
> image` hands the file to the graph as an opaque blob, so the bytes must
> already be exactly what `pixel_values` expects (1024×1536 `uint16`,
> 3,145,728 bytes, quantized with this ctx-bin's own scale/offset).

### 3.2 Test kit (needed for §6 tests T2, optional for T1)

| File | Count | Purpose |
|---|---:|---|
| `wx_*.raw` | 6 | Preprocessed weather/road images (3.0 MB each) |
| `wx_*.script` | 6 | One pipeline script per image — differs from the primary only in the image path |
| `wx_*.json` | 6 | Sidecars: grid, encoding, clip statistics, source licence |
| `wx_*.jpg` | 6 | The exact 512×512 image the device sees — for human eyes only, not used at runtime |
| `TEST_IMAGES.md` | 1 | Per-image expected captions (device-faithful **and** HF reference) |

### 3.3 Fallback (only if the primary fails to load — §8)

| File | Purpose |
|---|---|
| `genie_pipeline_qwen3vl_decodeonly.script` | Fallback script. Differs from the primary by exactly one line |
| `genie_text_generator_qwen3vl_4b_decodeonly.json` | Adds `load-select-graphs` / `execute-select-graphs` only |

### 3.4 Reference / documentation (safe to leave on the host)

| File | Purpose |
|---|---|
| `*.info.json` (3) | Graph/tensor dumps read back from the **final** binaries. Not read at runtime; this is how we prove the shipped bytes are the gated bytes |
| `sample_image.png`, `sample_image.json` | Source image and its sidecar |
| `README.md` (this file), `DEVICE_TEST.md` | Documentation |
| `genie-t2t-run` | Text-only driver, triage aid only. **Not** used by the pipeline |

---

## 4. Deploying to the board

```bash
# 1. push (either the folder, or the tarball if you have it)
adb push qwen3vl_4b_e2e_pipeline_v2 /data/local/tmp/qwen3vl

# 2. permissions
adb shell
cd /data/local/tmp/qwen3vl
chmod +x genie-app genie-t2t-run

# 3. sanity: the loader must find the .so files in the CWD
ls libGenie.so libQnnHtp.so libQnnHtpV81Skel.so
```

Requirements and cautions:

* **~6.2 GB free** in `/data/local/tmp` (plus KV cache at runtime).
* Run with `LD_LIBRARY_PATH=.` from **inside** the bundle directory. Do not set
  `ADSP_LIBRARY_PATH`; do not create a `lib/` subdirectory.
* The PD session is **unsigned** — the skel is `libQnnHtpV81Skel.so`.
* If the board has a governor/thermal policy that throttles aggressively, note
  it in the report; it changes every timing number.

---

## 5. Running and verifying

```bash
LD_LIBRARY_PATH=. ./genie-app -s genie_pipeline_qwen3vl.script
```

The prompt is *"Describe this image in one sentence."* over `sample_image.raw`
(a red circle and a blue square on a white background).

**Expected output** — recorded from the parity gate's `tierB` chain, which is
the past-KV prefill feed with deepstack zeroed, i.e. exactly what this bundle
runs:

> A red circle and a blue square are positioned side by side on a white background.

### How to judge it

**Do not expect a byte match.** That string came from an fp32 ONNX run; the
device is W8A16, so wording drifts.

| Verdict | Criterion |
|---|---|
| **PASS** | The caption names both shapes, both colours, and the white background |
| **FAIL** | Fluent text describing a *different* scene (→ image not reaching the tower, §8) |
| **FAIL** | Repetition until `Context Size was exceeded` |
| **FAIL** | Any crash, SIGSEGV, or `Failed to create the Genie Node` |

---

## 6. Test plan

Run in this order. **Stop and report at the first load failure** — do not
work around it silently.

### T1 — Smoke (required)

One run of the primary script on `sample_image.raw`. Judged by §5.
Capture full stdout/stderr, including everything before the first output token.

### T2 — Weather / road kit (required)

Six real photographs covering rain, fog, snow, clear, overcast and traffic:

```bash
for s in wx_*.script; do
  echo "===== $s"
  LD_LIBRARY_PATH=. ./genie-app -s "$s"
done
```

`TEST_IMAGES.md` gives, per image, the **device-faithful** expected caption and
the **HF fp32** reference. Diff against the device-faithful column.

Bar: **semantic agreement on the weather and the scene contents**. Wording will
not match. Flag an image only if the caption describes a different scene or
degenerates into repetition.

All six images were checked against the ViT's calibrated input range and clip by
at most 1.00 LSB, so none is an out-of-domain input.

### T3 — Warm/cold timing (required)

Cold start is materially different from warm. Run the smoke test **3 times
consecutively** without rebooting and record each separately. Report the first
run as *cold* and the rest as *warm*; do not average across them.

### T4 — Stability (recommended)

Run T2's loop **3 times back to back** (18 generations). Watch for: output
degrading over runs, memory growth, thermal throttling, or a crash on a later
iteration. Note the ambient/board temperature if available.

### T5 — Fallback (only if the primary fails to load)

See §8. If you run it, repeat T1 and T3 under the fallback so the two paths are
comparable.

---

## 7. Metrics to collect — and exactly how to define them

This section exists because a unit mismatch once produced a phantom "+134%
regression" in this project. **Use these definitions verbatim.**

| Metric | Definition | Unit |
|---|---|---|
| **Init time** | Process start → all three nodes created and the pipeline ready, *before* any prompt data is fed | ms |
| **TTFT** | First byte of prompt fed → **first generated token emitted**. Includes ViT, splice and all prefill calls | ms |
| **Prefill calls** | Number of text-graph executions consumed by the prompt. Expected: **3** for the 273-token prompt (`n_process` 128, 128, 17) | count |
| **Decode rate** | Generated tokens ÷ wall time spent generating, **excluding** TTFT | tok/s |
| **Total wall** | Process start → last token | ms |
| **Peak RSS** | Peak resident memory of the process | MB |
| **Peak DSP/VTCM** | If the platform exposes it | MB |

Rules:

1. **Never compare `init→first-logits` against `TTFT`.** They are different
   quantities. If your harness reports one, say which.
2. Report **cold and warm separately** (T3). Never average them.
3. Report **all three repetitions**, not just the best.
4. If you also run the fallback (T5), report the same metrics for it so the
   prefill benefit is measurable rather than assumed.

### What we expect, and what we deliberately do not predict

We predict the **structure**: a 273-token prompt should consume **3 prefill
calls**, not 273 decode steps. That is a property of qualla's strategy loop,
verified in source and reproduced in the parity gate.

We deliberately **do not quote a TTFT or tok/s figure**. No Qwen3-VL graph has
ever run on this device. The only decode-step measurement we hold is the 0.6B
text model's 155 ms/step, against which this model carries roughly 4.5× the
weight traffic across two shards — extrapolating that into a target would be
inventing a number, and a fabricated target is worse than none. **Your
measurement is the first data point.**

---

## 8. If the primary fails to load

Swap to the decode-only configuration — **one file, nothing else**:

```bash
LD_LIBRARY_PATH=. ./genie-app -s genie_pipeline_qwen3vl_decodeonly.script
```

It carries `load-select-graphs` / `execute-select-graphs: ["decode_0","decode_1"]`,
which filters the prefill graphs out before they ever become variants, so
nothing shape-validates them. The build lint proves this config differs from the
primary by exactly those two keys, and the script by exactly one line, so a
failure here is not confounded by anything else.

Cost: **all** prefill. Every prompt token goes through the AR=1 decode graph.
Output should be unchanged — the parity gate's `chain0-alldecode` is exactly
this path and is token-identical to HF — but expect it to be much slower to
first token.

**This fallback is untested on device** and carries one unverified risk: the
same graph-name list is handed to *both* contexts, and whether HTP tolerates an
enable-graphs name absent from a given binary is sealed inside `libQnnHtp`. **If
the fallback also fails, stop and report the error text** — do not keep swapping
configs.

### Triage

| Symptom | Most likely cause | Action |
|---|---|---|
| `ShapeError: attention_mask Expected [1,AR,CL] Found [1,AR,AR]` → `Failed to create the Genie Node (-1)` + SIGSEGV | The v1 failure. **Must not happen with v2** — the load gate covers exactly this | The pushed binary is not the gated one. Capture the log and the `*.info.json`, then stop |
| `Failed to create the Genie Node (-1)`, different tensor named | A validator rule the load simulation does not model | Capture the exact `Expected/Found` line — it maps onto a specific rule and tells us what to add |
| Loads, caption describes a different image | Image blob/encoding mismatch, or MRoPE not engaging | Confirm the `.raw` is 3,145,728 bytes; confirm `vision-param` is present in the image-encoder config |
| Caption fluent but generic ("a photo of a scene") | Image features not reaching the text tower | Check the ImageEncoder→TextGenerator connect line and that the ViT ctx-bin loaded |
| Repetition until `Context Size was exceeded` | Sampler config, not the model | Shipped generator is greedy, `context.size` 2048. Report and move on |
| Fallback also fails at load | The cross-context enable-graphs risk | Report the error verbatim. This is the documented unknown |

---

## 9. What the final report must contain

Please structure the report as below. Markdown is ideal; screen photographs are
fine — they get transcribed and the Markdown becomes the record.

1. **Environment**
   - Board / SoC revision, Android build, kernel
   - Free space in `/data/local/tmp`, RAM
   - Governor / thermal policy if non-default
   - Bundle identity: folder name and the `*.info.json` files from the bundle
     you **actually pushed** (this is how we confirm gated bytes == run bytes)

2. **T1 Smoke** — the command, the full output, PASS/FAIL against §5, and the
   complete stdout/stderr including everything before the first token.

3. **T2 Weather kit** — a table of image → caption produced → PASS/FAIL vs the
   device-faithful column of `TEST_IMAGES.md`, with a one-line note on any
   mismatch (different scene? wrong weather? repetition?).

4. **T3 Timing** — the §7 metric table, **three rows** (cold, warm-1, warm-2),
   with each metric named exactly as in §7 and the measurement method stated
   (which timestamps, from where).

5. **T4 Stability** — did output, memory or timing drift across 18 generations?
   Any crash, throttle, or slowdown, with the iteration number.

6. **T5 Fallback** — only if run. Why it was needed, plus T1/T3 repeated under it.

7. **Anomalies** — anything that surprised you, even if it did not fail the
   test. Warnings in the log, unusual pauses, odd token sequences.

8. **Raw artifacts** — `adb logcat` around the run, and the raw stdout files.
   Please attach rather than summarise: per-run detail is the point.

A negative result is a good result if it is precise. "Failed at load with
`<exact error>`" is far more useful than "didn't work".

---

## 10. Validation already performed (device-free)

| Gate | Result |
|---|---|
| Full path image → ViT → splice → text tower → tokens, vs `hf.generate` | **20/20 token-identical on 4 chains** |
| ↳ `chain0b-prefillkv` — **the device path**, past-KV prefill under the real chunk plan (128/128/17) | 20/20 |
| ↳ `chain0-alldecode` — worst case, every prompt row through decode | 20/20 |
| ↳ `chain1` — bertcache prefill + decode tail | 20/20 |
| ↳ `chain2` — fully-ONNX, features from the exported ViT graph | 20/20 |
| Per-chunk prompt logits vs HF (past-KV prefill) | 4.6e-03 / 1.3e-03 / 3.3e-04, **0 argmax mismatches** across all 273 rows |
| **Genie load simulation** (`validateModel` replay on the shipped `info.json`s) | **PASS** — and it reproduces the v1 failure on the v1 binaries, both shards, with the device's exact message |
| Interleaved MRoPE tables vs HF rotary | **bit-exact**, 0.000e+00 |
| ViT W8A16 quantsim vs fp32, 6 held-out photos | `image_features` cos **0.9975**; deepstack 0.9998 / 0.9986 / 0.9977 |
| ViT ctx-bin I/O contract | `pixel_values` + all 4 outputs `UFIXED_POINT_16`, scales byte-equal to the encodings |
| Text tower split vs unsplit | bit-identical — logits and all 72 KV outputs at 0.000e+00 |
| Weight sharing in the rebuilt ctx-bins | 1.70 GB / 2.42 GB shared |
| Bundle contract lint | 11 checks incl. load simulation, fallback integrity, kit closure |

The gate is non-vacuous six ways: zeroing deepstack moves row-127 logits by
1.4e+01; flat rope instead of MRoPE by 1.2e+01; zeroing the carried past moves
chunk-1 logits by 2.3e+01; dropping the position offset by 1.2e+01; the
zero-deepstack chains are live controls that flip 108/128 prefill-row argmaxes
while the gated chains stay exact; and the load simulation was accepted only
after reproducing the real v1 device failure.

---

## 11. Known limitations

**1. Never executed on target hardware.** See the banner at the top. The
internal W8A16 error is covered only at quantsim level, and the device memory
budget (~6.2 GB before KV) is unverified.

**2. Deepstack is fed zeros.** Genie's stock `ImageEncoder` publishes exactly
one output, so the ViT's three `deepstack_visual_embed` tensors have no route
into the text tower; Qwen3-VL normally injects them into the first three decoder
layers. Those graph inputs are explicitly zeroed at load, so this is a *defined*
degradation — exactly HF-minus-deepstack — not undefined behaviour. It costs
phrasing, not image understanding. **Every expected caption in this bundle was
generated under the same degradation**, so they are directly comparable.

**3. Sub-128-token prompts — CLOSED in v2.** The zero-fill of a shared input is
sized from the last registered variant, so a shared `deepstack_visual_embed_k`
was memset at decode's 1 row while prefill read 128. v1 was safe only because
its prefill never loaded. v2 renames the prefill graph's three deepstack inputs
(`..._p`), giving them their own allocations.

**4. First-token latency — ADDRESSED in v2.** v1's `[1,128,128]` prefill made
`ctx_size == AR`, so every prompt token went through the AR=1 decode graph (and
in the split tower it did not load at all). v2's past-KV prefill is selected.
No TTFT figure is quoted — see §7.

**5. Single image, first turn.** qualla's rope-delta continuation resets its
base on a second image, and `visionPos` is batch-local while the rope table is
indexed by absolute KV position. A second image, or an image in turn ≥2, lands
at wrong offsets.

**6. Fixed 512×512 input.** The graph is static: 1024 patches on a 32×32 grid →
256 embedding rows. Every image is resized to 512×512, distorting aspect ratio.

**7. The attention body runs fp16, not W8A16** — aimet-torch attaches quantizers
to module outputs and Qwen3-VL's attention is functional code. Not a correctness
issue; unexamined for performance.

---

## 12. Third-party components

Embeds Qualcomm QAIRT 2.48.40.260702 runtime binaries (7 aarch64 `.so`,
`genie-app`, `genie-t2t-run`) which this repository's licence tag does **not**
cover. Redistributed under Qualcomm's SDK licence terms.

Test-kit photographs are freely licensed works from Wikimedia Commons and COCO
val2017; per-image licence and author are recorded in `TEST_IMAGES.md` and in
each `wx_*.json` sidecar.
