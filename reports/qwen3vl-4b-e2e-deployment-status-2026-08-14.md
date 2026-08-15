# Qwen3-VL-4B-Instruct E2E Deployment on SA8797P — Status Report

> **Reconstructed** from the eight screen photographs at `reports/0814/IMG_3043..IMG_3050.HEIC`
> (2026-08-15). The photographs show a rendered Markdown document,
> `QWEN3VL_E2E_DEPLOYMENT_STATUS.md`, served from the device team's workspace at
> `/workspaces/cib/tree/data/sa8797-deploy-kit/docs/test/` — the path is legible in the
> browser URL bar of IMG_3046/3049/3050. The eight photographs are consecutive scroll
> captures covering §1–§8 with overlap; no section is missing.
>
> Body text is transcribed **verbatim**. Nothing has been re-worded, re-ordered or
> "corrected" in the transcription. Where a statement is ambiguous or where this repo's
> own docs corroborate or clarify it, that is recorded in **§9 Editorial notes** at the
> end and never inline.

**Date:** 2026-08-14 **Status:** Device test blocked — fix identified, pending hardware bring-up **Bundle:** `bundles/qwen3vl-4b-e2e-pipeline/`
**Target:** Qualcomm SA8797P (nordy / Gen5), Hexagon v81 HTP, unsigned PD, 16 MB VTCM **Runtime:** QAIRT 2.48.40 / QNN API v2.37.0 / libGenie 1.19.0 / `genie-app` pipeline

---

## 1. Bundle overview

Flat layout (all files in one directory, `LD_LIBRARY_PATH=.`), ~6.2 GB total.

| Component | File(s) | Size | Notes |
|---|---|---|---|
| ViT ctx-bin | `qwen3vl-4b-vit-w8a16_ctx.bin` | 413 MB | W8A16, graph `vit`, UFixed16 I/O |
| Text tower shard 1 | `qwen3vl-4b-w8a16_1_of_2.bin` | 1.85 GB | graphs: `prefill_0`, `decode_0` (18 heads/layer) |
| Text tower shard 2 | `qwen3vl-4b-w8a16_2_of_2.bin` | 2.63 GB | graphs: `prefill_1`, `decode_1` (18 heads/layer) + logits |
| Embedding LUT | `embedding_float32_lut.bin` | 1.55 GB | float32, [151936, 2560] |
| Tokenizer | `tokenizer.json` | 7 MB | Qwen3 tokenizer, vocab 151936 |
| Pipeline runner | `genie-app` | 1 MB | aarch64 native, executes `.script` files |
| Genie libs | `libGenie.so` + 6 libQnn*.so | ~115 MB | 7 ARM .so + 1 DSP skel (`libQnnHtpV81Skel.so`) |
| Pipeline script | `genie_pipeline_qwen3vl.script` | — | wires ImageEncoder → LUT TextEncoder → TextGenerator |
| Configs | `genie_image_encoder_qwen3vl.json`, `genie_text_encoder_qwen3vl.json`, `genie_text_generator_qwen3vl_4b.json` | — | see §3 |
| Perf configs | `htp_backend_ext_config_vit.json`, `htp_backend_ext_config_vltext.json` | — | O:3, vtcm_mb:16, burst, unsigned PD |
| Prompts | `prompt_seg1.txt` / `prompt_seg2.txt` | — | build the Qwen chat template |
| Sample image | `sample_image.raw` + `.json` + `.png` | 3 MB | UFixed16 512×512 red circle + blue square |

---

## 2. Architecture of the pipeline

Three Genie nodes wired by `genie_pipeline_qwen3vl.script`:

```
ImageEncoder    ── image_features [256, 2560] UFixed16 ──┐
                                                         ├─→ TextGenerator ──→ tokens
LUT TextEncoder ── text embeddings ──────────────────────┘
```

- **ImageEncoder** runs the ViT ctx-bin on HTP. Input is a raw pixel blob (UFixed16, scale=3.05e-05, offset=-32768 → fp32 range [-1,1]). Vision-param set to 32×32 patches so MRoPE engages correctly.
- **LUT TextEncoder** converts text to embeddings using the float32 embedding table. Fixed-point LUT is not used because the text tower's `inputs_embeds` is fp16 (and the float32 LUT requantizes automatically).
- **TextGenerator** is 2-shard tensor-parallel W8A16, context.size=2048, greedy sampling, MRoPE (rope-theta=5,000,000, mrope-section=[24,20,20], spatial-merge-size=2).

**Prompt construction** follows the Qwen chat template exactly:

```
<|im_start|>user\n<|vision_start|><|image_pad|>×256<|vision_end|>Describe this image in one sentence.<|im_end|>\n<|im_start|>assistant\n
```

This is split into two text segments (before / after the 256 image embeddings) so the pipeline can splice image features between them. Total prompt = **273 tokens**.

---

## 3. Known bundle limitations (documented by authors)

From `README.md` / `DEVICE_TEST.md` — these are expected, not bugs:

| # | Limitation | Impact |
|---|---|---|
| 1 | **Deepstack is fed zeros.** Genie's `ImageEncoder` only publishes one output; the 3 `deepstack_visual_embed_*` inputs to the text tower are zeroed by `initializeUnconnectedInputs`. | Quality: phrasing degradation, not image understanding loss. HF exactness drops 0→20/20 but semantic content is preserved. |
| 2 | **All-decode path only (~30 s TTFT).** Prefill graph has `attention_mask` `[1,128,128]` with no past-KV input, so Genie's strategy loop skips it for prompts >128 tokens. All 273 prompt tokens go through AR=1 decode one at a time. | Performance: ~27–30 s before first token. Fix requires text-tower re-export with past-KV prefill. |
| 3 | **Sub-128-token prompt hazard.** The deepstack memset uses the last graph variant's spec (`decode_0` AR=1 = 5120 bytes). Short prompts that select prefill would read uninitialized deepstack memory. | Safety: only affects sub-128-token prompts. Our 273-token prompt is safe. |
| 4 | **Single image, first turn only.** MRoPE rope-delta continuation resets on second image; visionPos is batch-local. | Capability: multi-image / multi-turn not supported. |
| 5 | **Fixed 512×512 input.** Static graph — 1024 patches → 256 embeddings. Aspect ratio distorted. | Quality: minimal for typical photos. |
| 6 | **Attention body is fp16, not W8A16.** AIMET quantizes module outputs; Qwen3-VL's attention is functional code so quantizers don't attach. | Performance: unexamined; not a correctness issue. |

---

## 4. Host-side validation (proven)

All gates below passed before the bundle was packaged. None required device access.

| Gate | Result |
|---|---|
| Full path: image → ViT → splice → text tower → generated tokens vs HF | **20/20 token-identical** (3 independent chains) |
| ↳ chain0-alldecode (the feed pattern device uses) | 20/20 |
| Per-row prompt logits vs HF (all 273 rows) | max abs 3.7e-04, 0 argmax mismatches |
| MRoPE tables vs HF rotary | **bit-exact** |
| ViT W8A16 quantsim vs fp32 (6 held-out photos) | image_features cos **0.9975**; deepstack 0.9998 / 0.9986 / 0.9977 |
| ViT ctx-bin I/O contract | all UFixed16, scales byte-equal |
| Text tower split vs unsplit | bit-identical logits + 72 KV outputs |
| Bundle contract lint | reference closure, graph binding, schema, vision-param, LUT, sample image, chat-template |

---

## 5. Device deployment attempt (2026-08-14)

### 5.1 What worked

- **Bundle successfully pushed to device** (~6.2 GB via ipc → adb push to `/data/local/tmp/qwen3vl_e2e/`).
- **ViT ImageEncoder node loads successfully** on HTP — we see the allocation line in logs.
- **3 FVC test images preprocessed** into UFixed16 blobs at `/mnt/code/build/vit_fvc_test/fvc_{0023,0036,0153}/` with per-image pipeline scripts ready.

### 5.2 Blocking issue: prefill validation error

**Error:**

```
ShapeError : attention_mask — Expected [ 1, 128, 2176] bitwidth=*. Found [ 1, 128, 128] bitwidth=2
ShapeError : attention_mask — Expected [ 1, 128, 2176] bitwidth=*. Found [ 1, 128, 128] bitwidth=2
Error validating model. Failed to create the Genie Node (-1).
Segmentation fault
```

### 5.3 Root cause

The text tower ctx-bin ships with `prefill_0` / `prefill_1` graphs where:

- `attention_mask` is `[1, 128, 128]` (self-attention only, no past KV)
- KV outputs are `[1, 8, 128, 128]` (only the 128 new tokens)

Genie sets `context.size = 2048`, so it validates every graph against a KV dimension of 2048 + AR = 2176. The prefill graphs fail this validation at node creation time, before any graph-selection logic runs.

### 5.4 Why the "obvious" fixes don't work

| Attempt | Why it fails |
|---|---|
| Remove prefill from `htp_backend_ext_config` `graph_names` | `graph_names` assigns per-graph perf options, not which graphs load. All graphs are always loaded and validated. |
| Set `context.size = 128` to match prefill | Prompt is 273 tokens; would overflow or truncate. Also decode's KV is sized for 2176. |
| Rebuild ctx-bin with `qnn-context-binary-generator` | Requires source DLCs, which aren't in the bundle. |
| Deserialize + re-serialize ctx-bin via QNN C API | `QnnContext_getBinary()` returns `QNN_COMMON_ERROR_OPERATION_NOT_PERMITTED` on contexts created from binary. |
| `qnn-context-binary-utility` extract graphs | The utility is read-only (JSON dump). No extraction mode. |

### 5.5 The fix: `execute-select-graphs` + `load-select-graphs`

Found in the Genie source (`examples/Genie/Genie/src/qualla/engines/qnn-htp.cpp:80-81`) — **undocumented** JSON keys that map to `QNN_CONTEXT_CONFIG_ENABLE_GRAPHS` passed to `QnnContext_createFromBinary`. When `load-select-graphs: true`, only the named graphs are deserialized from the ctx-bin; everything else is skipped, including validation.

**Applied to** `genie_text_generator_qwen3vl_4b.json` in the QnnHtp block:

```json
"execute-select-graphs": ["decode_0", "decode_1"],
"load-select-graphs": true
```

With this fix, only decode graphs are loaded. Their `attention_mask` is `[1, 1, 2176]` which matches Genie's expected spec exactly. The all-decode path (which the bundle was designed for) should work.

**Status:** Config updated locally and staged on ipc at `/tmp/qwen3vl_e2e_fix/`. **Not yet tested on device** because the device is currently offline (USB dropped, not visible in `lsusb` / `adb devices` on ipc).

---

## 6. Test plan (once device is back online)

### Step 1 — Smoke test with sample image

```
adb push genie_text_generator_qwen3vl_4b.json /data/local/tmp/qwen3vl_e2e/
adb shell "cd /data/local/tmp/qwen3vl_e2e && LD_LIBRARY_PATH=. ./genie-app -s genie_pipeline_qwen3vl.script"
```

**Expected output** (semantic match, exact wording may differ due to W8A16):

> A red circle and a blue square are positioned side by side on a white background.

**Expected timing:** ~30 s TTFT (all-decode path), then generation at estimated 4–6 tok/s.

### Step 2 — 3 FVC images

Push fvc UFixed16 blobs + per-image scripts, run each, capture stdout + timings.

### Step 3 — Profile

If loading succeeds, capture `--profile` output (for text-only genie-t2t-run) or use timing markers from genie-app.

---

## 7. Open questions / next steps

1. **Bring device back online.** USB dropped; need to power-cycle or re-plug the SA8797P board.
2. **Verify `execute-select-graphs` fix.** If it works, this is a general-purpose tool for fixing miscompiled multi-graph ctx-bins without rebuilding.
3. **Deepstack on-device quality check.** The bundle ships with zeroed deepstack; we should quantify how much caption quality degrades on real photos vs HF baseline with deepstack.
4. **Real prefill for 4B.** Re-export the text tower with a past-KV prefill graph sized for context=2048 to bring TTFT from ~30 s down to ~3–4 s (assuming ~80 tok/s prefill).
5. **Report creation.** Once the e2e pipeline works, produce the standalone FVC test report with device timings.

---

## 8. Key files

- Bundle root: `bundles/qwen3vl-4b-e2e-pipeline/`
- Author docs: `bundles/qwen3vl-4b-e2e-pipeline/README.md`, `DEVICE_TEST.md`
- Text generator config (modified): `bundles/qwen3vl-4b-e2e-pipeline/genie_text_generator_qwen3vl_4b.json`
- Pipeline script: `bundles/qwen3vl-4b-e2e-pipeline/genie_pipeline_qwen3vl.script`
- FVC test images: `docs/test/fvc_0023.jpg`, `fvc_0036.jpg`, `fvc_0153.jpg`
- Preprocessed fvc blobs + scripts: `/mnt/code/build/vit_fvc_test/fvc_{XXX}/`
- Prior FP16 ViT report: `docs/test/VLM_CAPTION_REPORT.md` (hybrid device/host)
- Bundle on device (when reachable): `/data/local/tmp/qwen3vl_e2e/`

---

## 9. Editorial notes (added during reconstruction — not in the photographed document)

These are transcription and cross-reference notes only. No figure, claim or conclusion
above has been altered.

1. **§3 row 1, "HF exactness drops 0→20/20", is transcribed verbatim and reads
   backwards.** The intended meaning is almost certainly *20/20 → 0*: exactness against HF
   is 20/20 **with real deepstack**, and the byte-exact match is lost when deepstack is
   zeroed. `scripts/validate/parity_e2e_vl.py:16-34` makes this explicit — the three gated
   chains (`chain0-alldecode`, `chain1-hf-vit`, `chain2-onnx-vit`) all run *real*
   deepstack, while `tierA-zero-deep` is **not gated** and is described there as "the text
   the device will actually produce". `docs/DEVICE_TEST_qwen3vl_e2e.md:45-49` gives both
   strings and concludes "Zeroing deepstack costs phrasing, not image understanding" —
   consistent with the second half of the cell. Read the §4 "20/20 token-identical" gate as
   the real-deepstack number, not as something the shipped bundle reproduces on device.
2. **§1 "Configs → see §3" points at the limitations table**, which does not describe the
   config files. Transcribed as printed; likely a stale cross-reference.
3. **§3 row 2 / §5.3 are independently corroborated in this repo.**
   `docs/DEVICE_TEST_qwen3vl_e2e.md:55-64` derives the same ~27 s figure and pins the
   mechanism to source: qualla takes a graph's context size from the attention mask's
   trailing dim (`nsp-graph.cpp:146-155`), so `[1,128,128]` registers `ctx_size == AR`
   (bertcache) and the strategy loop skips that bucket for prompts >128 tokens
   (`kvmanager.cpp:411-416`). `docs/NOTES-genie-pipeline.md:186-196` walks the same
   selection path. This matches the CLAUDE.md hard contract on `[1,AR,AR]` prefill masks.
4. **§4's ViT cosines match this repo's recorded gate exactly** —
   `docs/REFERENCE.md:284` lists 0.9975 / 0.9998 / 0.9986 / 0.9977 against a min-cos ≥ 0.99
   bar. The 273-token prompt length and the 2176 KV dimension also match
   `docs/NOTES-genie-pipeline.md`.
5. **§3 row 3's hazard is a known one here**: `docs/NOTES-genie-pipeline.md:65-70` notes
   that a prompt shorter than 128 tokens *does* select prefill and then reads deepstack
   sized for the wrong variant, and that the mitigation is per-graph-distinct deepstack
   tensor names.
6. **`execute-select-graphs` / `load-select-graphs` (§5.5) was new to this repo** and has
   since been folded into `docs/NOTES-genie-io.md` § "Split prefill is fatal at load"
   (2026-08-15), verified against the 2.48.40 sources. Two refinements came out of that
   verification, both of which correct §5.3/§5.5 above:
   - **The two keys are not interchangeable.** `execute-select-graphs` drops graphs before
     they enter `m_variant_list` (`nsp-model.cpp:314-318`, ahead of `:320`), which is what
     skips validation. `load-select-graphs` only adds
     `QNN_CONTEXT_CONFIG_ENABLE_GRAPHS` to `QnnContext_createFromBinary`
     (`QnnApi.cpp:120`) and is a **no-op unless the list is non-empty** — so §5.5's "when
     `load-select-graphs: true` … everything else is skipped, including validation" has
     the two backwards.
   - **The failure is split-specific, not a blanket `context.size + AR` check.** Shard 0's
     `prefill_0` emits `last_hidden_states` and no `logits` (confirmed here by
     `qnn-context-binary-utility` on the shipped bin), so it classifies `DECODER_PREFILL`
     (`nsp-graph.cpp:247-249`) and only *then* is its expected CL rewritten to the
     cache-group max (`nsp-model.cpp:604-605`), failing `checkShape` at `:858`. The map is
     keyed by the global `(AR, CL)` variant and filled from the first split only (`:608`),
     which is exactly why the log shows **two** identical ShapeErrors. An unsplit 0.6B
     build with the same `[1,128,128]` mask loads fine.
7. **Where the findings landed:** `docs/NOTES-genie-io.md` (full source chain + key
   semantics), `docs/NOTES-genie-pipeline.md` §C1 (correcting probe C's "silently skipped"
   premise and its `DEFAULT` classification claim), `docs/REFERENCE.md` §3.6 (hard
   contract), §5 (e2e gate row + the real-deepstack caveat) and corrections #20/#21.
8. **Source photographs deleted 2026-08-15** once transcribed (formerly
   `reports/0814/IMG_3043..IMG_3050.HEIC`, never tracked in git) — same disposition as the
   2026-08-10 batch. **This document is now the only record of them.**
