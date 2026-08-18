# Qwen3-VL-4B on SA8797P — bundle v5

Image + text in, description out, as one flat `genie-app` bundle for the
Qualcomm **SA8797P** (Hexagon v81 HTP, Android GVM), built against **QAIRT
2.48.40.260702 / libGenie 1.19**.

This file is the landing page: what is in the bundle, how to deploy it, what
has been validated, and what is still open. **The procedure lives elsewhere** —
see the document map below.

---

## Start here

| Read | For |
|---|---|
| **`V5_CHANGES.md`** | what this bundle is for — **read first** |
| **`SESSION_RUNBOOK.md`** | the ordered procedure for this session (probes A and C) |
| **`OPERATOR_GUIDE.md`** | full reference: install, metric definitions, pass/fail, triage, limitations |
| `V4_CHANGES.md` | the image fix (float32 blobs), still current — ctx-bin md5s live here |
| `TEST_IMAGES.md` | per-image expected captions for the weather kit |

> **v5 is a DIAGNOSTIC bundle.** The ctx-bins are unchanged from v3/v4 and no
> caption is expected from it. Its deliverable is a verdict on which stage
> breaks the text tower — see `V5_CHANGES.md`.

```bash
adb push qwen3vl_4b_e2e_pipeline_v5 /data/local/tmp/qwen3vl_v5
adb shell
cd /data/local/tmp/qwen3vl_v5 && chmod +x genie-app genie-t2t-run qnn-net-run
export LD_LIBRARY_PATH=.

ls -l sample_image_fp32.raw               # MUST read 6295552
# stale v3 blobs must not exist -- this must print NOTHING
find . -maxdepth 1 -name '*.raw' ! -name '*_fp32.raw' ! -name '*_u16.raw'

./genie-t2t-run -c genie_dialog_qwen3vl_4b.json -p "What is 2+2? Answer with one number." --profile v1.json
./genie-app -s genie_pipeline_qwen3vl.script
```

Expected caption (semantically — wording will differ):

> A red circle and a blue square are positioned side by side on a white background.

> **The image blob is float32 (`*_fp32.raw`, 6,295,552 bytes).** The
> `*_u16.raw` files are `qnn-net-run` triage inputs; feeding one to `genie-app`
> reproduces the v3 SIGSEGV. Delete any v3-era `sample_image.raw` / `wx_*.raw`
> from the device first — `V4_CHANGES.md` §1.

---

## 1. Status — what has and has not run on silicon

Earlier versions of this file claimed nothing here had ever executed on an
SA8797P. **That is no longer true**, and the current picture is:

| | Status |
|---|---|
| Text ctx-bins load and execute | ✅ confirmed on device (v2, v3 sessions) |
| ViT ctx-bin executes | ✅ confirmed under `qnn-net-run`, 4 outputs at 1,310,720 B each |
| Image path through Genie | ❌ never completed — crashed in `setData`. **v4 fixes the cause** (`V4_CHANGES.md` §1) |
| Text output quality on device | ❌ garbage in the v3 session. **Open.** One config defect fixed in v4; V1 is the retest |
| Any tok/s / TTFT figure | ⏳ never measured for this model on this silicon |

Everything not marked ✅ above rests on host-side numerical parity against
HuggingFace plus static contract checks on the final binaries (§4).

## 2. File manifest

**69 files, ~6.3 GB, flat on purpose.** Genie's loader resolves the `.so` files
and every config-referenced path from the bundle root, not from a `lib/`
subdirectory. **Do not reorganise into subfolders; do not remove anything.**

### 2.1 Required to run

| File | Size | Purpose |
|---|---:|---|
| `genie-app` | 1.0 MB | The driver. The only prebuilt binary exposing `GenieNode_*` / `GeniePipeline_*` and the image-encoder roles |
| `libGenie.so` | 9.8 MB | Genie runtime |
| `libQnnHtp.so` | 3.6 MB | QNN HTP backend |
| `libQnnHtpPrepare.so` | 84 MB | HTP graph preparation |
| `libQnnHtpV81Stub.so` | 0.7 MB | v81 stub (CPU side) |
| `libQnnHtpV81Skel.so` | 12 MB | v81 skeleton (DSP side) |
| `libQnnHtpNetRunExtensions.so` | 1.3 MB | Backend-extension loader |
| `libQnnSystem.so` | 3.9 MB | QNN system interface |
| `qwen3vl-4b-vit-w8a16_ctx.bin` | 414 MB | **Vision tower**, W8A16, `UFIXED_POINT_16` I/O |
| `qwen3vl-4b-w8a16_1_of_2.bin` | 1.8 GB | **Text shard 1** — `prefill_0`, `decode_0` (layers 0–17) |
| `qwen3vl-4b-w8a16_2_of_2.bin` | 2.5 GB | **Text shard 2** — `prefill_1`, `decode_1` (layers 18–35) + `lm_head` |
| `embedding_float32_lut.bin` | 1.5 GB | Token embedding table, float32. A fixed-point LUT silently no-ops against this graph |
| `tokenizer.json` | 6.7 MB | Tokenizer |
| `genie_image_encoder_qwen3vl.json` | <1 KB | ImageEncoder node config (carries `vision-param` in **patch** units) |
| `genie_text_encoder_qwen3vl.json` | <1 KB | LUT text-encoder node config |
| `genie_text_generator_qwen3vl_4b.json` | <1 KB | TextGenerator node config (context, sampler, MRoPE) |
| `genie_pipeline_qwen3vl.script` | 4 KB | **The primary script** |
| `htp_backend_ext_config_{vit,vltext}.json` | <1 KB | HTP tuning for the ViT / the four text graphs |
| `embedding_lut_params.json` | <1 KB | LUT dtype/size declaration |
| `prompt_seg{1,2}.txt` | <1 KB | Chat-template halves around the image. **Byte-exact, no trailing newline** |
| `sample_image_fp32.raw` | 6.0 MB | **The smoke-test pipeline input** — normalized float32; Genie quantizes it on device |

### 2.2 Test kit and triage inputs

| File | Count | Purpose |
|---|---:|---|
| `wx_*_fp32.raw` | 6 | **Pipeline inputs** — float32, 6,295,552 B each |
| `wx_*.script` | 6 | One pipeline script per image; differs from the primary only in the image path |
| `wx_*_fp32.json` | 6 | Sidecars: grid, encoding, clip statistics, source licence |
| `wx_*.jpg` | 6 | The exact 512×512 image the device sees — human reference, not used at runtime |
| `sample_image_u16.raw`, `wx_*_u16.raw` | 7 | **`qnn-net-run` triage only.** Exact tensor bytes (3,145,728 B). **Never feed these to `genie-app`** |
| `qnn-net-run` | 1 | Standalone graph runner — runs a ctx-bin with no Genie involved |
| `TEST_IMAGES.md` | 1 | Per-image expected captions (device-faithful **and** HF reference) |

### 2.3 Fallback (only if the primary fails at load)

| File | Purpose |
|---|---|
| `genie_pipeline_qwen3vl_decodeonly.script` | Differs from the primary by exactly one line |
| `genie_text_generator_qwen3vl_4b_decodeonly.json` | Adds `load-select-graphs` / `execute-select-graphs` only |

### 2.4 Reference

`*.info.json` (3) — graph/tensor dumps read back from the **final** binaries;
not read at runtime, this is how we prove the shipped bytes are the gated
bytes. `sample_image.png` — the source image. `genie-t2t-run` — text-only
driver, used by test V1. Documentation per the map at the top.

## 3. Deploying

```bash
adb push qwen3vl_4b_e2e_pipeline_v5 /data/local/tmp/qwen3vl_v5
adb shell
cd /data/local/tmp/qwen3vl_v5
chmod +x genie-app genie-t2t-run qnn-net-run
ls libGenie.so libQnnHtp.so libQnnHtpV81Skel.so    # loader sanity
```

* **~6.3 GB free** in `/data/local/tmp`, plus KV cache at runtime.
* Run with `LD_LIBRARY_PATH=.` from **inside** the bundle directory. Do not set
  `ADSP_LIBRARY_PATH`; do not create a `lib/` subdirectory.
* The PD session is **unsigned** — the skel is `libQnnHtpV81Skel.so`.
* If the board throttles aggressively, note it in the report; it changes every
  timing number.
* Reusing v3's ctx-bins to save the download is supported and encouraged —
  verify with the md5s in `V4_CHANGES.md` §4.

## 4. Validation already performed (device-free)

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
| Grouped-query attention on all four text DLCs | **0** replication ops, attention MatMul batch dim 8 |
| Weight sharing in the ctx-bins | 1.70 GB / 2.42 GB shared |
| Bundle contract lint | 11 checks incl. load simulation, fallback integrity, kit closure, fp32↔u16 blob agreement (0 LSB) |

The gate is non-vacuous six ways: zeroing deepstack moves row-127 logits by
1.4e+01; flat rope instead of MRoPE by 1.2e+01; zeroing the carried past moves
chunk-1 logits by 2.3e+01; dropping the position offset by 1.2e+01; the
zero-deepstack chains are live controls that flip 108/128 prefill-row argmaxes
while the gated chains stay exact; and the load simulation was accepted only
after reproducing the real v1 device failure.

**What no host gate covers:** the ctx-bin conversion step itself. Every
numerical gate above validates ONNX or replays a contract — none executes the
shipped `.bin`. That gap is the leading suspect for the open text issue.

## 5. Known limitations

1. **The text tower's device output quality is unresolved.** v3 produced
   garbage; v4 fixes one real config defect (`bos-token`) without claiming it
   is the whole cause. Test V1 is the retest.
2. **Deepstack is fed zeros.** Genie's stock `ImageEncoder` publishes exactly
   one output, so the ViT's three `deepstack_visual_embed` tensors have no
   route into the text tower; Qwen3-VL normally injects them into the first
   three decoder layers. Those inputs are explicitly zeroed at load, so this is
   a *defined* degradation — exactly HF-minus-deepstack — not undefined
   behaviour. It costs phrasing, not image understanding. **Every expected
   caption here was generated under the same degradation**, so they are
   directly comparable.
3. W8A16 throughout. Wording will differ from any fp32 reference.
4. `context.size` is 2048; longer prompts are refused.
5. **Single image, first turn.** qualla's rope-delta continuation resets its
   base on a second image, and `visionPos` is batch-local while the rope table
   is indexed by absolute KV position. A second image, or an image in turn ≥2,
   lands at wrong offsets.
6. **Fixed 512×512 input.** The graph is static: 1024 patches on a 32×32 grid →
   256 embedding rows. Every image is resized, distorting aspect ratio.
7. **The attention body runs fp16, not W8A16** — aimet-torch attaches
   quantizers to module outputs and Qwen3-VL's attention is functional code.
   Not a correctness issue; unexamined for performance.
8. The decode-only fallback is device-unvalidated by definition.

## 6. Third-party components

Embeds Qualcomm QAIRT 2.48.40.260702 runtime binaries (7 aarch64 `.so`,
`genie-app`, `genie-t2t-run`, `qnn-net-run`) which this repository's licence tag
does **not** cover. Redistributed under Qualcomm's SDK licence terms.

Test-kit photographs are freely licensed works from Wikimedia Commons and COCO
val2017; per-image licence and author are recorded in `TEST_IMAGES.md` and in
each `wx_*_fp32.json` sidecar.
