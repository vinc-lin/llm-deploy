# Qwen3-VL-4B on SA8797P — operator guide (bundle v4)

**This is the reference document: what the bundle is, how to install it, every
command, every metric definition, and the triage tree.** For the ordered
session plan with time budgets, see `SESSION_RUNBOOK.md`. For *why* v4 differs
from v3, see `V4_CHANGES.md`. Per-image expected captions are in
`TEST_IMAGES.md`.

## 0. Read this before anything else

**v4 changes two interface files and no ctx-bins.** All three binaries are
byte-identical to v3 (md5s in `V4_CHANGES.md` §4) — you can copy them from
your existing v3 deployment instead of re-downloading 6 GB.

Two v3 failures, two different states:

| v3 failure | Status in v4 |
|---|---|
| ImageEncoder `SIGSEGV` on `node set image` | **Root-caused and fixed host-side.** Genie reads the image file as **float32**, so v3's UFixed16 blob was a 2× (~3 MB) over-read. v4 ships float32 blobs |
| Text-only output was garbage | **One real defect fixed** (`bos-token`), **but not claimed solved.** V1 is the retest |

**The one thing that changes how you run this session:** run **V1 (text-only)
first**, because its result determines whether the image captions can be judged
at all. If the text tower still emits garbage, then V2/V3 will produce garbage
captions *even if the image path is working perfectly* — and in that case V2's
pass criterion is **"no SIGSEGV"**, nothing more. Judging a caption before
knowing V1's result will produce a wrong conclusion.

**Two blob formats ship per image, and only one is a pipeline input:**

| File | Bytes | Use |
|---|---:|---|
| `*_fp32.raw` | **6,295,552** | **The pipeline input.** `node set image` feeds only these |
| `*_u16.raw` | 3,145,728 | `qnn-net-run` triage only — **feeding one to genie-app reproduces the v3 SIGSEGV** |

**Delete every v3-era `sample_image.raw` / `wx_*.raw` from the device before
running v4.** They are the old format; if one shadows a new file, the crash
comes back and looks like the fix failed.

### What has and has not run on this silicon

Earlier statements that "nothing in this bundle has ever executed on an
SA8797P" are **no longer true** and should be disregarded:

* Both text ctx-bins **load and execute** on device (v2, v3 sessions).
* The ViT ctx-bin **executes correctly** under `qnn-net-run`, producing all
  four outputs at the expected sizes (v3 session, test T5).
* The **image path through Genie has never completed** — it crashed in
  `setData` before the graph ran.
* The text tower's **output quality on device is bad and unexplained** — the
  open item V1 retests.

Everything else below still comes from host-side numerical parity against
HuggingFace plus static contract checks on the final binaries.

---

## 1. What this is

An image→text pipeline running entirely on the HTP:

```
image (*_fp32.raw) → ImageEncoder (ViT, W8A16)
                         ↓ 256 embedding rows
                     splice into the prompt
                         ↓
                     TextGenerator (Qwen3-VL-4B text tower, W8A16,
                                    split across 2 weight-shared ctx-bins)
                         ↓
                     caption
```

Shapes that determine what you should see:

| | |
|---|---|
| Prompt | `Describe this image in one sentence.` — **273 tokens**, of which 256 are image rows |
| Text graphs | `prefill_0`/`prefill_1` (AR=128), `decode_0`/`decode_1` (AR=1), all CL=2176 |
| Prefill calls expected | **3** (`n_process` 128, 128, 17 — the last padded), *not* 273 decode steps |
| Context size | 2048 (longer prompts are refused) |
| Image blob fed | `*_fp32.raw`, exactly **6,295,552** bytes |

## 2. Files, by role

**69 files, flat by design.** Genie's loader resolves the `.so` files and every
config-referenced path from the bundle root. **Do not reorganise into
subdirectories and do not remove anything** — both break loading.

| Role | Files |
|---|---|
| Model | `qwen3vl-4b-vit-w8a16_ctx.bin` (414 MB), `qwen3vl-4b-w8a16_1_of_2.bin` (1.8 GB), `qwen3vl-4b-w8a16_2_of_2.bin` (2.5 GB), `embedding_float32_lut.bin` (1.5 GB) |
| Runtime | `genie-app`, `genie-t2t-run`, `qnn-net-run`, 7 × `lib*.so` |
| Configs | `genie_image_encoder_qwen3vl.json`, `genie_text_encoder_qwen3vl.json`, `genie_text_generator_qwen3vl_4b.json`, `htp_backend_ext_config_{vit,vltext}.json`, `embedding_lut_params.json` |
| Scripts | `genie_pipeline_qwen3vl.script` (primary), `wx_*.script` (6 kit images) |
| Text-only | `genie_dialog_qwen3vl_4b.json` (drives `genie-t2t-run`) |
| Fallback | `genie_text_generator_qwen3vl_4b_decodeonly.json`, `genie_pipeline_qwen3vl_decodeonly.script` |
| **Pipeline inputs** | `sample_image_fp32.raw`, `wx_*_fp32.raw`, `prompt_seg{1,2}.txt`, `tokenizer.json` |
| **Triage inputs (not pipeline)** | `sample_image_u16.raw`, `wx_*_u16.raw` |
| Reference | `*_fp32.json` sidecars, `*.info.json` (3), `sample_image.png`, `wx_*.jpg` |
| Docs | `V4_CHANGES.md`, this file, `SESSION_RUNBOOK.md`, `README.md`, `TEST_IMAGES.md` |

Total ≈ **6.3 GB** on device, before KV cache.

## 3. Install

```bash
adb push qwen3vl_4b_e2e_pipeline_v4 /data/local/tmp/qwen3vl_v4
adb shell
cd /data/local/tmp/qwen3vl_v4
chmod +x genie-app genie-t2t-run qnn-net-run
export LD_LIBRARY_PATH=.

# sanity — right CWD, right blob format, no stale blobs
ls libGenie.so genie_pipeline_qwen3vl.script
ls -l sample_image_fp32.raw          # MUST read 6295552
# stale v3 blobs must not exist -- this must print NOTHING
find . -maxdepth 1 -name '*.raw' ! -name '*_fp32.raw' ! -name '*_u16.raw'
```

That last check matters: a v3-era `sample_image.raw` left in the directory is
the one way to reproduce the old crash with a correct bundle.

If you are reusing v3's binaries to save the download, verify them:

```bash
md5sum qwen3vl-4b-vit-w8a16_ctx.bin qwen3vl-4b-w8a16_1_of_2.bin \
       qwen3vl-4b-w8a16_2_of_2.bin embedding_float32_lut.bin
```

against the md5s in `V4_CHANGES.md` §4.

Requirements: **~6.3 GB free** plus KV cache; run from inside the bundle
directory with `LD_LIBRARY_PATH=.`; do not set `ADSP_LIBRARY_PATH`. The PD
session is unsigned (`libQnnHtpV81Skel.so`). If the board throttles
aggressively, note it — it changes every timing number.

## 4. The tests

Full ordering, timing and stop conditions are in `SESSION_RUNBOOK.md`. The
commands themselves:

### V1 — text only (run first)

```bash
./genie-t2t-run -c genie_dialog_qwen3vl_4b.json \
    -p "What is 2+2? Answer with one number." \
    --profile v1_short_profile.json
```

`-c` config, `-p` prompt, `--profile FILE` writes JSON with `init`, TTFT, PPR
and decode rate. **Use `--profile`** — on Android `--log verbose` goes to
logcat, not stdout, so this is the only way to get numbers in the shell. There
is no `--max-tokens` flag (it would be `max-num-tokens` inside the JSON, which
this config does not set), and `-t` means `--embedding_table`, not timing.

This is a **retest of the v3 session's garbage output**, with `bos-token`
corrected. Report the generated text verbatim even if it is nonsense — and
note that the **timing numbers are valid regardless of output quality**, so V1
produces the first tok/s figure for a 4B two-shard W8A16 tower either way.

### V2 — image pipeline, sample image (the headline)

```bash
./genie-app -s genie_pipeline_qwen3vl.script
```

This is the exact sequence that crashed in v3, with only the blob format
changed. Expected caption for `sample_image_fp32.raw` (a red circle and a blue
square on white):

> A red circle and a blue square are positioned side by side on a white background.

### V3 — weather / road kit

```bash
for s in wx_*.script; do echo "== $s"; ./genie-app -s "$s"; done
```

`TEST_IMAGES.md` gives, per image, the expected **device-faithful** caption and
an HF fp32 reference. Diff against the device-faithful column; the HF column is
context only (see §8).

### V4 — decode-only fallback (only if V2 fails at *load*)

```bash
./genie-app -s genie_pipeline_qwen3vl_decodeonly.script
```

Filters the prefill graphs out before they become variants. Cost: no prefill —
every prompt token goes through the AR=1 decode graph, so TTFT will be far
longer. Output should be unchanged. **Never run on device**, and it carries one
unverified risk: the same graph-name list goes to both contexts. If it also
fails, report the error text and stop swapping configs.

### V5 — newer libGenie (optional, only if V2 still SIGSEGVs)

Only if a later QAIRT runtime is on hand. Swap **only** the runtime — never a
`.bin`:

```bash
cp libGenie.so libGenie.so.bak
# push the newer libGenie.so (and genie-app from the SAME SDK if a symbol/ABI
# error appears), then rerun V2. Revert with the .bak.
```

## 5. Metrics

**Use these definitions verbatim.** A unit mismatch once produced a phantom
"+134% regression" in this project.

| Metric | Definition | Unit |
|---|---|---|
| **Init time** | process start → all nodes created and pipeline ready, *before* any prompt data | ms |
| **TTFT** | first byte of prompt fed → **first generated token emitted**; includes ViT, splice and all prefill calls | ms |
| **Prefill calls** | text-graph executions consumed by the prompt. Expected **3** | count |
| **Decode rate** | generated tokens ÷ wall time generating, **excluding** TTFT | tok/s |
| **Total wall** | process start → last token | ms |
| **Peak RSS** | peak resident memory of the process | MB |

Rules: never compare `init→first-logits` against `TTFT` — different quantities;
say which your harness reports. Report cold and warm **separately**, never
averaged. Report **all repetitions**, not the best.

### Fill this in

| # | Run | Init (ms) | TTFT (ms) | Decode (tok/s) | Total (ms) | Peak RSS (MB) | Output |
|---|---|---|---|---|---|---|---|
| V1 | text-only, cold | | | | | | text verbatim |
| V1 | text-only, warm ×2 | | | | | | |
| V2 | image pipeline, cold | | | | | | caption verbatim |
| V2 | image pipeline, warm ×2 | | | | | | |
| V3 | kit, 6 images | | | | | | 6 captions |
| V4 | fallback — only if needed | | | | | | |

Also report once: **how many prefill calls** the 273-token prompt consumed, and
the raw profile JSONs.

### What we predict, and what we refuse to predict

We predict the **structure**: 3 prefill calls, not 273 decode steps — a
property of qualla's strategy loop, verified in source and reproduced in the
host parity gate. A fast TTFT means prefill was selected; a very long one
(tens of seconds) means it silently fell back to all-decode. Report which.

**We do not quote an expected tok/s or TTFT, deliberately.** Turning v3's
measured 34% byte reduction into a rate needs an effective-bandwidth figure,
and both available anchors are known-bad in this project's own records: the
"49 GB/s streaming ceiling" was measured under `qnn-net-run` at the default
clock (whose `--perf_profile` is a documented no-op) with a 1.95× swing to the
burst clock decode actually uses, and the byte figure usually paired with it is
pre-GQA-fix. Multiplying two unreliable numbers would be inventing a result.
**Your measurement is the first data point.**

## 6. Pass / fail

Judge each test against its own bar — and read V1's result before judging V2.

| Test | Pass | Fail |
|---|---|---|
| **V1** | Coherent answer ("4"; a sensible paragraph) | Garbage / repetition. **Not fatal to the session** — record it, continue to V2, and judge V2 by crash-behaviour only |
| **V2** | **No SIGSEGV, pipeline completes** — this alone clears the v3 blocker | Any SIGSEGV at `node set image`; `ShapeError`; failure to load |
| **V2 caption** *(only if V1 passed)* | Names both shapes, both colours, the white background | Fluent text describing a *different* scene; repetition to context exhaustion |
| **V3** *(only if V1 passed)* | Weather and scene contents semantically right | Different scene, or degeneration |

**Not a failure:** wording differing from the expected captions — those came
from fp32 ONNX runs and the device is W8A16. The bar is **semantic**. Also not
a failure: kit captions differing from the *HF fp32* column, which the device
cannot reach by design (§8).

## 7. Triage

| Symptom | Most likely cause | Action |
|---|---|---|
| `SIGSEGV (SEGV_ACCERR)` on `node set image` | A stale v3 blob is being fed, or a new mechanism | `ls -l` every `.raw`: the fed file must be `*_fp32.raw` at 6,295,552 B. Delete v3-era blobs and retry. If it still crashes, capture the tombstone and **stop** — it goes back to the bundle developer, not into more on-device experiments |
| `ShapeError: attention_mask Expected [1,AR,CL] Found [1,AR,AR]` → `Failed to create the Genie Node (-1)` | The v1-era failure. **Must not happen** — a load simulation gates exactly this | The pushed bytes are not the gated bytes. Capture the log + `*.info.json` and stop |
| `Failed to create the Genie Node (-1)`, different tensor named | A validator rule the load simulation does not model | Capture the exact `Expected/Found` line — it maps to a specific rule |
| Loads, caption describes a different image | Image not reaching the tower, or MRoPE not engaging | Confirm the blob is `*_fp32.raw` at 6,295,552 B; confirm `vision-param` is present in the image-encoder config |
| Caption fluent but generic ("a photo of a scene") | Image features not reaching the text tower | Check the ImageEncoder→TextGenerator connect line and that the ViT ctx-bin loaded |
| Garbage text in **both** V1 and V2 | The open text defect, not an image problem | Expected possibility — record both and report. Do not change configs chasing it |
| Repetition until `Context Size was exceeded` | Sampler config, not the model | Shipped generator is greedy, `context.size` 2048. Report and move on |
| Fallback (V4) also fails at load | The cross-context enable-graphs risk | Report the error verbatim. This is the documented unknown |

## 8. Known limitations

* **Deepstack is fed zeros.** A stock Genie pipeline has no path to deliver the
  ViT's three deepstack tensors to the text tower, so those inputs are
  zero-filled at load. A *defined* degradation, not a bug: it costs phrasing,
  not image understanding. Every expected caption shipped here was generated
  under the same degradation, which is why you diff against the device-faithful
  column and not the HF one.
* **The text tower's device output quality is unresolved** (§0). V1 is the
  retest; a second failure moves the investigation to the ctx-bin conversion
  numerics, which no host gate currently covers.
* W8A16 throughout; wording will differ from any fp32 reference.
* `context.size` is 2048.
* **Single image, first turn only.** qualla resets its rope-delta base on a
  second image, and `visionPos` is batch-local while the rope table is indexed
  by absolute KV position — a second image, or an image in turn ≥2, lands at
  wrong offsets.
* **Fixed 512×512 input**, 1024 patches → 256 rows. Images are resized,
  distorting aspect ratio.
* The decode-only fallback is device-unvalidated by definition.

## 9. What to send back

1. The §5 table, filled in, cold and warm separate, all repetitions.
2. Every profile JSON verbatim.
3. All generated text: V1's answers and all 7 captions (sample + 6 kit), as
   text — **including garbage output**, which is data.
4. Complete stdout/stderr for any failure, everything before the first error
   included, plus `adb logcat` around the run.
5. On any crash: the tombstone (signal line, fault address, backtrace) and
   `ls -l *.raw`.
6. The `*.info.json` files from the bundle you actually pushed — that is how we
   confirm the gated bytes are the run bytes.
7. Anything that surprised you, even if it did not fail a test.

A negative result is a good result if it is precise. "Failed at load with
`<exact error>`" is far more useful than "didn't work".

Screen photographs are fine; they get transcribed to Markdown and the Markdown
becomes the record.
