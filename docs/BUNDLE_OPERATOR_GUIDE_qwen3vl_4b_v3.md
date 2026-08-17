# Qwen3-VL-4B on SA8797P — operator guide (bundle v3)

**Read this file first.** It is the single entry point: what this bundle is,
how to run it, what to measure, and exactly what to send back. The other two
documents go deeper — `DEVICE_TEST.md` is the step-by-step runbook with
triage, `TEST_IMAGES.md` is the per-image expected-output table — but you can
run the whole session from this file alone.

**Nothing in this bundle has ever executed on an SA8797P.** HTP context
binaries cannot run on x86 and this SDK has no x86 path for W8A16, so every
claim below comes from numerical parity against HuggingFace at the ONNX level
plus static contract checks on the finalised binaries. **Your run is the first
real test.** That is also why the metrics section matters more than usual:
there is no prior measurement of this model on this silicon to compare against.

---

## 1. What this is

A multimodal image→text pipeline running entirely on the HTP:

```
image (.raw blob) → ImageEncoder (ViT, W8A16)
                        ↓ 256 embedding rows
                    splice into the prompt
                        ↓
                    TextGenerator (Qwen3-VL-4B text tower, W8A16,
                                   split across 2 weight-shared ctx-bins)
                        ↓
                    caption
```

Key shapes, because they determine what you should see:

| | |
|---|---|
| Prompt | `Describe this image in one sentence.` — **273 tokens**, of which 256 are image rows |
| Text graphs | `prefill_0`/`prefill_1` (AR=128) and `decode_0`/`decode_1` (AR=1), all CL=2176 |
| Prefill calls expected | **3** (`n_process` 128, 128, 17 — the last padded), *not* 273 decode steps |
| Context size | 2048 (longer prompts are refused) |
| Image blob | exactly **3,149,824 bytes** (3,145,728 payload + 4,096 pad — see §3) |

## 2. Files, by role

59 files, flat by design — Genie's loader resolves the `.so` files and every
config-referenced path from the bundle root, not from subdirectories.

| Role | Files |
|---|---|
| Model | `qwen3vl-4b-vit-w8a16_ctx.bin` (413 MB), `qwen3vl-4b-w8a16_1_of_2.bin` (1.72 GB), `qwen3vl-4b-w8a16_2_of_2.bin` (2.45 GB) |
| Runtime | `genie-app`, `genie-t2t-run`, 7 × `lib*.so` |
| Configs | `genie_image_encoder_qwen3vl.json`, `genie_text_encoder_qwen3vl.json`, `genie_text_generator_qwen3vl_4b.json`, `htp_backend_ext_config_{vit,vltext}.json` |
| Scripts | `genie_pipeline_qwen3vl.script` (primary), `wx_*.script` (6 kit images) |
| Text-only | `genie_dialog_qwen3vl_4b.json` (drives `genie-t2t-run`) |
| Fallback | `genie_text_generator_qwen3vl_4b_decodeonly.json`, `genie_pipeline_qwen3vl_decodeonly.script` |
| Inputs | `sample_image.{raw,png,json}`, `wx_*.{raw,jpg,json}`, `prompt_seg{1,2}.txt`, `tokenizer.json`, `embedding_float32_lut.bin` |
| Docs | this file, `DEVICE_TEST.md`, `README.md`, `TEST_IMAGES.md`, `*.info.json` |

**Do not remove anything and do not reorganise into subdirectories** — both
break loading.

## 3. What changed in v3, and why it matters to you

**(a) Image blobs are now padded.** v2 loaded and generated text on device but
crashed on `node set image` with `SIGSEGV (SEGV_ACCERR)` at
`GenieNode_setData+572`, fault address `base + 0x300000` — which is exactly
3,145,728, the blob size. Disassembly of the shipped `libGenie.so` showed
`setData` does a caller-sized heap allocation plus a `memcpy`, not DMA, so a
fault at exactly `base + size` is a Scudo guard page hit by a one-byte
over-read. Every `.raw` here is therefore payload + 4,096 zero bytes. The
runtime consumes only the payload, so the padding is inert — it just moves the
guard page out of reach. **If you see that crash again, go to §7.**

**(b) The text tower was rebuilt with grouped-query attention.** v2's attention
replicated its 8 KV heads up to 32 query heads — 36 `Expand` ops per shard,
which the converter lowers into broadcast multiplies whose large output the
attention MatMul re-reads every step. On the Qwen3-0.6B text model that same
defect was measured at **74.7% of decode DSP cycles**, and removing it took
that model from 6.836 to 44.707 tok/s. The VL chain could never produce the
grouped form because the flag was not wired through its exporter. It is now:
all four shipped graphs verify at **0 replication ops**, attention MatMul batch
dim 8 (was 32).

Measured consequence, from the converter's own DDR accounting:

| graph | v2 bytes read | v3 bytes read | delta |
|---|---|---|---|
| `prefill_0` | 3,637,106,688 | 2,820,626,432 | −22.4% |
| `decode_0` | 3,609,839,616 | 2,237,399,040 | −38.0% |
| `prefill_1` | 4,413,063,168 | 3,596,582,912 | −18.5% |
| `decode_1` | 4,387,743,744 | 3,015,303,168 | −31.3% |

One decode step executes both shards, so **per generated token the tower now
reads ~5.25 GB instead of ~8.00 GB — 34% fewer bytes.** See §6 for why we do
*not* turn that into a predicted tok/s.

## 4. Install

```bash
adb push qwen3vl_4b_e2e_pipeline_v3.tar.gz /data/local/tmp/
adb shell
cd /data/local/tmp && tar xzf qwen3vl_4b_e2e_pipeline_v3.tar.gz
cd qwen3vl_4b_e2e_pipeline_v3 && chmod +x genie-app genie-t2t-run
ls libGenie.so genie_pipeline_qwen3vl.script      # sanity: you are in the right CWD
```

Everything below assumes that working directory and `LD_LIBRARY_PATH=.`.

## 5. Usage — three ways to run it

### 5a. Text only (no image) — run this FIRST

```bash
LD_LIBRARY_PATH=. ./genie-t2t-run -c genie_dialog_qwen3vl_4b.json \
    -p "What is 2+2? Answer with one number." \
    --profile qwen3vl_4b_text_profile.json
```

`-c` config, `-p` prompt, `--profile FILE` writes a JSON with `init`, TTFT,
PPR and the decode rate. **Use `--profile`** — on Android `--log verbose` goes
to logcat, not stdout, so this is the only way to get numbers in the shell.
There is no `--max-tokens` flag (it would be `max-num-tokens` inside the JSON,
which this config does not set) and `-t` means `--embedding_table`, not timing.

Do this first because it is the one measurement guaranteed to produce a result
even if the image path fails, it takes minutes, and **no 4B two-shard W8A16
text tower has ever had its tok/s measured** — the number stands alone.

### 5b. Image pipeline — the primary path

```bash
LD_LIBRARY_PATH=. ./genie-app -s genie_pipeline_qwen3vl.script
```

Expected caption for `sample_image.raw` (a red circle and a blue square on
white):

> A red circle and a blue square are positioned side by side on a white background.

### 5c. Weather / road kit

```bash
for s in wx_*.script; do echo "== $s"; LD_LIBRARY_PATH=. ./genie-app -s "$s"; done
```

`TEST_IMAGES.md` gives, per image, the expected device-faithful caption **and**
the HF fp32 reference. They are different things — see §8.

## 6. Required result metrics

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

Rules: never compare `init→first-logits` against `TTFT` — different
quantities, say which your harness reports. Report cold and warm **separately**,
never averaged. Report **all repetitions**, not the best.

### Fill this in

| # | Run | Init (ms) | TTFT (ms) | Decode (tok/s) | Total (ms) | Peak RSS (MB) | Output OK? |
|---|---|---|---|---|---|---|---|
| T1 | text-only (§5a), cold | | | | | | |
| T2 | text-only, warm ×3 | | | | | | |
| T3 | image pipeline (§5b), cold | | | | | | |
| T4 | image pipeline, warm ×3 | | | | | | |
| T5 | kit, 6 images | | | | | | 6 captions attached |
| T6 | fallback (§7c) — only if needed | | | | | | |

Also report, once: **how many prefill calls** the 273-token prompt consumed,
and the raw `qwen3vl_4b_text_profile.json`.

### What we predict, and what we refuse to predict

We predict the **structure**: 3 prefill calls, not 273 decode steps. That is a
property of qualla's strategy loop, verified in source and reproduced in the
host parity gate.

**We do not quote an expected tok/s or TTFT, and that is deliberate.** Turning
the 34% byte reduction into a rate needs an effective-bandwidth figure, and
both available anchors are known-bad in this project's own records: the
"49 GB/s streaming ceiling" was measured under `qnn-net-run` at the default
clock, whose `--perf_profile` flag is a documented no-op, with a 1.95× swing to
the burst clock decode actually uses — so it is a floor on the ceiling, not the
ceiling. And the 961 MB/step byte figure often paired with it is *pre*-GQA-fix.
Multiplying two unreliable numbers to produce a target would be inventing a
result, and a fabricated target is worse than none. **Your measurement is the
first data point.** What we can say: the tower reads 34% fewer bytes per token
than v2, and if decode turns out compute-bound rather than byte-bound the gain
should exceed that — on the 0.6B model the same fix gave 6.54×.

## 7. Failure paths

### 7a. It crashes on `node set image`

v3 already ships the padded blobs that should fix this. If it still crashes at
`GenieNode_setData`, recreate the unpadded probe **from the blob already on
device** — nothing extra to push:

```bash
head -c 3145728 sample_image.raw > nopad.raw
```

Then follow `DEVICE_TEST.md` §3 and the decision tree in the imaging runbook.
If **both** the padded and unpadded blobs crash, stop: the overrun is past the
destination rather than the source, which is not host-fixable. Capture the
tombstone and report — that becomes a vendor escalation, not another on-device
experiment.

### 7b. It fails at load with a `ShapeError`

Should not happen — a load simulation replaying libGenie's own `validateModel`
gates this build and passes on the shipped binaries. If it happens anyway, the
bytes you ran are not the bytes we gated: capture the full log plus the
`*.info.json` files from the bundle you actually pushed, and stop.

### 7c. Fallback: decode-only

Only if the primary fails to load. **One file swap:**

```bash
LD_LIBRARY_PATH=. ./genie-app -s genie_pipeline_qwen3vl_decodeonly.script
```

It filters the prefill graphs out before they become variants. Cost: all
prefill — every prompt token goes through the AR=1 decode graph, so expect a
much longer TTFT. Output should be unchanged. **This fallback has never run on
device** and carries one unverified risk (the same graph-name list goes to both
contexts). If it also fails, report the error text and stop swapping configs.

## 8. Pass / fail

**Pass:**
- Both ctx-bins load; no `ShapeError`, no SIGSEGV
- The sample caption names both shapes, both colours, and the white background
- Kit captions get the weather and scene contents right
- The 273-token prompt consumes 3 prefill calls
- Metrics in §6 are captured

**Not a failure:**
- **Wording differs from the expected captions.** Those came from an fp32 ONNX
  run; on device both towers are W8A16. The bar is **semantic**.
- Kit captions differing from the *HF fp32* column — that column is context
  only, and the device cannot reach it (see §9).

**Fail:** fluent text describing a *different* image; repetition until context
exhaustion; either binary refusing to load.

## 9. Known limitations

- **Deepstack is fed zeros.** A stock Genie pipeline has no path to deliver the
  ViT's three deepstack tensors to the text tower, so those inputs are
  zero-filled at load. This is a *defined* degradation, not a bug: it costs
  phrasing, not image understanding. Every expected caption shipped here was
  generated **with the same degradation**, so they are directly comparable —
  which is why you diff against the device-faithful column, not the HF one.
- W8A16 throughout; wording will differ from any fp32 reference.
- `context.size` is 2048.
- The decode-only fallback is device-unvalidated by definition.

## 10. What to send back

1. The §6 table, filled in, cold and warm separate, all repetitions.
2. `qwen3vl_4b_text_profile.json` verbatim.
3. All 7 captions (sample + 6 kit), copied as text.
4. Complete stdout/stderr for any failure, including everything before the
   first error, plus `adb logcat` around the run.
5. The `*.info.json` files from the bundle you actually pushed — that is how we
   confirm the gated bytes are the run bytes.

Screen photographs are fine; they get transcribed to Markdown and the Markdown
becomes the record.
