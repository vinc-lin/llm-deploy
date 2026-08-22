# Device test — Qwen3-VL-4B end-to-end pipeline v3 (SA8797P)

**Nothing in this bundle has ever executed on an SA8797P.** HTP context binaries
cannot run on x86, and this SDK has no x86 execution path for W8A16 either
(`libQnnCpu` ships no 16-bit fixed-point kernels — see
`docs/NOTES-genie-pipeline.md`). Everything below comes from numerical parity
against HuggingFace at the ONNX level, plus static contract checks on the
finalised binaries. **This device run is the real test.**

## What changed since v2

v2's load-time failure (a `ShapeError` on `attention_mask`, root-caused in
`docs/REFERENCE.md` §3.6) is now history: v2 **did** load on device and
generate tokens. It crashed one step later, on `node set image`. v3 fixes
that crash and rebuilds the text tower on top of a defect found while
chasing it. Two things are different now:

1. **The `node set image` SIGSEGV is fixed at the root.** v2 crashed with
   `SIGSEGV (SEGV_ACCERR)` at `GenieNode_setData+572`, fault address exactly
   `base + 0x300000` — 3,145,728, the blob size. Disassembly of the shipped
   `libGenie.so` (`docs/DEVICE_TEST_qwen3vl_imgenc_sigsegv.md` §1.2b) showed
   `setData` does a caller-sized heap allocation and a `memcpy`, not DMA — so
   the fault is a Scudo guard page one byte past the end of the source buffer.
   **Every `.raw` in v3 is now payload + 4096 zero bytes** (3,149,824 bytes).
   The runtime still consumes only the first 3,145,728; the padding just moves
   the guard page out of reach. `lint_pipeline_bundle.py` enforces the padded
   size, so an unpadded blob cannot ship again.
2. **The text tower was rebuilt with grouped-query attention.** v2's tower
   replicated its 8 KV heads up to 32 query heads — 36 `Expand` ops per shard,
   which the QNN converter lowers into broadcast MULTIPLY ops whose large
   output the attention MatMul re-reads every step. The identical defect on
   the Qwen3-0.6B text model cost 74.7% of decode DSP cycles, and removing it
   took that model from 6.836 to 44.707 tok/s. The VL chain could never
   produce the grouped form before now because `--grouped-gqa` was not
   threaded through the VL exporter; it is now. Verified on the shipped DLCs:
   `4 DLC(s) checked, 0 failing`, replication ops **0**, attention MatMul batch
   dim **8** (was 32) — the decode MatMul shape went from `1x32x1x2176` to
   `1x8x4x2176`.

Converter `read_total_bytes`, v2 → v3 — **converter estimates for a build that
has never run on device.** They bound byte traffic; they do **not** predict
tok/s:

| graph | v2 | v3 | delta |
|---|---|---|---|
| `prefill_0` | 3,637,106,688 | 2,820,626,432 | −22.4% |
| `decode_0` | 3,609,839,616 | 2,237,399,040 | −38.0% |
| `prefill_1` | 4,413,063,168 | 3,596,582,912 | −18.5% |
| `decode_1` | 4,387,743,744 | 3,015,303,168 | −31.3% |

At roughly 4.3 GB of weights per token there is a byte floor near ~10 tok/s
that no topology change crosses, and whether VL-4B decode is byte-bound or
compute-bound on this silicon is unmeasured — so this table is not a tok/s
forecast, and none is given below either.

All device-free gates re-passed on the rebuilt tower: AIMET `--eval`
last-token argmax agreement **4/4** (unchanged from v2's bar); the full
6-chain `parity_e2e_vl.py` — all four gated chains **20/20** token-for-token
identical to HF, including `chain0b-prefillkv` (the device path, replaying
qualla's real three-call chunk plan); ctx-bin weight sharing 1.70 GB / 2.42 GB,
above the 1.4 / 2.0 floors; and the Genie load simulation (`validateModel`
replay) still **PASS**.

The expected caption is **unchanged from v2**, despite a completely new
quantization lineage — see the sample-image smoke test below.

None of this closes unknown failures. Fixing the two specific defects above
cannot rule out a third one: everything inside `libQnnHtp`/`libGenie` beyond
shape validation, the setData path, and GQA correctness is still unproven for
this configuration.

## Run it

```bash
adb push qwen3vl_4b_e2e_pipeline /data/local/tmp/qwen3vl
adb shell
cd /data/local/tmp/qwen3vl && chmod +x genie-app genie-t2t-run
```

The bundle is flat on purpose: Genie's loader resolves the `.so` files and every
config-referenced path from the bundle root, not from a `lib/` subdirectory.

## 1. Text-only first — tok/s and TTFT for the 4B text tower

**Do this before touching the image path.** It is the one measurement in this
session guaranteed to produce a result even if `node set image` still crashes,
it takes minutes rather than the whole session, and it does not depend on the
image path at all: no 4B two-shard W8A16 text tower has ever had its tok/s or
TTFT measured, so the number is independently valuable on its own.

```bash
LD_LIBRARY_PATH=. ./genie-t2t-run -c genie_dialog_qwen3vl_4b.json \
    -p "What is 2+2? Answer with one number." --profile qwen3vl_4b_text_profile.json
```

`-c` selects the dialog config, `-p` is the prompt string, and `--profile FILE`
writes a JSON performance file containing `init`, TTFT, PPR (prompt-processing
rate) and TGR (decode tok/s — the headline number) —
`docs/SA8797P_HTP_v81_Hardware_and_Deployment_Quantization_Reference_EN.md`
§10.3. That same section is why this is the invocation to use rather than
reading timing off the console: on Android, `--log verbose` sends logs to
`logcat`, not stdout, so `--profile` is the only way to get numbers directly
in the shell. There is no `--max-tokens` flag (`max-num-tokens` would need to
be set in the JSON, which this config does not set) and no `-n`; `-t` means
`--embedding_table`, not timing.

Report the `qwen3vl_4b_text_profile.json` contents verbatim alongside the
console output.

## 2. Smoke test — the sample image

```bash
LD_LIBRARY_PATH=. ./genie-app -s genie_pipeline_qwen3vl.script
```

`sample_image_fp32.raw` is a red circle and a blue square on a white background, the
same scene the parity gate used, so the output is directly comparable.
Prompt: *"Describe this image in one sentence."*

**Expected output** (recorded from the gate's `tierB` chain — the past-KV
prefill feed with deepstack zeroed, i.e. exactly what this bundle runs):

> A red circle and a blue square are positioned side by side on a white background.

**Do not expect a byte match.** That came from an fp32 ONNX run; on device both
towers are W8A16, so wording drifts. The bar is **semantic**: the caption must
name both shapes, both colours, and the white background. Fluent text describing
a *different* image is a failure — see triage.

## 3. If it still crashes on `node set image`

The v2/v3 crash mechanism is resolved: Genie interprets the image file as
float32 and reads 4 bytes per element, so the old UFixed16 blobs were a 2×
over-read — see `V4_CHANGES.md` §1. v4's `*_fp32.raw` blobs are the fix; the
padding theory (and its probes in `DEVICE_TEST_qwen3vl_imgenc_sigsegv.md`) is
superseded history.

If v4 crashes at `node set image` anyway, in order:

1. `ls -l *.raw` — the fed file must be `sample_image_fp32.raw` at exactly
   **6,295,552** bytes. A 3,145,728/3,149,824-byte file is a stale v3 blob
   shadowing the new one; delete every v3-era `sample_image.raw` / `wx_*.raw`
   and rerun.
2. Confirm the script line reads
   `node set image imageEncoder GENIE_NODE_IMAGE_ENCODER_IMAGE_INPUT sample_image_fp32.raw`.
3. If both check out and it still crashes, capture the tombstone (signal line,
   fault address, the `ls -l` output) and stop — that is a new mechanism, not
   a re-run candidate, and it goes back to the bundle developer with the
   tombstone rather than into more on-device experiments.

## 4. Weather / road kit

Six real photographs covering the deployment's scenes — road, surroundings,
weather (rain, fog, snow, clear, overcast, traffic). Each has its own script:

```bash
for s in wx_*.script; do
  echo "== $s"; LD_LIBRARY_PATH=. ./genie-app -s "$s"
done
```

`TEST_IMAGES.md` in the bundle carries, per image, the expected device-faithful
caption **and** the HF fp32 reference. Same semantic bar: the weather and the
scene contents must be right; wording will not match.

Every kit image was checked against the ViT's calibrated input range — all six
clip by at most 1.00 LSB, so none of them is out-of-domain for the encoder.

## 5. Timing — what to measure, and what we can and cannot predict

A 273-token prompt is processed as **three AR=128 prefill calls**
(`n_process` = 128, 128, 17 — the last padded), not as 273 sequential AR=1
decode steps. That is a structural fact of qualla's strategy loop
(`kvmanager.cpp:409,433`), verified in source and reproduced in the parity gate.

That structural fact makes the e2e run's own wall-clock a **prefill-selection
probe**, independent of whatever number it produces: a 273-token prompt should
be three AR=128 prefill calls, so a fast TTFT means prefill was selected as
designed, and a ~30 s one means it silently fell back to all-decode instead.
Which of those happens has never been observed on this device — report it
either way, it is a finding regardless of the raw number.

**We deliberately do not quote a TTFT number.** No Qwen3-VL graph has ever run
on this device, and the only decode-step measurement we have is the 0.6B text
model's 155 ms/step, against which this model carries roughly 4.5× the weight
traffic across two shards. Extrapolating that into a promise would be inventing
a number. What is defensible is the *ratio*: prefill replaces ~273 sequential
graph executions with 3, so TTFT should be far shorter than the all-decode
fallback below, and that difference is what we are asking you to measure.

Please report: init→first-token, tokens/s during generation, and the same two
numbers for the fallback if you end up running it. Compare cold-start numbers
only like-for-like — an init→first-logits vs TTFT unit mismatch once produced a
phantom "+134% regression" in this project.

## 6. If — and only if — the primary fails at load

Swap to the decode-only configuration. **One file, nothing else:**

```bash
LD_LIBRARY_PATH=. ./genie-app -s genie_pipeline_qwen3vl_decodeonly.script
```

It carries `load-select-graphs` / `execute-select-graphs: ["decode_0","decode_1"]`,
which filters the prefill graphs out before they ever become variants
(`nsp-model.cpp:314-318`), so nothing shape-validates them. The lint proves this
config differs from the primary by exactly those two keys, and the script by
exactly one line, so a failure here is not confounded by anything else.

Cost: **all** prefill. Every prompt token goes through the AR=1 decode graph.
Output should be unchanged — the gate's `chain0-alldecode` is exactly this path
and is token-identical to HF — but expect it to be *much* slower to first token.

**This fallback is untested on device and carries one unverified risk:** the same
graph-name list is handed to *both* contexts, and whether HTP tolerates an
enable-graphs name that is absent from a given binary is sealed inside
`libQnnHtp`. If the fallback *also* fails, **stop and report the error text** —
do not keep swapping configs.

## Triage

| Symptom | Most likely cause | Action |
|---|---|---|
| **`ShapeError: attention_mask Expected [1,AR,CL] Found [1,AR,AR]` → `Failed to create the Genie Node (-1)` + SIGSEGV** | The 2026-08-14 failure. **This must not happen with this bundle** — check 9 gates exactly this, and the shipped graphs carry `[1,128,2176]` | If it happens anyway, the shipped binary is not the one that was gated. Capture the full log and the `info.json`s and stop; do not try the fallback first |
| `Failed to create the Genie Node (-1)` with a *different* tensor named | A validateModel rule the load simulation does not model | Capture the exact `Expected/Found` line — it maps directly onto `nsp-model.cpp:844-917` and tells us which rule to add |
| **`SIGSEGV (SEGV_ACCERR)` at `GenieNode_setData+572` on `node set image`** | The v2 failure. **Should not happen with this bundle** — every shipped `.raw` is now padded | See §3, "If it still crashes on `node set image`" |
| Loads, but the caption describes a different image | Image blob / encoding mismatch, or MRoPE not engaging | Confirm `.raw` is **3,149,824 bytes** (payload + 4096-byte pad) — an unpadded blob is itself a finding, not just a size check; check `vision-param` is present in the image-encoder config (its absence silently drops image rows to plain rope) |
| Caption is fluent but generic ("a photo of a scene") | Image features not reaching the text tower | Check the ImageEncoder→TextGenerator connection line in the script and that the ViT ctx-bin loaded |
| Repetition until `Context Size was exceeded` | Sampler config, not the model | The shipped generator is greedy with `context.size 2048`; report it and move on |
| Fallback also fails at load | The cross-context enable-graphs risk above | Report the error text verbatim. This is the documented unknown |

## Known limitations of this bundle

- **Deepstack is fed zeros.** A stock Genie pipeline has no path to deliver the
  ViT's three deepstack tensors to the text tower, so those graph inputs are
  zero-filled at load (`initializeUnconnectedInputs`). This is a *defined*
  degradation, not a bug: it costs phrasing, not image understanding. Every
  expected caption in this bundle was generated **with the same degradation**,
  so they are directly comparable.
- The sub-128-token uninitialised-read hazard is **closed**: the prefill graph's
  deepstack inputs were renamed (`..._p`) so they get their own allocations and
  each is zero-filled at its own full size.
- W8A16 throughout. Wording will differ from any fp32 reference.
- `context.size` is 2048; prompts longer than that will be refused.

## What to capture on failure

1. The complete stdout/stderr, including everything before the first error.
2. `adb logcat` around the run.
3. Which script you ran, and whether the fallback was tried.
4. The `*.info.json` files from the bundle you actually pushed — that is how we
   confirm the gated bytes are the run bytes.

Screen photographs are fine; they get transcribed to Markdown and the Markdown
becomes the record.
