# Device test — Qwen3-VL-4B end-to-end pipeline v2 (SA8797P)

**Nothing in this bundle has ever executed on an SA8797P.** HTP context binaries
cannot run on x86, and this SDK has no x86 execution path for W8A16 either
(`libQnnCpu` ships no 16-bit fixed-point kernels — see
`docs/NOTES-genie-pipeline.md`). Everything below comes from numerical parity
against HuggingFace at the ONNX level, plus static contract checks on the
finalised binaries. **This device run is the real test.**

## What changed since the 2026-08-14 attempt

That bundle **never loaded**. Node creation died with two `ShapeError`s on
`attention_mask` (`Expected [1,128,2176]`, `Found [1,128,128]`) and a SIGSEGV,
before a single token. Root cause in `docs/REFERENCE.md` §3.6: in a *split*
tower, shard 0's prefill has no `logits`, so it classifies `DECODER_PREFILL`,
its expected CL is rewritten to the cache-group maximum, and an AR==CL
"bertcache" mask can never satisfy that.

Two things are different now:

1. **The text tower was rebuilt with a past-KV prefill.** `attention_mask` is
   `[1,128,2176]` and `past_key_N_in` is `[1,8,128,2048]` — byte-for-byte what
   the validator demanded. This is the same recipe as the device-proven 0.6B
   `ladekv` build, not a new idea.
2. **A load simulation now gates the build.** `lint_pipeline_bundle.py` check 9
   replays libGenie's own `validateModel` — graph classification, AR/CL
   derivation, the cache-group scatter/concat detection, the `DECODER_PREFILL`
   CL rewrite, and every `checkShape` — against the shipped `info.json`s. It was
   only accepted as a gate once it **reproduced the 2026-08-14 failure** on the
   old binaries, with the device's exact message, on both shards. It passes on
   what ships here.

That closes the specific failure. It cannot close unknown ones: everything
inside `libQnnHtp`/`libGenie` beyond shape validation is still unproven for this
configuration.

## Run it

```bash
adb push qwen3vl_4b_e2e_pipeline /data/local/tmp/qwen3vl
adb shell
cd /data/local/tmp/qwen3vl && chmod +x genie-app
LD_LIBRARY_PATH=. ./genie-app -s genie_pipeline_qwen3vl.script
```

The bundle is flat on purpose: Genie's loader resolves the `.so` files and every
config-referenced path from the bundle root, not from a `lib/` subdirectory.

## 1. Smoke test — the sample image

`sample_image.raw` is a red circle and a blue square on a white background, the
same scene the parity gate used, so the output is directly comparable.
Prompt: *"Describe this image in one sentence."*

**Expected output** (recorded from the gate's `tierB` chain — the past-KV
prefill feed with deepstack zeroed, i.e. exactly what this bundle runs):

> A red circle and a blue square are positioned side by side on a white background.

**Do not expect a byte match.** That came from an fp32 ONNX run; on device both
towers are W8A16, so wording drifts. The bar is **semantic**: the caption must
name both shapes, both colours, and the white background. Fluent text describing
a *different* image is a failure — see triage.

## 2. Weather / road kit

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

## 3. Timing — what to measure, and what we can and cannot predict

A 273-token prompt is processed as **three AR=128 prefill calls**
(`n_process` = 128, 128, 17 — the last padded), not as 273 sequential AR=1
decode steps. That is a structural fact of qualla's strategy loop
(`kvmanager.cpp:409,433`), verified in source and reproduced in the parity gate.

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

## 4. If — and only if — the primary fails at load

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
| Loads, but the caption describes a different image | Image blob / encoding mismatch, or MRoPE not engaging | Confirm `.raw` is 3,145,728 bytes; check `vision-param` is present in the image-encoder config (its absence silently drops image rows to plain rope) |
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
