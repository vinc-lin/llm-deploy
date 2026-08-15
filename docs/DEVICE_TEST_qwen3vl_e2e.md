# Device test — Qwen3-VL-4B end-to-end pipeline (SA8797P)

**Nothing in this bundle has ever executed on an SA8797P.** HTP context binaries
cannot run on x86, and this SDK has no x86 execution path for W8A16 either
(`libQnnCpu` ships no 16-bit fixed-point kernels — see
`docs/NOTES-genie-pipeline.md`). Every number below comes from numerical parity
against HuggingFace at the ONNX level plus static contract checks on the
finalised binaries. **This first device run is the real test.**

What *is* proven device-free: the full host-side path — image → ViT → splice →
text tower under the runtime's exact feed pattern — reproduces
`hf.generate` **token-for-token** on a real image+text prompt, across three
independent chains including the all-decode path the device actually takes
(`scripts/validate/parity_e2e_vl.py`).

## Run it

```bash
adb push qwen3vl_4b_e2e_pipeline /data/local/tmp/qwen3vl
adb shell
cd /data/local/tmp/qwen3vl && chmod +x genie-app
LD_LIBRARY_PATH=. ./genie-app -s genie_pipeline_qwen3vl.script
```

The bundle is flat on purpose: Genie's loader resolves the `.so` files and every
config-referenced path from the bundle root, not from a `lib/` subdirectory.

## What success looks like

The bundled `sample_image.png` is a red circle and a blue square on a white
background — the same scene the parity gate used, so the outputs are directly
comparable. The prompt is *"Describe this image in one sentence."*

**Expected output**, recorded from the E2E gate's Tier-A chain (the exact
configuration this bundle ships — deepstack fed zeros):

> A red circle and a blue square are positioned side by side on a white background.

**Do not expect a byte match.** That string came from an fp32 ONNX run of the
text tower with fp32 ViT features. On device both towers are W8A16, so wording
will drift. The success bar is **semantic**: the caption must correctly name
both shapes, both colours, and the white background. Anything that does that is
a pass. Fluent text that describes a *different* image is a failure — see triage.

For reference, the same pipeline with real deepstack (not reachable through a
stock Genie pipeline — see below) produces
*"The image displays a simple composition of a red circle and a blue square on a
white background."*, which is HF's exact output. Zeroing deepstack costs
phrasing, not image understanding.

## ⛔ SUPERSEDED BY THE 2026-08-14 DEVICE ATTEMPT — this bundle does not load

**Everything below about timing was written before the bundle met hardware, and
the prediction was wrong.** The run never reached a first token: node creation
died at load with

```
ShapeError : attention_mask — Expected [ 1, 128, 2176] bitwidth=*. Found [ 1, 128, 128] bitwidth=2   (x2, one per shard)
Error validating model. Failed to create the Genie Node (-1).
Segmentation fault
```

Cause: this is a **split** tower. The lm_head lives in the last shard, so
`prefill_0` emits no `logits`, classifies `DECODER_PREFILL`
(`nsp-graph.cpp:245-251`), and its expected CL is rewritten to the cache-group
max (`nsp-model.cpp:604-605`) — so the `[1,128,128]` mask fails `validateModel`
before any graph-selection logic runs. The reasoning below (bertcache graph
silently skipped, slow but working) is correct **only for an unsplit tower**.

**Do not re-run this bundle as shipped.** The fix is the past-KV prefill
re-export — see `docs/superpowers/plans/2026-08-15-qwen3vl-prefillkv-rebuild.md`.
A config-level escape hatch (`execute-select-graphs`) exists and is
**unverified on hardware**; see `docs/REFERENCE.md` §3.6.

Full mechanism: `docs/NOTES-genie-io.md` § "Split prefill is fatal at load",
`docs/NOTES-genie-pipeline.md` § C1. Device record:
`reports/qwen3vl-4b-e2e-deployment-status-2026-08-14.md`.

## Expected timing — *once a loadable bundle exists*

With the past-KV prefill rebuild, the 273-token prompt processes as
128 + 128 + 17 rows, so **TTFT should be ~3-4 s**, not 30.

The ~30 s figure below applied to the all-decode fallback path (every prompt
token through the AR=1 decode graph). It remains the expectation **if** you run
with the `execute-select-graphs` decode-only config, since that deliberately
drops the prefill graphs. At the 0.6B-measured ~10 tok/s that is ~27 s of
prompt processing before generation starts.

## Triage

| Symptom | First suspect | Check |
|---|---|---|
| `GENIE_STATUS_ERROR_JSON_SCHEMA` at load | a config declares `positional-encoding` **and** backend `pos-id-dim`/`rope-theta` | `Engine.cpp:159-161,677-680`. This exact bug shipped in the Stage 2 bundle. `lint_pipeline_bundle.py` check 3 |
| Load error naming a tensor or quant params | split encodings lineage — identically-named tensors across splits must carry byte-identical quant params | `docs/NOTES-genie-splits.md` |
| **`ShapeError: attention_mask Expected [1,AR,CL] Found [1,AR,AR]` → `Failed to create the Genie Node (-1)` + SIGSEGV** | **an AR==CL bertcache prefill in a SPLIT tower — OBSERVED 2026-08-14.** Shard 0's prefill has no logits → `DECODER_PREFILL` → expected CL rewritten to cache-group max → mask rejected at load | Not fixable by config alone. Rebuild with a past-KV prefill (`[1,AR,CL]`, `CL>AR`), or drop the prefill graphs via `execute-select-graphs` (untested). `REFERENCE.md` §3.6 |
| SIGSEGV on the first token | a graph not listed in `graph_names`, silently compiled with backend defaults (O=0, 4 MB VTCM) | `qnn-context-binary-utility --json_file`, compare against **both** `htp_backend_ext_config_*.json`. Precedent: BUILD_GUIDE §5.4b |
| Output is `!!!…` (token 0) | logits read past the end of a one-row buffer | prefill must emit all-position logits `[1,AR,vocab]` — `docs/NOTES-genie-io.md` |
| `"Unsupported requantization operation"` | an fp16 tensor reached the embedding accumulator | the ViT ctx-bin must be W8A16 with UFIXED_16 IO, not the old fp16 one. `Quantization.cpp:163-192` |
| Image encoder "succeeds" but the caption ignores the image | `setupInputFP16` silently discarded the pixels — you are running the **fp16** ViT | confirm `pixel_values` is `QNN_DATATYPE_UFIXED_POINT_16`; `nsp-image-model.cpp:526-530` |
| Fluent caption, wrong/hallucinated content | image embeddings not spliced, or spliced at the wrong rows | the two `pipeline connect` lines; segment order in the script |
| Caption degrades only when an image is present | MRoPE not engaging — `vision-param` missing, so `setVisionParam` is never called and image rows fall back to plain rope | `ImageEncoder.cpp:46-47,136-138`; `vision-param` must be `32`×`32` in **patch** units |
| Garbled/blank image understanding, no error | the raw blob is not what the graph expects | `sample_image.raw` must be exactly 3,145,728 bytes of uint16, quantized with the graph's own `pixel_values` scale/offset |
| Prompt appears to contain literal `\n` | segment files written with escapes, or via `node set text` instead of `node set textFile` | genie-app never unescapes (`main.cpp:655-658`); segment files must hold real newlines and **no trailing newline** |

## Known limitations of this bundle

**Deepstack is fed zeros.** Genie's stock `ImageEncoder` node publishes exactly
one output and throws on any other IO name (`ImageEncoder.cpp`), so the ViT's
three `deepstack_visual_embed` tensors have no route into the text tower. Those
graph inputs are explicitly memset to zero at load by qualla's
`initializeUnconnectedInputs` (`nsp-model.cpp:1442-1489`), so this is a defined,
intentional degradation — exactly HF-minus-deepstack — not undefined behaviour.

⚠ **One latent hazard**: that memset is sized from the *last* variant's spec
(`nsp-model.cpp:1481`), which for our graph order is `decode_0` at AR=1 — 5120
bytes of a prefill-AR-sized allocation. It is harmless **only because** prompts
over 128 tokens never touch the prefill graph. **A prompt shorter than 128
tokens would select prefill and read uninitialised memory beyond the first 5120
bytes.** If you test with a short prompt and see garbage, that is why. The fix
is per-graph-distinct deepstack tensor names.

**Single image, first turn only.** qualla's rope-delta continuation resets its
base to the raw row index on a second image (`nsp-model.cpp:3827`), and
`visionPos` is batch-local while the rope table is indexed by absolute KV
position — so a second image, or an image in turn ≥2, lands at the wrong
offsets. Details in `docs/NOTES-genie-pipeline.md` §B.

## What to capture on failure

1. Full stdout, including everything before the first token.
2. `adb logcat | grep -iE 'genie|qnn|qualla|htp'` from before the run.
3. `qnn-context-binary-utility --context_binary <bin> --json_file out.json` for
   whichever binary is implicated, and the exact script used.
4. If it loads but the caption is wrong: re-run with a **text-only** prompt
   through `genie-t2t-run` against `genie_dialog.json`. That isolates tower from
   pipeline — but note it proves nothing about MRoPE, which is unreachable from
   the dialog path (`TextGenerator.cpp:302,333` are the only callers of
   `setVisionParam`, and there is no C API for it).
