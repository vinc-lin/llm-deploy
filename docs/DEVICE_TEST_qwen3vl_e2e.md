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

## Expected timing

**First token will take roughly 30 seconds, and that is expected.**

The prompt is 273 tokens and every one of them goes through the **AR=1 decode
graph**, one at a time. The AR=128 prefill graph is never selected: qualla
derives a graph's context size from the attention mask's trailing dim
(`nsp-graph.cpp:146-155`), ours is `[1,128,128]`, so `ctx_size == AR` — the
bertcache shape — and the strategy loop skips that bucket for any prompt longer
than 128 tokens (`kvmanager.cpp:411-416`). At the 0.6B-measured ~10 tok/s that
is ~27 s of prompt processing before generation starts.

This is a throughput defect, not a correctness one, and the fix is a text-tower
re-export with a past-KV prefill graph. Do not chase it during bring-up.

## Triage

| Symptom | First suspect | Check |
|---|---|---|
| `GENIE_STATUS_ERROR_JSON_SCHEMA` at load | a config declares `positional-encoding` **and** backend `pos-id-dim`/`rope-theta` | `Engine.cpp:159-161,677-680`. This exact bug shipped in the Stage 2 bundle. `lint_pipeline_bundle.py` check 3 |
| Load error naming a tensor or quant params | split encodings lineage — identically-named tensors across splits must carry byte-identical quant params | `docs/NOTES-genie-splits.md` |
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
