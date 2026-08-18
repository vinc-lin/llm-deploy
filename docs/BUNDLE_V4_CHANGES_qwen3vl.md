# v4 — what changed and what to run

**Read this first. It supersedes the image-blob instructions in every other
document in this bundle.** v3's OPERATOR_GUIDE / SESSION_RUNBOOK / IMAGE_PROBE
remain included as the historical record of how the crash was diagnosed, but
where they tell you to feed `sample_image.raw` (a UFixed16 blob), that
instruction is exactly the bug v4 fixes — do not follow it.

v4 ships **the same three ctx-bins and the same LUT as v3** — no tower was
rebuilt. Both v3 failures traced to the *interface* files around them:

| # | v3 failure | root cause | v4 fix |
|---|---|---|---|
| 1 | ImageEncoder SIGSEGV at `GenieNode_setData` | Genie interprets the image **file as float32** and reads 4 bytes/element; our tensor-native UFixed16 blob made it read 6,291,456 B from a 3,145,728 B buffer — a 3 MB over-read | ship the image as **float32**: `*_fp32.raw` |
| 2 | Text-only generation produced garbage (T1) | still open — see §3; one confirmed config defect fixed, decisive on-device evidence needed | `bos-token` fix + retest |

## 1. The image crash: mechanism, finally

`QnnNspImageModel::setupInput` (nsp-image-model.cpp:501-524) branches on the
qualla context key `embedding-datatype`. That key **defaults to
`QNN_DATATYPE_FLOAT_32`** (context.cpp:31), and no key in an image-encoder
node config routes to it. So the shipped runtime always takes this branch:

```cpp
float* embeddingSrc = reinterpret_cast<float*>(inputs.data());
quantizeInput(embeddingSrc, name, 0, numElements);   // reads numElements * 4 bytes
```

It treats the blob as float32 and quantizes **on device** with the graph's own
scale/offset. Our v2/v3 blobs were already-quantized UFixed16
(numElements × 2 bytes), so this read exactly **2×** past the end of the
buffer. That mechanism reproduces every observation you reported:

* fault always at the page-rounded **buffer end** (`base+0x300000` for the
  3,145,728 B file; `0x303000` for the 3,158,016 B one) — a 3 MB over-read
  lands in the first guard page after the buffer wherever it ends;
* +4 KB / +12 KB padding changed nothing — the over-read is ~3 MB, not 1 byte;
* mmap vs heap changed nothing (T3);
* `qnn-net-run` ran the **same UFixed16 blob** natively and produced sane
  outputs (T5) — qnn-net-run copies the file into the tensor verbatim, so the
  blob was correct *for the graph*, wrong *for Genie's staging*.

**The fix is the blob, not the code**: `sample_image_fp32.raw` and
`wx_*_fp32.raw` hold the normalized float32 pixel values
(`[1024,1536]` fp32 = 6,291,456 B payload + 4,096 B inert pad = **6,295,552 B**).
Genie now reads exactly the payload and performs on device the same
quantization the host used to do — against the ctx-bin's own encoding, which
the bundle lint verifies.

**The `*_u16.raw` files are NOT pipeline inputs.** They are exact tensor bytes
(3,145,728 B, unpadded) for `qnn-net-run` triage only — the T5 flow. Feeding
one to `node set image` reproduces the v3 SIGSEGV byte for byte. The rename to
`_fp32`/`_u16` exists so a stale v2/v3 `sample_image.raw` lying in a deploy
directory can never be fed by accident: **delete old `sample_image.raw` and
`wx_*.raw` files from the device before running v4.**

## 2. What to run — image (V-tests, in order)

Same environment as v3 (`LD_LIBRARY_PATH=.`, all files flat in one directory).

* **V1 — the headline test.** `./genie-app -s genie_pipeline_qwen3vl.script`
  This is the exact sequence that crashed in v3, with only the blob format
  changed. Expected: no SIGSEGV; a caption is emitted.
  * If it emits a *coherent* caption: end-to-end image+text is working; go
    straight to V3.
  * If it emits a *garbage* caption but does not crash: the image blocker is
    fixed and the remaining fault is the same open text issue as §3 — report
    both, that is still major progress.
  * If it still SIGSEGVs: capture the tombstone; confirm the fed file is
    `sample_image_fp32.raw` at 6,295,552 bytes (`ls -l`), and that no stale
    blob shadowed it.
* **V2 — text retest** (§3): `./genie-t2t-run -c genie_dialog_qwen3vl_4b.json`
  with the same prompt as v3's T1. The configs changed; compare output
  against T1's garbage.
* **V3 — the weather kit.** `./genie-app -s wx_<scene>.script` per image;
  compare captions against TEST_IMAGES.md (semantic agreement, not wording).
* **Fallback** unchanged: if the primary script fails at *load*,
  `genie_pipeline_qwen3vl_decodeonly.script` (slow prefill, same output).

## 3. The text issue: status honest and current

What v4 changes: `bos-token` is now `-1` in all three text configs. The old
value prepended a spurious `<|endoftext|>` before the chat template on every
`genie-t2t-run` query (LUT.cpp:117-119 pushes any bos ≥ 0; Qwen3's template
has no BOS). Also added: explicit `n-embd: 2560` and `pad-token: 151643`,
matching the SDK's own working Qwen3 LUT config.

What v4 does **not** claim: that this explains T1's garbage. A bad BOS
degrades; it rarely garbles. The embedding chain (LUT stride → fp32 feed →
fp32→fp16 staging) was audited end-to-end against the SDK source and is
coherent, so the remaining suspects are the ctx-bin conversion numerics (the
one hop no host gate covers) or a shipped-libGenie divergence from the
example source. V2's result decides the next step; if still garbage, the next
kit will carry a `qnn-net-run` harness for the *text* decode graph — same
method that vindicated the ViT in T5 — to split converter numerics from Genie
runtime behavior.

## 4. Files: changed vs identical

**Byte-identical to v3** (verify with `md5sum`; safe to copy from your
existing v3 deployment instead of re-downloading):

```
b6c04d353e4233d37b07fa8d5bc7f97a  qwen3vl-4b-vit-w8a16_ctx.bin
065056baf6db142aa318ec0cc5662d42  qwen3vl-4b-w8a16_1_of_2.bin
0f1c86e89752b499eec09e9e10a73014  qwen3vl-4b-w8a16_2_of_2.bin
bf66c6ca56547c0f7eeea54c343579aa  embedding_float32_lut.bin
```

Also unchanged: all `.so` runtimes, `genie-app`/`genie-t2t-run`/`qnn-net-run`,
tokenizer.json, prompt segments, HTP extension configs, the `.info.json`
sidecars, `sample_image.png`, `wx_*.jpg`.

**Changed:** `genie_dialog_qwen3vl_4b.json`,
`genie_text_generator_qwen3vl_4b.json`,
`genie_text_generator_qwen3vl_4b_decodeonly.json` (context blocks),
both `genie_pipeline_qwen3vl*.script` (blob name + comments),
`wx_*.script`, `TEST_IMAGES.md`, this file, doc updates.

**New:** `sample_image_fp32.raw` + `sample_image_fp32.json` +
`sample_image_u16.raw`, and per kit image `wx_*_fp32.raw` + `wx_*_fp32.json` +
`wx_*_u16.raw`.

**Gone (delete on device):** `sample_image.raw`, `sample_image.json`,
`wx_*.raw`, `wx_*.json` from v3.

## 5. What to report back

Per test: the command, full stdout/stderr, and for V1/V3 the caption text
verbatim. On any crash: the tombstone (signal line, fault address, `ls -l` of
every `.raw` in the directory). For V2: the generated text plus the four
profile JSONs, same as T1.
