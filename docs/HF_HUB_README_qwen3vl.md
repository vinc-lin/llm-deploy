---
license: apache-2.0
base_model: Qwen/Qwen3-VL-4B-Instruct
tags:
  - qualcomm
  - qnn
  - genie
  - sa8797p
  - multimodal
---

# Qwen3-VL-4B-Instruct for SA8797P — QNN/Genie bundles

Device bundles for the Qualcomm SA8797P (Hexagon v81 HTP, Android GVM), built
against QAIRT 2.48.40.260702 / libGenie 1.19.

**Start with [`qwen3vl_4b_e2e_pipeline_v2/`](./qwen3vl_4b_e2e_pipeline_v2).**
Image + text in, description out, in one flat `genie-app` bundle, plus a
six-image weather/road test kit with per-image expected captions.

| Folder | What it is | Precision | Size |
|---|---|---|---|
| [`qwen3vl_4b_e2e_pipeline_v2/`](./qwen3vl_4b_e2e_pipeline_v2) | **Full pipeline — image+text → text.** Past-KV prefill; loads | W8A16 | ~6.2 GB |
| [`qwen3vl_4b_e2e_pipeline/`](./qwen3vl_4b_e2e_pipeline) | ⛔ **v1 — does NOT load.** Kept only so the 2026-08-14 failure report stays reproducible | W8A16 | ~6.2 GB |
| [`qwen3vl_4b_text_w8a16/`](./qwen3vl_4b_text_w8a16) | Text tower alone, 2-split ctx-bin | W8A16 | 6.16 GB |
| [`qwen3vl_4b_vit_fp16/`](./qwen3vl_4b_vit_fp16) | Vision tower alone | FP16 | 0.97 GB |

## Why there is a v2

The v1 e2e bundle **never loaded on device** (2026-08-14). Node creation died
with two `ShapeError`s on `attention_mask` (`Expected [1,128,2176]`, `Found
[1,128,128]`) and a SIGSEGV, before a single token.

Cause: in a *split* tower, shard 0's prefill graph has no `logits`, so libGenie
classifies it `DECODER_PREFILL` and rewrites its expected context length to the
cache-group maximum. An AR==CL "bertcache" prefill mask can never satisfy that.

v2 rebuilds the text tower with a **past-KV prefill** — `attention_mask
[1,128,2176]`, `past_key_N_in [1,8,128,2048]` — which is byte-for-byte what the
validator demanded, and is the same recipe as the device-proven 0.6B `ladekv`
build. A **load simulation** now replays libGenie's own `validateModel` against
the shipped `info.json`s as a build gate; it was only accepted once it
reproduced the v1 failure, on the v1 binaries, with the device's exact message.

A 273-token prompt now runs as three AR=128 prefill calls instead of 273
sequential AR=1 decode steps.

## Nothing here has run on an SA8797P

All of this was built and validated **without device access**. HTP context
binaries cannot execute on x86, and this SDK has no x86 path for W8A16 either
(`libQnnCpu` ships no 16-bit fixed-point kernels). So no gate in this repository
has exercised the Hexagon backend, HTP scheduling, or on-device memory. **No
tok/s or TTFT figure is quoted anywhere here** — none has been measured for this
model on this device, and extrapolating one would be inventing it.

What *is* established is numerical: the full host-side path — image → ViT →
splice → text tower, under the runtime's exact feed pattern including the real
chunked past-KV prefill — reproduces HuggingFace `generate` **token-for-token**,
across four independent chains. See the v2 folder's README for the full gate
table.

Treat the first device run as the real test. The v2 bundle ships a
`DEVICE_TEST.md` with a triage table and a one-file-swap fallback.

## Third-party components

Every bundle embeds Qualcomm QAIRT 2.48.40.260702 runtime binaries (7 aarch64
`.so`, `genie-app`, `genie-t2t-run`) which this repository's licence tag does
**not** cover. They are redistributed under Qualcomm's SDK licence terms.

The v2 test-kit photographs are freely licensed works from Wikimedia Commons and
COCO val2017; per-image licence and author are recorded in the bundle's
`TEST_IMAGES.md` and in each `wx_*.json` sidecar.
