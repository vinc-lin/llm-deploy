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

Device bundles for the Qualcomm **SA8797P** (Hexagon v81 HTP, Android GVM),
built against QAIRT 2.48.40.260702 / libGenie 1.19.

## → Start here: [`qwen3vl_4b_e2e_pipeline_v2/`](./qwen3vl_4b_e2e_pipeline_v2)

That folder is the deliverable and it is self-contained. Its `README.md` is the
full deployment and test guide: file manifest, how to deploy to the board, how
to run and verify, the test plan, the metrics to collect (with exact
definitions), and what the final report must contain.

```bash
adb push qwen3vl_4b_e2e_pipeline_v2 /data/local/tmp/qwen3vl
adb shell 'cd /data/local/tmp/qwen3vl && chmod +x genie-app && \
           LD_LIBRARY_PATH=. ./genie-app -s genie_pipeline_qwen3vl.script'
```

## Repository contents

| Folder | What it is | Status |
|---|---|---|
| [`qwen3vl_4b_e2e_pipeline_v2/`](./qwen3vl_4b_e2e_pipeline_v2) | **Full pipeline — image+text → text.** W8A16, past-KV prefill, 6-image weather/road test kit, decode-only fallback | **Current.** Load gate passes |
| [`qwen3vl_4b_e2e_pipeline/`](./qwen3vl_4b_e2e_pipeline) | v1 — **did not load.** Binaries removed; metadata kept so the failure stays reproducible | ⛔ Superseded stub |

The standalone single-tower folders (`qwen3vl_4b_text_w8a16/`,
`qwen3vl_4b_vit_fp16/`) have been **removed**:

* `qwen3vl_4b_text_w8a16/` shipped ctx-bins **byte-identical** to v1's and
  therefore carried the same fatal split-prefill shape — verified with the load
  simulator against its own dialog config. It could not have loaded, and its
  README claimed otherwise.
* `qwen3vl_4b_vit_fp16/` is an FP16 vision tower, which a stock Genie pipeline
  **cannot drive at all**: `setupInputFP16` is an empty stub that discards the
  pixel blob and returns success, and the requantize table has no `Float16`
  entries. It would have produced captions that silently ignore the image. The
  v2 bundle ships a W8A16 vision tower with `UFIXED_POINT_16` I/O instead.

## Nothing here has run on an SA8797P

All of this was built and validated **without device access**. HTP context
binaries cannot execute on x86, and this SDK has no x86 path for W8A16 either
(`libQnnCpu` ships no 16-bit fixed-point kernels). No gate in this repository
has exercised the Hexagon backend, HTP scheduling, or on-device memory.

**No tok/s or TTFT figure is quoted anywhere here.** None has been measured for
this model on this device, and extrapolating one from a different model would be
inventing it. The first device run produces the first data point.

What *is* established is numerical: the full host-side path — image → ViT →
splice → text tower, under the runtime's exact feed pattern including the real
chunked past-KV prefill — reproduces HuggingFace `generate` **token-for-token**
across four independent chains. The v2 folder's README has the complete gate
table.

## Third-party components

Every bundle embeds Qualcomm QAIRT 2.48.40.260702 runtime binaries (7 aarch64
`.so`, `genie-app`, `genie-t2t-run`) which this repository's licence tag does
**not** cover. They are redistributed under Qualcomm's SDK licence terms.

The v2 test-kit photographs are freely licensed works from Wikimedia Commons and
COCO val2017; per-image licence and author are recorded in the bundle's
`TEST_IMAGES.md` and in each `wx_*.json` sidecar.
