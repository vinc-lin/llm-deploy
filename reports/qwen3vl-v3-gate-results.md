# Qwen3-VL-4B v3 — device-free gate results

Rebuild of the VL text tower with grouped-query attention, plus 4 KB-padded
image blobs. Branch `qwen3vl-v3-gqa`. Host: `tank` unless noted.

**Why v3 exists.** v2 shipped a text tower whose attention replicated the 8 KV
heads up to 32 query heads — 36 replication ops per shard at a 4:1 ratio. The
same defect on Qwen3-0.6B was 74.7% of decode DSP cycles and its removal gave
6.54x. The VL chain could not produce the grouped form at all: `--grouped-gqa`
existed in `quantize_aimet.py` but `ExportQwen3.from_hf_vl_text` had no such
parameter, so it silently defaulted to `False`. v2 also shipped unpadded image
blobs, which is what blocked the device on 2026-08-15 with
`SIGSEGV (SEGV_ACCERR)` in `GenieNode_setData`.

---

## Status

| Gate | Result | Where |
|---|---|---|
| `test_vl_grouped_gqa.py` (local) | PASS — `Expand: replicating=4 grouped=0` | commit `900db5a` |
| `test_vl_grouped_gqa.py` (tank) | PASS — `max\|dlogits\|=1.490e-07` | 2026-08-16 |
| AIMET `--eval` last-token argmax | **4/4** | `logs/vl_gqa_quant.log` |
| Exported graph topology | **`Expand=0`** in prefill and decode | see below |
| Bundle lint rejects unpadded blobs | 7 failures → 0 after padding | commit `be8cddd` |
| GQA gate wired into ctx-bin build | present, fires both ways | commit `0db9643` |
| Past-KV prefill export | PASS — 7:36 wall, 66.7 GB peak RSS | `logs/vl_gqa_stage33.log` |
| Deepstack `_p` rename | PASS — 3 renamed, checker clean | same |
| Weight unification ×2 | PASS — `0 left untouched`, samples `MATCH=True` | same |
| Split at 18, no symlinks | PASS — 4 real graphs | `logs/vl_gqa_stage35.log` |
| **4 DLCs `lint_gqa_ops.py --layers 18`** | **PASS — 0 failing, batch dim 8** | same |
| ctx-bin readback (shapes, HTP bind, shared weights) | PASS — 1.70 / 2.42 GB pooled | `logs/vl_gqa_stage35.log` |
| Genie load simulation | PASS — replay clean | same |
| fp32 grouped exports | PASS — `Expand=0` ×3, wrapper-vs-HF 2.587e-05 | `logs/vl_gqa_stage4.log` |
| **`parity_e2e_vl.py` all 6 chains** | **PASS — 4/4 gated chains 20/20** | same |
| Kit captions (6 images, tierB) | *running* | |
| Bundle v3 lint | *pending* | |

---

## Phase 3.2 — quantization (grouped)

Invocation, from `~/llm-local/logs/vl_gqa_quant.log` (tank, 2026-08-16), via
`bash scripts/build/vl_text_build.sh qwen3vl-4b-w8a16-gqa 128 2048 --grouped-gqa`:

```
quantize_aimet.py --model .../Qwen3-VL-4B-Instruct --cl-prefill 128
  --out .../qwen3vl-4b-w8a16-gqa-prefill --eval --vl-text --n-deepstack 3
  --vl-calib .../qwen3vl-4b-calib-ar128.npz --device cpu --grouped-gqa
```

Wall clock ~53 min (18:47 → 19:40 UTC), exit 0. Calibration reused the existing
`qwen3vl-4b-calib-ar128.npz`.

**Quality gate — 4/4, the same bar v2 met:**

```
[eval] "EVAL img100 'What is happening in thi'": quant argmax == fp32 argmax; max|dlogits|=2.265
[eval] "EVAL img101 '这张图片里有几个人?'":      quant argmax == fp32 argmax; max|dlogits|=1.555
[eval] "EVAL text 'The capital of France is'":  quant argmax == fp32 argmax; max|dlogits|=0.943
[eval] "EVAL text '1+2+3+...+100 ='":           quant argmax == fp32 argmax; max|dlogits|=0.943
[eval] last-token argmax agreement: 4/4
```

Grouped attention did not cost quality. Encodings are a single lineage
(`gqa-prefill/model_filtered_renamed.encodings` serves both graphs), which is
the hard requirement — mixed encodings are a fatal Genie load error.

**Topology, measured on the AIMET exports rather than assumed:**

| graph | `Expand` | `MatMul` |
|---|---|---|
| `gqa-prefill/model_renamed.onnx` | **0** | 325 |
| `gqa-decode/model_renamed.onnx` | **0** | 325 |

Zero `Expand` is the whole point: the converter lowers each one into a
broadcast MULTIPLY whose output is re-read by the attention MatMul every step.

---

## Phase 3.6 — the topology gate, on the converted DLCs

This is the check v2 never ran against this tower. It now runs automatically
inside `vl_text_ctxbin_split.sh` (commit `0db9643`), on the DLCs just
converted, before any ctx-bin exists. From `~/llm-local/logs/vl_gqa_stage35.log`
(tank, 2026-08-16):

```
== GQA topology gate (grouped attention, per-shard) ==
PASS  prefill_0.dlc   replication ops: 0 (expected 0)   MatMuls: 36, batch dim ['8']
PASS  decode_0.dlc    replication ops: 0 (expected 0)   MatMuls: 36, batch dim ['8']
PASS  prefill_1.dlc   replication ops: 0 (expected 0)   MatMuls: 36, batch dim ['8']
PASS  decode_1.dlc    replication ops: 0 (expected 0)   MatMuls: 36, batch dim ['8']
4 DLC(s) checked, 0 failing
```

The MatMul shapes show the change directly. Batch is now the 8 KV heads, and
the query heads have folded into the row dimension as `rep * S`:

| graph | v2 (replicating) | v3 (grouped) |
|---|---|---|
| `decode_0` | `1x32x1x2176` | `1x8x4x2176` |
| `prefill_0` | — | `1x8x512x2176` (512 = 4 × 128) |

v2 additionally carried 36 `Expand` ops per shard, each lowered by the
converter into a broadcast MULTIPLY whose output the MatMul then re-read. Those
are gone: **0**.

## ctx-bin readback gates

From the same `vl_gqa_stage35.log`. Every structural gate the script enforces
passed, and the weight-sharing figures land exactly on v2's:

```
1_of_2: ['decode_0', 'prefill_0']   in=43 out=37 each   sharedWeights=1.70 GB
2_of_2: ['decode_1', 'prefill_1']   in=40 out=37 each   sharedWeights=2.42 GB
BOTH BINARIES VERIFIED
```

Floors are `{0: 1.4, 1: 2.0}` GB, so 1.70 / 2.42 clear them. Sizes 1.8 G and
2.5 G, unchanged from v2 — expected, since grouping changes op topology, not
the weight set.

Genie load simulation (replays `validateModel`, including the
`DECODER_PREFILL` CL rewrite that shipped the 2026-08-14 failure):

```
  decode_0:  shard 0, AR=1,   CL=2176, DECODER_PREFILL
  prefill_0: shard 0, AR=128, CL=2176, DECODER_PREFILL
  decode_1:  shard 1, AR=1,   CL=2176, DEFAULT
  prefill_1: shard 1, AR=128, CL=2176, DEFAULT
  cache group 'past_': ctx=2176, concat, variants {(1,2176), (128,2176)}
PASS: validateModel replay clean -- these ctx-bins would load
```

## Phase 4 — numerical parity, all six chains

`~/llm-local/logs/vl_gqa_stage4.log` (tank, 2026-08-16), wall 4638 s. Run with
no `--chains` filter, because a subset silently skips the mutation checks.

fp32 exports first: `Expand=0` in `prefill`, `decode` and `prefillkv`, and
`--parity-check` gave `wrapper-vs-HF max|dlogits| = 2.587e-05` against a
5e-3 tolerance. ViT deltas came out identical to v2 (`image_features`
1.551e-04, deepstack 4.625e-05 / 2.002e-04 / 3.700e-04), confirming the vision
tower is untouched by this rebuild.

```
== verdict ==
  HF reference     : 'The image displays a simple composition of a red circle and a blue square on a white background.'
  chain0-alldecode : 20/20 (100%)
  chain0b-prefillkv: 20/20 (100%)      <- THE DEVICE PATH
  chain1-hf-vit    : 20/20 (100%)
  chain2-onnx-vit  : 20/20 (100%)      <- bar is >=75%; got 100%
  tierA-zero-deep  : 0/20              <- expected, not gated
  tierB-prefillkv-zero-deep: 0/20      <- expected, not gated
PASS: full-path device-free parity (6 chains, n=273 prompt rows, 20 generated tokens)
```

Grouped attention is numerically exact end-to-end: image → ViT → splice → text
tower reproduces `hf.generate` token-for-token on every gated chain, including
`chain0b-prefillkv`, which replays qualla's real chunk plan (three AR=128
prefill calls, `n_process` 128/128/17, the last padded).

**The device-faithful caption for `sample_image`** (tierB — past-KV prefill
with zeroed deepstack, the deployed combination):

> `A red circle and a blue square are positioned side by side on a white background.`

Word-for-word identical to v2's, despite a completely new quantization
lineage — a useful stability signal, and the string that goes verbatim into
`DEVICE_TEST.md` as the expected result of the single smoke test.

## DDR bandwidth: v2 vs v3

Converter `read_total_bytes`, v2 from `~/llm-local/logs/splitkv.log`
(2026-08-15), v3 from `~/llm-local/logs/vl_gqa_stage35.log` (2026-08-16), both
on tank. Summaries appear in `--dlc_path` order within each ctx-bin, i.e.
prefill then decode.

| graph | v2 (replicating) | v3 (grouped) | delta |
|---|---|---|---|
| `prefill_0` | 3,637,106,688 | 2,820,626,432 | **−22.4%** |
| `decode_0` | 3,609,839,616 | 2,237,399,040 | **−38.0%** |
| `prefill_1` | 4,413,063,168 | 3,596,582,912 | **−18.5%** |
| `decode_1` | 4,387,743,744 | 3,015,303,168 | **−31.3%** |

Decode falls hardest, which is the expected shape of this fix: the replication
ops produced a large intermediate that had to be written and re-read every
step, and decode does the least useful work per step to amortise it.

⚠ **These are converter estimates for a build that has never run on device.**
They bound byte traffic; they do not predict tok/s. Whether VL-4B decode is
byte-bound or compute-bound on this silicon is unmeasured — at ~4.3 GB of
weights per token there is a byte floor near ~10 tok/s that no topology change
crosses. Treat these as evidence the defect is gone, not as a speedup claim.

---

## Notes for whoever reads this next

- `lint_gqa_ops.py` **must** be run with `--layers 18` on the split 4B tower.
  Its default is 28 (the 0.6B tower) and its pass criterion includes
  `len(matmuls) == 2 * layers`, so the default reports FAIL on a correct
  shard. Confirmed against the old v2 DLC: with `--layers 18` it prints
  `attention MatMuls: 36 (expected 36)` and fails only on topology
  (36 replication ops, batch dim 32 vs 8).
- Build scripts are **not executable on tank** — the repo lives on a
  Windows-backed mount where the exec bit never enters git. Invoke as
  `bash scripts/build/foo.sh`.
- The shipped v2 tree for comparison is
  `hf-staging-v2/qwen3vl_4b_e2e_pipeline_v2/` (58 files), **not**
  `bundles/qwen3vl_4b_e2e_pipeline/` (31 files, older, no kit, no fallback).
