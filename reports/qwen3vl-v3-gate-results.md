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
| ctx-bin readback + Genie load sim | *running* | |
| `parity_e2e_vl.py` all chains | *pending* | |

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

## DDR baseline to beat (v2, un-grouped)

Converter `read_total_bytes`, from `~/llm-local/logs/splitkv.log` (tank,
2026-08-15), in convert order. These are converter estimates for a build that
never ran on device — they bound byte traffic, they do not predict tok/s.

| graph | v2 `read_total_bytes` | v3 | delta |
|---|---|---|---|
| `prefill_0` | 3,637,106,688 | *pending* | |
| `decode_0` | 3,609,839,616 | *pending* | |
| `prefill_1` | 4,413,063,168 | *pending* | |
| `decode_1` | 4,387,743,744 | *pending* | |

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
