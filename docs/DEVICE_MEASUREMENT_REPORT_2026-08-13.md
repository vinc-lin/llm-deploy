# SA8797P Decode Step Investigation — 5-Test Measurement Report

**Date:** 2026-08-13
**Device:** SA8797P Hexagon v81 HTP (serial `REDACTED`), unsigned PD, 1 core, VTCM 16 MB
**Toolchain:** QAIRT 2.48.40.260702, QNN API v2.37.0, libGenie 1.19.0
**Bundles:** HF `vinccniv/sa8797p-qwen3-w8a16-bundles` (Qwen3-0.6B, W8A16 per-channel INT8 weights)
**Artifacts:** `docs/test_artifacts/measurement_2026-08-13/` — **not received; see §0.2**

---

## 0. Editorial annotations (added 2026-08-14, not part of the original report)

The report below is preserved as received, except that the device serial and the
jump-host access string are redacted (this repo's history has been scrubbed twice
for exactly that class of leak). Two corrections and one gap are recorded here
rather than edited into the body, so the original stays auditable.

### 0.1 Test 2's op attribution is wrong — the ops are GQA KV replication, not mask broadcast

The report attributes 261.8M cycles/step (74.7%) to broadcasting the causal
attention-mask scalar. **This is refuted by direct inspection of the shipped
`decode.dlc`** (`qairt-dlc-info`, 2026-08-14 — device-free, no new measurement
required). In the actual graph:

- The attention mask is **never expanded**. It enters as a `[1,1,1152]` graph
  input, gets one `Unsqueeze` to `[1,1,1,1152]` (op 54 in layer 0), and
  broadcasts *implicitly* inside the `Add` (op 55). There is no mask `Expand`.
- The 56 expensive ops (2 per layer × 28 layers) are **`repeat_kv` — GQA KV head
  replication**, materialising 8 KV heads into 16 Q heads:

  ```
  Expand    [1,8,1,128,1152] -> [1,8,2,128,1152] -> Reshape -> [1,16,128,1152]   (K, stored transposed)
  Expand_1  [1,8,1,1152,128] -> [1,8,2,1152,128] -> Reshape -> [1,16,1152,128]   (V)
  ```

- Their QNN type is **`Eltwise_Binary` with `operation: 13` = MULTIPLY**, against
  a `[1,1,2,1,1]` STATIC coefficient — i.e. the converter lowered ONNX `Expand`
  into a broadcast multiply-by-ones, not a fill or a copy.

**The report's "260 cycles/byte, extremely inefficient" conclusion is an
artifact of the wrong output shape.** It assumed the output was `[1,8,1,1152]`
= 18,432 B. The real output is 4,718,592 B — exactly 256× larger — which gives
**1.03 cycles/byte**: entirely ordinary throughput for a broadcast FP16
multiply. The cycle counts themselves are trustworthy; only the label was wrong.

Consequences for the report's recommendations:

- **Rec #1** ("fuse the mask into the Q@K kernel, Flash-Attention style") targets
  an op that does not exist. The correct fix is to stop materialising the
  replicated KV at all — feed the un-replicated `[1,8,…]` cache straight into a
  grouped/batched MatMul. Same ~75% cycle target, different mechanism.
- **Rec #1's context-length sub-point still holds**, and for a second reason:
  replication cost scales with CL, and so does the KV cache read.
- **"verify32 amortizes the broadcast" (Test 2, Batched Decode) is backwards.**
  Replication cost is AR-independent, which *strengthens* the case for
  speculative decoding rather than explaining it away.
- **The fusion and `lm_head` projections are renormalised.** Against the 88.5M
  cycles that remain once replication is removed, attention GEMV is 44.5%,
  weight GEMMs 35.3% and `lm_head` 6.9% — so QKV/Gate-Up fusion is worth ~10% of
  real compute, not "~2.5% marginal", and `lm_head` is not negligible.

### 0.4 Test 1's 11.72 tok/s is a phase blend, and it invalidates Tests 1, 3 and §6.3

*Added 2026-08-16. This is the largest correction to this report, and it was
found only after the 2026-08-15 measurements gave an independent control.*

`qwen3_06b_w8a16_local` is a **bertcache** bundle (prefill AR=128, CL=128). In
that topology Genie keeps generating *through the prefill graph*, re-processing
the whole 128-wide window once per token, until the KV cache passes 128
(`kvmanager.cpp:421-429`). Only then does the AR-1 decode graph take over. So
Test 1 Arm A's 128 generated tokens did not all run at one rate:

```
prompt 56 tokens, 128 generated, 10,837 ms total
  bertcache phase: tokens 1..72   (KV 56 -> 128) @ 40.1 ms  = 2,887 ms   <-- = this run's own TTFT
  AR-1 phase     : tokens 73..128 (56 tokens)               = 7,950 ms
                                             7,950 / 56     =  142.0 ms/step
```

**142.0 ms against 146.3 ms measured on 2026-08-15** for the pre-fix `ladekv`
bin in basic mode — a 3% agreement between two numbers from different days,
different bundles and different methods. The blend model closes, and the honest
pre-fix AR-1 rate is **6.84 tok/s**, not 11.72.

Consequences for this report:

- **Test 1's "LADE is −22%" is an artifact.** It compares LADE on `ladekv`
  against *blended* basic on `local`. Like-for-like on one bin and this same
  prompt, **LADE was 9.18 vs 6.84 = +34%**. (LADE does lose post-fix — 31.342
  vs 44.707 on 2026-08-15 — but for the different reason in `REFERENCE.md` §6.8,
  and that could not be known from this report.)
- **Test 3's "+51% faster than the device team's 7.79" is an artifact.**
  Like-for-like our pre-fix AR-1 is 6.84, i.e. ~12% *slower*. `REFERENCE.md`
  §8.8 is reopened rather than re-inverted, because their build's topology and
  provenance were never audited either.
- **§6.3's "~75% build gap between `local` and `ladekv`" does not exist.** The
  two decode graphs are structurally identical and share weights within 4 MB.
  Recommendation 2 ("investigate the build gap") should not be actioned.
- **Test 4's qh baseline is the blended number**, so the −43% is overstated by
  the blend on top of the confound §0.1 already flags.
- Test 2 (the op-level cycle profile) is **unaffected** — it ran under
  `qnn-net-run` on a single-graph decode-only bin, with no graph selection
  involved. It remains the most valuable measurement in this report.

Gate added so this cannot recur: `scripts/validate/lint_bundle_topology.py`
classifies any ctx-bin as pure or blended from its own graph shapes, and derives
the same 72/56 split independently. Full analysis in
`docs/MAX_TPS_QWEN3_0.6B_V4.md` §1 and `REFERENCE.md` §6.9 / corrections #22–25.

### 0.2 The raw artifacts were never received

`docs/test_artifacts/measurement_2026-08-13/` (the artifact index at the end of
this report) does not exist in this repo or anywhere on the build machine. Only
this narrative report was transferred. That is why §0.1's correction was derived
from the DLC rather than from `qnn-profile-viewer` output. Re-requesting the raw
`test2_decode_profile/qnn_profile_r*_viewer.txt` files is folded into the
device-team exchange (P0.4).

### 0.3 §6.1's SIGSEGV has been fixed build-side

`type: "lade"` + `max-num-tokens` shipped in `genie_dialog_demo.json` in three
bundles (fuseqkvgu, socmodel72, hvx8). Fixed 2026-08-14; the pair is now refused
by `scripts/validate/lint_bundle_dialogs.py`, which runs in `bundle.sh` and in
the documented hand-add recipe.

### 0.5 Outcome, and two things later readers got wrong (added 2026-08-16)

**This report's headline finding was correct and acting on it worked.** Removing
the 56 replication ops took decode from 6.836 → **44.707 tok/s**, +6.5%×, measured
2026-08-15 (`DEVICE_MEASUREMENT_REPORT_2026-08-15.md`). Test 2 is the most
valuable measurement in this project's history.

Two cautions for anyone quoting it:

1. **The step time is internally inconsistent.** §Test 2's preamble motivates the
   investigation against "approximately 155 ms per decode step", while the Note
   on Wall-Clock Time cites Genie at "~85 ms" and Test 1 measures 11.72 tok/s =
   85.3 ms — and per §0.4 that 85.3 ms is itself a *blend*, not a step time, so
   neither figure is a clean AR-1 rate (the honest one is 146.3 ms). 155 ms corresponds to the *superseded* 6.45 tok/s baseline. So the
   "roughly 100 ms of previously unexplained decode time" in the Executive
   Summary is computed against a step time this same report invalidates —
   against 85.3 ms the residue is ~65 ms. This does not affect the cycle
   percentages, which are what the recommendations rest on.
2. **Test 5 was read backwards downstream.** Its own conclusion is right and is
   stated correctly in the Executive Summary — `hvx_threads` is a **build-time**
   knob, and changing the runtime config did nothing. Later planning documents
   nonetheless cited "hvx 4 vs 8: −0.1%, no effect" as though the build-time A/B
   had been run. It has not been, on any lineage. Every shipping ctx-bin is
   compiled `numHvxThreads=4` against 8 available HVX units — see `REFERENCE.md`
   §8.9 and correction #29.

Test 4's −43% W8-head result carries its own confound section in the body
(different ctx-bins, different lineages) and should not be quoted as a clean
measurement of the head.

---

## Executive Summary

| # | Test | Headline Result |
| - | ---- | --------------- |
| 1 | Basic (AR-1) vs LADE | Basic: **11.72 tok/s**; LADE: **9.18 tok/s** (LADE −22% on this prompt) |
| 2 | **Op-level decode profile** ⭐ | **74.7% of cycles = attention-mask `Expand` broadcast** *(mis-attributed — see §0.1)*; weight GEMMs account for only 8.9% |
| 3 | Build gap (our arm only) | Our `local` bundle: **11.7 tok/s**, faster than the device team's reported 7.79 tok/s |
| 4 | W8 `lm_head` (qh) in basic mode | **6.70 tok/s vs 11.72 tok/s** (−43%; confounded by build differences, not a clean comparison) |
| 5 | Runtime `hvx_threads: 8` | No effect: **9.17 vs 9.18 tok/s** — the knob is build-time, not runtime |

### Key Finding

Approximately **three quarters of the DSP cycles in every decode step are spent
broadcasting the causal attention-mask value (`-10000.0`) to the full
`[1, 8, 1, 1152]` tensor through `Expand` operations**.
*(Corrected in §0.1: these are GQA KV-head replication ops of shape
`[1,8,2,128,1152]` / `[1,8,2,1152,128]`, not mask broadcasts.)*

Weight matrix multiplications — the part that had been the main optimization
focus — account for less than **9%** of total decode cycles.

This means the roughly **100 ms of previously unexplained decode time is not
primarily caused by weight-memory latency or per-op dispatch overhead.**

This is now the single largest optimization target.

---

# Methodology

## Device and Access Path

```text
workstation → ssh REDACTED → adb -s REDACTED → SA8797P (Android GVM, aarch64)
```

## Protocol

- **Warm state:** the first run after any reconnect is discarded because cold
  initialization takes approximately 1.8–2.0 s, versus approximately 800 ms when warm.
- **3 repetitions per arm**, using the same prompt and greedy sampling:
  `temp=0`, `top-k=1`, `top-p=1.0`, `seed=42`
- **Genie profiling:** `--profile profile.json` (aggregate KPI JSON from Genie's
  built-in profiler).
- **QNN profiling:** `qnn-net-run --profiling_level detailed`, post-processed with
  `qnn-profile-viewer`.
- Genie runs use `--prompt_file <file>` to avoid shell-quoting issues with
  multiline chat templates.
- Flat bundle layout: `LD_LIBRARY_PATH=.`
- For `qnn-net-run`, also set `ADSP_LIBRARY_PATH=.:/vendor/lib/rfsa/adsp`

## Prompt

Qwen3 chat template with an empty `<think>` block so that thinking mode never triggers:

```text
<|im_start|>system
You are a helpful assistant who gives detailed, structured answers.<|im_end|>
<|im_start|>user
Explain how a Hexagon DSP works, its key features, and benefits for AI workloads. Use numbered sections with multiple paragraphs.<|im_end|>
<|im_start|>assistant
<think>

</think>
```

The prompt contains **56 tokens**.

For Basic / qh Basic: `max-num-tokens: 128` (generation capped for rate comparisons).
For LADE: generation runs until the context is filled; greedy repetition; ~519 tokens generated.

## Bundles Used

| Bundle | Graphs | Ctx-bin Size | Purpose |
| ------ | -----: | -----------: | ------- |
| `qwen3_06b_w8a16_local` | prefill + decode (2) | 1087 MB | Test 1 Arm A (Basic), Test 3 (our arm) |
| `qwen3_06b_w8a16_ladekv` | prefill + decode + verify32 (3) | 1106 MB | Test 1 Arm B (LADE), Test 5 (hvx8) |
| `qwen3_06b_w8a16qh_ladekv` | prefill + decode + verify32, W8 `lm_head` (3) | 1094 MB | Test 4 (qh basic) |
| `qwen3-0.6b-w8a16-decodeonly_ctx.bin` | decode only (1) | 1075 MB | Test 2 (op-level profile) |

All were built from the same DLC base with `O=3`, `vtcm_mb=16`, unsigned PD,
weight sharing enabled.

---

# Test 1 — Basic vs LADE Genie Profile

## Configurations

**Arm A — Basic:** `type: "basic"`, `max-num-tokens: 128`, 2-graph ctx-bin,
`perf_profile: llm_decode_burst`, `enable-graph-switching: true`,
`cpu-mask: 0xe0`, `n-threads: 3`.

**Arm B — LADE:** `type: "lade"`, `window: 8`, `ngram: 3`, `gcap: 8`,
`update-mode: ALWAYS_FWD_ONE`, 3-graph ctx-bin, no `max-num-tokens` (using it
with LADE causes a SIGSEGV; see §6.1), same backend configuration as Arm A.

## Results

| Metric | Basic Rep 1 | Basic Rep 2 | Basic Rep 3 | Basic Mean | LADE Rep 1 | LADE Rep 2 | LADE Rep 3 | LADE Mean |
| ------ | ----------: | ----------: | ----------: | ---------: | ---------: | ---------: | ---------: | --------: |
| Init time (ms) | 813 | 843 | 862 | **839 ± 25** | 848 | 883 | 883 | **872 ± 21** |
| TTFT (ms) | 40.1 | 40.1 | 40.2 | **40.1 ± 0.1** | 185.7 | 186.6 | 186.8 | **186.3 ± 0.6** |
| Prompt tokens | 56 | 56 | 56 | 56 | 56 | 56 | 56 | 56 |
| Prompt rate (t/s) | 1397 | 1399 | 1395 | **1397 ± 2** | 302 | 300 | 300 | **301 ± 1** |
| Gen tokens | 128 | 128 | 128 | 128 | 519 | 519 | 519 | 519 |
| **Gen rate (tok/s)** | 11.72 | 11.73 | 11.71 | **11.72 ± 0.01** | 9.18 | 9.18 | 9.18 | **9.18 ± 0.00** |
| Gen time (ms) | 10835 | 10827 | 10849 | 10837 ± 11 | 56418 | 56435 | 56451 | 56435 ± 17 |

## Observations

1. **LADE is slower than Basic on this prompt** (9.18 vs 11.72 tok/s, −22%).
   The likely reason is the low n-gram match rate of this technical prompt:
   speculative decoding accepts too few tokens per verify call to amortize the
   cost of the verify step.
2. **LADE TTFT is 4.6× higher** (186 vs 40 ms). LADE must execute
   prefill → verify32 → graph-switch before returning the first token, while
   Basic only requires prefill → decode.
   *(§0.1 note: the dominant cause is the prefill graph's CL — `local`'s prefill
   is a CL=128 bertcache graph, `ladekv`'s attends over 1152 positions.)*
3. **LADE prompt-processing rate is 78% lower** (301 vs 1397 tok/s).
4. **Basic at 11.7 tok/s is significantly faster than the 6.3–6.5 tok/s
   previously reported** for `qwen3_06b_w8a16_ladekv` in Basic mode. See §6.3.
5. **Reproducibility is very strong** — generation rate identical to two decimal
   places across all three repetitions for both arms.

## Interpretation

LADE performance depends heavily on **acceptance rate**. For this technical
explanatory prompt many sentences are unique, n-gram repetition is low, a
`verify32` call costs ~180 ms, and graph-switching adds overhead. The number of
accepted tokens per verify call is therefore insufficient to amortize these
costs.

The previously reported **10.8 tok/s** LADE result used a different prompt and
is not directly comparable with Basic AR-1 on the same build.

---

# Test 2 — Op-Level Profile of One Decode Step ⭐

This is the **core test**. The objective was to determine where the
approximately **155 ms per decode step** is actually spent.

Previous microbenchmarks measured approximately **63 GB/s** weight-streaming
bandwidth during compute and approximately **961 MB of weights** read per decode
step, suggesting weight MatMuls should require only ~**15–20 ms** — leaving more
than **100 ms** unexplained.

## Setup

**Decode-only ctx-bin:** single `decode` graph, AR=1, CL=1152, 60 inputs,
57 outputs, `spill_bytes = 0` (zero VTCM spill). Same DLC and build settings as
`ladekv`: `O=3`, `vtcm_mb=16`, `hvx_threads=4`, weight sharing enabled.

**Inputs:** zero-filled KV past tensors; realistic `input_ids`; realistic
`attention_mask` (Qualla additive FP16); true cos/sin RoPE values at the
corresponding position; `rope_theta = 1e6`.

**Tool:** `qnn-net-run --profiling_level detailed` → `qnn-profile-viewer`.

**Backend configuration:** minimal — `perf_profile: llm_decode_burst`,
`rpc_polling_time: 9999`. Genie-style keys (`graphs`, `vtcm_mb`, `O`,
`hvx_threads`) produce `"Unknown Key"` warnings when passed directly to
`qnn-net-run` and are ignored; they are Genie backend extensions, not native QNN
backend configuration keys.

**Repetitions:** three. Cycle counts consistent within **±0.3%**.

## Note on Wall-Clock Time

`qnn-net-run` reports total execute time ≈ **1170 ms** and DSP cycles ≈ **350M**.
This is much slower than Genie (~**85 ms per decode step**) under the same
nominal performance profile. Possible reasons: the cycle-counter distribution is
reliable (on-DSP hardware counter); the `qnn-net-run` wall-clock execute window
may include initialization or power-on; `perf_profile` may not drive the HTP at
the same effective clock under the `qnn-net-run` execution model.

> The analysis therefore uses **cycle percentages**, not wall-clock seconds.

## Overall Cycle Count

**350,302,972 cycles**

## Category Breakdown

| Category | Cycles | % of Total | # Ops | Typical Op Example |
| -------- | -----: | ---------: | ----: | ------------------ |
| **Attention-mask Expand (broadcast)** *(= GQA replication, §0.1)* | **261,822,091** | **74.7%** | 56 | `/layers.12/self_attn/Expand_1` |
| Attention GEMV — Q @ K | 19,587,883 | 5.6% | 28 | Q×K^T MatMul |
| Attention GEMV — attn @ V | 19,766,563 | 5.6% | 28 | softmax × V MatMul |
| Weight GEMMs (q/k/v/gate/up/down/o) | 31,267,900 | 8.9% | 308 | All FC/MatMul weight ops |
| `lm_head` | 6,119,799 | 1.7% | 1 | Final logits projection |
| Elementwise (Add/Mul/Neg) | 5,261,468 | 1.5% | 392 | Scalar ops |
| Shape ops (T/R/S/C/U) | 2,855,013 | 0.8% | 280 | Transpose/Reshape/Slice/Concat/Unsqueeze |
| Softmax | 1,595,508 | 0.5% | 28 | Per-layer softmax |
| RMSNorm | 955,307 | 0.3% | 169 | Pre-attn, Q, K, post-attn norms |
| Cast/Convert | 810,787 | 0.2% | 56 | Dtype conversions |
| `embed_tokens` | 4,931 | 0.0% | 1 | Token embedding lookup |
| **Total** | **350,047,250** | **99.9%** | 1347 | |

## Top 10 Individual Ops

The single most expensive individual op is `lm_head` (6.1M cycles, 1.7%), but
there is only one of it. By comparison there are **56 `Expand` operations**
(2 per layer × 28 layers), each costing approximately **4.7–4.8M cycles** —
combined **261.8M cycles, 74.7% of the total**.

| Rank | Op | Cycles | % |
| ---- | -- | -----: | -: |
| 1 | `lm_head` | 6,119,799 | 1.7% |
| 2 | L12/attn/Expand_1 | 4,840,138 | 1.4% |
| 3 | L25/attn/Expand_1 | 4,830,756 | 1.4% |
| 4 | L16/attn/Expand_1 | 4,814,429 | 1.4% |
| 5 | L6/attn/Expand_1 | 4,812,039 | 1.4% |
| 6 | L18/attn/Expand_1 | 4,806,390 | 1.4% |
| 7 | L0/attn/Expand_1 | 4,801,456 | 1.4% |
| 8 | L10/attn/Expand_1 | 4,796,417 | 1.4% |
| 9 | L22/attn/Expand_1 | 4,773,888 | 1.4% |
| 10 | L27/attn/Expand_1 | 4,769,868 | 1.4% |

## Per-Layer Cost

Each of the 28 layers costs approximately **12.2–12.4M cycles (±1%)**.

| Layer | Total Cycles | Mask Expand *(= GQA repl.)* | Q@K | attn@V | Weight GEMMs | Softmax | RMSNorm | Other |
| ----- | -----------: | --------------------------: | --: | -----: | -----------: | ------: | ------: | ----: |
| L0 | 12,550,650 | 9,406,689 (74.9%) | 698,988 | 710,143 | 1,259,695 | 59,536 | 37,610 | 377,989 |
| L14 | 12,316,153 | 9,348,116 (75.9%) | 736,765 | 705,526 | 1,113,594 | 56,418 | 35,431 | 320,303 |
| L27 | 12,261,959 | 9,351,556 (76.3%) | 684,783 | 706,106 | 1,102,596 | 54,853 | 36,843 | 325,222 |
| **Average** | **12,252,930** | **9,351,000 (76.3%)** | **700k** | **705k** | **1,111,000** | **57k** | **34k** | **275k** |

Layer-to-layer variation is very small. There is no obvious outlier layer.

## What Is the `Expand` Operation?

> **Superseded by §0.1.** The original text of this subsection assumed the ops
> broadcast the causal-mask scalar `-10000.0` to `[1, 8, 1, 1152]` = 18,432 B,
> and concluded ~260 cycles/byte, i.e. "a highly suboptimal HVX scalar broadcast
> loop". The DLC shows the real output is `[1,8,2,128,1152]` / `[1,8,2,1152,128]`
> = 4,718,592 B (256× larger), giving 1.03 cycles/byte — ordinary throughput for
> a broadcast FP16 multiply. The op is GQA KV-head replication.

## Why This Matters

The previous assumption was that decode was **weight-memory-bound**, because
~961 MB of weights are streamed per step; at 63 GB/s that should cost only
15–20 ms. However the profile shows **weight GEMMs account for only 8.9% of
total cycles**. This changes the optimization priorities substantially.

- **QKV fusion:** reduces 2 of the 7 weight GEMMs per layer; expected total
  decode saving ~2.5%. *(§0.1: ~10% of post-replication compute.)*
- **Gate-Up fusion:** reduces 1 of the 3 MLP GEMMs; ~1.5%. *(Same renormalisation.)*
- **Flash-Attention-style fusion:** if Mask + Q@K + Softmax can be fused so the
  mask is applied inside the GEMV kernel, the separate Expand + Add could be
  eliminated — potential saving ~75% of decode time. **This is the largest
  optimization opportunity.** *(§0.1: correct target, wrong mechanism — the fix
  is to stop materialising replicated KV.)*
- **Smaller context:** Expand cost scales with `seq_len`; reducing 1152 → 256
  would reduce it ~4×, potentially ~50% decode-time saving.
- **Batched decode (`verify32`):** broadcast cost can be amortized across 32
  positions. *(§0.1: backwards — the cost is AR-independent.)*

## Reproducibility

| Rep | Total Cycles |
| --- | -----------: |
| 1 | 350,302,972 |
| 2 | 349,342,550 (burst mode) |
| 3 (burst) | 350,698,129 |

Variation across repetitions and configuration variants is **<0.3%**.

---

# Test 3 — Build Gap (Our Arm Only)

The original request was to A/B our unfused W8A16 build against the device
team's reported **7.79 tok/s** build. Currently only our arm is available.

## Our Result

`qwen3_06b_w8a16_local` AR-1 decode: **11.72 tok/s**, versus the device team's
reported **7.79 tok/s** — **+51% faster**, rather than ~20% slower as originally
assumed. The direction of the gap is the opposite of what was expected.

## The A/B Cannot Yet Be Completed

A clean comparison requires the device team's binary, build artifacts, converter
command lines, build-time HTP configuration, `ctx-bin-utility` JSON dump, and
the dialog JSON used when measuring 7.79 tok/s.

Possible explanations: different DLC/build flags; different ctx-bin graph count
(local 2 vs ladekv/qh 3); different performance profiles or runtime
configuration; different measurement protocol (cold start, different prompt
length, TTFT included in the average).

---

# Test 4 — INT8 Head (qh) in Basic Mode

## Setup

Bundle `qwen3_06b_w8a16qh_ladekv`: 3-graph ctx-bin, W8 `lm_head`, 1094 MB.
Configuration `type: "basic"` from `genie_dialog_basic.json` plus
`max-num-tokens: 128`. Same prompt, greedy sampling,
`perf_profile: llm_decode_burst`. Baseline is Test 1 Arm A
(`qwen3_06b_w8a16_local`, 2 graphs, FP16 `lm_head`).

## Results

| Metric | qh Rep 1 | qh Rep 2 | qh Rep 3 | qh Mean | Baseline Mean | Δ |
| ------ | -------: | -------: | -------: | ------: | ------------: | -: |
| Init time (ms) | 852 | 841 | 868 | **854 ± 14** | 839 ± 25 | +2% |
| TTFT (ms) | 188.5 | 188.2 | 188.0 | **188.2 ± 0.3** | 40.1 ± 0.1 | +369% |
| Prompt rate (t/s) | 297.2 | 297.6 | 298.0 | **297.6 ± 0.4** | 1397 ± 2 | −79% |
| Gen tokens | 128 | 128 | 128 | 128 | 128 | — |
| **Gen rate (tok/s)** | 6.69 | 6.70 | 6.70 | **6.70 ± 0.00** | 11.72 ± 0.01 | **−43%** |
| Gen time (ms) | 18975 | 18954 | 18965 | 18965 ± 10 | 10837 ± 11 | +75% |

## Key Finding

The HF README projected **+19%** decode throughput; the measurement shows **−43%**.

## ⚠️ This Is Not a Clean Comparison

1. **Different ctx-bins** — baseline is a 2-graph `local` bin, qh is a 3-graph
   `ladekv` bin. Basic mode on a 3-graph ctx-bin with graph switching may carry
   additional overhead.
2. **Potentially different builds** — the `local` bundle was built separately
   from the `ladekv`/`qh` bundles.
3. **TTFT matches `ladekv`** (188 vs 186 ms) versus Basic-local's 40 ms,
   confirming both use the same 3-graph structure and a similar prefill path.

## How to Obtain a Clean Result

Build a 2-graph (prefill + decode only) W8 `lm_head` variant ctx-bin, or extract
the decode graph from the qh bundle into a single-graph ctx-bin as in Test 2.

## Actual Impact of `lm_head`

Per Test 2, FP16 `lm_head` costs 6.1M cycles = **1.7%** of total. Even if W8
halved it, the saving is ~3M of 350M cycles, **<1% total**. The README's +19%
projection was based on an incorrect assumption about the dominant bottleneck.

*(§0.1 note: against post-replication compute `lm_head` is 6.9%, not 1.7%. And
in a weight-stream-bound regime the FP16 head is 311 of 751 MB — 41% of the
bytes — so its value must be re-derived after the replication fix, not inherited
from this table.)*

---

# Test 5 — Runtime `hvx_threads: 8`

## Setup

Same `ladekv` bundle and configuration as Test 1 Arm B. Only change: in
`htp_backend_ext_config.json`, `"hvx_threads": 4` → `"hvx_threads": 8`.
3 repetitions, warm, same prompt.

## Results

| Metric | hvx=4 Mean | hvx=8 Mean | Δ |
| ------ | ---------: | ---------: | -: |
| Init time (ms) | 872 ± 21 | 850 ± 18 | −3% |
| TTFT (ms) | 186.3 ± 0.6 | 176.0 ± 0.3 | −5% |
| Prompt rate (t/s) | 301 ± 1 | 318 ± 1 | +6% |
| Gen tokens | 519 | 519 | — |
| **Gen rate (tok/s)** | **9.18 ± 0.00** | **9.17 ± 0.00** | **−0.1%** |
| Gen time (ms) | 56435 ± 17 | 56485 ± 12 | +0.1% |

## Result: No Improvement in Generation Rate

Sustained generation rate is effectively identical. The QNN profile had already
reported `Number of HVX threads used: 4 count` regardless of the runtime
setting. This confirms `hvx_threads` is a **build-time parameter, not a runtime
parameter** — at runtime the graph uses however many HVX threads the ctx-bin was
compiled for.

To complete the A/B properly, a ctx-bin must be compiled with `hvx_threads: 8`
and compared against the original build. Since weight GEMMs are only 9% of
decode cycles, even a 2× improvement there would yield <5% total.

---

# Cross-Test Findings and Anomalies

## 6.1 `max-num-tokens` Causes SIGSEGV with LADE

When the dialog contains both `"max-num-tokens": 128` and `"type": "lade"`,
Genie crashes with exit code 139 (SIGSEGV). The same `max-num-tokens` setting
works correctly with `type: "basic"`. This is either a Genie bug or an
unsupported configuration combination. LADE was therefore tested without
`max-num-tokens`; generation continues until greedy repetition fills the context.

*(Fixed build-side 2026-08-14 — see §0.3.)*

## 6.2 LADE Acceptance Rate Varies Widely by Prompt

On the "Hexagon DSP" technical prompt, LADE = 9.18 vs Basic = 11.72 tok/s (LADE
slower). On earlier, simpler prompts, LADE = 10.8 vs Basic ≈ 6.5 tok/s (~+66%).

> The benefit of n-gram speculative decoding is highly prompt-dependent.

For production use, evaluate against the real target-workload prompt
distribution rather than synthetic benchmarks.

## 6.3 Build Gap Between `local` and `ladekv` / `qh` Bundles

`qwen3_06b_w8a16_local` (2 graphs) achieves **11.72 tok/s AR-1**, while
`qwen3_06b_w8a16_ladekv` / `qh_ladekv` (3-graph bins) achieve only ~**6.7 tok/s
in Basic mode** — a ~75% difference depending only on which ctx-bin is used.

Possible explanations: graph-switching overhead in Basic mode; different
encoding/weight layout; different weight sharing/memory layout forced by
accommodating three graphs.

*(§0.1 note: the 6.7 figure was measured on the **qh** bundle only, so this
comparison conflates W8 head, graph count, and build lineage. Basic on the plain
`ladekv` bin has never been measured — that single run splits the three.)*

## 6.4 `qnn-net-run` Ignores Genie-Style Backend Configuration Keys

Genie's `htp_backend_ext_config.json` keys (`graphs`, `graph_names`, `O`,
`vtcm_mb`, `hvx_threads`) generate `Unknown Key` warnings and are silently
ignored when passed to `qnn-net-run --config_file`. The QNN HTP backend uses a
flatter structure (`devices → cores → perf_profile`, `rpc_polling_time`).
The `graphs` section is a **Genie runtime extension** applied during ctx-bin
creation and graph registration, not part of native QNN backend configuration.

---

# Recommendations

1. ⭐ **Investigate the 75% `Expand` cost.** *(Retargeted by §0.1: eliminate GQA
   KV replication by feeding the un-replicated cache into a grouped MatMul.
   Mask fusion is not applicable — the mask is never expanded.)*
2. **Investigate the `local` vs `ladekv` build gap** (11.7 vs ~6.7 tok/s).
   Diff the builds via `ctx-bin-utility` JSON, shapes, dtypes, encodings.
3. **Perform a clean qh (W8 `lm_head`) comparison** with a 2-graph-only qh
   ctx-bin or an isolated decode graph.
4. **Perform a compile-time `hvx_threads=8` A/B.**
5. **Evaluate LADE on the real prompt distribution.**
6. **Test 3 follow-up:** send the measured 11.7 tok/s to the device team and
   request their unfused binary and build configurations.

---

# Artifact Index

> **Not received** — see §0.2. The paths below are the report's own index,
> retained so the re-request can name exact files.

All raw files were to be located under `docs/test_artifacts/measurement_2026-08-13/`:

| Path | Content |
| ---- | ------- |
| `SUMMARY.md` | Short summary, superseded by this report |
| `configs/` | All dialog and backend configurations used |
| `configs/orig_*_*` | Original as-shipped configurations from each bundle |
| `test1_basic/profile_basic_r{1,2,3}.json` | Test 1 Arm A Genie profiles |
| `test1_basic/stdout_basic_r{1,2,3}.txt` | Test 1 Arm A stdout |
| `test1_ladekv/profile_lade_r{1,2,3}.json` | Test 1 Arm B Genie profiles |
| `test1_ladekv/stdout_lade_r{1,2,3}.txt` | Test 1 Arm B stdout |
| `test2_decode_profile/qnn_profile_r{1,2,3}_viewer.txt` | Full `qnn-profile-viewer` output, burst mode |
| `test2_decode_profile/qnn_profile_llm_r{1,2,3}_viewer.txt` | Full `qnn-profile-viewer` output, `llm_decode_burst` mode |
| `test2_decode_profile/qnn_profile_*.bin` | Raw binary QNN profiling logs |
| `test2_decode_profile/execution_metadata_r1.yaml` | Graph I/O metadata |
| `test4_qhbasic/profile_qhbasic_r{1,2,3}.json` | Test 4 Genie profiles |
| `test4_qhbasic/stdout_qhbasic_r{1,2,3}.txt` | Test 4 stdout |
| `test5_hvx8/profile_hvx8_r{1,2,3}.json` | Test 5 Genie profiles |
| `test5_hvx8/stdout_hvx8_r{1,2,3}.txt` | Test 5 stdout |
