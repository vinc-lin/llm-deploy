# SA8797P LLM Deployment — Consolidated Reference

*Current truth as of 2026-08-12. Supersedes conflicting statements anywhere else
in this repo. Every number here is either device-measured, tool-measured on this
machine, or cited to SDK source — claims that could not be verified are marked
as such rather than repeated.*

**Read this first.** Then `docs/BUILD_GUIDE.md` for step-by-step recipes and
`docs/NOTES-genie-io.md` for the line-cited Genie contract.

---

## 0. Status board

| | |
|---|---|
| **Best sustained decode, 0.6B** | **10.8 tok/s** — LADE speculative decoding, `qwen3_06b_w8a16_ladekv` (2026-08-11) |
| Non-speculative AR-1 decode | 6.3–6.5 tok/s (~155 ms/step) |
| Bertcache early phase (topology A only) | ~23.8 tok/s for the first ~117 tokens, then falls to AR-1 |
| Output correctness | ✅ since v2 (2026-08-10) |
| Quantization | W8A16 (INT8 per-channel weights, FP16 activations) — the only recipe that works |
| Ctx-bin, 0.6B | ~1.09 GB, 2 or 3 weight-shared graphs |
| Device access | none from this machine — build + numerics only; device runs are done by a remote tester |
| Blocked on hardware | tok/s, DDR bandwidth, VTCM behavior, perf profiles, anything GVM |

**What moves the needle next:** multi-token decoding. Decode is 100% DDR-bound,
so the only lever that *divides* bytes/token is emitting more than one token per
weight-streaming pass. LADE already bought 1.7×; a learned draft head (`eaglet`,
`spd`) is the next step up, and acceptance rate — not per-call latency — is the
thing to optimize (§6.3).

---

## 1. Hardware and runtime reality

Inherited from the remote team's characterization (2026-08-09) and not
contradicted by anything since.

| Attribute | Value |
|---|---|
| SoC | SA8797P (nordy / Gen5 / Snapdragon Ride Flex) |
| HTP | Hexagon v81, 4 NSPs on silicon (~8 MB VTCM each, ~80 TOPS INT8 nominal) |
| **What the Android GVM guest actually gets** | **2 of 4 NSPs, 16 MB VTCM total, unsigned PD** |
| VTCM ceiling | 16 MB — `vtcm_mb: 24` is rejected at runtime (`0x138d`) |
| Runtime | QAIRT 2.48.40.260702 · QNN API v2.37.0 · libGenie 1.19.0 |
| DVFS | works via Genie `backend.extensions` JSON, 4 tiers, 1.95× swing. `qnn-net-run --perf_profile` is a no-op. Always use `llm_decode_burst`. |
| Burst BW, one large contiguous matmul | ~49 GB/s |
| Effective BW, real LLM decode | **~6–7 GB/s** |
| FastRPC per-call | ~220 µs (hypervisor-mediated `hfastrpc`) |
| Not available | zero-copy sharedbuf · DSP hardware queue · clock visibility |

**The central performance fact:** the ~7× collapse from 49 → 7 GB/s is
*access-pattern fragmentation* (28 layers of small MatMuls + KV traffic + per-op
sync), not a clock cap. Decode is per-token weight streaming over DDR. Compute is
not the bottleneck and neither is DVFS — both are already at ceiling.

Reference point for how much the hypervisor costs: the same class of model on a
QNX-native EVB with 4 cores runs **129.7 tok/s** (Qwen3-VL 4B, official Qualcomm
number). We are at ~7% of that, and most of the gap is environmental.

---

## 2. Build pipeline

```
HF checkpoint → export wrapper (PyTorch) → AIMET W8A16 quantsim + calibration
   → ONNX export → I/O rename → qairt-converter (one DLC per graph)
   → qnn-context-binary-generator (weight-shared ctx-bin) → bundle.sh
```

Scripts: `full_build.sh <name> <cl> <ctx>` → `lade_build.sh` (adds verify32) →
`ladekv_build.sh` (past-KV prefill, 3-graph) → `bundle.sh`. Flags after the
positional args pass through to `quantize_aimet.py`.

### 2.1 Two topologies

**A. Bertcache** (baseline, fused variants) — `prefill` AR=128 CL=128 no-past-KV,
`decode` AR=1 CL=1152. Genie keeps generating *through the prefill graph* one
token per step (~42 ms) until KV passes 128, then switches to AR-1 (~155 ms).
Gives the early-token burst; **cannot run LADE** (§3.4).

**B. All-past-KV** (`-ladekv`) — `prefill` AR=128 CL=1152 past=1024, `decode`
AR=1 CL=1152, `verify32` AR=32 CL=1152. **This is the reference topology for
anything new.** Enables LADE and prompts >128 tokens; gives up the bertcache
burst (generation runs at true decode speed from token 1).

### 2.2 Which graph actually serves a request

Genie picks by numeric best-fit, so topology B routes by prompt length:

| prompt tokens | graph that serves it |
|---|---|
| ≤ 32 | `verify32` |
| 33–128 | `prefill` |
| > 128 | `prefill`, chunked 128 at a time |
| all LADE verification batches | `verify32` |
| basic-mode decode | `decode` (AR-1) — **never invoked in LADE mode** |

This is why the two LADE reports show different cold starts: the 37-token prompt
ran through `prefill` (TTFT 458 ms), the 22-token prompt went straight to
`verify32` and paid a graph switch on step 0 (1070 ms). Both are correct
behavior, not a regression.

---

## 3. Hard contracts — violations are silent

Each is pinned to SDK source in `docs/NOTES-genie-io.md`. These produce binaries
that **load and run cleanly and emit garbage**, or SIGSEGV with no useful
message. All four have already cost a device cycle.

### 3.1 All-position logits

Every logit-producing graph MUST emit `[1, AR, vocab]`. Genie left-aligns tokens
and samples row `n_process − 1`; there is no last-token-only mode and no
load-time shape rejection — `numElements == vocab` is explicitly accepted.

A last-token-only head reads ~`(n_process−1)·vocab·2` bytes past a 1-row buffer →
zeros/noise → `argmax = token 0 = "!"`. **This was the v1 garbage bug**, and it
affected every v1 bundle regardless of fusion. Guard:
`scripts/validate/parity_qualla_read.py`.

Do not "fix" it by removing logits from prefill: a logits-less graph with
`input_ids` and no past-KV classifies as `GraphType::LUT`, not `DECODER_PREFILL`.

### 3.2 Cross-graph encodings identity

All graphs in one ctx-bin share weights, so every DLC must convert against **the
same encodings file** (the prefill run's). KV quant params must be byte-identical
across graphs for same-named tensors, or Genie fails the load. Never recalibrate
one graph of a set — that is what `--export-decode` / `--adopt-encodings` exist
for. (This is the failure class behind the remote team's "error 5005".)

### 3.3 Graph names — cosmetic to Genie, load-bearing for the backend

- **To Genie's picker:** names *are* cosmetic; selection is numeric best-fit on
  (AR, CL). Two graphs may never share an (AR, CL) pair — hard load error.
- **To the HTP backend:** names are how `htp_backend_ext_config.json` scopes its
  per-graph tuning. A graph not listed in `graph_names` silently compiles with
  **backend defaults — 4 MB VTCM, 24 MB spill** — with no warning and exit 0.
- The name is baked in at conversion time from the `--output_path` **basename,
  dots included**: converting to `decode.dlc.new` yields a graph named
  `decode_dlc`, and renaming the file afterwards does **not** change it.

Measured cost of getting this wrong (`docs/NOTES-vit-htp-config.md`, same DLC,
two configs): `spill_bytes` 4,194,304 → 1,446,117,376 — **345×** — from a build
whose log is clean. On the LADE build the same class of error produced a
null-pointer SIGSEGV on the first speculation step. Always convert straight to
the final filename and verify with `qnn-context-binary-utility --json_file`
before bundling.

### 3.4 LADE-specific

- **No AR==CL (bertcache) graph in the ctx-bin.** Only `ctx_size == variant`
  inflates `n_process` past the lade attention-map size, driving a heap OOB read
  whose garbage becomes a RoPE-table byte offset → host SIGSEGV.
- **Prompts must tokenize to ≥ 2 tokens.** `lhd_branch` warmup does
  `rand() % (tokens.size()−1)` — modulo zero on a 1-token prompt; aarch64 returns
  the dividend, giving index `1 + 0x6b8b4567` and a ~7 GB OOB read. This is an
  unconditional qualla bug, independent of topology. (0x6b8b4567 is `rand()`'s
  first output and matches the observed crash register exactly.)
- **Config guardrail:** `(ngram−1) × (window + gcap)` ≤ the verify graph's AR.
  Shipped config 8/3/8 = exactly 32. Oversized configs silently route batches to
  a graph that cannot serve them.

### 3.5 Past-KV prefill feeding contract (topology B)

What `parity_ladekv_read.py` reproduces:

- Tokens **left-aligned**, remainder = pad token, which defaults to the **first
  `eos-token`** entry (151645 for us).
- Mask is FP16 **additive**: allow `+0.0`, masked **`−1000.0`** (not `−inf`). Our
  encodings calibrate the mask at −100, so the device clips −1000 → −100 —
  e^−100 ≈ 0, harmless.
- Concat layout: mask cols `[0, past)` = past region, `[past, CL)` = new tokens.
- RoPE positions `iota(n_past + i)` on valid rows, 0 on pad rows.
- **The KV cache advances by `n_process`, not by the AR window.** A left-aligned
  n-token prompt in an AR=128 window enters only columns `0..n−1`; pad-slot KV is
  discarded. Host-side emulation must scatter new-slice KV at offset `n`, not AR.
- Chunking ceiling: accumulated KV ≤ `past_dim = CL − AR` (1024). Beyond that,
  `ContextLimitException`.

---

## 4. Quantization

**W8A16 via AIMET 2.36 PTQ** — per-channel symmetric INT8 weights, 16-bit
activations, `post_training_tf_enhanced`, calibrated on 10 mixed zh/en/code/math
prompts.

Quantizers **disabled (kept FP16)** on:

| what | why |
|---|---|
| `embed_tokens` | HTP v81 rejects `Gather` on INT16 weights (error `0xc26`) |
| final `norm` | quality |
| `lm_head` | default; see §6.4 — the *reason* usually given for this is wrong, but the default is still right |
| all K/V-projection outputs | cross-graph FP16 requirement (§3.2) |

Weight encodings are clipped to the ±0x7f7f-safe range. **HTP packed-pair
saturation is not modeled by quantsim**, so "passes quantsim, garbage on silicon"
is this bug's signature.

Fusion variants: Gate-Up keeps the fused `gate_up` output FP16 (requantized at
`down_proj`); QKV grafts the donor `q_proj` INT16 encoding onto the Q split with
K/V splits FP16, done at the **encodings** level by `qkv_surgery.py` (28/28).

### 4.1 Dead ends — do not re-run these

| Approach | Result | Real reason |
|---|---|---|
| **W8A8** | garbage, all variants (v15–v18) | v81 MatMul supports only per-tensor UINT8 asymmetric activations, which clip heavy-tailed LLM activations. Per-channel activations unsupported. |
| **W4A16 at 0.6B** | **0/4** on the argmax gate W8A16 passes 3/4 (`max\|Δlogits\|` 16–25 vs 1.3–1.7) | Accuracy, not kernel support. All three recipes fail: per-channel INT4, LPBQ block-64, LPBQ+SeqMSE. `--lpbq`/`--seq-mse` remain for larger models. |
| **QKV/Gate-Up fusion for speed** | builds fine, **no tok/s gain** (6.27–6.5 ≈ baseline) | The remote's 3.4× DDR reduction was measured at `vtcm_mb=24`, which unsigned PD rejects. At 16 MB every variant already shows **zero** VTCM spill — there is no spill left to remove. |
| **`--quant-head` (W8 lm_head) under LADE** | **−14% tok/s** | Costs ~10% n-gram acceptance; the DDR saving does not survive spec-decode amortization (§6.3). |
| **`sparse_weights_compression=1`** | 0 bytes saved | model isn't sparse |
| **2-core ctx-bin** | error 5005, or 3.96 tok/s — slower | Genie 1.19 creates a single-core device internally, no JSON override |
| **Multiple Genie instances** | 2 × 4.0 tok/s = 8 total | linear BW split — confirms decode is 100% DDR-bound |

---

## 5. Validation gates

Device-free, in order. Each has a known-good reference value; run them all before
shipping anything.

| Gate | Command | Pass |
|---|---|---|
| Wrapper vs HF | `export_qwen3.py … --parity-check` | max\|Δlogits\| ~4e-05 |
| Standard parity | `parity_onnx.py --onnx <dir> --cl-prefill 128 --ctx 1024` | prefill argmax match + 8-step greedy chain token-identical |
| **Device read pattern** (topology A) | `parity_qualla_read.py --onnx <dir> --cl-prefill 128` | 4/4 prompts |
| **Device read pattern** (topology B) | `parity_ladekv_read.py --onnx <model_renamed.onnx> --ar 128 --ctx 1024` | 6/6 (4 single-chunk + 2 chunked at 129 and 200 tokens) |
| Verify graph | `parity_verify.py` | batched rows match HF, ~3e-05 |
| Quant quality | `quantize_aimet.py … --eval` | last-token argmax ≥ 3/4 prompts |
| DLC shape | `qairt-dlc-info -i prefill.dlc \| grep logits` | `1,128,151936` — never `1,1,…` |
| **Graph names** | `qnn-context-binary-utility --json_file` → `graphName` | exactly matches both HTP configs (§3.3) |
| Quantized head | `qairt-dlc-info \| grep lm_head.weight` | `sFxp_8` with `--quant-head`, else `Float_16` |
| Ctx-bin | `qnn-context-binary-utility --json_file` | all graphs listed, logits dims per §3.1, ~1.09 GB for 0.6B |

---

## 6. Measured numbers

### 6.1 Device KPIs, Qwen3-0.6B W8A16

| | v1 fuseqkvgu (2026-08-10) | v2 baseline (2026-08-10) | ladekv (2026-08-11) | qh-ladekv (2026-08-12) |
|---|---|---|---|---|
| Output | ❌ garbage from token 1 | ✅ correct | ✅ correct | ✅ correct |
| Mode | basic | basic | **LADE** | **LADE** |
| Sustained tok/s | 6.27 | 6.5 | **10.8** | 9.3 |
| Per-step / per-call | 159.6 ms | ~155 ms | 180 ms (p50), σ≈3 ms | 187 ms (181/187/217) |
| Tokens per call | 1 | 1 | **~1.94** (see §6.2) | 1.74 |
| Init (dialog + backend) | ~770 ms | ~796 ms | — | — |
| Init → first logits | — | — | 1247 ms | 1070 ms |
| TTFT (prefill start → first token) | — | — | 458 ms | — |
| RAM allocated | 132 MB | 163 MB | 163 MB | — |
| VTCM spill | 0 | 0 | 0 | 0 |
| ctx-bin | ~1.4 GB (v1-era) | 1,087,074,304 | 1,106,276,352 | 1,093,767,168 |

⚠️ **The qh report's "+134% cold start" is a units mismatch.** It compares its own
init→first-logits (1070 ms) against ladekv's **TTFT** (458 ms), which excludes
~789 ms of init. Like-for-like from the ladekv init timeline — dialog config
loaded 07:20:26.377 → first verify32 done 07:20:27.624 = **1247 ms** — the qh
build actually reaches first logits ~177 ms *faster*. Row separated above so the
two are not read as comparable.

Bertcache phase (topology A only): prompt processing 265.6 / 266.5 tps (12 tokens
in 45 ms), then ~42 ms/step ≈ 23.8 tok/s for the first ~117 tokens. **Any tok/s
number for topology A is meaningless without saying which phase it refers to.**

LADE graph usage: `decode` (AR-1) is registered but **never invoked** — verify32
handles acceptances and rejections in one pass, with no AR-1 fallback.

### 6.2 The acceptance-rate correction: 2.05 → ~1.94

The ladekv report states "~2.05 accepted tokens/verify call". It does not
reconcile; **~1.94 does**, by four independent routes:

| Route | Computation | Result |
|---|---|---|
| Tokens ÷ calls | 635 / 327 | 1.94 |
| Stated accept distribution | 0.46(1) + 0.13(2) + 0.41(3) | 1.95 |
| Throughput identity | 10.8 tok/s ÷ (1000/180 calls per s) | 1.94 |
| Cross-check from the qh run | (1.74/1.94) × (180/187) = **−13.7%** vs measured −14% | 1.94 ✓ |

The last route is worth keeping: with 2.05 the same arithmetic predicts −18.3%,
which is not what the device measured. The qh regression independently confirms
the baseline was 1.94.

This *strengthens* the ladekv analysis. Re-running its speculative-speedup model
with the corrected rate: 1.94 × (156/180) = **1.68×** against a measured
**1.70×** — the formula now predicts the measurement almost exactly instead of
over-predicting at 1.78×.

### 6.3 Why spec-decode is acceptance-bound, not latency-bound

Sustained tok/s ≈ `accepted_tokens_per_call ÷ call_latency`. At 0.6B on this
device, per-call latency is pinned near 180–190 ms by DDR streaming and barely
moves; acceptance is the free variable. The qh experiment is the clean
demonstration: it traded ~10% acceptance for a per-call saving that never
materialized, and lost 14%.

**Consequence for future work:** optimize acceptance rate. A learned draft head
(`eaglet` / `spd`, both shipped in this SDK with Qwen3-4B-class example configs)
raises acceptance in a way n-gram matching cannot. Per-call micro-optimizations
are close to worthless here.

### 6.4 `--quant-head` (the `qh` variant), fully resolved

| | |
|---|---|
| Is the head actually INT8? | **Yes** — `lm_head.weight` = `sFxp_8` per-channel, verified in all three tested DLCs (`encoding for channel_0: bitwidth 8, min −0.117441192269, max 0.116523683071, scale 0.000917509315, offset 0`) |
| DLC size | 1,074,293,920 → 922,965,680 B (**−151.3 MB**, against a 155.6 MB ideal = 151936 × 1024) |
| ctx-bin size | 1,106,276,352 → 1,093,767,168 B (**−12.5 MB only**) |
| Device, LADE | **9.3 vs 10.8 tok/s (−14%)** |
| Device, verify latency | 187 vs 180 ms — went **up**, where the projection wanted −20 ms/call |
| Output quality | unchanged, greedy, 0.6B |

Two traps this variant exposed, both worth remembering:

1. **The flag silently did nothing for a while.** `filter_aimet_w8a16.py`
   stripped every `lm_head` encoding unconditionally, deleting the 8-bit
   per-channel weight encoding `--quant-head` had deliberately kept. AIMET
   emitted it correctly, the filter removed it, the converter emitted `Float_16`.
   No error anywhere. Fixed by `--keep-head-weight` (commit `f486eec`), which
   asserts the encoding is present instead of quietly producing an identical
   build. **Always verify the dtype, never trust the flag.**
2. **~139 MB of the DLC saving reappears at prepare time.** The DLC shrank
   151.3 MB but the ctx-bin only 12.5 MB. Since the `FullyConnected`'s input and
   output are both `Float_16`, the likely explanation is that HTP materializes
   the INT8 head back to 16 bits when preparing the context blob — in which case
   the on-device DDR traffic never changed at all, which fits the latency going
   up rather than down. **Unproven** (ctx-bin dumps expose no weight tensors:
   `numContextTensors: 0`); would need on-device DDR counters to settle.

Verdict: **do not ship `qh` for speculative decoding.** Untested in AR-1 basic
mode, where acceptance is irrelevant and the per-call saving — if it reaches the
device at all — would translate directly.

### 6.5 Build-time DDR (converter summary, vtcm 16, weight-shared)

| Variant | prefill | decode | VTCM spill |
|---|---|---|---|
| baseline W8A16 | 759 MB | 957 MB | 0 |
| + Gate-Up fusion | 769 MB | 961 MB | 0 |
| + QKV fusion | 763 MB | 961 MB | 0 |
| verify32 (AR=32) | — | 1,906 MB | **745/750 MB spill/fill** |

The verify32 spill is expected — AR=32 activations do not fit 16 MB VTCM — and it
is *cheap in practice*: spill/fill is contiguous DMA (~49 GB/s class) while weight
streaming is the fragmented ~7 GB/s kind. On raw bytes lookahead breaks even at
~2.0 accepted tokens/pass; on the device it wins at 1.94, which is exactly why the
byte-counting model under-predicts the real gain.

*(These are all pre-2026-08-11 builds. No converter DDR summary has been recorded
for a real `--quant-head` build.)*

---

## 7. Corrections ledger

Claims that were believed, are now known false, and may still be quoted in older
documents. Each historical report keeps its original text with a banner — this is
the index of what changed.

| # | The claim | Where it appeared | What is actually true |
|---|---|---|---|
| 1 | v1 garbage output was caused by the **fused QKV block emitting wrong K/V on HTP v81** | fuseqkvgu report §12–13 | Last-token-only prefill logits `[1,1,V]` vs qualla's row-`n_process−1` read. Affected **all** v1 bundles, fused or not. Encodings surgery was clean at JSON and DLC level. |
| 2 | The v1→v2 fix was in **weight layout / per-channel axis / encoding** | v2 report §4, §7.4 | It was a graph **shape** change (all-position logits). Quantization was not touched. |
| 3 | LADE SIGSEGV means a **missing draft head/verifier or ABI mismatch** | v2 report §3 | The verifier graph was in the ctx-bin all along. One missing `verify32` entry in `graph_names`, plus an independent `rand() % 0` qualla bug on 1-token prompts. |
| 4 | ladekv accepts **2.05 tokens/verify call** | ladekv report §1, §4 | ~**1.94** — four independent routes agree, including a cross-check from the qh run (§6.2). |
| 5 | `--quant-head` measures **961 → 763 MB/token (−20.6%)** | BUILD_GUIDE §5.7 | Fabricated from unrelated numbers: both figures are the prefill (763,410,432) and decode (961,130,496) `read_total_bytes` of **one non-qh build**, in `ctxbin-ws.log` dated 2026-08-10 — two days before `--quant-head` existed. Real effect: §6.4. |
| 6 | **`lm_head` INT8 degrades quality** | remote summary §2.2 | Not supported. 3/4 argmax locally, device parity confirmed at 0.6B greedy. Keep the head FP16 by default, but for the acceptance-rate reason (§6.4), not this one. |
| 7 | QKV fusion **"not yet done", needs ONNX surgery** | remote summary §3.1, §4.1 | Done at the encodings level (28/28 grafts), built and device-tested. It just buys nothing at vtcm 16. |
| 8 | W4A16 fails because **v81 has no INT4 MatMul kernel** | remote summary §3.1 | That row is also self-contradictory ("requires v75 or newer" — v81 *is* newer). Our result is about **accuracy**: 0/4 on the argmax gate, all three recipes. |
| 9 | ctx-bins are **1.5 GB** | LOCAL_ENV, remote summary §2.1 | ~**1.09 GB** for every current 0.6B build — measured 2026-08-12. See open question §8.2. |
| 10 | Graph names are **cosmetic** | BUILD_GUIDE §3.4, NOTES-genie-io | Cosmetic to Genie's picker only. Load-bearing for the HTP backend config (§3.3). |
| 11 | Disk: **flat 6 GB `disk_guard`, compact the VHD to recover** | BUILD_GUIDE §8 | `disk_guard <need_gb>` must be **sized to the step** (a 4B export writes 8.6 GB). No compaction needed — the vhdx is sparse and `/` is mounted `discard`. |
| 12 | Decode throughput **7.4–8.2 tok/s** | remote summary §2.1 | Never reproduced here; our builds measure 6.3–6.5 AR-1. |
| 13 | qh cold start is **+134% vs ladekv** (1070 vs ~458 ms) | qh report §1, §4 | Units mismatch — 1070 ms is init→first-logits, 458 ms is TTFT measured from prefill start. Like-for-like the ladekv build takes **1247 ms** to first logits, so qh is ~177 ms *faster*, not 134% slower (§6.1). |

---

## 8. Open questions

### 8.1 Where do the ~139 MB go? *(qh, §6.4)*
The DLC shrinks 151 MB, the ctx-bin only 12.5 MB. Hypothesis: HTP re-materializes
the INT8 head as 16-bit at prepare time because the surrounding activations are
FP16. If true, `--quant-head` cannot save DDR on this backend at all, and the
whole variant is moot. Needs on-device DDR counters or a prepare-time weight dump.

### 8.2 What actually changed between v1 (1.52 GB) and v2 (1.09 GB)?
A ~430 MB shrink is far too large for the all-position-logits fix, which should
have made the binary marginally *bigger*. Best candidate: **weight sharing became
effective**. Supporting evidence found today — the qh intermediates still show the
unshared signature: `qwen3-0.6b-w8a16qh_ctx.bin` is **1.84 GB** for 2 graphs and
`qwen3-0.6b-w8a16qh-lade_ctx.bin` is **2.16 GB** for 3, against 1.09 GB when
sharing works. Worth confirming, because it means a silently-unshared build is
still possible today and the only symptom is file size.

### 8.3 Does `qh` help in AR-1 basic mode?
The one configuration where the acceptance penalty does not apply. Cheap to test —
the bundle ships `genie_dialog_basic.json` against the same ctx-bin. Contingent on
§8.1: if the saving never reaches the device, the answer is no.

### 8.4 `soc_model: 0` at `O=3`
We build with `soc_model` unspecified. The SDK maps SA8797 → `soc_id 72`, and
Qualcomm's HTP docs state that specifying it at O=3 "could turn on additional
[algorithms] which may further improve inference performance". Real performance
possibly left on the table, for **all** builds, not just the ViT. Needs a measured
A/B before the next device run.

### 8.5 AR-32 ↔ AR-1 reshape churn
After every dialog-level KV update the cache is reshaped to the smallest
registered AR. In LADE that means an AR-32↔AR-1 reshape every iteration. If device
KPIs ever implicate it, the lever is a LADE-only ctx-bin with no AR-1 graph — the
graph is never invoked anyway (§6.1).

### 8.6 Environmental, unchanged from the remote team
4 NSPs instead of 2 (needs QNX-side VM config) · whether signed PD grants more
VTCM · `DDR_PERF_MODE` (C API option 7, not exposed in Genie JSON) · QNX shell
access via serial, the single highest-leverage change available (5–15× potential).

### 8.7 Qwen3-VL-4B stage 2
The W8A16 recipe is **validated at 4B** — 22 multimodal calibration windows,
396 weight tensors clipped, `--eval` **4/4** (bar is 3/4), including all-visual
windows whose activations span [−5.32, +4.97] vs text's [−0.146, +0.124]. What
fails is AIMET's *export*: OOM-killed twice inside
`_create_onnx_model_with_markers` at 37.4 and 45.4 GiB anon-rss against a 63 GB
budget. The cause is structural — the legacy `sim.export` path holds four fp32
copies of a 15.0 GiB graph. Allocator tuning bought ~3%. The real lever is
switching to `sim.onnx.export()`, which needs the encodings names and
`rename_aimet_io.py`'s positional assumptions revalidated.

---

## 9. Operational gotchas

**Disk / WSL2 — this has hard-crashed the VM three times.** `$LLMDEPLOY_DATA` sits
on an ext4.vhdx on Windows C:. If C: runs dry, the vhdx grow fails and this is
**not** ENOSPC: the guest still reports free space, the host write fails, and
every mmap'd page takes SIGBUS. PID 1 dies and the VM hard-crashes with no OOM
line anywhere. Dumps land in `%LOCALAPPDATA%\Temp\wsl-crashes`; the `-N` filename
suffix is the signal, `-7` = SIGBUS. Prevention: call `disk_guard <need_gb>`
before every multi-GB step, **sized to that step** — 6 GB is the converter floor,
a 4B export writes 8.6 GB and should ask 20. Recovery needs no compaction step:
the vhdx is sparse and `/` is mounted `discard`, so deleting in-guest returns the
space to C:. `ls` reports the ~448 GB virtual size and always will; `du -h <vhdx>`
without `--apparent-size` is the real consumption.

**HF uploads.** The local proxy (`http://127.0.0.1:17890`) drops long-lived
uploads. Use `scripts/util/hf_upload_watchdog.sh`, and:

1. **Set `SOCKET_CHECKS=999999`** — the CLOSE-WAIT detector false-positives
   through this proxy and kills healthy transfers; a partial blob restarts from
   byte 0 and can never finish. The progress-freeze detector (`STALL_SECS=240`) is
   the reliable signal.
2. **128 commits/hour hub limit.** Restart storms exhaust it; the commit phase
   then "hangs" with every byte already uploaded. Diagnose with one foreground
   `HfApi().upload_file` (a 429 surfaces in seconds). Recover by waiting ~1 h and
   committing one file at a time — blobs dedup, so each commit is instant.
3. **`hf upload-large-folder` silently flips a private repo PUBLIC** (it applies
   its default `private=False`). Re-check `HfApi().repo_info(repo).private` after
   every bulk upload. Single `upload_file` commits don't touch repo settings.

**Environment.** `source scripts/env.sh` first in every shell. `QUANT_DEVICE=cpu`
for anything >0.6B on this 8 GB-VRAM box. Hard pins: `onnx==1.19.0` in **both**
envs (≥1.20 removes `onnx.version` and breaks the converter *and* AIMET export),
`numpy<2` in the build env, `aimet-torch==2.36.0` (three of its bugs are patched
inside our scripts: LoRA attr shim, >2 GB protobuf `ByteSize` crash, unreliable
cross-variant `load_encodings`).

---

## 10. Document map

| Document | Status | Use it for |
|---|---|---|
| **`docs/REFERENCE.md`** (this file) | **current truth** | start here |
| `CLAUDE.md` | current | terse operating rules for agents |
| `docs/BUILD_GUIDE.md` | current | step-by-step recipes, per-variant commands, troubleshooting |
| `docs/NOTES-genie-io.md` | current, SDK-cited | the Genie/qualla contract — read before touching graph I/O |
| `docs/NOTES-vit-htp-config.md` | current | why graph names must appear in the backend config |
| `docs/LOCAL_ENV.md` | current + historical log | environment provenance, AIMET workarounds, progress log. Its ctx-bin sizes are marked stale. |
| `docs/SDK_INVENTORY.md` | current | what's in the QAIRT drop and what runs locally |
| `SA8797P_Deployment_Status_Summary.md` | **partly superseded** | the remote team's hardware/GVM characterization (§1, §3.2) is still the best there is. Its §2–4 carry corrections 6, 7, 8, 9, 12 above. |
| `reports/*-fuseqkvgu-*.md` | historical | v1 failure data. Root cause wrong — correction 1. |
| `reports/*-v2-lade-vs-baseline-*.md` | historical | first working bundles. Two conclusions wrong — corrections 2, 3. |
| `reports/*-ladekv-*.md` | historical | first working LADE. Acceptance figure wrong — correction 4. |
| `reports/*-qh-ladekv-*.md` | historical | the `--quant-head` experiment; §10 of it carries the DLC verification |
| `docs/superpowers/plans/` | working plans | VL-4B stage 1 (ViT) and stage 2 (text tower) |

**Convention for reports:** they are transcriptions of what a device run reported
at a point in time. They are never edited to correct errors — a banner at the top
points at the correction, and the analysis stays quarantined in its own marked
section. That keeps provenance intact. This file is where corrected facts live.

The source material is screen photographs taken by the remote tester, which are
**deleted once transcribed** — the report is thereafter the only record, so
transcription notes carry anything a re-reading of the image would otherwise
settle (uncertain glyphs, frame overlaps, gaps between frames).
