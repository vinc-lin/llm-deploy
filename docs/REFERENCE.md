# SA8797P LLM Deployment — Consolidated Reference

*Current truth as of 2026-08-13. Supersedes conflicting statements anywhere else
in this repo. Every number here is either device-measured, tool-measured on this
machine, or cited to SDK source — claims that could not be verified are marked
as such rather than repeated. 2026-08-13: reconciled against the device team's
measured-only characterization,
`docs/SA8797P_HTP_v81_Hardware_and_Deployment_Quantization_Reference_EN.md`
("the HTP doc" below) — corrections #14–18 and open questions §8.8–8.10 came
out of that reconciliation.*

**Read this first.** Then `docs/BUILD_GUIDE.md` for step-by-step recipes and
`docs/NOTES-genie-io.md` for the line-cited Genie contract.

---

## 0. Status board

| | |
|---|---|
| **Best sustained decode, 0.6B** | **44.707 ± 0.030 tok/s** (22.37 ms/step), basic mode, `qwen3_06b_w8a16_gqafix_ladekv`, 2026-08-15 (`DEVICE_MEASUREMENT_REPORT_2026-08-15.md`). Basic beats LADE post-fix (31.342); **LADE is parked** (§6.8) |
| Non-speculative AR-1 decode, pre-fix | **6.84 tok/s** (146.3 ms/step) — the honest like-for-like baseline, measured 2026-08-15 on the pre-fix `ladekv` bin. The GQA fix is therefore **6.54×** |
| ⚠️ **11.72 tok/s is NOT an AR-1 rate** | It is a **phase blend** on a bertcache bundle — see correction #22 and §6.9. Every comparison that used it as a decode rate produced a phantom finding (#23, #24, #25). Gate: `scripts/validate/lint_bundle_topology.py` |
| Bertcache early phase (topology A only) | ~23.8 tok/s for the first ~117 tokens, then falls to AR-1. **This is the other half of the blend** |
| Output correctness | ✅ since v2 (2026-08-10) |
| Quantization | W8A16 (INT8 per-channel weights, FP16 activations) — the only recipe that works |
| Ctx-bin, 0.6B | ~1.09 GB, 2 or 3 weight-shared graphs |
| Device access | none from this machine — build + numerics only; device runs are done by a remote tester |
| Blocked on hardware | tok/s, DDR bandwidth, VTCM behavior, perf profiles, anything GVM |

**What moves the needle next (revised 2026-08-16).** The GQA replication fix
shipped and measured 6.54× (6.84 → 44.707 tok/s), so the picture has changed
twice over:

- **"Decode is 100% DDR-bound" is no longer established.** At the post-fix
  operating point the byte and compute models are *numerically degenerate*:
  961 MB ÷ 22.37 ms = 43.0 GB/s (88% of the 49 GB/s streaming ceiling), **and**
  88.2M residual cycles ÷ 4 HVX ≈ 22.06 ms (1.4% off). Neither is confirmed.
  The discriminating experiment is a pair of arms whose predicted orderings
  differ — W8 head (+19.3% byte / +3.6% compute) versus CL=512 (+8.3% / +34.7%)
  — plus `hvx_threads: 8`, which changes zero bytes by construction. See
  `docs/MAX_TPS_QWEN3_0.6B_V4.md` §2.
- **LADE is parked** (§6.8). Post-fix break-even rose to 2.30 accepted
  tokens/call against ~1.6 measured, because the fix sped decode 6.54× but
  verify32 only 3.50×.
- **Multi-token decoding is still the right long-run lever** if the regime turns
  out to be byte-bound, and a learned draft head (`eaglet`, `spd`) now has a
  concrete bar to clear: **+43% acceptance just to reach parity** (§6.8).
- **QKV+Gate-Up fusion** remains untested *in combination with* the GQA fix —
  every fused build predates it (correction #15).

---

## 1. Hardware and runtime reality

Source: the HTP doc (device team, measured-only, 2026-08-12), which quotes the
official silicon spec and supersedes the 2026-08-09 remote summary's hardware
picture (correction #14).

| Attribute | Value |
|---|---|
| SoC | SA8797P (nordy / Gen5 / Snapdragon Ride Flex) |
| HTP | Hexagon v81 × 4 on silicon. Each HTP = 12-thread Q6 scalar core + **8× HVX** (1024-bit SIMD) + 1× HMX (INT8/16 + FP16 matrix sub-units) + **16 MB VTCM** + L2 |
| **What the Android GVM guest actually drives** | **one full HTP** — 8 HVX threads (fixed across all workloads), 16 MB VTCM (= exactly one HTP's), unsigned PD. `qnn-platform-validator` reports 4 cores *visible*; Genie creates a single-core device. Hypervisor allocation mask unknown (§8.6). |
| VTCM ceiling | 16 MB — `vtcm_mb: 24` compiles offline but is rejected at runtime (`0x138d`); `pd_session: "signed"` with an unsigned skel silently falls back to unsigned |
| Runtime | QAIRT 2.48.40.260702 · QNN API v2.37.0 · libGenie 1.19.0 |
| DVFS | works via Genie `backend.extensions` JSON, 4 tiers, 1.95× swing. `qnn-net-run --perf_profile` is a no-op. Always use `llm_decode_burst`. |
| Burst BW, one large contiguous matmul | 49–67 GB/s depending on timing definition (33.6 MB weight in 500–684 µs, excl-wait vs total) |
| Effective BW, real LLM decode — **pre-GQA-fix** | **~6–10 GB/s** — three ways: our converter DDR estimate (~957 MB/token), the device team's back-calculation (~924 MB/token × 7.4 tok/s), and the corrected 08-15 baseline (~1,489 MB ÷ 146.3 ms = **10.2 GB/s**) |
| Effective BW, real LLM decode — **post-GQA-fix** | **~43 GB/s** (961 MB ÷ 22.37 ms) — **88% of the 49 GB/s excl-wait ceiling.** The fix bought both fewer bytes and a far better access pattern |
| Peak FP16 compute (M ≥ 512 matmul) | ~2.7–2.8 TOPS — far below nominal; why is an open FAE question |
| FastRPC per-call | ~220 µs round-trip (hypervisor-mediated `hfastrpc`), ~50–60 µs fixed submit/sync, 30–60 µs steady-state inter-op wait |
| Graph-switch latency | 79–93 ms (prefill ↔ decode) |
| Prefill throughput | 266 tok/s at 12 prompt tokens → **1100+ tok/s** at 53+ (fixed overhead amortizes) |
| Init | cold (first run after reconnect) 1.8–2.0 s · warm 786–820 ms |
| Not available | zero-copy sharedbuf · DSP hardware queue · clock visibility · multi-HTP under Genie · KV quantization via config (§4.1) |

**The central performance fact — largely resolved 2026-08-15.** The ~7–10×
collapse from 49–67 → 7 GB/s was attributed here to *access-pattern
fragmentation* (28 layers of small MatMuls + KV traffic + per-op sync: ~220 µs
per RPC dispatch, 30–60 µs steady-state inter-op wait). **The dominant term was
in fact the GQA KV replication**, and removing it recovered the effective rate
from 10.2 to 43.0 GB/s — 88% of the contiguous-read ceiling — without touching
op count or dispatch. So fragmentation was real but second-order; the 56
replication ops were moving 528 MB/step *and* thrashing the stream.

What survives: fewer, larger ops still help twice (contiguous reads stream
nearer the ceiling, and every fused-away op refunds its dispatch), so fusion is
still worth testing — but note fusion's +15% was measured *pre-fix* at ~10 GB/s
effective, so it may have been recovering the same access-pattern loss the GQA
fix has now collected. That is an open A/B, not an inherited result.

**Compute is no longer known to be off the critical path.** See §0 — at 22.37
ms/step the byte and compute models both fit, and the pre-fix step is where they
diverge (the compute model under-predicts it by 1.67×, which is the signal that
the replication ops specifically were the memory-bound part).

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
for. (Historically blamed for the remote team's "error 5005"; the HTP doc §9
instead ties 5005 to NOT_SUPPORTED triggers — `vtcm_mb > 16`, multi-core — so
treat that attribution as unconfirmed. The mismatch itself is still a hard
load failure.)

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

### 3.6 An AR==CL prefill graph is FATAL in a split tower (2026-08-14)

Unsplit, an `attention_mask [1,AR,AR]` prefill registers `ctx_size == AR`
(bertcache) and is silently never selected — slow, not broken (§2.2,
`NOTES-genie-pipeline.md` probe C). **Split, the same graph refuses to load.**

In a 2-shard tower the lm_head is in the last shard, so shard 0's prefill emits
no `logits` → classifies `DECODER_PREFILL` (`nsp-graph.cpp:247-249`) → its
expected CL is rewritten to the cache-group max (`nsp-model.cpp:604-605`) →
`validateModel` rejects the `[1,AR,AR]` mask (`:858`) → `Failed to create the
Genie Node (-1)` + SIGSEGV. This killed the Qwen3-VL-4B e2e bring-up on
2026-08-14 (Expected `[1,128,2176]`, Found `[1,128,128]`, twice — once per
shard).

Splitting is mandatory ≳2B (`NOTES-genie-splits.md`), so **any ≳2B model must
ship a past-KV prefill** (`[1,AR,CL]`, `CL>AR`) or no prefill graph at all. The
0.6B topology-A pattern does not transfer.

Config-level escape hatch, if you are holding a ctx-bin you cannot rebuild:
`"execute-select-graphs": [...]` drops the offending graphs before validation
(`nsp-model.cpp:314-318`), optionally with `"load-select-graphs": true` to skip
deserializing them. Undocumented; **unverified on device**; full semantics and
the source chain are in `NOTES-genie-io.md` § "Split prefill is fatal at load".

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
| all K/V-projection outputs | **a build choice, not a contract.** §3.2 requires KV quant params be *byte-identical across graphs*, not that they be FP16. Quantizing them is reachable — see the note after §4.1 |

Weight encodings are clipped to the ±0x7f7f-safe range. **HTP packed-pair
saturation is not modeled by quantsim**, so "passes quantsim, garbage on silicon"
is this bug's signature.

Fusion variants: Gate-Up keeps the fused `gate_up` output FP16 (requantized at
`down_proj`); QKV grafts the donor `q_proj` INT16 encoding onto the Q split with
K/V splits FP16, done at the **encodings** level by `qkv_surgery.py` (28/28).

**Qwen3-VL ViT (W8A16, stage 3) inherits the same fused-QKV rule.** aimet-torch
only quantizes *module* outputs, so the whole attention body — rotary, both
MatMuls, Softmax, `attn.proj` — is functional code, carries no encodings, and
converts to FP16 (exactly why the text tower's K/V projections are FP16). The
ViT's QKV is one fused `nn.Linear`, so its output *is* a module output and *is*
encoded, and V reaches `attn @ v` through nothing but Reshape/Transpose/Split/
Squeeze — all of which merely inherit dtype. The converter dequantizes the Q/K
path (their `.float()` Casts force a fallback) but leaves V as `uFxp_16`, then
asks HTP for a `FLOAT_16 x UFIXED_16 -> FLOAT_16` MatMul that has no kernel:
`validateOpConfig failed 3110` / `Failed to validate op /blocks.0/attn/MatMul_1
with error 0xc26`, and ctx-bin generation dies at ComposeGraphs. No converter
flag inserts the missing dequantize. `vit_build_quant.sh` step 1a drops that one
activation encoding per block (24 of 243) — `qkv_surgery.py`'s step 1 verbatim,
without the donor graft, which would buy nothing here. The QKV **weights** stay
INT8 per-channel; the result is a strictly *higher*-precision graph than the
quantsim that was measured.

### 4.1 Dead ends — do not re-run these

| Approach | Result | Real reason |
|---|---|---|
| **W8A8** | garbage, all variants (v15–v19; v19 also slow — 4.32 tok/s, 730 MB spill) | v81 MatMul supports only per-tensor UINT8 asymmetric activations, which clip heavy-tailed LLM activations. Per-channel activations unsupported. |
| **W4A16, any size** | **0/4** on the argmax gate at 0.6B — and INT4 cannot execute anyway | Two independent kills. Accuracy: all three recipes fail at 0.6B (per-channel INT4, LPBQ block-64, LPBQ+SeqMSE; `max\|Δlogits\|` 16–25 vs W8A16's 1.3–1.7). Kernels: `htp_v2.json` has **zero INT4 input *datatypes*** for MatMul/FC (SDK 2.43 & 2.48) and qairt-converter folds s4 weights back to f16 (HTP doc §5.3/§10.1). ⚠️ It *does* declare 4-bit and 2-bit **block-quantization encodings** (`BW_AXIS_SCALE_OFFSET`) — see correction #26 and `NOTES-htp-kernel-table.md` before concluding this was mis-diagnosed. `--lpbq`/`--seq-mse` stay in the script only for a future SDK that ships the kernels. |
| **KV INT8 via Genie `kv-quantization: true`** | no effect on HTP | The flag exists only in the QnnGenAiTransformer **CPU** backend (HTP doc §5.4). **The *config* route is dead; the native route is not** — see the note below this table. |
| **`--quant-head` (W8 lm_head) under LADE** | **−14% tok/s** | Costs ~10% n-gram acceptance; the DDR saving does not survive spec-decode amortization (§6.3). |
| **`sparse_weights_compression=1`** | 0 bytes saved | model isn't sparse |
| **2-core ctx-bin** | error 5005, or 3.96 tok/s — slower | Genie 1.19 creates a single-core device internally, no JSON override |
| **Multiple Genie instances** | 2 × 4.0 tok/s = 8 total | linear BW split — confirms decode is 100% DDR-bound |
| **Executing any W8A16 graph on x86 (as a device-free numerics gate)** | impossible on this SDK | `libQnnCpu` has no 16-bit fixed-point kernels: a 4×8 single-Gemm probe composes with `uFxp_8` activations (per-tensor *and* per-channel weights) and fails `OpConfig validation … for FullyConnected` with `uFxp_16`, regardless of `--target_backend`. `libQnnGpu` refuses to initialise off-target ("TuningMode must be enabled on x86_64-linux-clang"); `libQnnHtpQemu` cannot create a context; a ctx-bin is backend-final, so `--retrieve_context` on CPU gives "Context de-serialization failed". Device-free gates must therefore run an **FP32 sibling** DLC (`parity_vit_dlc.py`, `parity_vit_quant.py`). |

**Native KV quantization is reachable — audited 2026-08-16.** The kernel table
(`docs/NOTES-htp-kernel-table.md`, read out of the converter's own
`htp_v2.json`) settles what was previously an FAE question:

- v81 has **exactly one** mixed float×fixed MatMul kernel: **`f16 × sfxp8 →
  f16`**, and its `input_quant_params` are *unconstrained* on input 1, so a
  per-tensor-quantized **dynamic** tensor — which is what a KV cache is — is
  legal there.
- Therefore the KV cache must be **signed** INT8. There is no `f16 × ufxp8`
  kernel, and AIMET defaults activations to *unsigned*, so this needs a
  symmetric signed 8-bit quantizer on the K/V-projection outputs specifically.
- **An INT16 KV cache is NOT reachable** while Q and the softmax output stay
  FP16 — there is no `f16 × ufxp16` or `f16 × sfxp16` kernel. This is the same
  gap that broke the Qwen3-VL ViT's V path (`FLOAT_16 x UFIXED_16 ->
  FLOAT_16`, §4 above), and it means "INT16 first as the safe fallback" is
  **backwards**: INT16 is the harder path, not the easier one.
- Supporting signal that the runtime anticipates this: `validateModel` Check 4
  already compares `(scale, offset)` byte-for-byte on KV tensors across graphs,
  and `checkShape` carries a per-tensor `bitwidth` — neither would exist for a
  KV cache that could only ever be FP16.

Remaining risk is accuracy and whether the converter *selects* the mixed kernel
rather than inserting a dequantize. Not yet built.

**Removed from this table 2026-08-13: QKV/Gate-Up fusion.** Our "no tok/s gain
(6.27–6.5 ≈ baseline)" verdict compared against a v1-era fused build that was
emitting garbage output, so it never actually measured fusion. The device
team's *working* fused build measures **8.98 vs 7.79 unfused (+15%)**, decode
DDR read 880 vs ~960 MB (§6.6, correction #15). Re-test on topology B;
first-class candidate for the 4B build.

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
| **Decode topology** | `lint_bundle_topology.py <ctxbin>` | `pure` for anything whose tok/s will be compared against another bundle; `BLENDED` bins carry an AR==CL graph and report a two-phase rate (§6.9) |
| Quantized head | `qairt-dlc-info -i <dlc> \| grep -oP 'lm_head\.weight \(data type: \K[A-Za-z_0-9]+'` | `sFxp_8` with `--quant-head`, else `Float_16`. ⚠️ a plain `grep lm_head.weight` reports the **activation** dtype and false-FAILs a correct build — `BUILD_GUIDE.md` §5.7 |
| Ctx-bin | `qnn-context-binary-utility --json_file` | all graphs listed, logits dims per §3.1, ~1.09 GB for 0.6B |
| **ViT fixed-point I/O** | `vit_build_quant.sh` step 3 (fails the build itself) | `pixel_values` + all 4 outputs `QNN_DATATYPE_UFIXED_POINT_16`, scale/offset byte-equal to `model.encodings`, one graph named `vit`, O=3 / vtcm 16 / 4 HVX read back out of the binary |
| **ViT quant numerics** | `parity_vit_quant.py` | min cos ≥ 0.99 on all four outputs (measured 0.9975 / 0.9998 / 0.9986 / 0.9977) |
| **Qwen3-VL e2e** (image → ViT → splice → text tower) | `parity_e2e_vl.py` (no `--chains` filter, or the mutation checks are skipped) | chain0/chain1 token-identical vs `hf.generate` (measured **20/20**); chain2 ≥ 75% step agreement |

⚠ **The 20/20 is a real-deepstack number and the shipped bundle does not
reproduce it.** All three gated chains feed *real* deepstack; the configuration
that actually ships — `tierA-zero-deep` — is **not gated**
(`parity_e2e_vl.py:16-35`), because a stock Genie pipeline has no deepstack path
and `initializeUnconnectedInputs` zeroes those three inputs
(`NOTES-genie-pipeline.md` §A). Zeroing costs phrasing, not image understanding:
HF's exact sentence vs the Tier-A sentence are both correct descriptions
(`DEVICE_TEST_qwen3vl_e2e.md`). So on device the bar is **semantic, not
token-exact** — do not quote 20/20 as the expected device behaviour, and do not
read a wording difference on device as a regression.

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

### 6.6 Device-team independent measurements (HTP doc, 2026-08-12)

Their own builds, not our HF bundles — their unfused ctx-bin is **1.01 GB vs
our 1.087 GB**, so the artifacts differ:

| Configuration (theirs) | tok/s | Notes |
|---|---|---|
| Unfused W8A16, graph-switching + mmap | **7.79–7.80** | init 786 ms, TTFT 805 ms, 128 tokens generated, coherent output |
| QKV + Gate-Up fused W8A16, vtcm 16 | **8.98 (+15%)** | coherent output; decode DDR read 880 MB; ctx-bin 1.086–1.09 GB |
| Unfused W8A8 v19 | 4.32 | garbled — W8A8 stays dead |

Two things do not reconcile with our measurements and stay open:

- **Their unfused AR-1 decode is ~20% faster than our v2 bundle** (7.8 vs 6.5).
  Not runtime config — their §8.2/§8.3 configs match our shipped
  `configs/genie_dialog_qwen3_0.6b.json` + `htp_backend_ext_config.json`
  field-for-field (`poll`, `cpu-mask 0xe0`, `n-threads 3`,
  `rpc_polling_time 9999`, `llm_decode_burst`). The difference is in the build
  (§8.8).
- **Their fused build gains +15% where our A/B showed nothing** — but our
  "fused" data point was a v1-era garbage-output build, so theirs is the only
  valid fused-vs-unfused measurement in existence (correction #15).

### 6.8 Post-fix decode, and why LADE is parked (2026-08-15)

Measured on `qwen3_06b_w8a16_gqafix_ladekv`, warm, greedy, 56-token technical
prompt — all three arms on **one binary**, so these are directly comparable:

| Arm | tok/s | per-step / per-call | Notes |
|---|---:|---:|---|
| basic (AR-1 decode) | **44.707 ± 0.030** | 22.37 ms | the baseline |
| LADE | 31.342 ± 0.090 | 51.4 ms/call | acceptance 1.61 tok/iter |
| pre-fix basic, same topology | 6.836 ± 0.000 | 146.3 ms | the honest 6.54× control |

The GQA fix sped **decode 6.54×** but **verify32 only 3.50×** (180 → 51.4 ms),
because replication cost is AR-independent and therefore dominated the AR-1
graph far more than the AR-32 one. So speculation's break-even moved:

```
break-even acceptance = verify call latency / decode step = 51.4 / 22.37 = 2.30 tokens/call
measured acceptance                                                      ≈ 1.61 (technical prompt)
                                                                         ≈ 1.94 (2026-08-11 simple prompt)
```

**LADE did not get worse; decode got better faster.** Keep `verify32` in the
bins (weight-shared, ~0 marginal cost, preserves optionality) and park the
workstream. A learned draft head (`eaglet`/`spd`, §6.3) needs **+43% acceptance
to reach parity** — that is the bar, and it is a much sharper target than
"optimize acceptance".

### 6.9 Topology A blends two decode rates — read this before quoting any tok/s

In topology A (bertcache, AR==CL=128) Genie keeps generating **through the
prefill graph**, re-processing the whole 128-wide window once per token, until
the KV cache passes 128 (`kvmanager.cpp:421-429`). Only then does AR-1 take
over. A tok/s figure measured over a fixed token budget on such a bundle is a
**time-weighted blend of two rates**, and it flatters itself because bertcache
steps attend over AR positions rather than CL.

Worked, from the 2026-08-13 Test 1 basic arm (`qwen3_06b_w8a16_local`, 56-token
prompt, 128 generated tokens, 10,837 ms):

```
bertcache phase: generated tokens 1..72   (KV 56 -> 128), 40.1 ms each  = 2,887 ms
AR-1 phase     : generated tokens 73..128 (56 tokens)                   = 7,950 ms
                                                    7,950 / 56          =   142.0 ms/step
```

**142.0 ms against 146.3 ms measured independently on 2026-08-15** on the
pre-fix `ladekv` bin — two numbers from different days, different bundles,
different methods, agreeing to 3%. The blend model closes.

`scripts/validate/lint_bundle_topology.py` classifies any ctx-bin as **pure** or
**blended** from its own graph shapes and derives the same 72/56 split. Run it
before comparing any two decode rates. Six of the eight 2026-08-14 `gqafix_*`
bundles are blended, which is why every variant in that drop had to be rebuilt
on the past-KV topology before it could be measured.

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
| 8 | (rewritten 2026-08-13 — this entry itself was wrong) W4A16 fails **solely on accuracy**; the "no INT4 kernel" claim was dismissed for bad sourcing | this file until 2026-08-13; remote summary §3.1 argued it badly | The kernel claim is real and now directly measured: **zero** INT4 MatMul/FC entries in `htp_v2.json` (SDK 2.43 & 2.48), and qairt-converter folds s4 weights to f16 (HTP doc §5.3/§10.1). Our 0/4 accuracy result at 0.6B also stands. W4A16 is dead on both grounds (§4.1). |
| 9 | ctx-bins are **1.5 GB** | LOCAL_ENV, remote summary §2.1 | ~**1.09 GB** for every current 0.6B build — measured 2026-08-12. See open question §8.2. |
| 10 | Graph names are **cosmetic** | BUILD_GUIDE §3.4, NOTES-genie-io | Cosmetic to Genie's picker only. Load-bearing for the HTP backend config (§3.3). |
| 11 | Disk: **flat 6 GB `disk_guard`, compact the VHD to recover** | BUILD_GUIDE §8 | `disk_guard <need_gb>` must be **sized to the step** (a 4B export writes 8.6 GB). No compaction needed — the vhdx is sparse and `/` is mounted `discard`. |
| 12 | (revised 2026-08-13) Decode throughput **7.4–8.2 tok/s** — dismissed here as "never reproduced" | remote summary §2.1; the dismissal was this file's until 2026-08-13 | Reproduced by the device team on **their own** unfused builds: 7.79–7.80 tok/s, runtime configs identical to ours (HTP doc §3.2). Our bundles still measure 6.3–6.5 — the ~20% delta is build-side and unexplained (§8.8). |
| 13 | qh cold start is **+134% vs ladekv** (1070 vs ~458 ms) | qh report §1, §4 | Units mismatch — 1070 ms is init→first-logits, 458 ms is TTFT measured from prefill start. Like-for-like the ladekv build takes **1247 ms** to first logits, so qh is ~177 ms *faster*, not 134% slower (§6.1). |
| 14 | The GVM guest gets **2 of 4 NSPs** (~8 MB VTCM each, "4 HVX threads per NSP") | remote summary §1.1–1.2; this file §1 until 2026-08-13 | Official spec: each HTP has **8 HVX units and 16 MB VTCM**. The observed 8 HVX threads + 16 MB VTCM is **one full HTP**, not two halves. 4 cores are visible to `qnn-platform-validator`; Genie drives 1. Theoretical multi-HTP upside is ×4, not ×2; the allocation mask is unknown (§8.6). |
| 15 | QKV/Gate-Up fusion **buys no tok/s** — dead end | this file §4.1 until 2026-08-13; BUILD_GUIDE §5.3 | The verdict was measured against a v1-era fused build that was emitting garbage. The device team's working fused build: **8.98 vs 7.79 unfused (+15%)**, decode DDR 880 vs ~960 MB (§6.6). Fusion is back on the table — re-test on topology B. |
| 16 | **LADE is broken on this platform** (SIGSEGV @ PC 0x4c2d58; "needs a Qualcomm fix") | HTP doc §7/§9/§12.6 | Stale by a day: LADE ran **10.8 tok/s on 2026-08-11** (9.3 for qh, 08-12) on the same device. The crash is config-side — the HTP doc's own §8.3 `graph_names` lists only `["prefill","decode"]`, and an unlisted verify graph null-pointers on the first speculation step (§3.3); the AR==CL and 1-token-prompt traps also apply (§3.4). |
| 17 | **`lm_head` must remain FP16** (else error 0xc26) | HTP doc §5.1 | 0xc26 is the **embedding Gather** restriction only. An INT8 (`sFxp_8`) lm_head builds, loads, and runs with unchanged quality — verified in the qh build (§6.4). We keep it FP16 by default for the LADE-acceptance reason, not supportability. |
| 18 | Unfused W8A16 at vtcm 16 spills **1.49 GB** at build time | HTP doc §4.2 ("older optctx2" row) | *Probable, not proven:* the graph-names-mismatch artifact — an unlisted graph gets 4 MB VTCM, and that exact failure measured 1.446 GB of spill here (`docs/NOTES-vit-htp-config.md`). Every correctly-configured vtcm-16 build, theirs and ours, spills ~0. |
| 19 | An oversized ctx-bin means **weight sharing is disabled** (check `weight_sharing_enabled`) | this file §8.2; MAX_TPS §2 A.4 as originally written | Incomplete. Weight sharing can be **on and working** and the bin still inflate, because dedup needs the graphs' exported **weights to be byte-identical**, and a *calibrated* export is not byte-identical to an `--export-decode` export of the same model — see §6.7. Check which export path each graph came from before touching the config. |
| 20 | An AR==CL prefill graph is **harmless dead weight — silently skipped, correctness unaffected** | `NOTES-genie-pipeline.md` probe C; §2.2/§3.3 as read until 2026-08-14 | True **unsplit** only. In a split tower shard 0's prefill has no logits, classifies `DECODER_PREFILL`, and its expected CL is rewritten to the cache-group max — the graph then **fails validation and the node never loads** (§3.6). Probe C also mis-stated that our prefill classifies `DEFAULT`: that holds for `prefill_1`, not `prefill_0`. |
| 21 | The Qwen3-VL e2e gate's **20/20 token-identical** describes what the shipped bundle does | 2026-08-14 status report §3–§4 | The three gated chains run *real* deepstack; the configuration that ships (`tierA-zero-deep`) is explicitly **not gated**. On device the bar is semantic, not token-exact (§5). The report's own "HF exactness drops 0→20/20" is the same point, written backwards. |
| **22** | **11.72 tok/s is the best sustained AR-1 decode rate** | this file §0 (twice) and §6.1; `MAX_TPS_V2` §0; 08-13 report Test 1/Test 3 | **It is a phase blend**, not a decode rate: ~72 bertcache steps at ~40 ms plus ~56 AR-1 steps at ~142 ms (§6.9). The honest pre-fix AR-1 rate is **6.84 tok/s**, measured 2026-08-15. §6.1's own rule — "any tok/s number for topology A is meaningless without saying which phase" — is what this violated. Gated now by `lint_bundle_topology.py`. |
| **23** | **LADE is −22% vs basic on the technical prompt** | 08-13 report Test 1 + §6.2; this file §0 | Compares LADE on `ladekv` against **blended** basic on `local`. Like-for-like on one bin and one prompt, **pre-fix LADE was 9.18 vs 6.84 = +34%**. LADE's loss is real but is a *post-fix* phenomenon (44.707 vs 31.342), first measurable 2026-08-15, and its cause is the break-even shift in §6.8 — not prompt-dependence. |
| **24** | **A ~75% build gap / 3-graph penalty exists between `local` and `ladekv`** | 08-13 report §6.3 + rec 2; `MAX_TPS_V2` §2.2; `MAX_TPS_V3` §7 item 4 ("64 ms unexplained term") | Same artifact as #22. The two decode graphs are structurally identical (CL=1152, spill 0, same vtcm/O/hvx) and share weights within 4 MB. **No graph-count or graph-switching penalty is in evidence, and the 64 ms term does not exist.** |
| **25** | **Our builds are +51% faster than the device team's 7.79 ("RESOLVED BACKWARDS")** | this file §8.8 | Blended-vs-AR-1 again. Corrected, our pre-fix AR-1 is **6.84** — roughly 12% *slower* than their 7.79, i.e. closer to the original premise than to its "resolution". Their build's provenance was never audited, so §8.8 is **reopened as an open question**, not resolved in either direction. |
| **26** | W4A16 is dead partly because there are **zero INT4 MatMul/FC entries** in `htp_v2.json` | this file §4.1 and correction #8 | Imprecise, though the conclusion holds. There are zero INT4 **input datatypes** (every entry is ≥8-bit) — that is the load-bearing half. But **31 MatMul / 21 FullyConnected `input_quant_params` entries declare `Bitwidth: "4"`** (and 15/13 declare `"2"`) under `BW_AXIS_SCALE_OFFSET`, i.e. LPBQ block-quantized sub-byte weights in an 8-bit container. Restated so nobody greps the file, finds `"4"`, and wrongly reopens the dead end. W4A16 stays dead on s4→f16 folding **and** 0/4 accuracy. See `docs/NOTES-htp-kernel-table.md`. |

---

## 6.7 Calibrated vs `--export-decode` weights are not byte-identical (2026-08-13)

Measured on the fused Qwen3-0.6B build, comparing the ONNX initializers the
converter actually consumes:

| Pair | `layers.0.mlp.gate_up_proj.weight` |
|---|---|
| calibrated prefill vs `--export-decode` decode | **3,505,204 / 6,291,456 elements differ** (~56%), ~1 quantization step, max rel 4.0e-02 |
| `--export-decode` decode vs `--export-decode` verify32 | **bit-identical** |
| prefill built **with** `--eval` vs **without** | **bit-identical** |

The calibration path runs `clip_weights_to_7f7f(sim)` after
`compute_encodings`; the `--export-decode` path calls `load_encodings` and
never clips. `--eval` was suspected and is **exonerated** — it is a pure
read-only evaluation and changes no exported byte.

Consequence for ctx-bin size, measured the same day at 0.6B fused:

| ctx-bin | Graphs | Size |
|---|---|---|
| `fuseqkvgu` (intermediate) | calibrated prefill + decode | 1,524,551,680 B |
| `fuseqkvgu-lade` (intermediate) | calibrated prefill + decode + verify32 | 1,536,610,304 B |
| **`fuseqkvgu-ladekv` (shipped)** | past-KV prefill + decode + verify32, **all `--export-decode`** | **1,102,467,072 B** |

So the LADE/LADEKV bundles dedup perfectly *because* all three of their graphs
come from `--export-decode` adopting one prefill's encodings. The two-graph
`_local` bundles mix the two paths and are the ones to watch. **The oversized
intermediates above are expected and disposable — do not "fix" them.**

Still unexplained: the 2026-08-10 fused two-graph bin measured 1,084,137,472 B
and the non-fused one 1,087,074,304 B, i.e. both deduped despite mixing the two
export paths. Their quant dirs are gone, so the difference cannot be
reconstructed. Treat two-graph bin sizes as uninformative until this is closed.

---

## 8. Open questions

*Outstanding: `docs/DEVICE_MEASUREMENT_REQUEST_2026-08-13.md` asks the device
team for run-now measurements covering §8.3, §8.8, §8.9, and the op-level
decode profile (where the ~155 ms/step actually goes).*

### 8.1 Where do the ~139 MB go? *(qh, §6.4)* — **independently reproduced 2026-08-16**
The DLC shrinks 151 MB, the ctx-bin only 12.5 MB. Hypothesis: HTP re-materializes
the INT8 head as 16-bit at prepare time because the surrounding activations are
FP16. If true, `--quant-head` cannot save DDR on this backend at all, and the
whole variant is moot. Needs on-device DDR counters or a prepare-time weight dump.

**Reproduced on a second, independent build** (`gqafix_qh_ladekv`, 3-graph
past-KV topology, 2026-08-16 — a different topology from the 08-12 build):

| | 2026-08-12 `qh_ladekv` | 2026-08-16 `gqafix_qh_ladekv` |
|---|---:|---:|
| DLC | −151.3 MB | −151.3 MB per graph |
| converter decode `read_total_bytes` | not recorded | 961,130,496 → **815,028,224** (−146.1 MB) |
| **ctx-bin** | **−12.5 MB** | **−8.4 MB** (1,086,570,496 → 1,078,185,984) |

So the effect is not an artifact of the 08-12 build. The head is verifiably
`sFxp_8` in all three DLCs, the converter's DDR model credits the full saving,
and the shipped context blob still does not shrink. **Treat "the qh arm measures
≈0% on device" as the leading hypothesis, not the null result** — and note the
converter's `read_total_bytes` is a graph-level estimate that evidently does not
describe what the prepared context streams. See `docs/VARIANT_PREDICTIONS.md`
§2b.

### 8.2 What actually changed between v1 (1.52 GB) and v2 (1.09 GB)?
A ~430 MB shrink is far too large for the all-position-logits fix, which should
have made the binary marginally *bigger*. Best candidate: **weight sharing became
effective**. Supporting evidence found today — the qh intermediates still show the
unshared signature: `qwen3-0.6b-w8a16qh_ctx.bin` is **1.84 GB** for 2 graphs and
`qwen3-0.6b-w8a16qh-lade_ctx.bin` is **2.16 GB** for 3, against 1.09 GB when
sharing works. Worth confirming, because it means a silently-unshared build is
still possible today and the only symptom is file size.

### 8.3 Does `qh` help in AR-1 basic mode? — now the **primary discriminating experiment**
The one configuration where the acceptance penalty does not apply. Post-fix this
question got much more load-bearing: it is one of the two arms that separates the
byte model from the compute model, which predict **+19.3%** and **+3.6%**
respectively (`MAX_TPS_QWEN3_0.6B_V4.md` §2.3). Still contingent on §8.1 — if the
151 MB never reaches the device, the answer is no, and *that is itself the byte
model's refutation on this arm*.
⚠️ The existing HF `gqafix_qh` bundle **cannot answer this**: it is a bertcache
2-graph bin, so its basic-mode rate is blended (§6.9). It must be rebuilt on the
3-graph past-KV topology first.

### 8.4 `soc_model: 0` at `O=3`
We build with `soc_model` unspecified. The SDK maps SA8797 → `soc_id 72`, and
Qualcomm's HTP docs state that specifying it at O=3 "could turn on additional
[algorithms] which may further improve inference performance". Real performance
possibly left on the table, for **all** builds, not just the ViT. Needs a measured
A/B before the next device run. The HTP doc's §8.4 "verified working" build sets
`soc_id: 72` **and** `soc_model: 72` explicitly — one more reason to just set
them and A/B once.

### 8.5 AR-32 ↔ AR-1 reshape churn
After every dialog-level KV update the cache is reshaped to the smallest
registered AR. In LADE that means an AR-32↔AR-1 reshape every iteration. If device
KPIs ever implicate it, the lever is a LADE-only ctx-bin with no AR-1 graph — the
graph is never invoked anyway (§6.1).

### 8.6 Environmental (revised 2026-08-13 per the HTP doc)
Multi-HTP for the guest — 4 cores are *visible* to `qnn-platform-validator`, but
Genie creates a single-core device and the observed 8 HVX + 16 MB VTCM is
exactly **one** HTP, so the theoretical upside is ×4, not ×2; the hypervisor
allocation mask is unknown (FAE) · whether signed PD grants more VTCM or
resources (probe so far: `pd_session: "signed"` with an unsigned skel silently
falls back, no error, no change) · whether DLBC is active and what it buys for
LLM traffic (FAE) · `DDR_PERF_MODE` (C API option 7, not exposed in Genie JSON)
· QNX shell access via serial, still the single highest-leverage environmental
change available.

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

Downstream of export, the 4B text tower **must** ship as ≥2 ctx-bins:
`qnn-context-binary-generator` has a hard 3.5 GiB per-graph serialization limit
and the 36-layer graph estimates 4.18 GiB (weights only — context length does
not move it). The split contract is in `docs/NOTES-genie-splits.md`.

### 8.8 Why are the device team's builds faster? — REOPENED (2026-08-16)

*Was "RESOLVED BACKWARDS (2026-08-13)". That resolution rested on the blended
11.72 figure and is withdrawn — see correction #25.*

The 2026-08-13 claim was: `qwen3_06b_w8a16_local` **11.72 tok/s** AR-1 vs the
device team's **7.79**, "+51% in our favour". But 11.72 is a phase blend (§6.9),
not an AR-1 rate. **Our pre-fix AR-1 rate is 6.84 tok/s** (measured 2026-08-15),
which is ~12% *slower* than their 7.79 — so the original premise was closer to
right than its retraction was.

Two caveats keep this genuinely open rather than merely re-inverted:

- Their 7.79 run reports PPR 7.45 tok/s on a 6-token prompt and TTFT 805 ms,
  which is decode-speed prefill — unlike anything we measure. Their bundle's
  topology, and therefore whether *their* number is itself blended, is unknown.
- Their ctx-bin is 1.01 GB against our 1.087 GB, so the artifacts differ.

Resolving it needs their binary, converter command lines, build-time HTP config,
`qnn-context-binary-utility` dump and dialog JSON — requested 2026-08-14
(`DEVICE_TEAM_EXCHANGE_2026-08-14.md` §4.2), still outstanding. **Do not quote a
direction for this gap until those land.**

The same 2026-08-13 measurement surfaced the real lead, which did pay out: 74.7%
of decode DSP cycles in GQA replication `Expand` ops → the `--grouped-gqa` fix →
**6.54× measured** (`DEVICE_MEASUREMENT_REPORT_2026-08-15.md`).

What the same measurement DID surface is the real lead: 74.7% of decode DSP
cycles in unreplicated-KV `Expand` ops — see the GQA replication fix
(`DROP_README_2026-08-14-gqafix.md`, `MAX_TPS_QWEN3_0.6B_V3.md`,
`--grouped-gqa` in `quantize_aimet.py`).

### 8.9 `hvx_threads`: 4 or 8? — promoted; it is now a model-falsification test
The runtime always reports **8 HVX threads in use**, on every workload size,
yet both teams build and configure with `hvx_threads: 4` (HTP doc open
question 7; 08-13 Test 5 showed only that the *runtime* knob is inert — it is
build-time only, `NOTES-htp-config-keys.md`).

Post-fix this is no longer a trimming knob. It **changes zero bytes by
construction**, so the byte model predicts exactly 0.0% while the compute model
allows up to 2× if the build-time 4 is binding. Any result above the rep spread
falsifies the byte model outright, and it costs one ctx-bin regenerate from
existing DLCs. It also resolves enough of the thread ambiguity to make the
per-op cycle profile interpretable — at 4 threads that profile reads
compute-bound and at 8 it reads byte-bound, so on its own it cannot adjudicate.

### 8.10 n-gram acceptance at 4B
Everything LADE buys at 4B hinges on acceptance holding near 0.6B's ~1.94
tokens/call, and 4B output distributions differ. Measure it in the first 4B
device run. If it sags, the SDK's learned-draft dialogs (`eaglet`, `spd`, with
Qwen3-4B-class example configs) are the designed answer (§6.3) — at ~4 GB
streamed per verify call, acceptance is worth far more per point than at 0.6B.

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
3. **`hf upload-large-folder` silently resets repo visibility and overwrites the
   hub README** (it applies its own defaults rather than preserving settings).
   Re-check `HfApi().repo_info(repo).private` and the README after every bulk
   upload — but **report a change, never "restore" it from assumption**.
   Visibility on the bundle repos is switched often and deliberately by the
   user, so **this document does not record it**: read it live with
   `HfApi().repo_info(repo).private` when it matters, change it only when
   asked in that message, and if a bulk upload flipped it, say so and stop.
   Acting on a remembered value has caused four incidents, in both directions.
   Single `upload_file` commits don't touch repo settings.
4. **Detach long uploads with `setsid`.** A backgrounded `upload_file` is killed
   when its parent shell exits — the log just stops mid-progress-bar with no
   error, which reads exactly like a proxy drop and sends you chasing the wrong
   bug. `setsid nohup … & disown` survives; through the local proxy a 1.08 GB
   blob then uploads in well under a minute.

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
| **`docs/MAX_TPS_QWEN3_0.6B_V4.md`** | **current — the 0.6B speed plan** | rev 2, 2026-08-16. The blend correction (§1), the byte-vs-compute degeneracy (§2), and the device session. **Start here.** Supersedes V3 §3–§7 and retires V3 §7 item 4 |
| `docs/DEVICE_MEASUREMENT_REPORT_2026-08-15.md` | current device truth | the GQA-fix result: 44.707 tok/s basic, 6.54× over the pre-fix control, LADE parked |
| `docs/NOTES-htp-kernel-table.md` | current, SDK-cited | every MatMul/FC dtype combination v81 supports. Read before any quantization-dtype decision — it answers KV INT8 and retro-explains the ViT V-path failure |
| `docs/VARIANT_PREDICTIONS.md` | current | the A2 gate: per-variant byte and cycle predictions, which knobs verifiably reached the artifact, and §2b's reproduction of §8.1 |
| `docs/DROP_README_2026-08-16-regime.md` · `kit-v2/` | current | the 2026-08-16 drop and its session kit. **Supersedes `kit/`**, whose priorities 4–5 point at blended bundles |
| `docs/DEVICE_TEAM_EXCHANGE_2026-08-16.md` | current | the blend correction sent to the device team, the P1 retraction, and the three artifacts still outstanding |
| `docs/MAX_TPS_QWEN3_0.6B_V3.md` | **superseded by V4** for §3–§7 | §0 (the inversion principle) and its build discipline still current. Its §10b execution log is the record of the gqafix build |
| `docs/MAX_TPS_QWEN3_0.6B_V2.md` | current for §0–§2 only | measured baselines, revised performance model, ctx-bin forensics |
| `docs/MAX_TPS_QWEN3_0.6B.md` (V1) | **superseded** | the original 10.8 tok/s recipe. Its headline predates the 2026-08-13 measurement that showed basic AR-1 at 11.72 on the same device. Historical. |
| `docs/NOTES-genie-pipeline.md` | current, SDK-cited | the multimodal pipeline contract — image-encoder dtype, MRoPE, deepstack-by-zeros, and the split-prefill load failure (§C1) |
| `docs/NOTES-htp-config-keys.md` | current, SDK-cited | which HTP backend-extension keys are real, audited against the SDK's own `config.py` / `QnnHtpGraph.h` |
| `docs/DEVICE_MEASUREMENT_REPORT_2026-08-13.md` | current device truth | the 5-test measurement run: 11.72 basic vs 9.18 LADE, and the 74.7% `Expand` finding |
| `docs/DROP_README_2026-08-14-gqafix.md` | current | the GQA replication fix (`--grouped-gqa`) and the bundles it produced |
| `reports/qwen3vl-4b-e2e-deployment-status-2026-08-14.md` | current device truth | the Qwen3-VL e2e attempt — **failed at load**, see §3.6 |
| `docs/NOTES-genie-io.md` | current, SDK-cited | the Genie/qualla contract — read before touching graph I/O |
| `docs/NOTES-vit-htp-config.md` | current | why graph names must appear in the backend config |
| `docs/NOTES-genie-splits.md` | current, SDK-cited | the multi-ctx-bin (split) contract — required for any graph over the 3.5 GiB serialization limit, i.e. every text tower ≳2B |
| `docs/SA8797P_HTP_v81_Hardware_and_Deployment_Quantization_Reference_EN.md` | current, annotated | the device team's measured-only hardware/runtime truth (2026-08-12) — best hardware ground source. Three superseded claims are flagged in its header annotation (corrections #16–18). |
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
