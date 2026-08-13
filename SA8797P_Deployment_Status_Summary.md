# SA8797P Deployment Status Summary

*Last updated: 2026-08-09*

> ## ⚠️ Inherited document — partly superseded
>
> This is the **remote team's** status as of 2026-08-09, written for a different
> environment (`/mnt/code/…`, `sa8797_deploy_kit`, conda envs). It remains the best
> record of the *hardware and GVM* findings in §1 and §3.2, which nothing since has
> contradicted. But several §2–§4 claims have since been disproved or completed on
> this project. Corrections are marked inline below and consolidated in
> **`docs/REFERENCE.md`**, which is the current-truth document. The files it
> references in its header (`SA8797P_DEPLOYMENT_KNOWLEDGE.md`, `GETTING_STARTED.md`,
> `TROUBLESHOOTING.md`, the dated investigations) are **not present in this repo**.
>
> Headline reversals:
> - **"`lm_head` INT8 degrades quality" (§2.2) is wrong** — device-measured parity at 0.6B.
> - **QKV fusion is done** (§3.1 says "not yet") — encodings-level surgery, built and device-tested.
> - **Decode is ~6.3–6.5 tok/s** on our builds, not §2.1's 7.4–8.2.
> - **All ctx-bins are ~1.09 GB**, not 1.5 GB.

This is a concise status of what we have, what works, what doesn't, and what's next for LLM deployment on the Qualcomm SA8797P (nordy / Gen5) automotive SoC via QAIRT 2.48.x + Genie T2T 1.19.0.

For detailed step-by-step procedures, error catalogs, and dated investigation logs, see:

- `SA8797P_DEPLOYMENT_KNOWLEDGE.md` — full reference
- `GETTING_STARTED.md` — environment setup
- `TROUBLESHOOTING.md` — error catalog (note: some stale entries, see below)
- `THROUGHPUT_INVESTIGATION_2026-08-05.md`, `GVM_BANDWIDTH_INVESTIGATION_2026-08-07.md` — dated investigations

---

# 1. Hardware & Runtime Environment

## 1.1 Silicon

| Attribute | Value |
|---|---|
| SoC | SA8797P (nordy / Gen5 / Snapdragon Ride Flex) |
| HTP | Hexagon v81 (NSP-based) |
| Silicon NSP count | **4 NSPs** (4 HVX threads each = 16 HVX total) — confirmed by `qnn-platform-validator` |
| VTCM per NSP | ~8 MB |
| Peak INT8 aggregate | ~80 TOPS (silicon nominal) |
| SDK used | QAIRT 2.48.40.260702, QNN API v2.37.0, libGenie 1.19.0 |

> **Correction (2026-08-13):** per the official silicon spec quoted in
> `docs/SA8797P_HTP_v81_Hardware_and_Deployment_Quantization_Reference_EN.md`
> §1, each HTP has **8 HVX units** (not 4) and **16 MB VTCM** (not ~8 MB). The
> "2 of 4 NSPs" reading in §1.2 below is therefore wrong: the observed 8 HVX
> threads + 16 MB VTCM is **one full HTP**. 4 cores are visible to the
> validator; Genie drives 1, so the theoretical multi-core upside is ×4, not
> ×2. See `docs/REFERENCE.md` §1 and correction #14.

---

## 1.2 What we actually get from Android GVM

The Android image runs as a **guest VM (GVM)** under QualVM hypervisor on top of QNX. The hypervisor partitions hardware; our guest sees only a subset:

| Resource | GVM reality | Native HLOS expectation |
|---|---|---|
| NSP cores available | **2 of 4** (8 HVX threads, fixed across all workloads) | Up to 4 (16 HVX) |
| VTCM available (unsigned PD) | **16 MB total** (2 NSP × 8 MB; requesting 24 MB is rejected at runtime with err `0x138d`) | Up to 32 MB at full allocation |
| DVFS control | **Works via Genie `backend.extensions` JSON** (4 tiers, 1.95× swing default→burst). `qnn-net-run --perf_profile` CLI flag is a no-op. | Full guest-side control |
| Clock visibility | `/sys/kernel/debug/clk/clk_summary` shows only virtio proxy (~3 lines); `/sys/class/devfreq/` empty | Full HTP/CC/DDR clock tree |
| sharedbuf (zero-copy CPU↔DSP) | Not supported | Supported |
| DSP hardware queue | `dspqueue_create` failure → CPU polling fallback | Hardware queue |
| FastRPC transport | `hfastrpc` (hypervisor-mediated) | direct FastRPC |
| FastRPC per-call overhead | **~220 µs** (zero-compute op roundtrip) | Lower |
| Burst-mode weight-streaming BW (single large matmul) | **~49 GB/s** (`lin_big_m1`, 33.6 MB FP16, burst) | Higher |
| Effective LLM decode BW (28 fragmented layers) | **~6–7 GB/s** | Higher |
| Guest OS | Android 16 (kernel 6.12.38 android16-5) | — |
| Host/Hypervisor | QNX (QualVM) | — |

**Key insight:** the silicon can do ~49 GB/s on contiguous reads under burst profile — DDR/HTP clocks are **NOT** stuck low. The ~7× collapse to ~7 GB/s on LLM decode is due to **access-pattern fragmentation** (28 layers × small MatMuls + KV traffic + per-op sync), not a clock cap. The fundamental bottleneck is **per-token weight streaming over DDR**, not raw DSP compute.

---

## 1.3 Access & toolchain

| Item | Value |
|---|---|
| Device access | `ssh <JUMPHOST>` → `adb -s REDACTED` |
| Host build root | `/mnt/code/build/` (`/` is 100% full — `TMPDIR=/mnt/code/build/tmp_claude` mandatory) |
| SDK path | `/mnt/code/toolchains/qairt/2.48.40/` |
| Python envs | `qwen3-deploy` (py3.10, torch+CUDA+AIMET, numpy<2) for export/quant/ctxbin/parity; `qairt-py312` (py3.12, numpy 2.x) for `qairt-converter` only |
| Required ONNX version | **1.19.0** (≥1.20 breaks AIMET export) |
| Bundle layout | **Flat directory** — all `.so`, binaries, configs, `.bin` in one folder, `LD_LIBRARY_PATH=.`, no `lib/` subdir, no `ADSP_LIBRARY_PATH` needed for unsigned PD |
| Required ARM `.so` (7) | `libGenie.so`, `libQnnHtp.so`, `libQnnSystem.so`, `libQnnHtpPrepare.so`, `libQnnHtpNetRunExtensions.so`, `libQnnHtpV81Stub.so`, `libQnnHtpV81Skel.so` (DSP6 skel from `lib/hexagon-v81/unsigned/`) |
| Large adb push caveat | >500 MB can trigger USB disconnects; push as `.tar.gz` and extract on-device; `adb reconnect` recovers |
| Device `/data` | Can fill to 98–99%; clean old bundles first |

---

# 2. What We Have Successfully Validated (Best Results)

## 2.1 Best on-device result: Qwen3-0.6B W8A16 single-core

| Metric | Value |
|---|---|
| Quantization | Weights INT8 per-channel symmetric, activations FP16 (W8A16) |
| Multi-graph ctx-bin | Prefill + decode, `enable-graph-switching: true`, `use-mmap: true`, `spill-fill-bufsize: 0`, `mmap-budget: 25` |
| Build-time config | `O: 3`, `vtcm_mb: 16`, `num_cores: 1`, `hvx_threads: 4`, `sparse_weights_compression: 1`, `rpc_polling_time: 9999`, `pd_session: unsigned`, `extended_udma: true` |
| Runtime perf profile | `llm_decode_burst` (highest tier) |
| Dialog QnnHtp block | `pos-id-dim: 64`, `kv-dim: 128`, `rope-theta: 1000000.0`, `cpu-mask: "0xe0"`, `poll: true`, `allow-async-init: true` |
| **Decode throughput** | **7.4–8.2 tok/s** (warm, burst) — ⚠️ *our builds measure **6.3–6.5 tok/s** AR-1 on device (2026-08-10/11), and **10.8 tok/s** with LADE speculative decoding (2026-08-11). The 7.4–8.2 figure has never been reproduced here.* |
| Perf profile ladder (warm) | burst / `llm_decode_burst` = 7.4–7.5 tok/s → high_perf/powersave = 6.3 → balanced = 5.4 → default/low_power_saver = 3.8 (1.95× swing) |
| Prefill / TTFT | Not precisely measured; short-context TTFT is acceptable |
| Output quality | Coherent (no quantization-induced garbage) |
| ctx-bin max-CL | 1152 (`dialog context.size` must be ≤ ~1024 for headroom) |
| Bundle tarball | `qwen3_06b_w8a16_bundle.tar.gz` (887 MB); binary 1.5 GB |

---

## 2.2 Quantization pipeline that works (AIMET PTQ W8A16)

The validated pipeline (scripted in `scripts/quant/quantize_aimet.py`):

1. **Export torch → ONNX**
   - split cos/sin RoPE
   - no fused `nn.RMSNorm` — opset 17 compatible
   - attention mask shape fixed to CL+AR

2. **AIMET Quantsim** with `htp_quantsim_config_v81_per_channel_linear.json`:
   - `default_param_bw=8`
   - `default_output_bw=16`
   - `quant_scheme=post_training_tf_enhanced`
   - Calibration: ~10 mixed Chinese/English/code/math prompts
   - `clip_weights_to_7f7f(sim)` to avoid INT8 saturation
   - RMSNorm forced to 16-bit
   - **Critical:** `embed_tokens`, final `norm`, `lm_head` kept in FP16 (their quantizers disabled). Otherwise HTP v81 rejects `Gather` on INT16 weight with error `0xc26`; ~~`lm_head` INT8 degrades quality~~.

   > **Correction (2026-08-12):** the `lm_head`-INT8 quality claim is **not
   > supported**. An INT8 per-channel head (`--quant-head`) held 3/4 argmax
   > locally and showed no visible degradation on device under greedy sampling
   > at 0.6B. It is still not worth shipping, for an unrelated reason: under
   > LADE it costs ~10% n-gram acceptance and nets −14% tok/s. Keeping the head
   > FP16 remains the default; the *reason* in this line is wrong. See
   > `reports/qwen3-0.6b-w8a16qh-ladekv-test-report.md`.
   - K/V projection outputs kept in FP16 (cross-graph KV cache cannot have mismatched INT16 scales between prefill and decode).

3. **Encodings filter** (`scripts/quant/filter_aimet_w8a16.py`)
   - strips any remaining encodings for embed/norm/lm_head.

4. **qairt-converter** (in `qairt-py312` env):

```bash
python3 $QAIRT/bin/x86_64-linux-clang/qairt-converter \
    --input_network model.onnx --output_path model.dlc \
    --quantization_overrides model_filtered.encodings \
    --float_bitwidth 16 --target_backend HTP \
    -d input_ids 1,128 ...  # dynamic dims for all inputs
```

- `--target_backend HTP` must be uppercase.
- `--float_fallback`, `--target_soc_model`, `--input_list` do **NOT** exist on this converter.

5. **ctx-bin generation**
   - native ELF binary, not Python
   - in `qwen3-deploy` env or with `LD_LIBRARY_PATH=$QAIRT/lib/x86_64-linux-clang`
   - **CWD must be the directory containing `htp_config.json` and `perf_config.json`** (relative `config_file_path`).
   - `--binary_file` value must **NOT** include `.bin` (auto-appended).
   - `--model libQnnModelDlc.so --dlc_path prefill.dlc,decode.dlc` (DLCs, not `.so` directly as `--model`).

6. **Bundle flat layout** with all 7 `.so` + `genie-t2t-run` + `tokenizer.json` + dialog JSON + `htp_backend_ext_config.json`.

---

## 2.3 Other things proven to work

- **Perf-profile voting via Genie ext JSON** — 4 reproducible tiers; always use `llm_decode_burst`.

- **Graph switching for prefill+decode in one ctx-bin** — works with `use-mmap: true`; must **NOT** set `groupContext.share_resources: true` (incompatible).

- **`rpc_polling_time: 9999`** — reduces idle overhead (adopted in all configs).

- **Gate-Up fusion (`fuse_gate_up`)** — ONNX-export-level fusion of `gate_proj + up_proj` into one MatMul; bit-exact in FP16; implemented in `sa8797_deploy_kit/export/models/qwen3.py`; AIMET quantsim path working (`gate_up_proj` output quantizer disabled = FP16 internal, requantize at `down_proj`); DLCs built (1074 MB each, 28 `gate_up` outputs in FP16). **Pending on-device smoke test** (ctx-bin rebuild with `vtcm_mb=16` needed).

- **Qwen3-1.7B W8A16 build** — pipeline works, artifacts exist in `build/qwen3-1.7b-*/`; device testing not yet performed.

- **QNX native binaries exist in QAIRT 2.43 drop** at `/mnt/code/qairt/2.43.0/{bin,lib}/aarch64-qnx800/` — including `genie-t2t-run`. Official EVB numbers (Qwen3-VL 4B, QNX native, 4 cores): **129.7 tok/s** — confirms GVM virtualization is the major bottleneck.

- **AIMET encodings piped directly into qairt-converter** via `--quantization_overrides` produces better-packed binaries (1.1 GB vs 1.5 GB for the same topology) than hand-rolled per-tensor encodings.

---

## 2.4 Parity / quality

- W8A16 (per-channel symmetric weights, FP16 activations, FP16 embed/lm_head/norm) produces coherent text with no visible quality degradation at our tested prompts.
- Earlier W8A8 attempts (per-tensor UINT8 asymmetric activations) all produced garbage (§3.3).
- Split cos/sin RoPE + preserved Q/K RMSNorm in ONNX export was validated for parity against torch.

---

# 3. What Doesn't Work / Key Bottlenecks

## 3.1 Quantization dead ends

| Approach | Result | Why |
|---|---|---|
| **W8A8** (INT8 activations, per-tensor UINT8 asymmetric) | All variants (v15–v18) produce **garbage output** | Per-tensor UINT8 asymmetric (the only activation quant HTP v81 MatMul supports) clips heavy-tailed LLM activations (attention scores, residuals). Per-channel activations not supported on v81. Axis fixes alone don't resolve. |
| **W4A16 / INT4 blockwise weights** | Converter **folds s4 weights back to FP16**; `htp_v2.json` has zero INT4 MatMul kernels | INT4 MatMul requires HTP v75 (Snapdragon 8 Gen 3) or newer. v81 has no INT4 MatMul support. |
| ~~**QKV fusion + W8A16**~~ **— SOLVED, see correction below** | Build-time OK with `vtcm_mb=24` (zero spill), but two runtime blockers: (1) `vtcm_mb=24` rejected on unsigned PD; (2) Q/K/V share one output tensor and one output quantizer — Q needs INT16 for attention MatMul, K/V must stay FP16 across the prefill/decode graph boundary. Single-quantizer choice → either cross-graph INT16 scale mismatch (error 5005) or garbage output if all-FP16. | ~~Needs ONNX graph surgery: `qkv_proj → DQ → Split → Quantize(Q only)`; not yet done.~~ |
| **`--preserve_io_datatype` on qairt-converter** | `input_ids` stays INT64 → unsupported by HTP; combining with `--source_model_input_datatype int32` errors. Even after ONNX cast to INT32, combined with disabled qkv output quantizer gave garbage. | Let converter auto-cast INT64→INT32. |
| **2-core ctx-bin (`num_cores=2`)** | Compiles (1.5 GB binary) but Genie T2T 1.19 fails with error 5005 (`QNN_ERROR_NOT_SUPPORTED`) or runs at **3.96 tok/s** (slower than single-core). | Multi-core needs custom QNN API code or a newer Genie. Genie creates a single-core device internally with no JSON option to override. |
| **Multiple concurrent Genie instances** | 2 instances → 4.0 tok/s each (8 total) — linear BW split, no per-sequence gain. | Confirms decode is 100% DDR-bound. |
| **`sparse_weights_compression=1`** | Build reports `sparse_weights_compression bytes saved: 0`. | Model isn't sparse enough; no gain. |

> **Corrections (2026-08-12):**
>
> - **QKV fusion is done.** Solved not with ONNX surgery but at the **encodings**
>   level (`scripts/export/qkv_surgery.py`, 28/28 grafts): the donor `q_proj`
>   INT16 encoding is grafted onto the Q split, K/V splits stay FP16. Converter
>   and generator accept it at `vtcm_mb=16`, and it ran on device.
>   *(Revised 2026-08-13: the "buys nothing — 6.27–6.5 ≈ baseline" verdict that
>   used to stand here compared against a garbage-output v1 build. The device
>   team's working fused build measures **8.98 vs 7.79 unfused, +15%** —
>   `docs/REFERENCE.md` §6.6, correction #15.)*
> - **W4A16 dead end, restated.** This row argued it badly ("requires v75 or
>   newer" vs "v81 has no INT4 support" — v81 *is* newer than v75), but
>   *(revised 2026-08-13)* its conclusion was right: `htp_v2.json` has **zero**
>   INT4 MatMul/FC kernels (SDK 2.43 & 2.48) and qairt-converter folds s4 → f16.
>   Independently, accuracy also fails: per-channel INT4, LPBQ block-64, and
>   LPBQ+SeqMSE all score **0/4** on the argmax gate W8A16 passes 3/4. Dead end
>   on both grounds, at any size — `docs/REFERENCE.md` §4.1, correction #8.
| **QAIRT 2.43 AUTO + built-in quantizer** | Same per-tensor UINT8 activations → same W8A8 garbage; no `llm_decode_*` perf profiles. | No improvement. |

---

## 3.2 System-level bottlenecks (fundamental)

**Decode is 100% DDR memory-bound.** W8A16 decode reads ~880 MB/token (fused) / ~1369 MB/token (non-fused). At 7.8 tok/s, single-core DDR BW = 10.7 GB/s sustained. To reach 20 tok/s at 1369 MB/token would need ~27 GB/s from one core — well above what a single HTP v81 core delivers in fragmented access.

Breaking down the gap between nominal silicon and observed performance:

| Limitation | Impact | Addressable from GVM guest? |
|---|---|---|
| **2/4 NSPs (hypervisor partition)** | ~2× compute loss, ~2× BW loss | No — requires QNX-side VM config change (`dsp_cores` mask in VM XML) |
| **GVM/IOMMU translation + fragmented access** | Single large matmul hits 49 GB/s; full LLM collapses to 7 GB/s (~7×) | No for the fundamental fragmentation; partial mitigation via op fusion |
| **VTCM hard cap 16 MB (unsigned PD)** | Single-layer weights ~6–24 MB → every layer re-streams weights from DDR; prevents VTCM weight reuse across layers | No without signed PD + larger VTCM (unclear if signed PD grants more VTCM) |
| **FastRPC 220 µs/call overhead** | Negligible for full-graph execution; ~6 ms per token if dispatching per-layer | N/A (we run full-graph) |
| **DVFS already at burst** | No further uplift from `perf_profile` | Already at ceiling |
| **No zero-copy (sharedbuf)** | Activation copies over RPC; small impact on decode (KB-sized activations), matters for prefill (MB) | No |
| **CPU-polled DSP queue** | Slight overhead vs hardware queue | No |

---

## 3.3 Bottlenecks we have partially addressed

- **Op fusion (Gate-Up, future QKV with ONNX surgery)** reduces DMA-launch count and increases tile size → better BW utilization. Build-time with `vtcm_mb=24` showed 3.4× DDR-read reduction and zero VTCM spill; on-device validation pending `vtcm=16` rebuild + QKV fix.

- **AIMET per-channel encodings + embed/lm_head FP16** reduced binary size and eliminated a `0xc26` error.

- **Graph switching + mmap + rpc_polling** eliminated RAM bloat from loading both graphs simultaneously and reduced idle overhead.

---

# 4. Remaining Goals & Open Issues

## 4.1 Immediate (next steps when device is available)

- [ ] **Gate-Up-only W8A16 on-device test** — ctx-bin rebuild with `vtcm_mb=16`, push, smoke-test, profile. Expected modest tok/s gain (+0–15%) plus TTFT improvement.

- [ ] **QKV fusion ONNX surgery** — insert:

  `qkv_proj → Dequant → Split(Q,K,V) → Quantize(Q only)`

  after the fused MatMul. This is the main unlock for zero-spill on the real 16 MB VTCM.

- [ ] Once QKV+GateUp both work, **rebuild with `vtcm_mb=16` and measure real decode/TTFT**. Build-time DDR report suggests 3.4× DDR-read reduction; real tok/s could be **12–16 tok/s if BW scales**.

- [ ] **W8A16 Qwen3-1.7B on-device smoke test** — confirm pipeline scales.

---

## 4.2 Medium-term performance directions

| Path | Target tok/s (0.6B) | Effort | Notes |
|---|---:|---|---|
| Gate-Up fusion (no QKV) | ~8–9 tok/s | Low | Built, pending device validation |
| QKV+GateUp fusion (after ONNX surgery) | ~12–16 tok/s | Medium | High confidence if ONNX surgery works and fits in 16 MB |
| W8A8 re-test after fusions | ~1.5–2× of baseline (if quality holds) | Medium | Zero-spill may reduce requantization error accumulation vs non-fused W8A8; prior non-fused W8A8 was garbage, but spill-induced scale/offset error may have contributed |
| `fp16_relaxed_precision=1` + O:0/1/2 variants | +0–10% | Low | Quick A/B once baseline is re-established |
| `DDR_PERF_MODE` (C API option 7) | +0–30% (unknown) | Medium | Requires custom C code; not exposed in Genie JSON; needs bus corner = MAX (burst likely already does this) |
| Custom QNN multi-core runtime (drive 2 NSPs directly) | up to ~1.5–2× (12–20 tok/s for fused 0.6B) | Large | Bypasses Genie; needs QNN device/context API + custom dispatch |
| Signed PD investigation | unknown (possibly unlocks larger VTCM / QoS) | Medium | Needs Hexagon SDK + OEM keys; unlikely to change NSP count but may change VTCM/QoS |
| QNX native execution (bypass GVM entirely) | potential 5–15× (EVB 4B @ 129.7 tok/s reference) | High — needs QNX shell access | Physical serial or Ethernet to QNX; ctx-bin rebuilt against QAIRT 2.43 QNX binaries; skel goes to `/mnt/etc/images/dsp/` |
| AIMET AdaRound / CLE (advanced PTQ) | quality/quant tradeoff | Medium | May enable W8A8 to work at acceptable quality |
| Newer QAIRT / Genie version | unknown | Depends on Qualcomm | May add native multi-core Genie |
| Smaller model (0.4B) or MoE with small active params | linear with param count | Model-dependent | New model work |

---

## 4.3 Open technical questions

1. **Does Gate-Up fusion (and QKV fusion once fixed) produce measurable real tok/s gain on device, or did build-time DDR read reductions not translate to BW win under GVM fragmentation?**

2. **Can we get 4 NSPs via GVM VM config change?** Needs BSP/FAE engagement on QNX host. `qpv` confirms 4 NSPs exist; we only get 2.

3. **Does `DDR_PERF_MODE` (C API option 7) give a BW boost on v81 Android beyond what `llm_decode_burst` already does?** Needs custom backend extension code.

4. **Does signed PD grant more VTCM or other resources on SA8797P GVM?** We observed silent fallback with unsigned skel + `pd_session:"signed"`; needs Hexagon SDK + real OEM-signed skel to test.

5. **Can W8A8 (INT8 activations) be made to work at acceptable quality on the fused graph?** Non-fused attempts all failed; zero-spill + per-channel activation-like hacks might help but v81 MatMul only supports per-tensor UINT8 asymmetric.

6. **Is `vtcm_mb=16` really the hard cap for unsigned PD on SA8797P?** The error says `"requested 24 MB unsupported"`; we haven't tested intermediate values. The math (2 NSP × 8 MB) strongly suggests 16 is the true cap.

7. **Can we get QNX shell access via serial/UART?** EVB should have a micro-USB debug port; this is the highest-leverage environment change (unlocks 4 NSP + no hypervisor overhead).

8. **Why does `qnn-net-run --perf_profile` CLI flag silently no-op while Genie's ext-JSON equivalent works?** Acceptable limitation but worth a note to Qualcomm.

9. **How much does prefill actually benefit from fusion + VTCM fit?** We don't have a baseline TTFT number yet.

---

## 4.4 Documentation / tooling debt

- [ ] `TROUBLESHOOTING.md` still prescribes the pre-2.43 `lib/` subdir and `ADSP_LIBRARY_PATH` for error 14001 — **stale**, will mislead users.

- [ ] `GETTING_STARTED.md` §2 lists conda env names as `sa8797-torch-export` / `sa8797-qairt-convert` with python versions reversed (says torch=py3.12, qairt=py3.10); actual names are `qwen3-deploy` (py3.10) / `qairt-py312` (py3.12). §9 shell commands also use stale `lib/` layout.

- [ ] `GVM_BANDWIDTH_INVESTIGATION_2026-08-07.md` TL;DR claims `"perf_profile completely ignored in GVM"` — this was a cold-start artifact; needs a correction addendum at the top.

- [ ] `configs/boards/sa8797p.yaml` still has `vtcm_mb: 24` — must be changed to **16** for unsigned PD on-device builds.

- [ ] `--fuse-gate-up` / `--fuse-qkv` flags exist in PipelineConfig but are not fully plumbed through `sa8797 convert` / `sa8797 quantize` / `sa8797 ctxbin` CLI.

- [ ] Multi-core / QNX bundle targets not yet supported in `package/bundle.py`.

---

# 5. Reference: Key Build Artifacts

| Artifact | Location | Status |
|---|---|---|
| Working W8A16 0.6B bundle (on-device 7.4–8.2 tok/s) | `qwen3_06b_w8a16_bundle.tar.gz` (887 MB) | ✅ Validated |
| W8A16 GateUp-fused ctx-bin (needs vtcm=16 rebuild) | `build/qwen3-0.6b-qnn-fusegu/` | ⏳ Built, needs ctxbin @ vtcm=16 + device test |
| W8A16 QKV+GateUp-fused (vtcm=24, fails to load) | `bundles/qwen3_06b_w8a16_fuseqkv_fusegu/` (939 MB tar) | ❌ Loads fails (err `0x138d`); QKV quant conflict also present |
| W8A16 1.7B WIP | `build/qwen3-1.7b-qnn/`, `bundles/w8a16_17b/` | ⏳ Built, not tested |
| FP16/AIMET quant scripts | `scripts/quant/quantize_aimet.py`, `scripts/quant/filter_aimet_w8a16.py` | ✅ Working for non-fused + fusegu paths |
| HTP microbenchmarks (FP16 Linear) | `build/htp_microbench/v2/` (`tiny`, `lin_m1...m2048`, `lin_big_m1`) | ✅ For BW/overhead characterization |
| QNX 2.43 native binaries | `/mnt/code/qairt/2.43.0/{bin,lib}/aarch64-qnx800/` | ✅ Present, awaiting QNX shell access |
