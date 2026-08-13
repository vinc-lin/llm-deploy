# SA8797P HTP v81 Hardware Description and Deployment/Quantization Reference

**Date**: 2026-08-12 (v3: inference/estimation content removed; only measured/verified data retained)  
**SoC**: SA8797P (nordy, Gen5)  
**HTP Version**: Hexagon v81  
**SDK**: QAIRT 2.48.40 Premium, QNN API v2.37.0, libGenie 1.19.0  
**Runtime Environment**: Android 16 GVM (SA_QX_VM) under the QualVM hypervisor on QNX  
**Purpose**: Reference for model deployment, quantization, and performance analysis. This document **contains only directly measured data, configuration validation results, and actually observed facts**. It does not include speculation, estimation, or prediction. Section §12 lists items that still require confirmation.

> **Repo annotation (llm-deploy, 2026-08-13)** — kept verbatim as received;
> three items are superseded by measurements newer than this document
> (details in `docs/REFERENCE.md` §7, corrections #16–18):
>
> - **§7 / §9 / §12.6 — LADE is not a platform bug.** The SIGSEGV at PC
>   0x4c2d58 is the unlisted-`graph_names` failure: the §8.3 config lists only
>   `["prefill","decode"]`, and a verify graph absent from `graph_names`
>   null-pointers on the first speculation step. With the verify graph listed,
>   no AR==CL graph in the ctx-bin, and prompts ≥ 2 tokens, LADE ran at
>   **10.8 tok/s on this device on 2026-08-11** (9.3 for the quant-head
>   variant, 08-12). See `docs/REFERENCE.md` §3.3–3.4.
> - **§5.1 — `lm_head` does not have to stay FP16.** Error 0xc26 is the
>   embedding *Gather* restriction only; an INT8 (`sFxp_8`) lm_head builds and
>   runs with unchanged quality (REFERENCE §6.4). It is kept FP16 by default
>   because quantizing it costs LADE acceptance (−14% tok/s), not because it
>   is unsupported.
> - **§4.2 — the 1.49 GB spill at `vtcm_mb=16` ("older optctx2") is probably
>   not real VTCM behavior** but the same graph-names artifact: an unlisted
>   graph silently gets 4 MB VTCM, measured here to produce 1.446 GB of spill
>   (`docs/NOTES-vit-htp-config.md`). Correctly-configured vtcm-16 builds
>   spill ~0.
>
> §11 describes the device team's workstation; none of it applies to this
> repo's environment (see `docs/LOCAL_ENV.md`).

---

## 1. SA8797P HTP Hardware Specifications (Official Data)

> Source: SA8797P hardware specification materials. The following are official silicon specifications, not measured results.

The SA8797P integrates **4 independent HTPs** (Hexagon Tensor Processors). The internal structure of each HTP is:

```text
                One HTP
    ┌─────────────────────────────┐
    │          Q6 DSP             │
    │   Control / Scheduling /    │
    │      Scalar Compute         │
    │              │              │
    │      ┌───────┴───────┐      │
    │      ↓               ↓      │
    │   HVX ×8          HMX       │
    │ Vector Compute   Matrix     │
    │                  Compute    │
    │                  ├─ INT     │
    │                  └─ FLOAT   │
    │                             │
    │       VTCM      L2 Cache    │
    │ High-Speed       Cache      │
    │ Workspace                   │
    └──────────┬──────────────────┘
               │
               ↓
              DDR
```

**Components in each HTP**:

| Component | Specification | Function |
|---|---|---|
| **Q6 DSP** | 12-thread scalar processor | Control, scheduling, and non-parallel logic (Hexagon Q6 core) |
| **HVX** | 8 vector co-processors | Vector compute (1024-bit SIMD) |
| **HMX** | 1 matrix co-processor | Matrix multiplication acceleration, containing two sub-units: INT (INT8/INT16) and FLOAT (FP16) |
| **VTCM** | 16 MB | Vector Tightly Coupled Memory SRAM (high-speed workspace for HVX/HMX) |
| **L2 Cache** | HTP-internal L2 | Cache |
| **DDR** | Off-chip memory | Accessed through the system bus |

The HTP also integrates the following hardware acceleration functions, including DLBC and UBWC:

| Hardware Acceleration Unit | Function |
|---|---|
| DLBC | Deep Learning Bandwidth Compression; inter-layer DDR compression/decompression |
| UBWC | Universal Bandwidth Compression; pixel-frame compression |
| Depthwise Conv acceleration | Dedicated hardware for depthwise convolution |
| Activation acceleration | Hardware implementation of activation functions such as ReLU/SiLU |

---

## 2. Runtime Environment: GVM Virtualization (Measured)

All items below are states directly observed from inside the Android GVM.

### 2.1 HTP Core Visibility

| Measurement | Observed Result | Measurement Method |
|---|---|---|
| Total NSP/DSP core count reported by `qnn-platform-validator` | 4 available | `qnn-platform-validator --backend dsp --testBackend` |
| Number of HVX threads actually used at runtime | **8** (fixed across all workload sizes) | QNN profile log: `Number of HVX threads used: 8 count` (8 from tiny add through M=2048 matmul) |
| Effect of explicitly specifying a multi-core JSON config (4 cores) | No effect (still uses 8 HVX) | Comparative experiment |

### 2.2 Clock / DVFS

| Measurement | Observed Result | Measurement Method |
|---|---|---|
| Genie backend extension JSON `perf_profile` field | **Effective**, with 4 stable performance tiers (see §3.3) | Genie warm run, 5 inferences × 3 reps |
| `qnn-net-run --perf_profile` CLI flag | Ineffective (difference <0.5%, within noise) | Direct qnn-net-run comparison |
| Devices under `/sys/class/devfreq/` | Empty | `adb shell ls` |
| Clocks visible in `/sys/kernel/debug/clk/clk_summary` | Only `virtio_clk` proxy (about 3 lines) | adb shell inspection |
| debugfs | Not mounted by default | Requires manual `mount -t debugfs none /sys/kernel/debug` |

### 2.3 Other GVM Environment Observations

| Measurement | Observed Result | Measurement Method |
|---|---|---|
| sharedbuf (zero-copy) | Not supported | `qnn-platform-validator` report |
| DSP hardware queue | `dspqueue_create` fails → CPU queue polling fallback | `qnn-platform-validator` + verbose log |
| FastRPC zero-op round-trip overhead | ~220 µs/call (DSP compute is only 6 µs; the rest is RPC/submit/sync) | 4-float add microbenchmark |
| Fixed FastRPC submit/sync overhead | ~50–60 µs/call | Consistent across workloads |
| IOMMU/SMMU | Enabled (all DDR access goes through hypervisor translation) | GVM architecture |

---

## 3. Measured Performance Data

### 3.1 Microbenchmark: qnn-net-run FP16 Linear (burst profile, qnn-net-run, not Genie)

Tool: `qnn-net-run --profiling_level basic` + `qnn-profile-viewer`  
Note: These tests use `qnn-net-run --perf_profile` (the CLI flag is known to be ineffective), so they run at the GVM default clock rather than the highest Genie burst-profile frequency.

| Benchmark | Shape (M, K, N) | Weight MB | DSP Compute µs | NetRun Total µs | Effective FP16 TOPS/s |
|---|---|---|---|---|---|
| tiny (4-float add) | 1,1,4 | ~0 | 6 | 222 | — |
| lin_m1 | 1,1024,3072 | 6.3 | 100 | 400 | 0.03 |
| lin_m8 | 8,1024,3072 | 6.3 | 144 | 552 | 0.17 |
| lin_m32 | 32,1024,3072 | 6.3 | 217 | 668 | 0.46 |
| lin_m128 | 128,1024,3072 | 6.3 | 189 | 736 | 2.13 |
| lin_m256 | 256,1024,3072 | 6.3 | ~450 | 1270 | 1.79 |
| lin_m512 | 512,1024,3072 | 6.3 | 600 | 1280 | **2.68** |
| lin_m1024 | 1024,1024,3072 | 6.3 | ~2050 | 5460 | 1.57 |
| lin_m2048 | 2048,1024,3072 | 6.3 | ~2300 | 10834 | **2.80** |
| lin_big_m1 | 1,2048,8192 | 33.6 | 500 | 708 | 0.03 |

**Observed facts**:

- At large M (M≥512), aggregate peak FP16 compute is approximately 2.7–2.8 TOPS/s.
- For lin_m1 / lin_big_m1 (M=1, decode-like, one-time weight reads), compute times are 100 µs / 500 µs respectively → approximately 63–67 GB/s of weight streaming bandwidth.
- Using excl-wait timing, lin_big_m1 takes 684 µs → 49 GB/s. The difference between the two bandwidth figures comes from whether wait time is included.
- Under qnn-net-run, the difference between `perf_profile` burst and low_power_saver is <0.5%.

### 3.2 LLM Inference Speed (Genie T2T, Device-Measured)

**Qwen3-0.6B W8A16, `llm_decode_burst` profile, warm state**:

| Configuration | TGR (tok/s) | TTFT (ms) | Init (ms) | Prefill PPR (tok/s) | Quality |
|---|---|---|---|---|---|
| Unfused W8A16, graph-switching + mmap | 7.79–7.80 | 805 | 786 | 7.45 | Coherent Chinese |
| QKV + Gate-Up fused W8A16, vtcm=16 | **8.98** | 1667 (14 prompt tok) | — | 8.40 | Coherent English |
| Unfused W8A8 v19 (quality already broken) | 4.32 | 1619 | — | — | Garbled output |

**Detailed Qwen3-0.6B W8A16 burst-profile timing** (128 generated tokens, 6 prompt tokens):

- init: 786 ms
- TTFT: 805 ms
- TGR: 7.79 tok/s
- PPR: 7.45 tok/s

**Prefill throughput**:

- Short prompt (12 tok): 266 tok/s
- Long prompt (53+ tok): 1100+ tok/s

**Prefill/Decode graph-switch latency**: 79–93 ms

### 3.3 Perf Profile Tier Comparison (Genie Backend Extension JSON, W8A16, Warm, 3 Reps)

| Tier | Included Profile Names | tok/s | Relative to burst |
|---|---|---|---|
| 1 (highest) | burst, llm_decode_burst | 7.43–7.45 | 1.00× |
| 2 | high_perf, powersaver, sustained_high_perf, llm_prefill_burst | 6.32–6.34 | 0.85× |
| 3 | balanced, low_balanced | 5.37–5.38 | 0.72× |
| 4 (lowest) | default, low_power_saver | 3.81 | 0.51× |

The burst → low_power_saver range is **1.95×**.

### 3.4 Latency

| Scenario | Latency |
|---|---|
| Cold-start init (first run after USB reconnect) | 1.8–2.0 s |
| Warm-start init (DSP already awake) | 786–820 ms |
| One-shot CLI end-to-end (warm) | 3–5 s (20-token output) |
| Hot daemon stdio (KV reuse) | 1.2–2.0 s / query |
| genie-service + prefix cache | 1.0–1.5 s / query |
| Initial VTCM acquire (cold) | 1.9–5.4 ms |
| Initial VTCM acquire (warm) | ~200 µs |
| HVX+HMX power-up acquire (cold) | 7–21 ms |
| HVX+HMX power-up acquire (warm) | ~9 µs |
| Steady-state inter-op wait | 30–60 µs / execute |

### 3.5 Bandwidth Observations (Indirectly Calculated)

| Scenario | Observed Bandwidth | Data Source |
|---|---|---|
| Single large MatMul sequential weight stream (33.6 MB) | 49–67 GB/s (depending on timing definition) | qnn-net-run microbenchmark |
| Steady-state effective LLM decode bandwidth | ~6–7 GB/s | Back-calculated from ~924 MB DDR read per token × 7.4 tok/s |

---

## 4. Memory and VTCM

### 4.1 VTCM Configuration Limits

| Configuration | Observed Result |
|---|---|
| `vtcm_mb: 16` (unsigned PD, runtime) | Runs normally |
| `vtcm_mb: 24` (unsigned PD, runtime) | Error 5005; logcat: `Request feature vtcm size with value 25165824 unsupported` (err 0x138d) |
| `vtcm_mb: 24` (build-time offline compilation) | Compilation succeeds, but runtime fails with error 5005 |
| `pd_session: "signed"` with unsigned skel | No error; silently falls back to unsigned PD (no performance change) |

### 4.2 Build-Time VTCM Spill/Fill/DDR Report (qnn-context-binary-generator Output)

**Unfused W8A16 (`vtcm_mb=24` build, cannot run)**:

| Graph | Spill | Fill | DDR Read | DDR Write |
|---|---|---|---|---|
| Prefill | 38.9 MB | 38.9 MB | 1.35 GB | 92.5 MB |
| Decode | 0 | 0 | 1.31 GB | 0.4 MB |

**Unfused W8A16 (`vtcm_mb=16` build, older optctx2)**:

| Graph | Spill | Fill |
|---|---|---|
| Prefill | 1.49 GB | 1.54 GB |
| Decode | 666 KB | 1.3 MB |

**QKV + Gate-Up fused W8A16 (`vtcm_mb=24` build, cannot run)**:

| Graph | Spill | Fill | DDR Read |
|---|---|---|---|
| Prefill | 0 | 0 | 920 MB |
| Decode | 0 | 0 | 880 MB |

**QKV + Gate-Up fused W8A16 (`vtcm_mb=16` build, runs normally on device @ 8.98 tok/s)**:

| Graph | Spill | ctx-bin Size |
|---|---|---|
| Decode | 0 | 1.086 GB |

**Unfused W8A8 v19 (`vtcm_mb=16`, quality collapsed)**:

| Metric | Value |
|---|---|
| Decode spill | 730 MB |
| TGR | 4.32 tok/s |

### 4.3 Memory Usage

| Item | Value | Condition |
|---|---|---|
| CPU-side allocation | 171,082,240 bytes (~163 MB) | Genie startup log: `Allocated total size = 171082240 across 1 buffers` |
| ctx-bin size (unfused W8A16) | 1.01 GB | — |
| ctx-bin size (fused W8A16) | 1.09 GB | embed/lm_head retained as FP16 |
| Bundle tarball (unfused W8A16 local) | 885 MB | ~1.2 GB after extraction |

---

## 5. Quantization Experiment Results

### 5.1 W8A16 (INT8 Per-Channel Symmetric Weight + FP16 Activation)

| Item | Result |
|---|---|
| Quality (Qwen3-0.6B) | Normal, coherent Chinese/English output |
| Decode TGR (unfused) | 7.8 tok/s |
| Decode TGR (QKV + Gate-Up fused) | 8.98 tok/s |
| MatMul INT8 per-channel kernel in htp_v2.json | Present |
| Modules that must remain FP16 | `embed_tokens.weight`, all `norm.weight`, and `lm_head.weight` (otherwise embed Gather reports error 0xc26) |
| AIMET encodings requirement | v2.0.0 format + explicit `axis` field (without `axis`, v1.0.0 causes the converter to infer the wrong weight axis, especially for non-square `k_proj`/`q_proj`) |
| AIMET `EXPORT_TO_ONNX_DIRECT` | Must be set to False (True causes double quantization through Q/DQ + `quantization_overrides`, producing garbled output) |

### 5.2 W8A8 (INT8 Activations) — All Failed

All experiment results:

| Version | Activation | Weight | Result |
|---|---|---|---|
| v15, v16 | UINT8 per-tensor asymmetric | INT8 per-channel symmetric | Garbled output |
| v17 | INT8 per-tensor symmetric | INT8 per-channel symmetric | Runtime error `Failed to query` |
| v18 (embed FP16, lm_head INT8) | UINT8 per-tensor asymmetric | INT8 per-channel symmetric | Garbled output |
| AIMET W8A8 v19 (unfused) | — | — | 4.32 tok/s, 730 MB spill, garbled output |

Additional fact: v81 has no per-channel activation kernel for MatMul.

### 5.3 W4A16 / INT4 Weight Quantization

| Item | Result |
|---|---|
| INT4 kernel entries for MatMul/FC in htp_v2.json | 0 (same in SDK 2.43 and 2.48) |
| qairt-converter encountering s4 weights | Automatically folds them back to FP16: `Constant folded static tensor ... from s4 to f16` |

### 5.4 KV Cache

| Item | Result |
|---|---|
| Genie dialog JSON `kv-quantization: true` on QnnHtp backend | Ineffective (this flag exists only in the QnnGenAiTransformer CPU backend) |
| FP16 KV cache size (ctx=1024) | ~130 MB |

---

## 6. Multi-Core and Concurrency

| Configuration | Observed Result |
|---|---|
| 2-core ctx-bin (`num_cores=2, core_id=[0,1]`) run through Genie T2T | Error 5005 (QNN_ERROR_NOT_SUPPORTED) |
| One actual run of a 2-core ctx-bin (v17 version) | 3.96 tok/s (slower than the 1-core 7.4 tok/s result) |
| `groupContext.share_resources: true` + `enable-graph-switching` | Incompatible (OOM or other failures) |
| 2 concurrent W8A16 Genie processes | ~4.0 tok/s each, ~8 tok/s total (bandwidth divided approximately linearly) |

---

## 7. LADE (Speculative Decoding)

| Configuration | Observed Result |
|---|---|
| `type: "lade"` | SIGSEGV (SEGV_MAPERR, PC 0x4c2d58); verify32 graph exists, but crashes on the first step |
| `type: "basic"` | Works normally |
| Additional PD memory usage of LADE bundle | ~80 MB (verify32 graph is unused in basic mode but still occupies memory) |

---

## 8. Verified Working Configuration

### 8.1 Bundle Layout (Flat Layout)

All files (binary, 7 `.so` files, `genie-t2t-run`, tokenizer, and configs) are placed in the same directory with `LD_LIBRARY_PATH=.`. There is no `lib/` subdirectory, and `ADSP_LIBRARY_PATH` is not required.

**Required 7 `.so` files**:

1. `libGenie.so` (~9.8 MB)
2. `libQnnHtp.so` (~3.6 MB)
3. `libQnnSystem.so` (~3.9 MB)
4. `libQnnHtpPrepare.so` (~84 MB)
5. `libQnnHtpNetRunExtensions.so` (~1.4 MB)
6. `libQnnHtpV81Stub.so` (~760 KB)
7. `libQnnHtpV81Skel.so` (~13 MB, DSP skel)

If any one of these is missing, error 14001 may occur.

### 8.2 Key Genie Dialog JSON Parameters (Measured and Verified Effective)

```json
{
  "dialog": {
    "type": "basic",
    "max-num-tokens": 40,
    "context": {
      "size": 1024,
      "n-vocab": 151936,
      "bos-token": 151644,
      "eos-token": [151645, 151643, 151647],
      "n-embd": 2048
    },
    "sampler": {
      "seed": <random>,
      "temp": 0.85,
      "top-k": 50,
      "top-p": 0.9
    },
    "tokenizer": {"path": "tokenizer.json"},
    "engine": {
      "n-threads": 3,
      "backend": {
        "type": "QnnHtp",
        "QnnHtp": {
          "use-mmap": true,
          "spill-fill-bufsize": 0,
          "mmap-budget": 25,
          "poll": true,
          "cpu-mask": "0xe0",
          "kv-dim": 128,
          "pos-id-dim": 64,
          "rope-theta": 1000000.0,
          "enable-graph-switching": true,
          "allow-async-init": true
        },
        "extensions": "htp_backend_ext_config.json"
      },
      "model": {
        "type": "binary",
        "binary": {"ctx-bins": ["qwen3-0.6b-w8a16_ctx.bin"]}
      }
    }
  }
}
```

Note: Running the `genie_dialog.json` bundled with the HF package directly (`bos-token: -1`, only 2 `eos-token` values, `temp: 0.0`, and no `max-num-tokens`) causes repetitive looping output until `Context Size was exceeded`. The ADAS configuration must be used, or a similar configuration that adds `max-num-tokens`, correct EOS values, and non-greedy sampling.

### 8.3 htp_backend_ext_config.json (Runtime Performance Configuration)

```json
{
  "graphs": [{"graph_names": ["prefill", "decode"], "O": 3, "vtcm_mb": 16, "hvx_threads": 4}],
  "devices": [{
    "dsp_arch": "v81",
    "pd_session": "unsigned",
    "cores": [{"core_id": 0, "perf_profile": "llm_decode_burst", "rpc_polling_time": 9999}]
  }]
}
```

### 8.4 Key Build-Time perf_config.json Parameters

| Parameter | Value | Notes |
|---|---|---|
| O | 3 | Highest optimization level |
| vtcm_mb | 16 | Runtime limit under unsigned PD |
| soc_id | 72 | Corresponds to SA8797P |
| soc_model | 72 | — |
| dsp_arch | `"v81"` | String |
| pd_session | `"unsigned"` | — |
| weight_sharing_enabled | true | Prefill/decode weight sharing |
| hvx_threads | 4 | — |
| graph_names | Must exactly match the internal DLC graph names | Mismatch = silent fallback to defaults (VTCM=4 MB, O=2, extremely slow) |

### 8.5 Qwen3 Architecture Parameters (Must Match)

| Parameter | Value |
|---|---|
| head_dim | 128 |
| rotary_dim | 64 (head_dim / 2) |
| kv-dim | 128 |
| pos-id-dim | 64 |
| rope-theta | 1,000,000 (1M, not Qwen2's 100K) |
| context size (dialog) | ≤ 1024 (ctx-bin max-CL = 1152, leaving headroom) |

### 8.6 Qwen3 Chat Template Requirements

Use `<|im_start|>/<|im_end|>`. The assistant prefix must include an empty `<think>\n\n</think>\n\n` block to disable thinking mode. If this is omitted, thinking mode is triggered, making the output longer and increasing latency.

---

## 9. Observed Errors

| Error | Observed Trigger |
|---|---|
| **14001** (QAIRT_DEVICE_ERROR_INVALID_CONFIG) | Missing one or more of the 7 `.so` files; malformed `htp_backend_ext_config.json`; graph-switching enabled without use-mmap; graph_names mismatch |
| **5005** (QNN_ERROR_NOT_SUPPORTED) | `vtcm_mb > 16` under unsigned PD; Genie multi-core; unsupported feature combination |
| **0xc26** (UNSUPPORTED op) | Quantized `embed_tokens` weights (Gather does not support INT16/INT8 input); unsupported dtype + kernel combination |
| **Context Size was exceeded** | No `max-num-tokens` + greedy sampling causes repetitive looping; incorrect chat template triggers unbounded continuation |
| **SIGSEGV** in libGenie.so @ PC 0x4c2d58 | LADE (`type: "lade"`) mode |

**Note**: Error 14001 does not mean "library not found"; it indicates an invalid DSP performance-infrastructure configuration.

---

## 10. Known Toolchain Behavior (Measured)

### 10.1 qairt-converter

- `--target_backend` must use uppercase `HTP` (lowercase causes an error).
- The `--float_fallback` argument **does not exist** in qairt-converter (it belongs to qairt-quantizer).
- The `--target_soc_model 8797` argument **does not exist** (rejected by backend_awareness validation).
- The `--input_list` argument **does not exist** (available only in the legacy qnn-onnx-converter).
- `--float_bitwidth 16` is a string.

### 10.2 qnn-context-binary-generator

- Native ELF binary (not Python). RUNPATH is misconfigured, so `LD_LIBRARY_PATH=$QAIRT/lib/x86_64-linux-clang` is required.
- Must use `--model libQnnModelDlc.so --dlc_path <dir>`; a `.dlc` file **cannot** be passed directly to `--model`.
- The `--binary_file` value **must not include** the `.bin` suffix because the tool appends it automatically.
- The `perf_config.json` path is relative to the **CWD**, not relative to the `--model` argument.

### 10.3 genie-t2t-run CLI

- There is **no** `--max-tokens` argument (`max-num-tokens` must be set in JSON).
- `-t` means `--embedding_table` (T2E mode), not timing.
- There is no `-n` argument.
- On Android, `--log error|warn|info|verbose` sends logs to logcat, not stdout.
- `--profile FILE` writes a JSON performance file containing init, TTFT, PPR, TGR, and traceEvents.

### 10.4 General

- Build SDK and runtime `.so` versions must match exactly (2.48.40). Mixing 2.43 and 2.48 silently falls back to CPU execution.
- Internal DLC graph name = basename of `--output_path` used during conversion, not the `.dlc` filename. Renaming the `.dlc` afterward does not change the internal graph name.
- `--preserve_io_datatype` + INT64 `input_ids` causes an error because HTP does not support INT64 activations.
- HTP PMU counters **do not include** DDR bandwidth / NoC / LLC miss events.
- `QNN_SDK_LOG_LEVEL=verbose` and Genie `--log verbose` provide limited additional information.

---

## 11. Environment / Infrastructure Notes

| Item | Observation |
|---|---|
| Workstation root partition `/` | Can reach 100% usage; all build/tmp data must use `/mnt/code/build/`, with `TMPDIR=/mnt/code/build/tmp_claude` |
| Two conda environments | `qwen3-deploy` (py3.10, torch+aimet+export, numpy<2) and `qairt-py312` (py3.12, converter) cannot be merged because of the numpy 2.x vs <2 conflict |
| ADB connection | Via SSH jump host: `ssh <JUMPHOST> "adb -s REDACTED ..."`; pushing large files (>500 MB) can easily trigger USB disconnects; `adb reconnect` restores the connection |
| Device `/data` partition | Frequently 98–99% full; old bundles must be cleaned before pushing a new bundle; the original tar.gz can be deleted after extraction |
| Device clock | Approximately 2 days behind the host (file timestamps are unreliable) |
| Onnx version | 1.19.0 (≥1.20 breaks AIMET export) |
| ctx-bin build time (prefill + decode, two graphs) | ~16–20 minutes (Graph Opt ~670s, Sequencing ~620s, Parallelization ~78s, VTCM Alloc ~13s, Finalizing ~21s) |

---

## 12. Open Questions Requiring Qualcomm FAE / Platform-Team Support

The following items cannot be independently verified under the current conditions and require assistance from the platform team or Qualcomm FAE:

1. **HTP core allocation**: Can multiple HTPs be allocated to the Android guest in the GVM configuration? Runtime observation currently shows 8 HVX threads, which according to the official architecture corresponds to all 8 HVX units of a single HTP. QNN reports 4 NSP cores as available, but Genie can use only 1. What is the hypervisor HTP allocation mask?
2. **Signed PD**: What is the process for obtaining/using an OEM-signed skel? What is the `vtcm_mb` limit under signed PD? Does signed PD unlock additional HTP resources?
3. **DLBC**: Is DLBC enabled under the current QNN/Genie configuration? If enabled, what compression ratio does it achieve for LLM activations/KV, and how can this be confirmed?
4. **Actual HTP runtime frequency**: What HTP/DDR frequencies correspond to the `llm_decode_burst` profile? Why does the microbenchmark reach only 2.7 TOPS/s FP16? Is the burst profile already the maximum hardware frequency?
5. **HMX utilization**: Does W8A16 MatMul actually run on the HMX INT sub-unit or on HVX? Is there a profiling method to identify which execution unit runs each operator?
6. **Future SDK support**: Which QAIRT version will add an INT4 MatMul kernel for v81, fix LADE, and support multi-HTP Genie execution?
7. **Correct `hvx_threads` value in perf_config**: The official architecture provides 8 HVX units per HTP, while our build configuration works with `hvx_threads: 4`. Is setting it to 8 effective?
8. **Native KV INT8**: What ONNX modifications and QNN configuration are required? Is the relevant kernel path available in 2.48?

---

## 13. Current Device State (Verified 2026-08-11)

- Device serial number: `REDACTED` (accessed via `ssh <JUMPHOST> "adb -s REDACTED ..."`)
- Deployed W8A16 baseline bundle: `/data/local/tmp/qwen3_06b_w8a16_local/`
- Deployed W8A16 LADE bundle: `/data/local/tmp/qwen3_06b_w8a16_lade/`
- Both bundles come from HF `vinccniv/sa8797p-qwen3-w8a16-bundles` (2026-08-10 v2)
- With the ADAS configuration + Qwen3 chat template, output is correct: `"前方有鹿横穿，注意减速避让。"` ("A deer is crossing ahead; slow down and avoid it.")

---

## 14. Reference Documents

This document consolidates measured/verified data from the following three source documents:

- `docs/SA8797P_DEPLOYMENT_KNOWLEDGE.md` (most detailed build logs, microbenchmarks, and error records)
- `docs/SA8797P_LLM_HARDWARE_LIMITATIONS_2026-08-11.md` (summary of hardware limitations)
- `docs/THROUGHPUT_INVESTIGATION_2026-08-05.md` (microbenchmarks and GVM bottleneck investigation)

Other references:

- ADAS deployment guide: `docs/ADAS_ANIMAL_WARNING_ON_SA8797P.md`
- HF bundles test report: `docs/SA8797P_Qwen3-0.6B_W8A16_HF_Bundles_Test_Report_2026-08-11.md`
- Qualcomm reference quantization project: `qwen3-vl-quantization/` (W8A16 + SpinQuant, single HTP)
- Troubleshooting: `docs/TROUBLESHOOTING.md`
- Knowledge base: `docs/KNOWLEDGE_BASE.md`
