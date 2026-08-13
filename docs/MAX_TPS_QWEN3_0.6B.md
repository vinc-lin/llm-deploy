# Qwen3-0.6B maximum-TPS build — the complete process

*2026-08-13. Companion to `docs/REFERENCE.md` (facts, corrections) and
`docs/BUILD_GUIDE.md` (per-variant recipes). This document is the one path to
the fastest 0.6B bundle we know how to build, with every measured alternative
that was rejected and why. Section references like §6.6 point into
REFERENCE.md unless marked otherwise.*

---

## 0. The target configuration, and the evidence for it

**Highest measured decode throughput on the SA8797P GVM guest: 10.8 tok/s
sustained** — `qwen3_06b_w8a16_ladekv`, LADE speculative decoding, device run
2026-08-11. Everything below either reproduces that build or stacks a lever on
top of it.

| Decision | Choice | Evidence |
|---|---|---|
| Quantization | **W8A16** — INT8 per-channel symmetric weights, FP16 activations | The only recipe that passes quality gates. W8A8: garbage all 5 variants. W4A16: fails accuracy **and** v81 has no INT4 kernels (§4.1). |
| `lm_head` | **FP16** — do NOT pass `--quant-head` | qh is a measured **−14%** under LADE (9.3 vs 10.8): it costs ~10% n-gram acceptance and the DDR saving never reaches the device (§6.4). |
| Topology | **B: all-past-KV, 3 graphs** — `prefill` AR=128 CL=1152, `decode` AR=1 CL=1152, `verify32` AR=32 CL=1152 | Required for LADE (an AR==CL bertcache graph SIGSEGVs it, §3.4). Costs the ~23 tok/s bertcache early phase in basic mode — irrelevant, we ship LADE. |
| Decoding mode | **LADE** (`type:"lade"`), window 8 / ngram 3 / gcap 8 | 1.7× over AR-1: ~1.94 accepted tokens per 180 ms verify call (§6.2). The single biggest lever that exists on this device. |
| Fusion | **QKV + Gate-Up — the untested +15% candidate** (Phase B) | Our old "no gain" verdict compared against a garbage-output v1 build. The device team's *working* fused build: **8.98 vs 7.79 tok/s (+15%)**, decode DDR 880 vs ~960 MB (§6.6, correction #15). Never yet combined with LADEKV. |
| Build config | `O:3`, `vtcm_mb:16`, weight sharing ON, + two A/Bs: `soc_model:72`, `hvx_threads:8` (Phase C) | 16 MB VTCM is the unsigned-PD cap. `soc_model` is currently 0 (§8.4); runtime always uses 8 HVX regardless of the configured 4 (§8.9). |
| Perf profile | `llm_decode_burst`, `rpc_polling_time: 9999`, `poll: true`, `cpu-mask: "0xe0"` | Tier 1 of 4; burst → low_power_saver spans 1.95× (HTP doc §3.3). Already in our shipped configs. |

**Expected outcomes** (only the first row is measured; the rest are labeled
projections — remember the qh projection failed on-device, §6.4):

| Build | tok/s | Status |
|---|---|---|
| Phase A: unfused LADEKV | **10.8** | measured 2026-08-11; `qwen3_06b_w8a16_ladekv.tar.gz` |
| Phase B: fused LADEKV | ~11.5–12.4 | **built and shipped 2026-08-13** (`qwen3_06b_w8a16_fuseqkvgu_ladekv.tar.gz`); the number is still a projection: 1.94 acc/call ÷ (180 ms × 880/957 … ÷1.15) — must be device-measured |
| Phase C: + `soc_model`/`hvx_threads` | unknown, ≥ 0 | free if flat, cheap to try |
| Hard ceiling (AR-1 basic, any build) | ~7–8 | ~880–960 MB streamed/token at 6–7 GB/s effective DDR (§1) |

---

## 1. Prerequisites

```bash
source scripts/env.sh          # ALWAYS first — QAIRT_SDK, PY envs, LD_LIBRARY_PATH
disk_guard 20                  # each export stage re-checks, but start clean
ls $LLMDEPLOY_DATA/models/Qwen3-0.6B/   # HF checkpoint + tokenizer.json present
```

- Envs: `qwen3-deploy` (AIMET/torch/onnx) and `qairt-py312` (converter) — both
  managed by `env.sh`, no activation needed.
- Time budget: Phase A ≈ 2.5 h total (full_build ~1 h, lade ~45 min, ladekv
  ~45 min; each ctx-bin generation is ~15–20 min of that). Phase B ≈ +2 h.
- Disk: exports are the fat step. `disk_guard` is called inside every script;
  never bypass it — C: running dry is a VM hard-crash, not an error (§9).

---

## 2. Phase A — the proven 10.8 tok/s build (unmodified scripts)

Three chained scripts. Do not skip the first two: `ladekv_build.sh` consumes
their artifacts (`decode.dlc`, `verify32.dlc`, the prefill quant dir's
encodings) and adopts — never recalibrates — the baseline's quantization, which
is what keeps KV scales byte-identical across all three graphs (§3.2).

### A.1 Baseline (donor + decode graph)

```bash
./scripts/build/full_build.sh qwen3-0.6b-w8a16 128 1024
```

Gate before continuing: the run prints the `--eval` argmax score — needs
**≥ 3/4**. `qairt-dlc-info -i $LLMDEPLOY_DATA/work/dlc/qwen3-0.6b-w8a16/prefill.dlc | grep logits`
must show `1,128,151936` (never `1,1,…` — that is the v1 garbage bug, §3.1).

### A.2 Verify graph (AR=32)

```bash
./scripts/build/lade_build.sh qwen3-0.6b-w8a16 128 1024 32
```

Gate: `parity_verify.py` batched rows match HF (~3e-05). The 3-graph ctx-bin
this stage emits is an **intermediate** — its bertcache prefill cannot ship for
LADE.

### A.3 Past-KV prefill + the real 3-graph ctx-bin

```bash
./scripts/build/ladekv_build.sh qwen3-0.6b-w8a16 128 1024 128
```

The script runs its own gates: `parity_ladekv_read.py` must pass **6/6**
(4 single-chunk + 2 chunked prompts — this replays qualla's exact device feed
pattern), converter output must show `logits 1,128,151936` *and*
`past_key_0_in`, and the final stage prints all three graphs.

### A.4 Verify the ctx-bin before it goes anywhere

```bash
CTX=$LLMDEPLOY_DATA/work/ctxbin/qwen3-0.6b-w8a16-ladekv
python3 -c "
import json; d = json.load(open('$CTX/info.json'))
for g in d['info']['graphs']: print(g['info']['graphName'])"
```

- Graph names must be exactly `prefill`, `decode`, `verify32` — matching
  `graph_names` in **both** `configs/htp_backend_config.json` (build-time) and
  `configs/htp_backend_ext_config.json` (device). A missing name silently
  compiles that graph with 4 MB VTCM / 24 MB spill and, for the verify graph,
  is precisely the LADE SIGSEGV the device team hit (correction #16). Names
  come from the DLC **filenames at conversion time** — renaming files after
  the fact does nothing (§3.3).
- Size ≈ **1.10–1.11 GB**. If it is ≈ the sum of the DLCs (1.8–2.2 GB),
  weight sharing silently failed — check `context.weight_sharing_enabled` in
  `configs/htp_backend_config.json` and rebuild (§8.2).

### A.5 Bundle

```bash
./scripts/build/bundle.sh qwen3_06b_w8a16_ladekv \
    $LLMDEPLOY_DATA/work/ctxbin/qwen3-0.6b-w8a16-ladekv/qwen3-0.6b-w8a16-ladekv_ctx.bin \
    $LLMDEPLOY_ROOT/configs/genie_dialog_qwen3_0.6b_lade.json
```

Then add the basic-mode config to the same bundle (BUILD_GUIDE §7 has the
snippet) so the tester can A/B lade vs basic on one binary. **Also add a demo
config**: the shipped dialogs are greedy parity configs (temp 0, no
`max-num-tokens`) and will loop until `Context Size was exceeded` in free-form
use — the device team hit exactly this. For demos: `max-num-tokens` set,
temp 0.85 / top-k 50 / top-p 0.9 (verified on device, HTP doc §8.2).

---

## 3. Phase B — the +15% candidate: fused LADEKV

**Status: BUILT 2026-08-13, shipped, not yet device-measured.**
`qwen3_06b_w8a16_fuseqkvgu_ladekv.tar.gz` (933,127,630 B) is on the HF repo
with its own README. QKV+Gate-Up fusion measured +15% in basic mode on a
working build (§6.6); this is the first build to combine it with topology B.
Since per-call verify latency is the same weight stream as decode, the gain
should carry — but the qh episode proved projections die on this device, so
treat it as an experiment with Phase A as the fallback ship.

Device-free gates, all passed (see §3.3 for the full list):

| Gate | Result |
|---|---|
| fused AIMET `--eval` last-token argmax | 3/4 — identical to the non-fused donor, same prompt failing |
| fused export wrapper vs HF | max abs 4.67e-05 |
| fused AR=32 batched verify vs HF, all positions | max abs 3.05e-05, 8/8 argmax |
| QKV encodings surgery | 28/28 layers |
| `parity_ladekv_read.py` on the fused past-KV prefill | 6/6 (4 single-chunk + 2 chunked) |
| ctx-bin | 1,102,467,072 B, graphs `prefill`/`decode`/`verify32`, `lm_head` `Float_16` |

**What is NOT known:** whether fusion's basic-mode +15% survives lookahead
decoding. LADE is acceptance-bound, so the device report must include accepted
tokens per verify call, not just tok/s.

### B.1 What the scripts needed — implemented 2026-08-13

Until 2026-08-13 `lade_build.sh` and `ladekv_build.sh` hardcoded two things
that are wrong for a fused variant:

1. Their AIMET export calls passed no fusion flags, so they exported *unfused*
   verify/prefill graphs against fused donor artifacts. The export wrapper's
   structure comes straight from `--fuse-qkv` / `--fuse-gate-up`
   (`quantize_aimet.py` `build_wrapper`), including on the `--export-decode`
   path — so this is a structural mismatch, not a cosmetic one.
2. They converted against `$QP/model_filtered_renamed.encodings`; a fused
   build's truth is the **surgery** file, `$QP/model_surgery.encodings` (the
   donor `q_proj` INT16 encoding grafted onto the Q split — `qkv_build.sh`
   stage 5).

Both scripts now take two optional env vars, and are **exact no-ops when
unset** — the non-fused path is byte-for-byte the command it always was:

| Var | Effect |
|---|---|
| `FUSE_FLAGS` | word-split and appended to every `quantize_aimet.py` call |
| `ENC_SRC` | replaces the encodings file used for the rename *and* the conversion. It is passed through unchanged because surgery encodings are already in renamed-I/O space — the same handling as `qkv_build.sh` stages 6–7. Checked for existence up front. |

Nothing else in either script is fusion-sensitive: graph names still come from
the DLC filenames (`prefill.dlc` / `decode.dlc` / `verify32.dlc`), so the
`graph_names` lists in both HTP configs stay correct for fused builds.

### B.2 The build sequence

```bash
# 1. Fused 2-graph build (needs the Phase-A baseline as encodings donor)
./scripts/build/qkv_build.sh qwen3-0.6b-w8a16-fuseqkvgu \
    $LLMDEPLOY_DATA/work/quant/qwen3-0.6b-w8a16-prefill 128 1024 --fuse-gate-up

# 2. Fused verify32 (extended lade_build.sh)
FUSE_FLAGS="--fuse-qkv --fuse-gate-up" \
ENC_SRC=$LLMDEPLOY_DATA/work/quant/qwen3-0.6b-w8a16-fuseqkvgu-prefill/model_surgery.encodings \
./scripts/build/lade_build.sh qwen3-0.6b-w8a16-fuseqkvgu 128 1024 32

# 3. Fused past-KV prefill + 3-graph ctx-bin (extended ladekv_build.sh)
FUSE_FLAGS="--fuse-qkv --fuse-gate-up" \
ENC_SRC=$LLMDEPLOY_DATA/work/quant/qwen3-0.6b-w8a16-fuseqkvgu-prefill/model_surgery.encodings \
./scripts/build/ladekv_build.sh qwen3-0.6b-w8a16-fuseqkvgu 128 1024 128
```

The encodings lineage rule (§3.2) is the thing to protect: all three DLC
conversions must use the **same** surgery encodings file. Mixed
surgery/non-surgery conversions produce KV quant params that differ across
graphs — a hard Genie load error at best, silent garbage at worst.

### B.3 Gates, beyond the Phase-A set

- `parity_ladekv_read.py` and `parity_verify.py` must pass against the
  **fused** ONNX — fusion changes the graph, so the v1-era lesson applies:
  never assume a fused build is numerically the same build.
- Confirm the graft actually happened: the build should report 28/28 QKV
  grafts (from `qkv_surgery.py`), and the ctx-bin should land near the fused
  signature ≈ 1.09–1.11 GB.
- On device, measure **acceptance** (tokens/verify-call), not just tok/s:
  fusion must not repeat the qh failure mode of buying per-call bytes with
  acceptance. If acceptance holds ~1.94 and calls get faster, ship it; if
  acceptance drops, Phase A remains the ship.

---

## 4. Phase C — config A/Bs (cheap, possibly free performance)

**Status: both built and shipped 2026-08-13**, as
`qwen3_06b_w8a16_fuseqkvgu_ladekv_socmodel72.tar.gz` and
`…_hvx8.tar.gz`. Each is a config edit + ctx-bin regeneration (converter,
weights and encodings untouched) + re-bundle, so any device delta is
attributable to the single variable. Change **one variable per device run**,
and measure the plain Phase B bundle first — it is the reference these are
an A/B against.

| # | Change | Where | Rationale |
|---|---|---|---|
| C1 | `soc_model: 0` → `72` (add `soc_id: 72` if the generator accepts it) | `configs/htp_backend_config.json` `devices[0]` | We build genericized; SA8797P is soc 72, and Qualcomm's docs say O=3 + soc_model can enable further optimizations. The device team's verified build sets both (§8.4). |
| C2 | `hvx_threads: 4` → `8` | both `configs/htp_backend_config.json` and `configs/htp_backend_ext_config.json` | The runtime uses 8 HVX threads no matter what is configured; whether the *compiler* schedules better when told 8 is unmeasured (§8.9). |

Regenerate with the stage-5 command from `ladekv_build.sh` (the
`qnn-context-binary-generator` call), re-run the A.4 checks, re-bundle. If a
variant is flat or negative on device, revert — neither is load-bearing.

---

## 5. On-device configuration and measurement

Runtime rules — every one of these has a measured failure behind it:

- **LADE config guardrail:** `(ngram−1) × (window + gcap) ≤ 32` (the verify
  AR). Shipped 8/3/8 = exactly 32. Oversizing silently routes verification
  batches to a graph that cannot serve them.
- **Prompts must tokenize to ≥ 2 tokens** — a 1-token prompt hits `rand() % 0`
  in qualla's warmup and reads ~7 GB out of bounds (§3.4).
- **`htp_backend_ext_config.json` must list all three graph names** — this
  file ships in the bundle; if a variant renames graphs, regenerate it.
- Dialog `context.size` ≤ 1024 (ctx-bin max CL is 1152; prompts > 128 tokens
  are chunked automatically by the past-KV prefill).
- Apply the Qwen3 chat template with the **empty `<think>\n\n</think>` block**
  in the assistant prefix — otherwise thinking mode triggers and TTFT/latency
  balloon (HTP doc §8.6).
- Keep `perf_profile: "llm_decode_burst"` (tier 1). The 4-tier ladder spans
  1.95×; `default` alone costs half the throughput.

Measurement protocol (so numbers stay comparable across builds):

```bash
LD_LIBRARY_PATH=. ./genie-t2t-run -c genie_dialog.json \
    -p "<templated prompt>" --profile profile.json
```

- Report **sustained tok/s** from the profile's TGR on warm runs, 3 reps,
  fixed prompt set. Warm ≠ cold: first run after a reconnect pays 1.8–2.0 s
  init vs ~790 ms warm.
- For LADE builds also report **accepted tokens per verify call** (total
  tokens ÷ verify calls) and the per-call latency — tok/s ≈ acceptance ÷
  latency, and acceptance is the variable that moves (§6.3).
- Compare cold-start numbers only like-for-like: init→first-logits vs the
  same, never vs TTFT (that units mismatch produced a phantom "+134%
  regression", correction #13).

---

## 6. Do not do (all measured, none worth re-running)

| Temptation | Result |
|---|---|
| `--quant-head` to cut lm_head DDR | **−14% tok/s** under LADE; ctx-bin shrinks only 12.5 MB; the DLC saving re-materializes at prepare time (§6.4) |
| W8A8 activations | Garbage output, five variants (v15–v19) — v81 MatMul has no per-channel activation kernel (§4.1) |
| W4A16 / INT4 weights | Fails accuracy at 0.6B **and** cannot execute: zero INT4 kernels in `htp_v2.json`, converter folds s4 → f16 (§4.1) |
| Keeping the bertcache (AR==CL) graph for its 23 tok/s early phase | SIGSEGVs LADE (§3.4). Two graphs sharing an (AR, CL) pair is also a hard load error. |
| `kv-quantization: true` in the dialog JSON | No-op on QnnHtp — the flag only exists in the CPU backend (§4.1) |
| 2-core ctx-bin / multi-core JSON | Error 5005 or *slower* (3.96 tok/s); Genie 1.19 is single-core (§4.1) |
| `vtcm_mb: 24` | Compiles, then error 5005 at runtime on unsigned PD |
| Renaming DLCs to fix graph names after conversion | Names are baked at conversion from the output basename; rename does nothing (§3.3) |

---

## 7. If Phase A–C is still not enough

In descending order of expected value:

1. **The ~20% build-side mystery (§8.8).** The device team's unfused builds
   decode at 7.8 tok/s where ours measure 6.5, identical runtime configs;
   their ctx-bin is 77 MB smaller. Obtain their exact converter/quantizer
   invocations and diff a `qnn-context-binary-utility --json_file` dump of
   their bin against ours. If it transfers, it stacks on everything above —
   potentially ~13 tok/s fused-LADE.
2. **Learned draft head (`eaglet` / `spd`).** LADE is acceptance-bound at
   ~1.94; per-call latency is pinned by DDR. A learned draft raises acceptance
   in a way n-gram matching cannot, and the SDK ships example configs. This is
   the only known software lever past ~12 tok/s.
3. **LADE-only ctx-bin (drop the AR-1 decode graph).** The decode graph is
   registered but never invoked in LADE mode, yet forces an AR-32↔AR-1 KV
   reshape every iteration (§8.5). Untested; costs basic-mode fallback in the
   same bundle.
4. **Environmental (not ours to fix):** the guest drives 1 of 4 HTPs; signed
   PD, DLBC status, and the hypervisor HTP mask are FAE questions (§8.6). A
   QNX-native 4-core EVB runs a 4B model at 129.7 tok/s — the gap is the
   environment, not the model.
