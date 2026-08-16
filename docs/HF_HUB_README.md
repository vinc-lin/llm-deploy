---
license: apache-2.0
base_model: Qwen/Qwen3-0.6B
tags:
  - qualcomm
  - qnn
  - genie
  - sa8797p
  - hexagon-v81
  - w8a16
  - automotive
---

# SA8797P Qwen3 W8A16 Deployment Bundles

> ## 📌 Current baseline: **44.707 tok/s** — updated 2026-08-16
>
> The GQA KV-replication fix shipped and was measured on device on 2026-08-15.
>
> | | tok/s | |
> |---|---:|---|
> | **`qwen3_06b_w8a16_gqafix_ladekv`, basic mode** | **44.707 ± 0.030** | ✅ **the ship configuration** |
> | same bundle, LADE mode | 31.342 | ❌ parked — post-fix break-even rose to 2.30 accepted tok/call |
> | pre-fix `ladekv`, basic mode | 6.836 | the like-for-like control → the fix is **6.54×** |
>
> **⚠️ Two older numbers in this README are wrong and are being corrected.**
> **11.72 tok/s was never an AR-1 decode rate** — it is a *phase blend* measured
> on a bertcache bundle, where Genie generates through the CL=128 prefill graph
> for the first ~72 tokens before switching to AR-1. The honest pre-fix AR-1
> rate is 6.84. Three previously-reported findings were artifacts of quoting the
> blend as a decode rate: the "~75% build gap", the "LADE is −22%" result, and
> "our builds are +51% faster than the device team's".
>
> **Practical rule:** only compare decode rates between bundles that are
> *topologically pure* (no AR==CL graph). Of the 2026-08-14 `gqafix_*` bundles,
> only **`gqafix_ladekv`** and **`gqafix_pastkv2g`** are pure — the other six
> carry a bertcache prefill and will report a flatteringly fast blended number.
> The build-side gate is `scripts/validate/lint_bundle_topology.py`.
>
> ### 📦 Latest drop: [`2026-08-16-regime/`](2026-08-16-regime)
>
> **Start at [`2026-08-16-regime/DEPLOYMENT_AND_TEST_GUIDE.md`](2026-08-16-regime/DEPLOYMENT_AND_TEST_GUIDE.md)** —
> one self-contained document: deployment, test procedure, the exact metrics to
> record, and what every outcome means.
>
> Eight topologically-pure bundles that separate the two remaining performance
> models, plus `kit-v2/` (the scripted form of the same procedure). The
> `2026-08-14-gqafix/` drop's priorities 4 and 5 are **superseded — do not run
> them**; six of its eight bundles are blended.
>
> Full analysis: `docs/MAX_TPS_QWEN3_0.6B_V4.md` §1–§2 and `docs/REFERENCE.md`
> §6.8–§6.9 in the source repo.

Push-ready Genie T2T device bundles for the Qualcomm **SA8797P** (nordy / Gen5,
Hexagon v81) automotive SoC, Android GVM, built **entirely off-device** on
2026-08-10 with **QAIRT 2.48.40.260702** (exact match to the target runtime,
libGenie 1.19.0).

## Bundles

**Start here (2026-08-14): [`2026-08-14-gqafix/`](2026-08-14-gqafix)** — the
whole drop lives in that one folder, with its own README, the eight bundles, a
self-contained device session (`kit/`), and the profiling material. Everything
below this section is history kept for provenance.

These remove **GQA KV-head replication**, which an op-level profile on
2026-08-13 showed was consuming **74.7% of every decode step's DSP cycles**
(261.8M of 350.3M). That profile labelled the ops "attention-mask broadcast";
inspecting the shipped `decode.dlc` showed the mask is never expanded and the
56 ops are `repeat_kv`, materialising 8 KV heads into 16 so a 16-head MatMul
can consume them. That is ~264 MB of intermediate tensor per step, but it never
reached DDR (decode `write_total_bytes` = 419,840 B) — the cost was cycles, not
bytes. The attention
MatMuls now batch over the 8 KV heads directly (`1x8x2x1152` instead of
`1x16x1x1152`).

## Current baseline: 44.707 tok/s

**Device-measured 2026-08-15**: `2026-08-14-gqafix/qwen3_06b_w8a16_gqafix_ladekv`
in **basic** mode — **44.707 ± 0.030 tok/s**, TTFT 103 ms, init 796 ms. That is
**+6.5×** over the same topology pre-fix (6.836 tok/s), and it is the ship
configuration.

**Run it in basic mode, not LADE.** On the same binary LADE measures 31.342
tok/s — a 30% regression. Post-fix the decode step is 22.4 ms, so there is no
longer a slow call for speculation to amortise.

Every other number below is **pre-fix history**, kept for provenance. The
earlier baselines were 11.72 tok/s (basic `local`) and 9.18 (LADE), both
2026-08-13.

Rows marked *measured* are device numbers. Rows marked *projection* are not —
and two projections in this table have already been falsified on device, so
please treat the remaining ones as hypotheses.

| File | Variant | Status |
|---|---|---|
| **[`2026-08-14-gqafix/`](2026-08-14-gqafix)** | **GQA replication fix — 8 bundles + kit + profiling** | ✅ **shipped and measured: 44.707 tok/s basic, the current baseline.** Use `qwen3_06b_w8a16_gqafix_ladekv` with `genie_dialog_basic.json`. ⛔ Do not use `gqafix_hybrid` — it emits degenerate output |
| `qwen3_06b_w8a16_fuseqkvgu_ladekv.tar.gz` | ladekv + QKV + Gate-Up fusion (2026-08-13) | **superseded — pre-GQA-fix.** Its ~11.5–12.4 tok/s projection is about a quarter of the measured baseline. Fusion remains an open lever, but must be rebuilt on the gqafix base to mean anything |
| `qwen3_06b_w8a16_fuseqkvgu_ladekv_socmodel72.tar.gz` | config A/B **C1**: `soc_model 0 → 72` at build time | same weights/DLCs as the row above, ctx-bin recompiled — isolates one variable |
| `qwen3_06b_w8a16_fuseqkvgu_ladekv_hvx8.tar.gz` | config A/B **C2**: `hvx_threads 4 → 8`, build **and** runtime | Pre-fix base, **never measured**. Note every other bin here is compiled `numHvxThreads=4` against 8 available HVX units, and `hvx_threads` is build-time only — so this A/B, rebuilt on gqafix, is now the top open experiment |
| `qwen3_06b_w8a16_ladekv.tar.gz` | 0.6B lade-fix (2026-08-11): past-KV prefill AR=128 CL=1152 + decode AR=1 + verify AR=32 | Pre-fix reference. 10.8 tok/s LADE on a simple prompt; its **basic** rate was later measured at **6.836 tok/s**, which is the number the 44.707 baseline improves on |
| `qwen3_06b_w8a16qh_ladekv.tar.gz` | **ladekv + W8 lm_head (2026-08-11)**: same 3 past-KV graphs, lm_head weight INT8 | **9.3 tok/s — measured: −14% vs `ladekv`, a REGRESSION.** It buys ~155 MB/token of DDR but costs ~10% n-gram acceptance, and acceptance dominates. Kept only as evidence; do not ship. (This row previously read "est. +19%" — that projection was wrong.) |
| `qwen3_06b_w8a16_local.tar.gz` | Qwen3-0.6B W8A16 baseline (2-graph, bertcache prefill) | Pre-fix reference; 11.72 tok/s basic (2026-08-13). Note topology-A rates blend a fast bertcache phase with slow AR-1 — quote the phase |
| `qwen3_06b_w8a16_fusegu_local.tar.gz` | + Gate-Up fusion | projection: ~8–9 tok/s + TTFT gain |
| `qwen3_06b_w8a16_fuseqkv_local.tar.gz` | + QKV fusion (encodings surgery) | isolates QKV effect |
| `qwen3_06b_w8a16_fuseqkvgu_local.tar.gz` | + QKV + Gate-Up, basic mode | fusion measured **+15%** (8.98 vs 7.79 tok/s) on an equivalent working build. The older "~12–16 tok/s if BW scales" projection in this row was wrong — fusion does not reduce bytes/token, it improves access pattern |
| `qwen3_17b_w8a16_local.tar.gz` | Qwen3-1.7B W8A16 baseline (3.9 GB ctx-bin) | **do not test** — still carries the broken last-token-only prefill head (see 2026-08-11 note) |
| `qwen3_06b_w8a16_lade.tar.gz` | 0.6B baseline + AR=32 verification graph | **SUPERSEDED by `ladekv` for lade mode** — its bertcache prefill crashes `type:"lade"` (see below); basic mode fine |

Each bundle: a ctx-bin (two graphs for the `_local` variants; **three** —
prefill + decode + verify32 — for the `lade*` variants), cross-graph
weight-shared, `vtcm_mb=16`, unsigned PD, O3; the 7 required ARM64 libraries,
`genie-t2t-run` (aarch64-android), `tokenizer.json`, dialog JSON, and
`htp_backend_ext_config.json` — flat layout, no `lib/` subdir.

> **Third-party components.** The `apache-2.0` tag covers this repository's own
> scripts, configs and documentation. It does **not** cover the Qualcomm QAIRT
> 2.48.40.260702 runtime binaries that each bundle embeds for convenience —
> the 7 ARM64 `.so` files (`libGenie.so`, `libQnnHtp.so`, `libQnnSystem.so`,
> `libQnnHtpPrepare.so`, `libQnnHtpNetRunExtensions.so`, `libQnnHtpV81Stub.so`,
> `libQnnHtpV81Skel.so`) and `genie-t2t-run`. Those remain subject to the
> Qualcomm AI Engine Direct / QAIRT SDK licence you accepted when obtaining the
> SDK. Model weights derive from Qwen/Qwen3-0.6B under its own licence. If you
> need bundles without the Qualcomm binaries, say so and they can be rebuilt
> with the runtime omitted.

Quantization: AIMET 2.36 PTQ, per-channel symmetric INT8 weights + 16-bit
activations; `embed_tokens`/final-norm/`lm_head`/K-V-proj outputs kept FP16;
weight clip to ±127 steps applied.

## Deploy

```bash
adb push qwen3_06b_w8a16_local.tar.gz /data/local/tmp/
adb shell 'cd /data/local/tmp && tar xzf qwen3_06b_w8a16_local.tar.gz'
adb shell 'cd /data/local/tmp/qwen3_06b_w8a16_local && LD_LIBRARY_PATH=. ./genie-t2t-run -c genie_dialog.json -p "<prompt>"'
```

Smoke-test the **baseline** bundle first so any failure isolates to a fused
variant rather than the environment.

> **2026-08-11 — all 0.6B bundles rebuilt (prefill logits fix).** The first
> on-device run exposed that the original prefill graphs emitted
> last-token-only logits `[1,1,V]`, which Genie's basic dialog cannot use
> (it samples logits row `n_process-1` of an all-position tensor over
> left-aligned input — the mismatch is silent and produces garbage output
> from the first token). Every bundle now ships a prefill graph with
> all-position logits `[1,128,V]`. Bundles downloaded before this date
> produce garbage on device — re-download.
> **`qwen3_17b_w8a16_local.tar.gz` is NOT yet rebuilt** (deferred until the
> 0.6B fix is device-validated) and still carries the broken prefill head —
> do not test it.

## The 2026-08-14 device session

Everything for it is under **[`2026-08-14-gqafix/`](2026-08-14-gqafix)** —
read that folder's `README.md` first, then its `kit/runsheet.md`. It is
self-contained and should need no round-trip with the build side.

The decisive run is **not** in `run_all.sh` (it uses `qnn-net-run`, not
`genie-t2t-run`): profile
`2026-08-14-gqafix/profiling/qwen3-0.6b-w8a16-gqafix-decodeonly_ctx.bin`
against the 350,302,972-cycle baseline. **Expectation: ~90M.**

If you have time for exactly one Genie run, make it `p3_a1_ladekv_basic` —
basic mode on the plain `qwen3_06b_w8a16_ladekv` bundle you already have. It
has never been measured, and it retroactively de-confounds every 3-graph
number taken so far, because the 6.70 tok/s in Test 4 came from the **qh**
bundle and therefore conflates W8 head, graph count and build lineage.

## Validation performed (local, device-free)

- Export wrapper vs HF forward: max |Δlogits| = 4.12e-05
- ONNX prefill + 8-step greedy decode chain: token-identical to HF `generate`
- **Genie-runtime read pattern parity** (left-aligned input, logits row n−1 —
  exactly what qualla samples on device): greedy argmax matches HF on all
  reference prompts (`scripts/validate/parity_qualla_read.py`)
- Quantsim vs FP32 last-token argmax agreement: 3/4 reference prompts
- **Grouped-GQA equivalence (2026-08-14):** max |Δ| = 6e-16 in float64 against
  the replicating form, across decode AR=1, verify AR=32 and prefill AR=128
  both with and without past, fused and unfused; bit-identical for decode
- **Grouped-GQA ONNX parity:** wrapper-vs-HF 2.12e-05, prefill argmax identical,
  8-step greedy decode chain token-identical to HF
- **Topology assertion per DLC** (`scripts/validate/lint_gqa_ops.py`): all four
  exported graphs — bertcache prefill, decode, verify32, past-KV prefill — carry
  0 replication ops and attention MatMuls batched over 8 KV heads
- **qualla-read parity on the gqafix build:** 6/6 (4 single-chunk + 2 chunked
  through growing past-KV)
- Genie graph-I/O contract (tensor names, transposed-key KV layout, new-slice
  outputs) extracted from the SDK's shipped qualla engine source — see `docs/`

### Lookahead-decoding bundles (lade / ladekv)

**2026-08-11 device runs SIGSEGV'd `type:"lade"` on `qwen3_06b_w8a16_lade`.
Root cause (two independent qualla bugs, confirmed in SDK source):**

1. The old bundle's prefill graph is "bertcache" style (AR==CL==128, no
   past-KV inputs). qualla's strategy planner re-processes prompt history
   through such a graph, inflating `n_process` beyond the lade attention map —
   heap out-of-bounds position ids become RoPE-table byte offsets → SIGSEGV
   (kvmanager.cpp:421-429 → attention-mask.cpp:236-240 → nsp-model.cpp:2196).
2. Independently, lade with a **1-token prompt** (e.g. `-p "Hi"`) executes
   `rand() % (tokens.size()-1)` = modulo zero (lhd-dec.cpp:120) — the crash
   register `x0=0x6b8b4567` is exactly `rand()`'s first output. **No bundle
   fixes this; lade prompts must tokenize to ≥ 2 tokens.**

**`qwen3_06b_w8a16_ladekv.tar.gz` fixes (1)**: its prefill is a past-KV graph
(AR=128, CL=1152, past=1024, all-position logits), so the bertcache code path
no longer exists. All three graphs are prepared with `vtcm_mb=16`/O3 (the old
bundle's verify32 silently got defaults: 4 MB VTCM + 24 MB spill; now 0 spill).
Graph routing is numeric best-fit: prompts ≤32 tokens run on the AR-32 graph,
33–128 on AR-128, and prompts >128 now **chunk** (128/step, up to ~1024 total).

**Testing `ladekv`** (config window 8 / ngram 3 / gcap 8, max batch 32):
- `genie_dialog.json` = lade mode, `genie_dialog_basic.json` = basic, same bin.
- Use prompts of **≥ 2 tokens** in lade mode (bug 2 above).
- Expected in basic mode: the ~23 tok/s "chunked-prefill decode" phase is GONE
  (it was an artifact of the bertcache graph) — generation runs ~6.5 tok/s
  from token 1. That trade is intentional; lade mode is where the win is.
- Report `tokenAcceptance` from the KPI block + tok/s, and please run one
  prompt >128 tokens (basic mode) to confirm chunked prefill.
- Quality note: Qwen3-0.6B is post-trained — apply the chat template
  (`<|im_start|>user\n...<|im_end|>\n<|im_start|>assistant\n`) to kill the
  preamble/repetition seen in raw-prompt runs; EOS 151645 then fires naturally.

### `qwen3_06b_w8a16_fuseqkvgu_ladekv` — the fused LADE build (2026-08-13)

Identical topology, encodings recipe and runtime configs to `ladekv`; the only
change is **QKV + Gate-Up fusion**, applied to all three graphs with encodings
surgery on 28/28 layers. It is the first build to combine fusion with lookahead
decoding, and it is **not yet measured** — `ladekv` (10.8 tok/s) stays the
reference until it is.

It ships **three** dialog configs on one binary, and its own `README.md`:

| Config | Mode | Use for |
|---|---|---|
| `genie_dialog.json` | lade, greedy, unbounded | the throughput number |
| `genie_dialog_basic.json` | basic AR-1, greedy | the A/B baseline on identical weights |
| `genie_dialog_demo.json` | lade, temp 0.85 / top-k 50 / top-p 0.9 | anything interactive |

**Use `genie_dialog_demo.json` for free-form runs.** The greedy parity configs
(temp 0) are correct for measurement but produce repetitive output until
`Context Size was exceeded` — that is the shipped default's fault, not the
model's.

> **2026-08-14 — `genie_dialog_demo.json` was fixed; re-pull if you have an
> older copy.** It carried `"max-num-tokens": 256` alongside `"type": "lade"`,
> which SIGSEGVs (exit 139) on the first speculation step. That pair shipped in
> the `fuseqkvgu`, `socmodel72` and `hvx8` bundles, so **every demo run of those
> three died**. Generation is now bounded by `context.size` and EOS instead, and
> the combination is refused by a linter that runs during bundling.

Please report, for both `genie_dialog.json` and `genie_dialog_basic.json`:
sustained tok/s, **accepted tokens per verify call**, and per-call latency.
`tok/s ≈ acceptance ÷ latency`. If fusion buys latency but costs acceptance it
is a net regression — that is exactly how the `qh` variant failed at −14%.

#### The two `_socmodel72` / `_hvx8` config A/Bs

Same fused DLCs, same weights, same encodings — **only the ctx-bin was
recompiled**, with exactly one build-config variable changed. So a device delta
is attributable to that variable and nothing else.

| Bundle | Variable | ctx-bin |
|---|---|---|
| `qwen3_06b_w8a16_fuseqkvgu_ladekv` | reference | 1,102,467,072 B |
| `…_socmodel72` | `soc_model: 0 → 72` (build only) | 1,104,285,696 B |
| `…_hvx8` | `hvx_threads: 4 → 8` (build **and** the shipped runtime ext config) | 1,104,973,824 B |

**Test order matters:** measure the plain fused bundle first — it is the
reference these are an A/B against — then one variant per run. Do not compare
either of these directly against the unfused `ladekv`, which differs in two
ways at once. If a variant is flat or negative, discard it; neither knob is
load-bearing and both exist only to see whether they are free.

## Known unknowns (need hardware)

- **Does fusion's +15% survive lookahead decoding?** That is the whole point of
  `qwen3_06b_w8a16_fuseqkvgu_ladekv`. Fusion does **not** reduce bytes/token
  (~960 MB/token decode, all variants — the old 3.4× figure was at the
  unavailable `vtcm 24`); its gain is access-pattern quality, measured at +15%
  in basic mode. LADE is **acceptance**-bound rather than purely latency-bound,
  so the gain may or may not carry. Report accepted-tokens-per-verify-call
  alongside tok/s or the result cannot be interpreted.
- The ~20% build-side gap: the device team's unfused builds decode at 7.8 tok/s
  where equivalent local builds measure 6.5, with identical runtime configs and
  a ctx-bin 77 MB smaller. Cause unknown; if found it stacks on everything else.
- ~~~100+ ms per decode step is unaccounted for~~ **Answered 2026-08-13/14:**
  74.7% of decode cycles were GQA KV replication (see the `_gqafix_` bundles
  above). What is *not* yet known is whether removing it converts into
  throughput, or whether decode then hits a weight/KV streaming floor — the
  step must still move 751 MB of weights plus a 132 MB KV read. The kit is built
  to answer exactly that; `2026-08-14-gqafix/kit/decision_table.md`
  pre-commits both branches.
- **Weight sharing regresses in 2-graph gqafix bins.** Any bin containing the
  CL=128 bertcache prefill under grouped attention drops ~444 MB of INT8
  decoder weights out of the shared pool into per-graph constants, so they are
  stored twice (1.523 GB vs 1.087 GB). Bins without that graph share perfectly.
  The weight *bytes* are unchanged, so per-step traffic should not move, but it
  costs disk and possibly init time. Root cause not established — it is in the
  ctx-bin generator's layout decision, not the graph topology, which gates
  clean. The single-graph `profiling/` bin is unaffected.

See `docs/LOCAL_ENV.md` for full provenance, reconstruction caveats, and the
aimet-torch 2.36 workarounds baked into the build scripts.
