# Qwen3-0.6B standalone build kit — design

*Spec, 2026-08-22. Extracts the known-optimal SA8797P 0.6B path out of
`llm-deploy` into a self-contained tree that runs on a fresh Linux box with
QAIRT 2.48.40 and nothing else from this repo.*

Authority: `docs/REFERENCE.md` is the source for every measured number quoted
here. Where this spec and REFERENCE disagree, REFERENCE wins.

---

## 1. Purpose

Produce `kit-06b/` — a standalone build kit with two targets:

| Target | Output | Status of the recipe |
|---|---|---|
| **`ship`** | `qwen3_06b_w8a16_gqafix_ladekv` bundle | device-measured **44.707 ± 0.030 tok/s**, 2026-08-15 |
| **`variants`** | the ranked speed-experiment arms | predicted only — none has been run on device |

The kit must run without `scripts/`, `configs/`, or `docs/` from this repo. It
still requires the QAIRT SDK, a Python toolchain, and the HF checkpoint; it does
not require a device (building is device-free) and it cannot measure tok/s.

## 2. What "optimal" means — the settled decision set

These are decided by measurement, not open for re-exploration by the kit. The
kit's job is to **encode** them, not re-litigate them.

| Dimension | Choice | Evidence against the alternative |
|---|---|---|
| Weights / activations | **W8A16** (INT8 per-channel weights, FP16 activations) | W4A16 scores 0/4 on quality *and* v81's `htp_v2.json` ships zero INT4 matmul kernels; the converter folds s4→f16 (REFERENCE §4.1) |
| Attention | **`--grouped-gqa`, mandatory** | Without it: 56 KV-replication ops = 74.7% of decode DSP cycles, **6.836 tok/s instead of 44.707** (6.54×) |
| Topology | **B — `ladekv`, 3 graphs**: past-KV prefill (AR=128, CL=1152, past=1024) + decode (AR=1, CTX=1152) + verify32 (AR=32) | Bertcache prefill caps prompts at 128 and classifies `ctx_size == AR`; its ~23.8 tok/s early burst is now *slower* than plain decode |
| Runtime mode | **basic** (`genie_dialog_basic.json`) | LADE post-fix is 31.342 vs 44.707 — a 30% regression; break-even rose to 2.30 accepted tokens/call against ~1.6 measured (REFERENCE §6.8) |
| `lm_head` | **FP16** (no `--quant-head`) | ≈−2% in basic mode; −14% under LADE. Suspected not to reach the device at all (§8.1) |
| HVX threads | **4** in `ship` (matches the measured bin); 8 is a `variants` arm | see §6 |

`verify32` stays in the shipping bin because it is weight-shared and costs ~0 —
preserving the LADE option, not exercising it.

**Model geometry** (0.6B-specific, hard-coded by design — this is a 0.6B kit):
28 layers, 8 KV heads, head_dim 128, rope_dim 64, rope_theta 1e6,
vocab 151936, CL=128, CTX=1024 → ctx-bin CL 1152, decode past 1151.

## 3. Gaps in the current path that the kit must close

Found by reading the four scripts on the winning path. A naive copy would carry
all of these forward.

1. **`--grouped-gqa` is passed two different ways.** `full_build.sh` takes it as
   a positional pass-through; `lade_build.sh` / `ladekv_build.sh` need
   `FUSE_FLAGS="--grouped-gqa"`. Both re-export graphs, so omitting it there
   silently ships pre-fix attention in `verify32` and the past-KV prefill while
   decode looks correct. This is the single most-documented footgun in the repo.
   **The kit removes the choice: grouped GQA is structural, not a flag.**
2. **Two ctx-bins are built and discarded.** `full_build.sh` emits a 2-graph
   bertcache bin and `lade_build.sh` a 3-graph lade bin; neither reaches the
   ship bundle. Each costs ~20 min and ~1–2 GB. `full_build.sh`'s `prefill.dlc`
   likewise never enters the shipping bin — it exists only to satisfy
   `lade_build.sh`'s prerequisite check.
   ⚠ The CL=128 prefill **quantization** run is *not* droppable: it is the
   calibration run, and every other export adopts its encodings via
   `--export-decode`. Only its conversion and the two intermediate bins go.
3. **The winning dialog file is hand-made and ungated.** `genie_dialog_basic.json`
   is produced by a heredoc in `BUILD_GUIDE.md:509` *after* `bundle.sh` runs, so
   `bundle.sh`'s own `lint_bundle_dialogs.py` call cannot see it. A
   lade-config-carrying-`max-num-tokens` pair (device exit 139, SIGSEGV) reached
   three shipped bundles through exactly this ungated step. The content exists
   already — `configs/genie_dialog_qwen3_0.6b.json` is `"type": "basic"` — only
   the gated emission is missing.
4. **`env.sh` is this-box-shaped**: hard-coded `/home/vinc/llm-local`, a WSL
   `/mnt/c` `disk_guard` target, an extracted-libc++ `LD_LIBRARY_PATH`, and two
   named uv envs.
5. **The shipped backend config carries three inert keys.** `memory.extended_udma`
   (the `memory` section is `extra="forbid"` with exactly one field, so it never
   applied), `graph_configs_extra.sparse_weights_compression` (measured: 0 bytes
   saved), and `fp16_relaxed_precision: 0`. Proven inert — see §8.
6. **`coord.py` (612 lines) is repo-local multi-session dedup.** Out of scope for
   a standalone tree. It fails open, so removal is safe.
7. **Several gates exist only as prose** in `BUILD_GUIDE.md` tables: graph names
   vs `graph_names`, pooled weight-sharing fraction, `numHvxThreads` readback,
   per-graph byte accounting.

## 4. Architecture

```
kit-06b/
├── README.md            # 0.6B truth only: the decision set, the variant ledger, how to run
├── env.sh               # parameterized; sourced by every script
├── setup/
│   ├── check_sdk.sh     # QAIRT 2.48.40 presence + tool/lib layout
│   ├── make_envs.sh     # the two Python envs (torch+AIMET+onnx; converter-only)
│   └── fetch_model.sh   # HF Qwen3-0.6B checkpoint + tokenizer.json
├── build/
│   ├── quantize.sh      # stage 1: calibration + 3 adopted exports
│   ├── convert.sh       # stage 2: 3 DLCs
│   └── ctxbin.sh        # stage 3: one ctx-bin from an explicit config
├── variants/
│   ├── arms.tsv         # the ledger: arm, class, knob, byte %, compute %
│   └── build_arm.sh     # builds one arm by name
├── bundle/
│   ├── bundle.sh        # flat device bundle, both dialogs, then lint
│   └── configs/         # htp_config, htp_backend (generated), both dialog JSONs
├── gates/               # every check, individually runnable
└── build.sh             # top-level: `./build.sh ship` | `./build.sh variants [arm…]`
```

Each unit has one job and a defined interface: `quantize.sh` produces renamed
ONNX + one encodings file; `convert.sh` consumes those and produces DLCs;
`ctxbin.sh` consumes DLCs + a config and produces a bin; `bundle.sh` consumes a
bin. Any stage can be run alone against a prior stage's output directory.

### 4.1 Ship pipeline

```
HF Qwen3-0.6B
  └─ quantize.sh
       ├─ [calib]  AIMET W8A16 quantsim, CL=128            → encodings (the donor)
       ├─ filter + canonical I/O rename
       ├─ [decode] --export-decode <calib> --decode-ar 1   → decode ONNX
       ├─ [verify] --export-decode <calib> --decode-ar 32  → verify32 ONNX
       └─ [prefkv] --export-decode <calib> --decode-ar 128 → past-KV prefill ONNX
  └─ convert.sh   → decode.dlc · verify32.dlc · prefill.dlc   (one encodings file for all three)
  └─ ctxbin.sh    → qwen3-0.6b-w8a16-gqafix-ladekv_ctx.bin    (3 graphs, weight-shared)
  └─ bundle.sh    → qwen3_06b_w8a16_gqafix_ladekv.tar.gz
```

Four AIMET runs, three conversions, **one** ctx-bin — versus the current
chain's four runs, four conversions and three bins.

**Cross-graph rule, preserved verbatim:** all three DLCs convert against the
*same* encodings file. Mixed encodings are a fatal Genie load error — KV quant
params must be byte-identical.

**Graph-name rule, preserved verbatim:** a graph's name is baked in at
conversion time from the `--output_path` basename, dots included. Convert
straight to the final filename; never rename after.

### 4.2 One config generator for both targets

`ctxbin.sh` generates its backend config from the graph names it is given (the
`ctxbin_variant.sh` approach) rather than reading a checked-in
`htp_backend_config.json`. This is safe and is what §8 proves: the config the
generator emits produces a **byte-identical** bin to the checked-in one.

Emitted config, `ship` target:

```json
{"graphs":[{"graph_names":["prefill","decode","verify32"],
            "O":3,"vtcm_mb":16,"hvx_threads":4}],
 "devices":[{"dsp_arch":"v81","soc_model":0,"pd_session":"unsigned",
             "cores":[{"core_id":0,"perf_profile":"burst",
                       "rpc_control_latency":100,"rpc_polling_time":9999}]}],
 "context":{"weight_sharing_enabled":true}}
```

The three inert keys of §3.5 are dropped. `vtcm_mb: 16` is the ceiling — 24
compiles offline and is rejected at runtime (`0x138d`).

## 5. Variant arms

Two classes, and the distinction drives the whole `variants` design because the
cost differs by an order of magnitude.

**Class A — ctx-bin-only knobs.** Same DLCs, regenerate the bin: **~5 min, no
re-export.**

| Arm | Knob | Byte model | Compute model | Artifact changed? |
|---|---|---:|---:|---|
| `hvx8` | `hvx_threads: 8` | **0% by construction** | **large** | ✅ +2,277,376 B |
| `socmodel72` | `soc_model` **and** `soc_id` = 72 | 0% | unknown | ✅ +249,856 B |
| `udma` | `extended_udma` in **`context`** (not `memory`) | 0% | unknown | ✅ +212,992 B |
| `dlbc` | `dlbc: 1` — activation (inter-layer) compression | 0% (measured) | unknown | ⚠️ 0 — unproven |
| `wpack` | `weights_packing` — never tried | 0% (measured) | unknown | ⚠️ 0 — unproven |

⚠ `dlbc` compresses *activations*, not the weight stream, so it is unlikely to
touch the term that matters. Its sibling `dlbc_weights` is **not**
weight-sharing-compatible on SDK ≥ 2.36 — the kit must never set it, since every
0.6B bin depends on a healthy shared pool.

**Class B — lineage variants.** Need their own export + conversion: **~1 h each.**

| Arm | Change | Byte model | Compute model | Note |
|---|---|---:|---:|---|
| `cl512` | `context.size` 512 → ctx-bin CL **640**, decode past **639** | +10.1% | **+26.0%** | caps context at 512 |
| `fuseqkvgu` | `--fuse-qkv --fuse-gate-up` on the gqafix base | +9.1% | ~+11% | **never built** — the two winning changes have never been combined |
| `qh` | `--quant-head` (+ `--keep-head-weight` on the filter) | +17.9% | +3.6% | confounded — §8.1 predicts **~0%** on device |

`arms.tsv` carries these predictions as data. Each arm's build records its own
`read_total_bytes` and diffs the bin against the `ship` control, so an arm whose
bytes did not move — or whose knob did not reach the artifact — fails loudly.

### 5.1 On exceeding 50 tok/s

50 tok/s is **+11.8%** over 44.707. The kit **cannot verify** any of this: both
build hosts are device-free.

`hvx8` is the best single shot and is free — every shipping bin uses **4 of the
8 available HVX units**, and 2026-08-16 confirmed the build-time value is really
consumed (`numHvxThreads` reads back 8; `hvx_threads` is inert at *runtime*,
which is a different question). Under the compute model it clears 50 alone.
`cl512` lands ~56 tok/s under the compute model, ~49 under the byte model.

But whether decode is compute-bound is the project's #1 open question (§8.11):
at the current operating point the two models are numerically degenerate —
961 MB ÷ 22.37 ms = 43.0 GB/s and 88.2M residual cycles ÷ 4 HVX ≈ 22.06 ms,
1.4% apart. `hvx8` vs `hvx4` is precisely the experiment that separates them.

Two caveats the kit's README must carry: the device rep spread on a *single*
binary was 23.4 / 44.5 / 29.3, so any ">50" claim needs the B0 protocol (5 reps,
median, fixed thermal state, deltas inside the spread decide nothing); and the
byte model has already mispredicted badly once — V3 predicted 18.1 tok/s where
reality was 44.7.

**The kit's contribution is to make every candidate one command away, not to
promise a number.**

## 6. Gates

None skippable. Ship fails, not warns.

| Gate | Check |
|---|---|
| ONNX parity | `parity_ladekv_read.py` — qualla feed pattern incl. two-chunk prompts; argmax matches HF on all prompts |
| Quant quality | `quantize_aimet.py --eval` ≥ 3/4 last-token argmax agreement |
| **GQA removal** | `lint_gqa_ops.py` — **0** `Eltwise_Binary` `operation: 13` in **every** graph, not just decode |
| Prefill contract | `logits` is `[1,128,151936]` **and** `past_key_0_in` exists (all-position logits; not bertcache) |
| Graph names | bin's actual `graphName` list == configured `graph_names`, exactly |
| (AR, CL) uniqueness | no two graphs share a pair — Genie dispatches by numeric best-fit |
| Config bind-back | `numHvxThreads` / `vtcmSize` / `optimizationLevel` read back from the finalized bin match what was asked |
| Weight sharing | pooled fraction, **not** `constSize == 0`. `ship` expects ~100% (const 256 B). A hard-zero gate would reject good bins: `--quant-head` moves ~144 MB private by design, bertcache forces a private 444 MB copy |
| Byte accounting | record `read_total_bytes` / `write_total_bytes` per graph from the converter's `====== DDR bandwidth summary ======`, **with log name and date** |
| Bundle topology | `lint_bundle_topology.py --require-pure` — a blended bin cannot be compared against 44.707 at all |
| Bundle dialogs | `lint_bundle_dialogs.py` runs **after all dialogs are written**, not before |

Key-name hazards to encode, since they are not guessable and have bitten this
repo: `vtcmSize` (not `vtcmSizeInMB`), `graphBlobInfoV2` (not `graphBlobInfo` —
V1 has no weight fields at all), `sharedWeightsSize` printed identically on every
graph (never sum it).

## 7. Portability contract

`env.sh` exposes and validates: `KIT_ROOT` (self-derived), `KIT_DATA` (all heavy
output), `QAIRT_SDK`, `PY_DEPLOY`, `PY_QAIRT`, `QUANT_DEVICE`.

`disk_guard [need_gb]` is retained — it is load-bearing, not hygiene. On WSL a
failed vhdx grow does **not** surface as ENOSPC: the guest still reports free
space, the host write fails, and every mmap'd page takes SIGBUS, killing PID 1
and hard-crashing the VM (3× on 2026-08-12). The portable version guards the
volume the writes actually land on — Windows `C:` when `/mnt/c` exists, else
`$KIT_DATA` — and is sized per step (6 GB converter floor; export peaks higher).

Sizing targets: 0.6B quant dirs ~8.6 GB each of which ~135 MB is irreplaceable;
ctx-bin ~1.09 GB; bundle tarball ~930 MB.

## 8. Why one config generator is provably safe

`qwen3_06b_w8a16_gqafix_ctrl_ladekv` was built through `ctxbin_variant.sh`'s
*generated* config; the shipped `qwen3_06b_w8a16_gqafix_ladekv` was built through
`ladekv_build.sh` reading the checked-in `configs/htp_backend_config.json`. The
two ctx-bins are **byte-identical**: md5 `9c6024ad5b141137fbe22f3a4972eb96`, and
both report decode `read_total_bytes` = 961,130,496.

That single fact establishes three things the kit relies on: the generated
config path is equivalent to the checked-in one; the three extra keys of §3.5
are inert; and ctx-bin generation is deterministic, so md5 reproduction is a
valid acceptance test.

## 9. Success criteria

1. **`./build.sh ship` on a clean box reproduces ctx-bin md5
   `9c6024ad5b141137fbe22f3a4972eb96`.** This is the headline gate: it proves the
   restructuring (dropped conversion, dropped intermediate bins, generated
   config) changed nothing that reaches the device.
2. Every gate in §6 runs and passes on that build.
3. Decode `read_total_bytes` = **961,130,496**, `write_total_bytes` = **419,840**.
4. The bundle contains both `genie_dialog.json` and `genie_dialog_basic.json`,
   both pointing at the real ctx-bin filename, and the dialog linter ran after
   both existed.
5. `./build.sh variants hvx8` reproduces a bin **+2,277,376 B** vs the ship bin
   with `numHvxThreads` reading back 8 and byte-identical DDR figures.
6. No path in the kit can produce a bin without `--grouped-gqa` applied to all
   three graphs.
7. The kit runs with this repo absent from the filesystem.

## 10. Non-goals

- Qwen3-VL / multimodal anything — separate chain, separate kit.
- LADE tuning, learned draft heads (`eaglet`, `spd`) — parked by measurement.
- W4A16 / INT4 — no kernels on v81.
- Multi-core / multi-process — Genie returns 5005.
- KV INT8 via Genie config — the flag exists only in the CPU backend.
- `coord.py` multi-session coordination, HF upload/watchdog machinery, the
  device-report and exchange-protocol docs.
- Measuring tok/s. The kit builds candidates; a device team measures them.
- Re-deriving the decision set of §2.

## 11. Risks

| Risk | Handling |
|---|---|
| Dropping the bertcache conversion / intermediate bins changes the ship bin | Criterion §9.1 — md5 must reproduce. If it does not, restore the discarded steps and re-test one at a time |
| A knob silently fails to bind and reads as "the arm did nothing" | Config bind-back gate (§6) — the most expensive possible failure, because it looks like a measurement |
| `dlbc` / `wpack` change no artifact | Kept in `arms.tsv` ranked last, flagged unproven. "Identical" does not prove a no-op — some keys are runtime hints |
| Variant predictions are quoted as results | `arms.tsv` and README label every variant number as *predicted*; only 44.707 is measured |
| SDK version drift | `check_sdk.sh` pins 2.48.40.260702 / QNN API v2.37.0 / libGenie 1.19.0 and fails on mismatch — quantization behavior is SDK-specific |
