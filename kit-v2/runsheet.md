# Runsheet v2 — SA8797P decode-regime session

**Supersedes `kit/runsheet.md` (2026-08-14).** That kit's priorities 4 and 5
pointed at bundles that cannot answer the questions they were written for — see
[§0.2](#02-the-rule-that-changed) — so do not run it.

**Read §0 before running anything.** It is short, and it is the difference
between a result and a wrong result.

---

## 0. What changed since the last session

### 0.1 The GQA fix worked, and we have a real baseline now

| Arm (all on `qwen3_06b_w8a16_gqafix_ladekv`, one binary) | tok/s |
|---|---:|
| **basic mode** | **44.707 ± 0.030** (22.37 ms/step) |
| LADE mode | 31.342 (acceptance 1.61 tok/iter) |
| pre-fix `ladekv`, basic — the control | 6.836 (146.3 ms/step) |

**6.54×.** Basic beats LADE, so basic is the ship configuration and the LADE
workstream is parked. Nothing in this session re-tests either of those.

### 0.2 The rule that changed

**A bundle containing a graph whose AR equals its CL reports a blended decode
rate, not a decode rate.** In that topology Genie keeps generating *through* the
prefill graph, re-processing the whole window once per token, until the KV cache
passes AR — only then does the AR-1 decode graph take over. A tok/s figure
measured over a fixed token budget is a time-weighted average of the two, and it
comes out **flatteringly fast**.

That is what 11.72 tok/s was. It produced three phantom findings before anyone
noticed (a "~75% build gap", "LADE is −22%", and "our builds are +51% faster
than yours"). All three are withdrawn.

> ### 🔴 The one rule for this session
> **Never compare a decode rate from a BLENDED bundle against one from a PURE
> bundle.** Every arm below is labelled. If you add an arm, classify it first:
> ```sh
> lint_bundle_topology.py <ctx-bin>     # prints "pure" or "BLENDED"
> ```

Of the 2026-08-14 bundles, only `gqafix_ladekv` and `gqafix_pastkv2g` are pure.
`gqafix_local`, `gqafix_qh`, `gqafix_cl512`, `gqafix_dlbc`, `gqafix_udma` and
`gqafix_hybrid` are all blended — which is why they have been rebuilt for this
kit on the pure 3-graph past-KV topology.

### 0.3 What this session is actually deciding

Post-fix, two performance models both fit the measurement exactly:

| | fits 22.37 ms/step? |
|---|---|
| **Byte model** — 961 MB/step ÷ 22.37 ms = 43.0 GB/s | yes, 88% of the 49 GB/s streaming ceiling |
| **Compute model** — 88.2M residual DSP cycles ÷ 4 HVX @ ~1 GHz = 22.06 ms | yes, to 1.4% |

They are **degenerate at this operating point**. They stop being degenerate the
moment you perturb the model, because they predict opposite orderings:

| Arm | Byte model says | Compute model says |
|---|---:|---:|
| **W8 `lm_head`** | **+17.9%** | +3.6% |
| **`cl512`** (`context.size` 512 → ctx-bin CL 640) | +10.1% | **+26.0%** |
| **`hvx_threads: 8`** | **0.0%, by construction** | up to large |

**Priority 1 is `hvx_threads: 8`** — the only arm that varies compute while
holding DDR bytes *exactly* constant, so it needs no assumption about which
model is right. It costs ~10 minutes. `cl512` is second (it discriminates by
magnitude); the W8 head is **third and optional**, because it changes bytes *and*
is independently suspected of never reaching the device, which confounds the
question it was meant to answer (`REFERENCE.md` §8.11).

---

## 1. Protocol — changed, and it comes first

The last session measured `pastkv2g` at **23.43 / 44.54 / 29.34 tok/s on one
arm, one binary**. That spread is larger than every effect this session is
chasing, so the old 3-rep protocol cannot resolve anything.

| Rule | Value |
|---|---|
| **Reps** | **5 per arm** — report the **median** and **every raw value** |
| Warm-up | 1 discarded run per bundle (cold init ~1.8–2.0 s vs ~800 ms warm) |
| Sampling | greedy (`temp=0`, `top-k=1`, `top-p=1.0`, `seed=42`) |
| Prompt | `prompts/technical.txt` (56 tokens) unless the arm says otherwise |
| `perf_profile` | `llm_decode_burst`, `rpc_polling_time: 9999` — change nothing else |
| **Thermal** | record device temperature before and after each arm; fixed cool-down between arms |
| TTFT | report separately from init time — never compare one against the other |

> **If an arm's rep spread exceeds the delta you are measuring, that arm decided
> nothing.** Re-run it; do not interpret it. This applies especially to
> priority 1, where the two models differ by ~11 percentage points on the qh arm.

**Run `p0_rebaseline` first.** Every delta this session reports is computed
against it, and 44.707 may have come from an unusually cool device.

---

## PART 1 — Runnable now, with bundles you already have

No new artifacts needed beyond this kit. ~45–60 minutes.

### P0 — Pull last session's results off the device ⚠️ do this before anything else

`/data` runs 98–99% full and the 2026-08-15 per-op record is one cleanup away
from being gone forever.

```sh
adb pull /data/local/tmp/results ./results-2026-08-15
```

### P0b — Re-baseline `gqafix_ladekv` basic, 5 reps

| Arm | Bundle | Dialog | Topology |
|---|---|---|---|
| `p0_rebaseline` | `qwen3_06b_w8a16_gqafix_ladekv` | `genie_dialog_basic.json` | **pure** |

Expected ~44.7. If the median lands far off, something environmental changed and
the rest of the session is not comparable to 2026-08-15 — say so in the report
rather than proceeding silently.

### P1 — The decode-only cycle profile ⭐ (this failed last time; here is why it should not again)

**Not in `run_all.sh`** — it uses `qnn-net-run`, not `genie-t2t-run`.

Last session reported this blocked by "pre-fix format" profiling inputs. **That
diagnosis was wrong.** The pre-fix and post-fix decode graphs have byte-identical
input contracts (60 inputs, same names, shapes, dtypes — the GQA fix is
graph-internal and KV I/O was frozen by design), and the shipped inputs score
**60/60** against the gqafix bin. In particular `position_ids_cos` is 128 bytes
because it is `[1,1,64]` fp16 — 128 B is *correct*.

So the real cause is still unknown. **Verify first, then run:**

```sh
cd /data/local/tmp/profiling
tar xzf decode_profile_inputs.tar.gz          # -> ar1_decode/

# NEW: proves the package matches the bin, or names the exact mismatch
python3 verify_profile_inputs.py \
    --bin qwen3-0.6b-w8a16-gqafix-decodeonly_ctx.bin \
    --inputs ar1_decode/inputs

cd ar1_decode
export ADSP_LIBRARY_PATH=.:/vendor/lib/rfsa/adsp
LD_LIBRARY_PATH=. qnn-net-run \
    --backend libQnnHtp.so \
    --retrieve_context ../qwen3-0.6b-w8a16-gqafix-decodeonly_ctx.bin \
    --input_list input_list.txt \
    --output_dir out_gqafix \
    --profiling_level detailed \
    --config_file ../htp_config_decodeonly.json

qnn-profile-viewer --input_log out_gqafix/qnn-profiling-data_0.log \
    > profile_gqafix_viewer.txt
```

If `verify_profile_inputs.py` passes and `qnn-net-run` still fails, the cause is
environmental. Check in this order, and **report which one it was**:

1. the `--retrieve_context` path actually resolves after extraction
2. `ADSP_LIBRARY_PATH` is set
3. `/data` had room — the package plus bin is ~1.1 GB and a truncated extract is
   silent
4. the two `Unknown Key` warnings for `memory.extended_udma` and
   `graph_configs_extra.sparse_weights_compression` are **expected here and are
   not the error** — that config is kept byte-identical to the one behind the
   350,302,972-cycle baseline on purpose, so the comparison stays valid

3 reps. Send `profile_gqafix_viewer.txt` and the raw `.log` for each.

**What to look for:** aggregate cycles (**~88M** = the fix landed; ~350M = it did
not); **zero `Expand` ops**; and **the new top-20** — with replication gone, the
shape of the profile is what decides where the next round of work goes.

⚠️ This profile **cannot on its own** settle byte-vs-compute: at 4 HVX threads
~88M reads compute-bound, at 8 it reads byte-bound, and the runtime reports 8
while every build compiles for 4. Priority 3 below is what disambiguates it.

### P2 — `pastkv2g` rep-variance isolation

| Arm | Bundle | Dialog | Topology |
|---|---|---|---|
| `p2_pastkv2g_variance` | `qwen3_06b_w8a16_gqafix_pastkv2g` | `genie_dialog_basic.json` | **pure** |

**8 reps**, not 5, with temperature logged before and after each. Last session:
23.43 / 44.54 / 29.34, and init time tracked it weakly (873 / 811 / 854 ms — the
fastest rep also initialised fastest). If this is thermal, the whole
measurement protocol needs a cool-down discipline; if it is not, something in
the 2-graph layout is non-deterministic and that matters more.

### P3 — Prompt distribution on the baseline

| Arm | Bundle | Prompt | Generated tokens | Topology |
|---|---|---|---|---|
| `p3_basic_structured` | `qwen3_06b_w8a16_gqafix_ladekv` | `prompts/structured.txt` | ~150–250 | pure |
| `p3_basic_simple` | `qwen3_06b_w8a16_gqafix_ladekv` | `prompts/simple.txt` | **~10–30 ⚠️** | pure |

44.707 is one prompt. The record needs the distribution before it is quoted as
"the" number.

> ⚠️ **`simple` generates too few tokens for a trustworthy rate.** Its reference
> continuation is one sentence, so the run is dominated by prefill and
> first-token latency and its tok/s is **indicative only** — do not average it
> with the others or quote it as a decode rate. It is included because its TTFT
> and prompt-rate numbers are still useful, and because it is the prompt class on
> which LADE historically won. `structured` is the one that carries the
> distribution claim.

---

## PART 2 — The discriminating arms (needs the bundles shipped with this kit)

### P4 — ⭐ The pair that decides the plan

Run **both**. Either alone is half an answer, because the verdict is in the
*ordering*, not the magnitudes.

| Arm | Bundle | Dialog | Topology | byte / compute |
|---|---|---|---|---|
| `p4_qh_ladekv` | `qwen3_06b_w8a16_gqafix_qh_ladekv` | `genie_dialog_basic.json` | **pure** | +17.9% / +3.6% |
| `p4_cl512_ladekv` | `qwen3_06b_w8a16_gqafix_cl512_ladekv` | `genie_dialog_basic.json` | **pure** | +10.1% / +26.0% |

Both are same-lineage rebuilds of the 3-graph past-KV topology — identical to
the baseline except for the one variable. The old `gqafix_qh` and
`gqafix_cl512` bundles are blended and **must not** be substituted.

### P5 — The null test

| Arm | Bundle | Dialog | Topology |
|---|---|---|---|
| `p5_hvx8` | `qwen3_06b_w8a16_gqafix_hvx8_ladekv` | `genie_dialog_basic.json` | **pure** |

Identical DLCs to the baseline; the only difference is `hvx_threads: 8` at build
time. **Verified two ways:** `numHvxThreads` reads back as **8** out of the
finalized binary (every other bundle reads 4), and the bin is 2,277,376 bytes
larger. It moves **zero DDR bytes** — both report `read_total_bytes =
961,130,496` — so the byte model predicts exactly 0.0%. Anything above the rep
spread falsifies it outright, and adopting it would be free.

Compare against **`p0_rebaseline`**. There is no separate control arm: the `ctrl`
bundle is byte-identical to the baseline (md5 `9c6024ad…` on both ctx-bins —
ctx-bin generation is deterministic), so running it would measure the same binary
twice.

| Arm | Bundle | Purpose |
|---|---|---|
| ~~`p5_ctrl`~~ | `qwen3_06b_w8a16_gqafix_ctrl_ladekv` | **skip** — byte-identical to the baseline; `p0_rebaseline` is the control |

### P6 — Cheap knobs, ~5 min each

| Arm | Bundle | Tests | Offline evidence |
|---|---|---|---|
| `p6_udma` | `..._gqafix_udma_ladekv` | `extended_udma` — **its first real A/B ever**; the key sat in the wrong config section in every previous build | binary changed (+212,992 B) — consumed |
| `p6_socmodel72` | `..._gqafix_socmodel72_ladekv` | `soc_model: 72`; Qualcomm document extra O=3 algorithms behind naming the SoC | — |
| `p6_dlbc` | `..._gqafix_dlbc_ladekv` | activation DLBC | binary **identical** to control — unproven |
| `p6_wpack` | `..._gqafix_wpack_ladekv` | `weights_packing`, never tried before | binary **identical** to control — unproven |

`dlbc` and `wpack` are low-expectation: they produced byte-identical binaries
offline. Run them only if time allows.

### P7 — Fusion, combined with the GQA fix for the first time

> ⚠️ **Not in this drop.** `gqafix_fuseqkvgu` needs a full rebuild from
> scratch — no quant dir exists for that flag combination — so it is deferred
> rather than rushed. `run_all.sh` will log `SKIP p7_fuseqkvgu` and move on.
> Listed here so the arm is not forgotten, not so it is chased.

| Arm | Bundle | Topology |
|---|---|---|
| `p7_fuseqkvgu` | `qwen3_06b_w8a16_gqafix_fuseqkvgu_ladekv` | pure (when built) |

Both models predict ~+10%, so this is a ship candidate rather than evidence.
Open question: fusion's +15% was measured *pre-fix* at ~10 GB/s effective, so it
may have been recovering the same access-pattern loss the GQA fix already
collected.

---

## Quality check — before trusting any speed number

`expected/` holds the greedy continuation each bundle should produce per prompt,
from the local ONNX parity harness. W8A16 bugs historically produce *fluent but
wrong* output, not obvious garbage. Diff the first ~30 tokens of each arm's
`stdout_r1.txt` against `expected/`; flag any arm that diverges early.

Speed on a broken graph is meaningless — and the `gqafix_hybrid` bundle already
produced an infinite `"and parallel, and parallel…"` loop last session, so this
is not hypothetical.

## What to send back

The whole `results/` directory: per-rep `--profile` JSON, stdout, and the dialog
plus backend configs each arm actually used. Plus the P1 viewer outputs, and the
temperature log from P2.

`MANIFEST.txt` records what ran and what was skipped. **Please also record, per
arm, all 5 raw rep values** — not just the mean. The 2026-08-15 `pastkv2g`
spread is the reason.

If anything in §0.2 seems wrong to you, say so before running — that rule is
new, and it is doing a lot of work in this session's design.
