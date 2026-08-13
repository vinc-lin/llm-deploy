# Decision table — what each outcome means, decided in advance

Every row was agreed **before** the session so nobody has to interpret results
on the spot, and so no result needs a round-trip. Find your measurement, read
the action.

Baselines to compare against, all from 2026-08-13, warm, greedy, 56-token
technical prompt:

| Baseline | Value |
|---|---|
| Basic AR-1, `local` (2-graph) | **11.72 ± 0.01 tok/s** |
| LADE, `ladekv` (3-graph), technical prompt | 9.18 ± 0.00 tok/s |
| LADE, earlier simple prompt | 10.8 tok/s |
| Decode-only aggregate DSP cycles | **350,302,972** (±0.3% reproducible) |
| Decode step time (Genie) | ~85 ms |

---

## 1. The primary 2×2 — B7a (cycles) × B7b (tok/s)

Neither measurement alone is decisive. **Together they separate the two
competing models of the machine**, which is the entire point of the session.

| | **tok/s rises** (≥ 14) | **tok/s flat** (11.5–13) |
|---|---|---|
| **cycles fall** (≈ 90M, −75%) | **A. Compute-bound, fix works.** The plan's central case. → `gqafix` becomes the ship base. Proceed to priority 4 to pick the final config, then re-tune LADE (D1/D2) on the new base. | **B. Byte floor is real.** The DSP got 4× cheaper and the step did not. Decode is bound by streaming 883 MB (751 weights + 132 KV), not by compute. → Priority 4 becomes the *lead* workstream, not a contingency: W8 head (−156 MB) and CL=512 (−73 MB) are now the only levers that matter. Do **not** spend further effort on compute-side optimisation. |
| **cycles flat** (≈ 350M) | **C. Impossible — investigate.** Cycles unchanged but time improved means the arms differ in something other than the graph. Check you ran the `gqafix` bundle, and check the bin's graph names against `htp_backend_ext_config.json`. Report before concluding anything. | **D. The fix did not reach the device.** The most likely cause is a build/config error, not a wrong hypothesis. Verify with `qnn-context-binary-utility --json_file` that the decode graph's attention MatMuls are `1x8x2x1152`, not `1x16x1x1152`. Do not report this as "the GQA fix does not help" — report it as a build defect. |

**Sanity check before trusting any of the above:** the fixed decode graph must
contain **zero** `Eltwise_Binary` ops with `operation: 13` whose output is
`[1,8,2,...]`. If any remain, the export flag did not propagate to that graph
and the arm is invalid.

---

## 1b. Known confound — `gqafix_local` is 1.52 GB, not 1.09 GB

Build-side finding, 2026-08-14. In the **2-graph** gqafix bin, ~444 MB of INT8
decoder weights fell out of the shared-weights pool and became **per-graph
constants**, so they are stored twice:

| bin | sharedWeightsSize | constSize (per graph) | file |
|---|---|---|---|
| baseline `local` (2-graph) | 1,063 MB | 4 MB | 1.087 GB |
| **gqafix `local` (2-graph)** | **623 MB** | **444 MB × 2** | **1.523 GB** |
| gqafix `ladekv` (3-graph) | 1,067 MB | 0 | 1.087 GB |

The weight *bytes* are unchanged (623 + 444 ≈ 1,067 MB); only their
classification is. The 3-graph bin shares perfectly, so this is specific to the
2-graph combination (bertcache CL=128 prefill + decode) under grouped
attention. Root cause not yet established — it is in the ctx-bin generator's
layout/sharing decision, not in the graph topology, which gates clean.

**What this means for the session:**

- Each graph still streams the same weights per step, so this should not change
  decode traffic — but it is +436 MB of storage on a device whose `/data` runs
  98–99% full, and it may affect init time and mmap behaviour.
- **`p2_gqafix_local_basic` vs the 11.72 tok/s baseline is therefore not
  size-matched.** If that arm underperforms, do not attribute it to the GQA fix
  without checking the arm below.
- **The clean comparison is `p2_gqafix_ladekv_basic` (priority 2) against
  `p3_a1_ladekv_basic` (priority 3).** Same topology, same graph count, same
  1.087 GB bin — the only difference is the attention. Treat that pair as the
  primary evidence for box A vs box B above, and `gqafix_local` as corroboration.
- Please report the observed init time for `gqafix_local` separately; that is
  the measurement most likely to show the size cost.

## 2. Individual arms

### A1 — basic mode on the pre-fix plain `ladekv` bin (priority 3)

This retroactively de-confounds every previous 3-graph measurement.

| Observation | Pre-committed conclusion |
|---|---|
| **≈ 11.7 tok/s** (matches `local`) | There is **no** 3-graph penalty. The "75% build gap" in report §6.3 was the W8 `lm_head` all along, since 6.70 was measured only on the `qh` bundle. → Close the build-gap question. The C1 2-graph past-KV bin is shelved unrun. Earlier 6.3–6.5 tok/s figures need re-examining. |
| **≈ 6.7 tok/s** (matches `qh`) | There **is** a real 3-graph / graph-switching penalty independent of the head. → Run the `enable-graph-switching: false` toggle on the same bin in this session, and the C1 bin next session. This becomes a first-class workstream. |
| Between 8 and 11 | Partial penalty; both effects present. → Report the number; we will need the C1 bin to separate them. |

### W8 `lm_head` on a clean 2-graph bin (priority 4)

The 2026-08-13 −43% is **not** evidence about the head: it compared a 3-graph qh
bin against a 2-graph FP16 bin. This arm is the clean comparison, against
`gqafix_local`.

| Observation | Pre-committed conclusion |
|---|---|
| **≥ +5%** | Byte-bound regime confirmed from a second direction. Adopt W8 head for **basic-mode products**. LADE keeps the FP16 head — the −14% acceptance regression still stands and there is no speculation to reject in basic mode. |
| **−2% … +5%** | Neutral. Keep FP16 head everywhere; it is the simpler lineage. |
| **≤ −5%** | Head quantisation genuinely hurts. Park `--quant-head` permanently and record it next to the W4A16 dead end. |

### CL=512 (priority 4)

Cuts the KV read 132 → 59 MB and shrinks attention GEMV.

| Observation | Pre-committed conclusion |
|---|---|
| **≥ +8%** | The KV read is a live term. Ship CL-sized product variants; build CL=768 next. |
| **+2% … +8%** | Present but small. Offer CL variants only where the product allows a short context. |
| **< +2%** | KV traffic is not binding. Drop workstream F. |

### LADE acceptance map (priority 6)

Three prompt classes on one binary. LADE was a **regression** on the technical
prompt (9.18 vs 11.72), so this decides whether it ships at all.

| Observation | Pre-committed conclusion |
|---|---|
| LADE beats basic on ≥2 of 3 prompt classes | LADE is the default for those workloads. Proceed to the parameter sweep (D2), optimising **acceptance**, not call latency. |
| LADE wins only on `structured` | Ship LADE only for structured/repetitive workloads; basic is the default. |
| LADE loses on all three post-fix | **Basic is the ship configuration.** Park LADE entirely: post-fix, verify32's cost regains AR-scaling (replication was AR-independent and paid for speculation), so the 1.68× multiplier measured pre-fix does not survive. Do not run D2/D3. |

### `dlbc: 1` and hybrid prefill (priorities 5, 7)

| Observation | Pre-committed conclusion |
|---|---|
| `dlbc` ≥ +3% | Adopt as a build default. |
| `dlbc` < +3% | Leave off; it costs a ctx-bin rebuild for nothing. |
| Hybrid prefill TTFT ≈ 40 ms with full context | Ship it for all **basic-mode** products (never lade — the AR==CL bertcache graph breaks speculation). This is a user-visible latency win independent of decode rate. |
| Hybrid prefill TTFT > 100 ms | The bertcache graph is not being selected by Genie's (AR, CL) best-fit. Send us the bin's graph list. |

---

## 3. Things that would invalidate a result

Report these rather than working around them:

- **Any arm exits 139 (SIGSEGV).** Should not happen — the `lade` +
  `max-num-tokens` pair that caused this is fixed and now linted — but if it
  does, send the dialog JSON that was used.
- **Init time > 1.2 s**, which means the run was not warm. Re-run.
- **Generated token count differs between arms** being compared on rate.
- **`Unknown Key` warnings** from the backend config: the key was silently
  ignored and that arm is not testing what it claims. This has bitten us three
  times; see `docs/NOTES-htp-config-keys.md`.
- **Quality regression** — output that is fluent but wrong, or degenerate.
  Compare against `expected/` before trusting any speed number from that arm.
