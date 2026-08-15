# MAX TPS Qwen3-0.6B on SA8797P — Plan V4 (revision 2)

**Date:** 2026-08-16 (rev 2; rev 1 same day, superseded in full)
**Supersedes:** `MAX_TPS_QWEN3_0.6B_V3.md` §3–§7 (the gqafix trunk shipped and was
measured; V3's decision tree is resolved). V3 §0 (the inversion principle) and its
build discipline carry forward unchanged. V3 §7's item 4 (the "64 ms unexplained
term") is **retired by §1 below**, not carried.

**Analytical basis — read in this order:** `REFERENCE.md` (consolidated truth) ·
`DEVICE_MEASUREMENT_REPORT_2026-08-15.md` (post-fix rates) ·
`DEVICE_MEASUREMENT_REPORT_2026-08-13.md` (per-op cycles, byte accounting) ·
`MAX_TPS_QWEN3_0.6B_V2.md` §1 (the compute model) ·
`SA8797P_HTP_v81_Hardware_and_Deployment_Quantization_Reference_EN.md` (bandwidth,
DVFS, build-time DDR reporting) · `NOTES-genie-io.md` §"Graph selection" (why
topology A blends two rates) · `NOTES-htp-config-keys.md` (which knobs are real).

---

> ### What changed in revision 2
>
> Revision 1 was written before `REFERENCE.md`, `V2` and `NOTES-genie-io.md` were
> read, and it asserted a byte-bound regime as settled fact. Reading them
> produced a correction that is larger than the plan itself: **the 11.72 tok/s
> figure that has anchored every comparison in this project since 2026-08-13 is
> not an AR-1 decode rate.** It is a blend of two different graphs running at two
> different speeds (§1). Once that is removed, three separate "open questions"
> collapse into one artifact, the pre-fix bandwidth figure moves from 17.5 to
> 10.2 GB/s, and the byte and compute models turn out to be *numerically
> degenerate* at the current operating point rather than one being right (§2).
>
> The plan's **structure** survives: device-free Part A with a build-time gate,
> one pre-scripted Part B. Its **direction** changed: the W8 head is no longer
> the headline, `hvx_threads: 8` and the CL ladder are promoted, and the primary
> device experiment is now a *pair* of arms chosen because the two surviving
> models predict opposite orderings for them.

---

## 0. The measured state this plan starts from

All 2026-08-15, warm, greedy, 56-token technical prompt, `gqafix_ladekv`
(3-graph past-KV, W16 head, 1.087 GB, weight-shared, spill 0) unless noted:

| Fact | Value | Consequence |
|---|---|---|
| `gqafix_ladekv` basic | **44.707 ± 0.030 tok/s** = 22.37 ms/step, TTFT 103 ms | the baseline, and the **first unblended AR-1 rate this project has measured** |
| pre-fix `ladekv` basic (P3) | 6.836 ± 0.000 = 146.3 ms/step | the honest pre-fix AR-1 rate; the GQA fix is **6.54×** |
| `gqafix_ladekv` LADE | 31.342, acceptance 1.61 tok/iter | LADE parked — but see §2.4, the reason in the report is not the operative one |
| `pastkv2g` per-rep | 23.43 / **44.54** / 29.34 | **rep variance exceeds every effect this plan chases** |
| `gqafix_hybrid` | degenerate output | mechanism unestablished; A1.4 |
| P1 cycle profile | not run (shipped inputs were pre-fix format) | A1.6 makes it runnable |
| P4/P5 variant arms (qh, cl512, dlbc, udma, fuseqkvgu) | skipped | **fortunate** — see §1.3, they were unmeasurable as built |

## 1. The correction that reorders the plan: 11.72 tok/s is two graphs, not one

### 1.1 The arithmetic

`REFERENCE.md` §6.1 states the rule and then §0 breaks it: *"Any tok/s number for
topology A is meaningless without saying which phase it refers to."* In topology A
(bertcache, AR==CL=128) Genie keeps generating **through the prefill graph**, one
token per step, re-processing the whole 128-wide window, until the KV cache passes
128 — then switches to the AR-1 decode graph (`kvmanager.cpp:421-429`;
`REFERENCE.md` §2.1).

Test 1 Arm A on 2026-08-13 ran `qwen3_06b_w8a16_local` — a 2-graph **bertcache**
bin — with a 56-token prompt, generating 128 tokens in 10,837 ms:

```
bertcache phase : generated tokens 1..72        (KV 56 → 128)
AR-1 phase      : generated tokens 73..128      (56 tokens)

one bertcache step = one full AR=128 window pass = TTFT = 40.1 ms   (measured, same run)

72 × 40.1 ms                        =  2,887 ms
10,837 − 2,887 = 7,950 ms ÷ 56      =   142.0 ms per AR-1 step
```

**142.0 ms against the 146.3 ms measured independently on 2026-08-15 (P3, pre-fix
`ladekv` basic, 6.836 tok/s) — a 3% agreement between two numbers taken on
different days, on different bundles, by different means.** The blend model closes.

11.72 tok/s is therefore `128 ÷ (72 × 40.1 + 56 × 142.0) ms` — a **time**-weighted
composite of a ~25 tok/s bertcache phase and a ~7 tok/s decode phase, not an
average of the two rates. It is a real number for a 128-token completion on that
bundle. It is not a decode rate, and nothing that compares against it as one is
valid.

### 1.2 Three open questions were one artifact, and all three close

| Claim | Where | What it actually was |
|---|---|---|
| "~75% build gap: `local` 11.72 vs 3-graph ~6.7" | 08-13 report §6.3; `REFERENCE.md` §8.8 | 11.72 (blended) vs 6.84 (AR-1). Same decode graph, ±4 MB shared weights (V2 §2.2). **No graph-count penalty exists.** |
| "A 64 ms unexplained term" | V3 §7 item 4 | 142 − 85 = the blend gap. There is no unexplained term; the ceiling model needs no slot for it. |
| "Our builds are +51% faster than the device team's 7.79" | `REFERENCE.md` §8.8 ("RESOLVED BACKWARDS") | Blended-vs-AR-1. Like-for-like our pre-fix AR-1 is **6.84**, i.e. ~12% *slower* than their 7.79 — the original premise was closer to right than the resolution was. Their build's own provenance is still unaudited, so this reopens as a question, not as a conclusion. |

A fourth follows in §2.4.

### 1.3 The trap this creates for every future measurement — and the six bundles already in it

Six of the eight `gqafix_*` bundles on HF carry the CL=128 bertcache prefill:
`gqafix_local`, `gqafix_qh`, `gqafix_cl512`, `gqafix_dlbc`, `gqafix_udma`,
`gqafix_hybrid` (they are the ~1.32 GB ones — the same graph also costs them the
444 MB weight-sharing dup, V3 §10b / open Q10). **A basic-mode rate measured on any
of them is phase-blended and cannot be compared to 44.707.**

That is the entire P4/P5 variant program. Had the device team run it as instructed,
every arm would have returned an uninterpretable number and — worse — a *flattering*
one, because the bertcache phase is fast. The one variant arm that did run,
`pastkv2g`, is the one built on the past-KV topology, and its best rep (44.54)
lands on the baseline (44.707) exactly.

**Consequence for Part A:** every variant must be rebuilt on the 3-graph past-KV
(`ladekv`) topology before it is worth a device minute. This is a stronger reason
than revision 1's ("the 444 MB dup means it can't ship") — a bundle that cannot
ship is a nuisance; a bundle that returns a confidently wrong number is a hazard.

**Consequence for the repo:** this failure class needs a gate, not a memo. A1.2.

## 2. The two performance models are degenerate at this operating point

### 2.1 Corrected regime table

Using the *corrected* pre-fix step time throughout. Pre-fix bytes = 961 MB base
+ 264 MB replication write + 264 MB replication re-read (V2 §1.2).

| | step | bytes/step | effective BW | DSP cycles | aggregate cycle rate |
|---|---:|---:|---:|---:|---:|
| pre-fix AR-1 (08-15 P3) | 146.3 ms | ~1,489 MB | **10.2 GB/s** | 350,302,972 (measured) | 2.39 Mcyc/ms |
| post-fix AR-1 (08-15 P2) | 22.37 ms | 961,130,496 | **43.0 GB/s** | 88.2M (predicted) | 3.94 Mcyc/ms |

Revision 1 and the 08-15 report's own §0.2 annotation both put the pre-fix rate at
~17.5 GB/s. That used `local`'s blended 85.3 ms with `ladekv`'s byte count — two
different arms. **The corrected 10.2 GB/s is consistent with `REFERENCE.md` §1's
independently established ~6–7 GB/s effective decode bandwidth** instead of
contradicting it, which is the first sign the correction is the right one.

### 2.2 Both models fit the post-fix point, and neither can claim it

**Byte model.** 961 MB ÷ 22.37 ms = 43.0 GB/s, against the hardware doc's 49 GB/s
excl-wait microbenchmark ceiling (§3.1/§3.5) — **88% of the conservative streaming
ceiling.**

**Compute model** (V2 §1.2). Removing 261.8M replication cycles from 350.3M leaves
**88.2M** (re-summed from the 08-13 category table; V2 rounded to 88.5M). At 4 HVX
threads and ~1 GHz: 88.2 / 4 = **22.06 ms against 22.37 measured — 1.4%.**

Both land. They are not distinguishable by any measurement taken so far, and the
plan must stop pretending otherwise. Two further notes keep this honest:

- The compute model **fails on the pre-fix point**: it predicts 350.3M / 4 =
  87.6 ms against 146.3 measured, off by 1.67×. Read forward, that is not a defect
  — it says the replication ops achieved only ~60% of the DSP's compute-limited
  rate, i.e. *they* were the memory-bound part, and what survives them is not.
  The regime genuinely flipped; the argument is about where it landed.
- Both models' projections below are **upper bounds**, for opposite reasons. The
  byte model holds 43 GB/s fixed while removing the single largest *contiguous*
  read (the head), which `REFERENCE.md` §1 says should lower the average rate. The
  compute model assumes perfect thread scaling and zero stall. The *ordering* of
  their predictions is robust even though the magnitudes are not — which is what
  §2.3 exploits.

Adjudication is additionally blocked by two unknowns `REFERENCE.md` already flags:
the runtime reports **8** HVX threads in use on every workload (§1, §8.9) while both
teams build with `hvx_threads: 4`, and the clock is invisible under the GVM (only
`virtio_clk`; HTP doc §2.2, open question 4). At 8 threads the compute model
predicts 11 ms and would over-predict throughput by 2×.

### 2.3 What separates them: pick perturbations whose predicted *orderings* differ

This is the plan's core. Each candidate change is scored under both models against
the 22.37 ms / 88.2M-cycle / 961 MB baseline. Cycle shares are the 08-13 category
table renormalized over 88.2M: attention GEMV 44.6%, weight GEMMs 35.4%, `lm_head`
6.9%, everything else 13.1%.

Percentages below are **model-internal** — each is `1/(1 − Δ/baseline) − 1` within
its own model, so the two columns are directly comparable to each other and to a
measured delta against a re-baselined control (B0).

| Arm | Byte model | Compute model | Discriminates? |
|---|---:|---:|---|
| **W8 `lm_head`** (−155.6 MB = −16.2% bytes; 6.9% of cycles) | 805.5 MB → 18.75 ms → **+19.3%** | halve `lm_head` → −3.5% cycles → **+3.6%**, or **0%** if `REFERENCE.md` §8.1 holds and the head is re-materialized to FP16 at prepare time | ✅ **strongly** |
| **CL=512** (−73.4 MB KV; GEMV+softmax scale with CL) | 887.7 MB → 20.67 ms → **+8.3%** | GEMV 39.35M → 17.47M, softmax −0.89M → −25.8% cycles → **+34.7%** | ✅ **strongly, and in the opposite direction** |
| **`hvx_threads: 8`** at build time | zero bytes change → **0.0%** | up to +100% if the build-time 4 is binding; sub-linear in practice | ✅ **null test** |
| QKV + Gate-Up fusion | −80 MB measured by the device team (880 vs 960) → **+9.1%** | ~10% of cycles (V2 §1.3) → **+11%** | ❌ — ship candidate, not evidence |
| KV INT8 | −66.0 MB → **+7.4%** | GEMV byte-side only; small | weakly; and gated on a kernel that may not exist (A1.3) |
| `soc_model: 72`, `extended_udma`, DLBC, `weights_packing` | unknown, plausibly 0 | unknown, plausibly 0 | ❌ — cheap lottery tickets |

**The W8-head and CL=512 arms predict opposite orderings** (+19 vs +4, and +8 vs
+35). Running both settles the regime with no assumption about clock or thread
count, which is exactly what the cycle profile cannot do alone. That pair is
priority 1 in Part B, and building them is priority 1 in Part A.

**`hvx_threads: 8` is the cheapest experiment in the plan** — a ctx-bin regenerate
from existing DLCs, no re-export, no requantization — and its byte-model prediction
is exactly zero. Any result above noise falsifies the byte model outright. It also
partially resolves the thread ambiguity, which is what makes the cycle profile
interpretable afterwards. Revision 1 buried this in a footnote; it is now a
first-class build.

### 2.4 The LADE decision is right, for a reason the report does not give

The 08-15 report parks LADE because 31.342 < 44.707 and acceptance is "only" 1.61.
The operative quantity is the break-even, and it moved:

```
post-fix verify32 call latency = 1.61 accepted ÷ 31.342 tok/s = 51.4 ms
post-fix decode step                                          = 22.37 ms
break-even acceptance                = 51.4 / 22.37           = 2.30 tokens/call

the GQA fix sped decode   146.3 → 22.37 ms  = 6.54×
the GQA fix sped verify32   180 → 51.4 ms   = 3.50×
```

Replication cost is AR-independent, so removing it helped the AR-1 graph far more
than the AR-32 one — **exactly what V3 §7 predicted would happen to the LADE
multiplier.** Break-even rose from ~1.2 to 2.30 accepted tokens/call while
acceptance stayed near 1.6–1.9. LADE did not get worse; decode got better faster.

This also produces the fourth correction promised in §1.2: the 08-13 report's
headline *"LADE is slower than Basic (9.18 vs 11.72, −22%)"* compares LADE on
`ladekv` against blended basic on `local`. **Like-for-like on one bin and one
prompt, pre-fix LADE was 9.18 vs 6.836 = +34%.** LADE's technical-prompt loss is a
post-fix phenomenon only, first measurable on 2026-08-15.

Consequence: park LADE, keep `verify32` in the bins (weight-shared, ~0 cost,
preserves optionality), and record the bar a learned draft head (`eaglet`/`spd`,
`REFERENCE.md` §6.3) would have to clear: **+43% acceptance to reach parity, more
to win.** That is a sharper target than "optimize acceptance".

## 3. What the split can and cannot settle

**Device-free, quantitatively:** every *byte* claim, and now every *cycle* claim
too. The converter reports `read_total_bytes` per graph; `qnn-context-binary-
generator` emits a build-time DDR read/write and spill/fill report per graph (HTP
doc §4.2); and `qairt-dlc-info` gives op counts and output shapes, which map onto
the 08-13 category table to predict a cycle delta. So each variant can be required
to state **two** predictions before it ships (A2).

**Device-gated, irreducibly:** which of the two predictions the silicon honours.
No substitute exists — the x86 HTP path is closed (`libQnnHtpQemu.so` rejects v81
ctx-bins, `Request feature arch with value 81 unsupported`, settled 2026-08-14,
V3 §10b).

**Therefore Part A builds every branch unconditionally**, as V3 did: local compute
and disk are cheap, device-hours are scarce, a wasted build costs disk and a wasted
session costs weeks. Only KV INT8 is conditional, gated by a device-free SDK check.

---

# PART A — Executable now, without the device

## A1. Desk work first (no builds; highest value per hour)

| # | Item | Decides | Method |
|---|---|---|---|
| **A1.1** | **Propagate the blend correction** | Whether every future comparison is valid | §1 changes four documents' load-bearing numbers. `REFERENCE.md` §0 (11.72 as "best sustained decode"/"new best AR-1"), §6.1, §8.8 (resolve it *forwards* this time, and reopen the device-team comparison); a §0-style banner on the 08-13 report (its §6.3 and Test 1 interpretation); `HF_HUB_README.md`; `DROP_README_2026-08-14-gqafix.md`; `BUNDLE_README_gqafix_ladekv.md` §5.4. Per repo convention reports are **never edited** — banner + annotation only. Add corrections #22–25 to `REFERENCE.md` §7 |
| **A1.2** | **`lint_bundle_topology.py`** (new gate) | Kills the §1.3 failure class permanently | Classify each ctx-bin from its `qnn-context-binary-utility` dump: any graph with `AR == CL` ⇒ **blended**; past-KV-only ⇒ **pure**. Fail the build if a bundle README or kit runsheet quotes a decode rate for a blended bin, and stamp the classification into the bundle manifest. Cheap, and it is the only durable fix |
| **A1.3** | **KV-quantization kernel check** | Whether A3.8 exists at all | Revision 1 framed this as "read the qualla source". Sharper: the Genie contract does **not** forbid quantized KV — `NOTES-genie-io.md` requires only that KV quant params be *byte-identical across graphs* (`nsp-model.cpp:922-961`), and FP16 K/V-proj outputs are a build **choice** (`REFERENCE.md` §4). The real blocker is kernels. Grep `htp_v2.json` for INT8/INT16 MatMul entries on the KV-consuming shapes — **the same method that killed W4A16** (zero INT4 entries). No entries ⇒ A3.8 is dead device-free, and it becomes FAE open question 8, not a build |
| **A1.4** | **Hybrid degenerate-output mechanism** | Whether the TTFT product survives | Reproduce device-free by replaying Genie's graph-selection sequence (`kvmanager.cpp:388-409`) through a `parity_ladekv_read.py`-style feed on the hybrid variant lattice. Leading suspect: with `ctx_size == variant` the bertcache path inflates `n_process` by `n_past` (`:421-429`), and `getLogits` samples row `n_process − 1` (`nsp-model.cpp:3294`) — past the end of a 128-row logits buffer ⇒ garbage token ⇒ the observed self-reinforcing loop. **Unproven**; the probe is to reproduce it as an argmax divergence. Note the product case has weakened on three axes: TTFT is already 186 → 103 ms post-fix, the bin costs the 444 MB dup, and it now blends its own decode rate |
| **A1.5** | **Regenerate `decode_profile_inputs`** against the gqafix decode graph | Makes Part B's cycle profile runnable | `gen_decode_profile_inputs.py` ships in the package; the 08-15 miss was a build-side packaging defect (64-dim KV / 64-byte `position_ids` vs the pre-fix 128). **Add to the drop checklist: regenerate profiling inputs whenever graph I/O changes** |
| **A1.6** | **Weight-dup root cause** (open Q10) | Bertcache-carrying products only | ctx-bin forensics across the built bins. **Demoted** from revision 1: §1.3 makes bertcache-carrying products less attractive independently, so this is now documentation rather than a blocker. "Documented, not fixable" is a valid outcome — it is closed-source layout logic |
| **A1.7** | **Correction note to the device team** | Unblocks Part B's agenda | Their rec 5 ("P4 unnecessary — decode is clearly compute-bound") inherits quadrant A's label without §2's arithmetic. Send: the blend correction, the degeneracy, the two-arm discriminator, and the news that the P4/P5 bundles they were asked to run are being rebuilt because as-built they cannot answer the question |

## A2. The dual-prediction gate (replaces revision 1's byte gate)

Every variant records **both** predictions before it is allowed near the device, and
the recorded pair is what Part B adjudicates. A variant whose bytes or op-counts did
not move **did not do what it claims** and must not ship — the device-free analogue
of the `Unknown Key` rule.

**Byte side** — converter `read_total_bytes` (decode graph) + the ctx-bin
generator's per-graph DDR report:

| Variant | Predicted `read_total_bytes` | Δ vs baseline |
|---|---:|---:|
| baseline `gqafix` decode (**measured**, `ctxbin-ws.log`) | 961,130,496 | — |
| W8 head | 805,548,032 | −155,582,464 |
| CL=512 (KV term only; see note) | 887,730,176 | −73,400,320 |
| KV INT8 @ CL=1152 | 895,127,552 | −66,002,944 |
| W8 head + KV INT8 | 739,545,088 | −221,585,408 |
| `hvx_threads: 8`, `soc_model: 72`, `extended_udma` | 961,130,496 | **0 — by construction** |

*KV read = 2 × 28 layers × 8 heads × 128 dim × past × 2 B; past = 1151 → 132,005,888,
past = 511 → 58,605,568. Revision 1 carried −88,082,432 for CL=512 attributed to V3;
it is not derivable from V3 and the KV term accounts for only −73.4 MB. **Record the
converter's own figure; do not assert this one.***

**Cycle side** — `qairt-dlc-info` op-count and output-shape diff, mapped onto the
08-13 category table, giving the predicted Δcycles from the 88.2M post-fix budget.
`lint_gqa_ops.py` already proves 0 replication ops; this extends the same idea to
the changed category.

A variant ships to Part B carrying `(Δbytes, Δcycles, byte-model tok/s,
compute-model tok/s)`. When the two disagree, that variant is evidence; when they
agree, it is a ship candidate. Nothing else earns a device slot.

## A3. Builds (device-free; full V3 build discipline applies)

`FUSE_FLAGS="--grouped-gqa"` is **load-bearing** on every `lade_build.sh` /
`ladekv_build.sh` call — they re-export `verify32` and the past-KV prefill, and
omitting it silently ships old attention in those graphs (`lint_gqa_ops.py` gates
it). `disk_guard` sized per step (6 GB converter floor, 20 for an export); convert
straight to final filenames — never rename a DLC; `qnn-context-binary-utility`
check per bin; no duplicate (AR, CL) pairs; no AR==CL graph in any lade bundle.

**Every variant is built on the 3-graph past-KV (`ladekv`) topology** so its
basic-mode rate is directly comparable to 44.707 (§1.3). That is not optional.

### A3a — ctx-bin-only rebuilds (no re-export, no requantization, ~20 min each)

`scripts/build/ctxbin_variant.sh <out> <name> <dlc_csv> <graph_names_csv> <json>`
regenerates a bin from the existing gqafix DLCs with an explicit graph config and
verifies the graph names back out of the binary. These are the cheapest builds in
the plan and revision 1 under-weighted all of them.

| # | Build | Override | Why |
|---|---|---|---|
| **A3.1** | **`gqafix_hvx8_ladekv`** | `{"hvx_threads": 8}` | **The null test (§2.3).** Byte model predicts exactly 0. `REFERENCE.md` §8.9: the runtime reports 8 in use on every workload while every build compiles for 4; 08-13 Test 5 only proved the *runtime* knob is inert (`NOTES-htp-config-keys.md`: build-time only). Potentially the largest single unexplored lever in the plan |
| **A3.2** | **`gqafix_udma_ladekv`** | `{"__context": {"extended_udma": true}}` | Its **first real A/B ever** — the key sat in the `"memory"` section, which is `extra="forbid"` with one field, in every prior build, so it has never applied. A v81-and-above feature on a v81 part |
| **A3.3** | **`gqafix_socmodel72_ladekv`** | `devices[].soc_model: 72` | `REFERENCE.md` §8.4: never A/B'd; the device team's own verified config (HTP doc §8.4) sets `soc_id` **and** `soc_model` to 72, and Qualcomm documents extra O=3 algorithms behind it. **Note `ctxbin_variant.sh` hardcodes `soc_model: 0` in its `devices` block — this one needs a small script change, not just an override** |
| **A3.4** | **`gqafix_dlbc_ladekv`** | `{"dlbc": 1}` | Rebuild of the existing bertcache-blended `gqafix_dlbc` onto a measurable topology. Activations only; `dlbc_weights` is weight-sharing-incompatible (SDK ≥2.36) |
| **A3.5** | **`gqafix_wpack_ladekv`** | `{"weights_packing": true}` | Surfaced by the 2026-08-14 config audit, **never tried**. Real field on `HtpGraphConfig` |
| **A3.6** | **`gqafix_sparse_ladekv`** | `{"finalize_config": {...OPTIMIZATION_TYPE 6}}` | The *correct* route to sparse-weights compression — it is a graph optimization type, not a config key (`QnnHtpGraph.h:52`). Expectation is low: `REFERENCE.md` §4.1 records `sparse_weights_compression=1` → **0 bytes saved, "model isn't sparse"**. Build it because it is 20 minutes, not because it is promising |

### A3b — Full rebuilds

| # | Build | Why | Risk |
|---|---|---|---|
| **A3.7** | **`gqafix_qh_ladekv`** — W8 `lm_head` | **Discriminator arm 1** (§2.3): +19.4% byte vs +3.6% compute. `full_build.sh … --quant-head` → `lade_build.sh` → `ladekv_build.sh` (a bitwidth change needs the full chain, not the §5.6 fast path). Verify `qairt-dlc-info \| grep lm_head.weight` → `sFxp_8`, never trust the flag. Also finally records a converter DDR summary for a real qh build — none has ever existed (`REFERENCE.md` §6.5) | Low build risk; **`REFERENCE.md` §8.1 stands** — 151.3 MB left the DLC but only 12.5 MB left the ctx-bin, so the bytes may never reach the device. That is a *result*, not a reason to skip it |
| **A3.8** | **`gqafix_cl512_ladekv`**, then `cl768` | **Discriminator arm 2** (§2.3): +8.3% byte vs +34.7% compute — the opposite ordering. New conversions, same post-B encodings lineage. Doubles as the product variant | Low |
| **A3.9** | **`gqafix_fuseqkvgu_ladekv`** | The two winning changes have **never been combined** — every fused build predates the fix. Both models say ~+10%, so this is a ship candidate | **Gain uncertain.** Fusion's +15% was measured pre-fix at ~10 GB/s effective; if it was access-pattern recovery, the fix already collected it. Same reasoning that moved LADE |
| **A3.10** | **The stack** — winner(s) of A3.1–A3.9 combined, then the CL ladder on it | Ship candidate | Low |
| **A3.11** | **KV INT8** (INT16 first) — *conditional on A1.3* | −66 MB and the byte side of attention GEMV | **Highest.** Cross-graph rule bites hardest: KV encodings must be byte-identical across all three graphs (`nsp-model.cpp:922-961`, enforced at load). Kill criteria: `--eval` ≥3/4 **and** `parity_ladekv_read.py` 6/6. INT16 fallback still buys −33 MB |

Per-build local gates, none skippable: numerical equivalence → ONNX parity vs HF →
`--eval` ≥3/4 → `parity_ladekv_read.py` 6/6 incl. chunked → `lint_gqa_ops.py` 0
replication ops → `lint_bundle_topology.py` **pure** → graph names + weight sharing
(expect shared ≈1,067 MB, const ≈0, spill 0) → **A2 dual-prediction gate**.

## A4. Documentation and kit (device-free)

- **A1.1's correction propagation is the priority**, ahead of the baseline refresh:
  a doc that says 44.707 but still anchors on 11.72 elsewhere is worse than one
  that is uniformly stale.
- Then: 44.707 is the baseline, LADE is parked with the §2.4 break-even recorded,
  basic is the ship config. `BUNDLE_README_gqafix_ladekv.md` §5.4's decision matrix
  is written as an open question that 2026-08-15 closed — resolve it in place.
- **Kit v2**, implementing Part B: runsheet + decision table + `run_all.sh` +
  `expected/`. Two new hard rules in the runsheet: *no arm may be compared against
  44.707 unless `lint_bundle_topology.py` says its bin is pure*, and *5 reps,
  median, all raw values*. Kit hygiene: no serials, no ssh strings, no terminal
  chrome.
- Upload per standing rules: single-file commits via `hf_upload_file.py`, watchdog
  with `SOCKET_CHECKS=999999` for tarballs, well under 128 commits/hour, and
  **visibility read live before and after and never changed unasked** — four prior
  incidents, in both directions.

## A5. What Part A delivers on its own

Even if the device never becomes available: the blend correction propagated and
gated so it cannot recur; three long-standing "open questions" formally closed; the
KV-quantization question resolved device-free either way; a full set of variants
rebuilt on a topology that can actually be measured, each carrying two falsifiable
predictions; the hybrid failure either reproduced or documented; and a session kit
that runs without us. **That is a complete piece of work, not a stalled one.**

---

# PART B — The device session (one sitting, ~2 h)

## B0. Protocol change — mandatory, and it comes first

The `pastkv2g` spread (23.43 / 44.54 / 29.34 tok/s **on one arm, one bin**) is
larger than every effect this plan chases. Init time tracks it weakly (873 / 811 /
854 ms — the fastest rep also initialized fastest), which is suggestive of thermal
or DVFS state rather than anything in the binary. Until it is isolated:

- **5 reps per arm; report the median and every raw value.**
- Record thermal state before and after each arm; fixed cool-down between arms.
- **Never compare arms measured in different thermal regimes.**
- **An A/B whose delta falls inside the rep spread decides nothing** — re-run, do
  not interpret.
- **Never compare a blended bin against a pure one** (§1.3). The runsheet states
  each arm's classification next to its name.

Re-measure the `gqafix_ladekv` basic baseline under this protocol **first** — 44.707
may have come from an unusually cool device, and every delta in the session is
computed against it.

## B1. Priority ladder

| Pri | Arm | Decides |
|---|---|---|
| **0** | **Pull `/data/local/tmp/results/` off the device** | `/data` runs 98–99% full; the 08-15 per-op record is one cleanup away from gone. **Before anything else** |
| **1** | **`gqafix_qh_ladekv` AND `gqafix_cl512_ladekv`, both vs the re-baselined `gqafix_ladekv`** | **The whole plan** (§2.3). The two models predict opposite orderings (qh +19/+4, cl512 +8/+35). Assumption-free — needs no clock or thread-count knowledge. Run them as a pair; either alone is half an answer |
| **2** | **`gqafix_hvx8_ladekv`** | The null test. Byte model says exactly 0.0%. Also resolves enough of the thread ambiguity to make priority 3 interpretable |
| 3 | P1 cycle profile with A1.5's regenerated inputs | Confirms ~88M cycles and zero replication ops, and — the real value — yields **the new top-10**, which is the input every subsequent plan needs. Demoted from revision 1's "unambiguous priority 1": at 4 threads it reads compute-bound and at 8 it reads byte-bound, so it does **not** discriminate on its own |
| 4 | `gqafix_fuseqkvgu_ladekv` | Ship candidate; does fusion survive the fix? (A3.9's open risk) |
| 5 | `socmodel72`, `udma`, `dlbc`, `weights_packing`, `sparse` | Four ctx-bin-only lottery tickets, ~5 min each once the bins are on the device |
| 6 | KV INT8 (if A1.3 said the kernels exist and A3.11 built) | −66 MB, and validates the quantized-KV contract end to end |
| 7 | `cl768`; the stacked winner (A3.10) | Product variants and the ship configuration |
| 8 | Basic-mode rates on `simple` / `structured` prompts | 44.707 is one prompt; the record needs the distribution |
| 9 | Hybrid prefill TTFT (only if A1.4 fixed it) | 103 → ~30 ms, basic mode only, never lade. **Blended bin — never quote its tok/s** |

## B2. Pre-committed decisions

| Observation | Action |
|---|---|
| **qh ≥ +12% and cl512 ≤ +12%** | **Byte-bound.** qh is the ship base; KV INT8 and compression lead; CL is a product knob only |
| **cl512 ≥ +25% and qh ≤ +8%** | **Compute-bound.** The CL ladder leads; qh is parked next to W4A16; attention-side compute (fusion, GEMV shape, `hvx_threads`) is the workstream |
| both ≥ their low bars, neither at its high bar | Mixed regime; re-derive the ladder from priority 3's per-op table before building anything further |
| **both < +5%** | Neither model holds. Something outside both — dispatch overhead (~220 µs RPC × op count), DVFS, or the `pastkv2g` variance mechanism — is binding. Priority 3's profile becomes the only admissible next input, and open question 14 (is `llm_decode_burst` the real maximum clock?) escalates to the platform team |
| **hvx8 > +5%** | Byte model **falsified** regardless of anything else — zero bytes changed. Adopt it in every build immediately; it is free |
| hvx8 ≈ 0 within the rep spread | Build-time 4 is not binding, or 8 units are already in use. Interpret priority 3 at 8 threads |
| fusion ≥ +5% | Fold into the ship base |
| fusion flat | It was pre-fix access-pattern recovery, already collected; retire it as a lever |
| any ctx-bin-only knob ≥ +5% | Adopt immediately — zero build cost, zero risk, no quality gate needed |
| compression moves the DDR report but not tok/s | The ~440 MB weight floor is not the binding term; stop pursuing weight bytes |
| **any arm's spread > its delta** | Undecided. Re-run under B0; do not interpret |
| any blended bin's tok/s quoted in a comparison | Reject the comparison (§1.3), re-run on the pure rebuild |

## B3. If the device is available for only 15 minutes

Re-baseline `gqafix_ladekv` (5 reps), then **`gqafix_cl512_ladekv`** (5 reps). One
arm, one comparison, and the two models differ by 4× on it — the sharpest single
question in the plan. If there are 25 minutes, add `gqafix_hvx8_ladekv`: it is free
to build, free to run, and a positive result falsifies half the plan on its own.

---

## 4. What we explicitly do NOT do

- **LADE tuning of any kind** — parked by measurement 2026-08-15 and quantified in
  §2.4. `verify32` stays in the bins (weight-shared, ~free); the workstream does
  not. Revisit only if a learned draft head can clear **2.30 accepted tokens/call**.
- **W4A16 / INT4 anything** — v81 ships zero INT4 MatMul/FC kernels in `htp_v2.json`
  (2.43 and 2.48) and the converter folds s4 → f16. Dead on kernels *and* accuracy.
- **Multi-core / multi-process** — Genie multi-core returns 5005, a 2-core ctx-bin
  measured *slower* (3.96 vs 7.4 tok/s), and two concurrent processes split
  bandwidth ~linearly. Not an untapped lever; the ×4 upside is a hypervisor
  allocation question for the platform team (HTP doc open question 1).
- **More compute-removal in attention micro-ops** — softmax, RMSNorm, cast and
  elementwise together are 9.8% of post-fix cycles. The 74.7% mine is exhausted.
- **x86 HTP simulation** — closed 2026-08-14, `Request feature arch with value 81
  unsupported`.
- **Measuring, or quoting, any variant on a bertcache-carrying bin** (§1.3).
- **Editing `REFERENCE.md`'s cycle-level claims before priority 3 lands** — but
  §1's blend correction is *not* cycle-level and ships now (A1.1). It rests on two
  device measurements and a graph-selection rule read out of the SDK source, none
  of which is contingent on Part B.

## 5. Risk register

| Risk | Mitigation |
|---|---|
| Rep variance swallows every A/B | B0 protocol; baseline re-measured first; no interpretation when spread > delta |
| **Another blended-topology comparison slips through** | `lint_bundle_topology.py` (A1.2) as a hard build gate, plus per-arm classification in the runsheet — a memo would not have caught the first one |
| Both discriminator arms land in the mixed band | B2 row 3 routes to the per-op profile rather than to a guess; priority 3 is in the same session |
| §8.1 is right and the W8 head's bytes never reach the device | Then qh reads +0–4%, which is *itself* the byte model's refutation on that arm. Not a wasted build |
| Fusion's +15% does not survive the fix | Built anyway (cheap); priority 4 is explicitly a test |
| KV INT8 breaks the cross-graph encodings contract | A1.3 kernel check gates the build; INT16 fallback; parity gates are the backstop; dead-end recording if both fail |
| Weight-dup root cause sits in closed-source layout logic | "Documented, not fixed" is a valid A1.6 outcome, and §1.3 lowers its stakes |
| Part B never happens | Part A is scoped to stand alone (A5) |
| Build fan-out vs disk (the 08-12 vhdx crash class) | `disk_guard` sized per step; delete intermediates; `du -h` the vhdx, never `ls`. Note A3a's six builds are ctx-bin-only and cheap; only A3.7/A3.8/A3.11 re-export |
| Upload flips repo visibility | Read live pre/post; report and stop; never "restore" from memory |

## 6. Open questions ledger

- **9** (*where do the bytes bind post-fix*) — reframed by §2: the question is not
  *where* but *whether*. B1 priority 1's pair answers it.
- **10** (*bertcache weight-dup root cause*) — A1.6; **demoted**, since §1.3
  independently devalues bertcache-carrying products.
- **11** (*rep-variance cause: thermal, scheduler, or bin layout*) — B0 captures the
  data; init time correlates weakly and is the first thing to check.
- **12** (*is quantized KV reachable?*) — A1.3, and it is a **kernel** question, not
  a Genie-contract question. Resolvable device-free.
- **13** (*what produced the ~4× effective-bandwidth recovery — overlap, or
  de-thrashing the weight stream?*) — B1 priority 3's per-op table vs the 08-13 one.
- **14** (*is `llm_decode_burst` actually the maximum HTP/DDR frequency?*) — HTP doc
  open question 4. Only `virtio_clk` is visible under the GVM, so this is a
  **platform/hypervisor question, not a build one**. Orthogonal to everything above
  and potentially larger than all of it. Route to the device/platform team.
- **15 (new)** (*does the build-time `hvx_threads` value bind, given the runtime
  reports 8 on every workload?*) — A3.1 + B1 priority 2. One ctx-bin rebuild.
- **16 (new)** (*is the device team's 7.79 tok/s build actually faster than ours?*)
  — reopened by §1.2. Their number is AR-1; ours, correctly stated, is 6.84. Needs
  their binary and build config (the 2026-08-14 request, still outstanding).

## 7. Corrections this plan hands to `REFERENCE.md` §7

Drafted here, to be applied by A1.1 rather than asserted from this document.

| # | The claim | Where it appears | What is actually true |
|---|---|---|---|
| 22 | **11.72 tok/s is the best sustained AR-1 decode rate** | `REFERENCE.md` §0 (twice) and §6.1; V2 §0; 08-13 report Test 1/3 | It is a phase blend of ~72 bertcache steps at ~40 ms and ~56 AR-1 steps at ~142 ms (§1.1). The honest pre-fix AR-1 rate is **6.84 tok/s**, measured 2026-08-15. `REFERENCE.md` §6.1 states the rule that this violates |
| 23 | **LADE is −22% vs basic on the technical prompt** | 08-13 report Test 1 + §6.2; `REFERENCE.md` §0 | Compares LADE on `ladekv` against blended basic on `local`. Like-for-like on one bin, pre-fix LADE was 9.18 vs 6.84 = **+34%**. LADE's loss is real but is a *post-fix* effect only (44.707 vs 31.342), and its cause is the break-even shift in §2.4 |
| 24 | **A ~75% build gap / 3-graph penalty exists between `local` and `ladekv`** | 08-13 report §6.3 + rec 2; V2 §2.2; V3 §7 item 4 ("64 ms unexplained") | Same artifact as #22. The decode graphs are structurally identical and share weights within 4 MB. **No graph-count or graph-switching penalty is in evidence**, and the 64 ms term does not exist |
| 25 | **Our builds are +51% faster than the device team's 7.79 ("RESOLVED BACKWARDS")** | `REFERENCE.md` §8.8 | Blended-vs-AR-1. Corrected, our pre-fix AR-1 is 6.84 — ~12% *slower*. §8.8 should be reopened as open question 16, not resolved in either direction, since their build's provenance was never audited |
