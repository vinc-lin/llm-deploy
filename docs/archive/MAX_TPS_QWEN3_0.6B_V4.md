# MAX TPS Qwen3-0.6B on SA8797P — Plan V4 (byte-floor descent)

**Date:** 2026-08-16 · **Supersedes:** `MAX_TPS_QWEN3_0.6B_V3.md` §3–§7 (the gqafix
trunk shipped and was measured; V3's decision tree is resolved). V3 §0 (the
inversion principle) and its build discipline carry forward unchanged.
**Analytical basis:** `DEVICE_MEASUREMENT_REPORT_2026-08-13.md` (per-op cycles,
byte accounting) + `DEVICE_MEASUREMENT_REPORT_2026-08-15.md` (post-fix rates) +
`SA8797P_HTP_v81_Hardware_and_Deployment_Quantization_Reference_EN.md` (bandwidth,
multi-core, build-time DDR reporting).

**Structure:** this plan is split into **Part A — everything executable now,
device-free**, and **Part B — one pre-scripted device session**. The split is not
cosmetic: §2 establishes which claims can be settled without silicon and which
cannot, and Part A is designed to be worth executing in full even if Part B never
happens.

---

> ## ⚠ STATUS: DRAFT — four known errors, revision pending (2026-08-16)
>
> This plan was written before `REFERENCE.md` and `MAX_TPS_QWEN3_0.6B_V2.md` were
> read. Both contradict it in load-bearing places. **The structure (device-free
> Part A with a byte gate; one pre-scripted Part B) stands; the direction needs
> rebalancing.** Do not execute A3.4 or A3.5, and do not quote §1's ladder, until
> this is revised.
>
> | # | Claim in this plan | What the repo already establishes |
> |---|---|---|
> | 1 | KV INT8 (§1, A1.1, A3.4) is an open lever | **Half-dead.** `kv-quantization: true` exists only in the **CPU** `QnnGenAiTransformer` backend, not HTP (`REFERENCE.md` §4.1, and §1 "Not available"). The ONNX-level path survives but needs an *unconfirmed kernel path* — an FAE question, not a build |
> | 2 | Sparse weights compression (A3.5) is "the only lever touching the irreducible 440 MB" | **Measured dead: `sparse_weights_compression=1` → 0 bytes saved, "model isn't sparse"** (`REFERENCE.md` §4.1). Only `weights_packing` (untried) and DLBC (open FAE question, §8.6) remain of A3.5 |
> | 3 | W8 head saves 155.6 MB of device DDR traffic (§1 ladder, A3.1, B1 pri-1) | **May never reach the device.** The DLC shrinks 151.3 MB but the ctx-bin only **12.5 MB**; HTP likely re-materializes the INT8 head to 16-bit at prepare time, which fits verify latency going *up* (`REFERENCE.md` §6.4, open question §8.1). **A2's byte gate is the missing test and should move to A1** — no converter DDR summary has ever been recorded for a real `--quant-head` build (§6.5) |
> | 4 | Decode is byte-bound at ~43 GB/s (§1) | **One of three live models, not established.** V2 §1.2 fits the same data with a *compute* model (350M ÷ 4 HVX @ ~1 GHz ≈ 87.5 ms vs 85 measured; extended: 88.5M ÷ 4 ≈ 22.1 ms vs **22.37 measured**), and `REFERENCE.md` §1 attributes the 49–67 → 7 GB/s collapse to *access-pattern fragmentation* (~220 µs RPC, 30–60 µs inter-op), which would survive the GQA fix. The byte model has already failed once here — V3 §7 predicted "883 MB → ~18.1 tok/s"; reality was 44.7. Adjudication is blocked because REFERENCE §1 reports **8** HVX threads always in use while both teams build with `hvx_threads: 4`, and clock is invisible under the GVM |
>
> Consequences for the revision: **the P1 cycle profile becomes unambiguous
> priority 1** (it is the only measurement that discriminates the three models);
> the direction rebalances away from the W8 head toward **fusion**, **CL
> reduction**, and the two free knobs `REFERENCE.md` flags as never A/B'd —
> `soc_model: 72` (§8.4) and `hvx_threads: 8` (§8.9).
>
> Still unread when this was written: `NOTES-genie-io.md`, `NOTES-htp-config-keys.md`,
> `BUILD_GUIDE.md`, and the full hardware doc (grepped only).

---

## 0. The measured state this plan starts from

All 2026-08-15, warm, greedy, 56-token technical prompt:

| Fact | Value | Consequence |
|---|---|---|
| `gqafix_ladekv` basic | **44.707 ± 0.030 tok/s**, TTFT 103 ms | the baseline |
| `gqafix_ladekv` LADE | 31.342, acceptance 1.61 tok/iter | **LADE parked** |
| `pastkv2g` per-rep | 23.43 / **44.54** / 29.34 | **rep variance exceeds every effect this plan chases** |
| `gqafix_hybrid` | degenerate output | TTFT workstream blocked on a wiring bug |
| P1 cycle profile | not run (inputs were pre-fix format) | regime confirmation owed |
| P4 (qh, cl512 arms) | skipped as "unnecessary" | see §1 — the inference was wrong |

## 1. The core claim: the regime flipped, and the next multiplier is bytes

```
step time      = 1000 / 44.707     = 22.37 ms
bytes streamed = read_total_bytes  = 961,130,496 B   (converter accounting)
effective rate                     ≈ 43.0 GB/s
```

Against the hardware doc's own microbenchmarks — **49 GB/s** (excl-wait, the
honest sustained figure) and 63–67 GB/s (incl-wait) — decode now runs at **~88%
of the conservative streaming ceiling**. The same doc back-calculates the
*pre-fix* steady state at **~6–7 GB/s**. The GQA fix therefore bought roughly
**4× in effective bandwidth**, not merely a byte reduction.

Consequence: compute-removal is exhausted as a strategy, and the two remaining
levers are **fewer bytes** and **a better access pattern** — with the second one
worth at most 43→49 GB/s (~+14%), not the larger figure an earlier draft implied.

Where the 961 MB goes:

| Component | Bytes/step | Share | Lever | Reachable in this plan? |
|---|---:|---:|---|---|
| INT8 decoder weights | ~440 MB | 45.8% | weight-stream compression | **only via A3.5** (W4A16 is a recorded dead end) |
| **FP16 `lm_head`** | **311 MB** | **32.4%** | W8 head | **yes — A3.1** |
| **FP16 KV read** (CL=1152) | **132 MB** | 13.7% | KV INT8 | yes if feasible — A3.4 |
| mask, activations, misc | ~78 MB | 8.1% | CL ladder | product variants — A3.3 |

**The W8-head objection is dead.** The measured −14% was an *n-gram acceptance*
penalty under LADE; LADE is parked, and in basic mode there is no speculation to
reject. The 08-13 report's −43% Test 4 is explicitly annotated as confounded.

**Projection ladder** — hypotheses, not specs (two projections in this project
have already been falsified on device):

| Stack | Bytes/step | @43 GB/s | @49 GB/s |
|---|---:|---:|---:|
| today | 961 MB | **44.7 (measured)** | 51 |
| + W8 head | 805 MB | 53 | 61 |
| + KV INT8 | 739 MB | 58 | 66 |

Honest expected landing: **50–56 tok/s**. The 58–66 end requires both KV INT8
landing *and* the access-pattern gap closing.

## 2. What the split can and cannot settle

**Device-free, quantitatively:** every *byte* claim in §1. The converter reports
`read_total_bytes` per graph, and `qnn-context-binary-generator` emits a
build-time **DDR Read/Write and spill/fill report per graph** (hardware doc §4.2).
So "did this change actually remove the bytes it was supposed to?" is answerable
on the build machine, per variant, before anything ships. This is the closest
thing to a device-free proxy we have, and Part A makes it a formal gate.

**Device-gated, irreducibly:** whether removed bytes convert into throughput.
There is no substitute — the x86 HTP path is closed (`libQnnHtpQemu.so` rejects
v81 ctx-bins outright, `Request feature arch with value 81 unsupported`, settled
2026-08-14).

**Therefore Part A builds every branch unconditionally**, as V3 did: local
compute and disk are cheap, device-hours are the scarce resource, and a wasted
build costs disk while a wasted session costs weeks. The only conditional build
in Part A is KV INT8, gated by a *device-free* SDK recon (A1.1).

---

# PART A — Executable now, without the device

## A1. Desk work first (no builds, ~2–3 days, highest value per hour)

| # | Item | Decides | Method |
|---|---|---|---|
| A1.1 | **KV dtype contract recon** | Whether Phase A3.4 exists at all | Does qualla/Genie accept quantized KV I/O, or is the KV contract FP16-fixed? Read `docs/NOTES-genie-io.md`, the shipped qualla engine source, and the requantize-table behavior. The current recipe keeps K/V-proj outputs FP16 *by choice* — establish whether that choice is also a constraint |
| A1.2 | **Weight-dup root cause** (open Q10) | Unblocks every bertcache-carrying product incl. the hybrid | ctx-bin forensics across the seven bins already built. The predictor is isolated (presence of the CL=128 bertcache prefill); what is missing is *why* the generator gives that graph a private 444 MB pool. May terminate in "documented, not fixable" — it is in closed-source layout logic |
| A1.3 | **Hybrid prefill wiring bug** | TTFT 103 → ~40 ms product win | Reproduce device-free through the bertcache graph's exact I/O with a `parity_ladekv_read.py`-style feed. Suspects in order: per-graph input naming, the position-id path under grouped attention, graph selection handing the wrong prefill its mask. The device saw an infinite `"and parallel, and parallel…"` loop — a parity harness should reproduce it as an argmax divergence |
| A1.4 | **Compression-lever audit** | Whether A3.5 is real | `QNN_HTP_GRAPH_OPTIMIZATION_TYPE_ENABLE_SPARSE_WEIGHTS_COMPRESSION = 6` via `graphs[].finalize_config`; `HtpGraphConfig.weights_packing` (exists, never tried); DLBC's actual scope (the hardware doc defines it as *inter-layer* DDR compression, so it likely touches activations, not the 440 MB weight stream — establish this before spending a build) |
| A1.5 | **Correction note to the device team** | Unblocks Part B's agenda | Their rec 5 ("P4 unnecessary, decode clearly compute-bound") inherits the quadrant-A label without the §1 arithmetic. Send the bandwidth derivation and request the qh/cl512 arms explicitly |
| A1.6 | **Regenerate `decode_profile_inputs`** from the gqafix decode graph's real I/O | Makes Part B's cycle profile runnable | `gen_decode_profile_inputs.py` ships in the package; the 08-15 miss was a build-side packaging defect. **Add to the drop checklist: regenerate profiling inputs whenever graph I/O changes** |

## A2. The byte-accounting gate (new, applies to every build below)

For each variant, record the converter's `read_total_bytes` and the ctx-bin
generator's DDR-read report, and check against the prediction. A variant whose
bytes did not move **did not do what it claims**, and must not reach the device —
this is the device-free analogue of the `Unknown Key` rule.

| Variant | Predicted `read_total_bytes` | Δ vs baseline |
|---|---:|---:|
| baseline `gqafix` decode (known) | 961,130,496 | — |
| W8 head | **805,548,032** | −155,582,464 |
| CL=512 (known from V3) | 873,048,064 | −88,082,432 |
| KV INT8 @ CL=1152 | **895,070,208** | −66,060,288 |
| W8 head + KV INT8 | **739,487,744** | −221,642,752 |

## A3. Builds (device-free; full V3 build discipline applies)

`FUSE_FLAGS="--grouped-gqa"` is **load-bearing** on every `lade_build.sh` /
`ladekv_build.sh` call — they re-export verify32 and the past-KV prefill, and
omitting it silently ships old attention in those graphs (`lint_gqa_ops.py`
gates this). `disk_guard` sized per step; convert straight to final filenames;
`qnn-context-binary-utility` check per bin; no duplicate (AR, CL) pairs.

| # | Build | Why | Risk |
|---|---|---|---|
| A3.1 | **`gqafix_qh_ladekv`** — 3-graph past-KV, W8 `lm_head` | **The headline candidate.** The existing HF `gqafix_qh` is a bertcache 2-graph bin: it can answer the science but carries the 444 MB dup and cannot ship. Verify `qairt-dlc-info \| grep lm_head.weight` → `sFxp_8`. Keep verify32 (weight-shared, ~0 cost, preserves optionality) | Low |
| A3.2 | **`gqafix_fuseqkvgu_ladekv`** — QKV + Gate-Up fusion on the gqafix base | The two winning changes have **never been combined** — every fused build predates the fix | **Gain uncertain.** Fusion's +15% was measured pre-fix at ~10 GB/s effective; if it was access-pattern recovery, the GQA fix already collected it. Same reasoning that killed LADE |
| A3.3 | **`gqafix_qh_fuseqkvgu_ladekv`** + CL ladder (`cl768`, `cl512`) on the winner | Ship candidate + product variants | Low |
| A3.4 | **KV INT8** (INT16 first, then INT8) — *conditional on A1.1* | Halves KV read and the byte side of attention GEMV | **Highest.** Cross-graph rule bites hardest here: KV encodings must be byte-identical across all three graphs. Kill criteria: `--eval` ≥3/4 and `parity_ladekv_read.py` 6/6. INT16 is the fallback (still −33 MB) |
| A3.5 | **Compression variants** — sparse weights compression via `finalize_config`, `weights_packing`, plus the already-built-but-unmeasured `gqafix_dlbc` | **The only lever touching the irreducible ~440 MB of decoder weights.** Cost is a ctx-bin rebuild, not a quantization campaign — and A2's DDR report makes the effect **visible device-free** | Medium — may simply not engage |

Per-build local gates, none skippable: numerical equivalence → ONNX parity vs HF
→ `--eval` ≥3/4 → `parity_ladekv_read.py` 6/6 incl. chunked →
`lint_gqa_ops.py` 0 replication ops → graph names + weight sharing (expect shared
≈1,067 MB, const ≈0, spill 0) → **A2 byte gate**.

## A4. Documentation and kit (device-free)

- Update the HF repo README, the gqafix drop README, and
  `BUNDLE_README_gqafix_ladekv.md`: 44.707 is the baseline, LADE is parked, basic
  is the ship config. All three currently state superseded numbers, and the
  bundle README's §5.4 is written as an open question that is now closed.
- Assemble **kit v2** (runsheet + decision table + `run_all.sh` + `expected/`)
  implementing Part B below. Kit hygiene: no serials, no ssh strings.
- Upload per standing rules — single-file commits via `hf_upload_file.py`,
  visibility read live before/after and **never changed unasked**, watchdog for
  tarballs, well under 128 commits/hour.

## A5. What Part A delivers on its own

Even if the device never becomes available: every byte claim in §1 confirmed or
refuted quantitatively; the KV-contract, weight-dup, hybrid-bug and
compression-lever questions answered or documented as dead ends; a full set of
gated, shippable bundles covering every branch; corrected docs; and a session kit
that runs without us. **That is a complete piece of work, not a stalled one.**

---

# PART B — The device session (one sitting, ~2 h)

## B0. Protocol change — mandatory, and it comes first

The `pastkv2g` spread (23.43 / 44.54 / 29.34 tok/s **on one arm**) is larger than
every effect this plan chases. Until the cause is isolated:

- **5 reps per arm, report the median and every raw value.**
- Record thermal state before and after each arm; fixed cool-down between arms.
- **Never compare arms measured in different thermal regimes.**
- **An A/B whose delta falls inside the rep spread decides nothing** and must be
  re-run, not interpreted.

Re-measure the `gqafix_ladekv` basic baseline under this protocol *first* — the
44.707 figure may have come from an unusually cool device, and every delta in
this session is computed against it.

## B1. Priority ladder

| Pri | Arm | Decides |
|---|---|---|
| **1** | **`gqafix_qh_ladekv` basic vs `gqafix_ladekv` basic** | **The whole plan.** Same lineage, same 3-graph past-KV, same size — only the head dtype differs. −155.6 MB (−16.2% of the stream). This is the clean single-variable byte test the 08-15 session skipped |
| 2 | P1 cycle profile, with A1.6's regenerated inputs | Confirms ~90M cycles / zero replication ops; decomposes the 22.4 ms into compute vs streaming — the input every future plan needs |
| 3 | `gqafix_fuseqkvgu_ladekv` basic | Does fusion survive the fix? (A3.2's open risk) |
| 4 | A3.5 compression variants + `gqafix_dlbc` | The only lever on the 440 MB floor |
| 5 | KV INT8 variant (if A3.4 built) | −66 MB, and validates the quantized-KV contract end to end |
| 6 | CL ladder (`cl512`, `cl768`) | Product variants; ceiling is −9.2% at CL=512 |
| 7 | Hybrid prefill TTFT (if A1.3 fixed it) | 103 → ~40 ms, basic mode only, never lade |
| 8 | Basic-mode rates on `simple` / `structured` prompts | 44.707 was one prompt; the record needs the distribution |
| — | **Pull `/data/local/tmp/results/` off the device** | `/data` runs 98–99% full; the per-op record is one cleanup away from gone. Do this **first**, not last |

## B2. Pre-committed decisions

| Observation | Action |
|---|---|
| qh ≥ +12% | Byte-bound confirmed. qh is the ship base; proceed down the ladder; KV INT8 and compression become the lead workstreams |
| qh +5…+12% | Partially byte-bound. Ship qh; expect sub-linear returns from every further byte lever; re-derive the ladder from priority 2's profile |
| qh < +5% | **§1 is wrong.** Not byte-bound at 43 GB/s despite the arithmetic — priority 2's per-op profile becomes the only admissible next input. Park the head; record next to W4A16 |
| fusion ≥ +5% | Fold into the ship base |
| fusion flat | Access pattern is saturated; retire fusion as a lever (it was pre-fix bandwidth recovery, already collected) |
| compression moves DDR-read but not tok/s | The 440 MB floor is not the binding term — stop pursuing weight bytes |
| KV INT8 parity green + ≥ +5% | Fold in; open the CL=2048 product investigation |
| any arm's spread > its delta | Undecided. Re-run under B0; do not interpret |

## B3. If the device is available for only 15 minutes

Run priority 1, 5 reps, both arms. It settles byte-boundedness, which gates
everything else in this plan and most of what would follow it.

---

## 3. What we explicitly do NOT do

- **LADE tuning of any kind** — parked by measurement 2026-08-15. The `verify32`
  graphs stay in the bins (weight-shared, free); the workstream does not.
- **W4A16 / INT4 anything** — v81 ships zero INT4 matmul kernels.
- **Multi-core / multi-process** — closed by the hardware doc: Genie multi-core
  returns 5005 (`QNN_ERROR_NOT_SUPPORTED`), a 2-core ctx-bin measured *slower*
  (3.96 vs 7.4 tok/s), and two concurrent processes split bandwidth roughly
  linearly (~4 tok/s each). Not an untapped lever.
- **More compute-removal in attention** — the 74.7% mine is exhausted; softmax,
  RMSNorm and elementwise sum to <5% of post-fix compute.
- **x86 HTP simulation** — closed 2026-08-14.
- **Editing REFERENCE.md's cycle-level claims before priority 2 lands.**

## 4. Risk register

| Risk | Mitigation |
|---|---|
| Rep variance swallows every A/B | B0 protocol; no interpretation when spread > delta; baseline re-measured first |
| Fusion's +15% does not survive the fix | Built anyway (cheap); priority 3 is explicitly a *test*, not an assumption |
| KV INT8 breaks the qualla contract | A1.1 desk recon gates the build; INT16 fallback; dead-end recording if both fail |
| Weight-dup root cause sits in closed-source layout logic | Accept "documented, not fixed" as a valid A1.2 outcome |
| Part B never happens | Part A is scoped to stand alone (A5) |
| Phase fan-out vs disk (the 08-12 vhdx crash class) | `disk_guard` per step; delete intermediates; `du -h` the vhdx, never `ls` |
| Upload flips repo visibility | Read live pre/post; report and stop; never "restore" from memory |

## 5. Open questions ledger

- **9** (*where do the bytes bind post-fix*) — §1 answers analytically; B1 priority 1 answers empirically.
- **10** (*bertcache weight-dup root cause*) — A1.2; blocks the hybrid product.
- **11** (*rep-variance cause: thermal, scheduler, or bin layout*) — B0 captures the data.
- **12** (*is the qualla KV contract FP16-fixed?*) — A1.1; decides A3.4 in an afternoon of SDK reading.
- **13** (*what produced the 4× bandwidth recovery — overlap, or de-thrashing the weight stream?*) — B1 priority 2's per-op table vs the 08-13 one.
- **14 (new)** (*is `llm_decode_burst` actually the maximum HTP/DDR frequency?*) — hardware doc open question 4; only `virtio_clk` is visible under the GVM, so this is a **platform/hypervisor question, not a build one**. Orthogonal to everything above, and potentially larger than all of it. Route to the device/platform team, not to a build.
