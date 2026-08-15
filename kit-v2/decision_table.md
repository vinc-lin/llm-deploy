# Decision table v2 — every outcome has a pre-committed meaning

Agreed in advance so the session needs no round-trip. If you hit a state that is
not in here, record it and stop rather than improvising — that is a finding too.

**Supersedes `kit/decision_table.md` (2026-08-14).**

---

## 0. The gate that applies to every row below

| Observation | Action |
|---|---|
| **an arm's rep spread > the delta being measured** | **Undecided.** Re-run under the §1 protocol. Do not interpret. This outranks every row below |
| a blended-topology bundle's tok/s appears in any comparison | **Reject the comparison.** Re-run on the pure rebuild (runsheet §0.2) |
| `p0_rebaseline` median is not ≈44.7 | Something environmental changed. Report it; the session is not comparable to 2026-08-15 until it is explained |
| any arm's first ~30 tokens diverge from `expected/` | **Quality failure — the speed number is meaningless.** Report the divergence, not the tok/s |

---

## 1. P4 — the pair that decides the plan

Read the **ordering**, not the magnitudes. The models' absolute predictions are
both upper bounds (the byte model holds 43 GB/s fixed while removing the largest
contiguous read; the compute model assumes perfect thread scaling and zero
stall), but they disagree about *which arm wins*, and that is robust.

| `p4_qh_ladekv` | `p4_cl512_ladekv` | Verdict | What happens next |
|---|---|---|---|
| **≥ +12%** | ≤ +12% | **Byte-bound** | `qh` becomes the ship base. KV signed-INT8 (kernel confirmed available) and weight-stream compression become the lead workstreams. CL stays a product knob |
| ≤ +8% | **≥ +19%** | **Compute-bound** | The CL ladder leads — build `cl768`, and re-open CL=256 for short-context products. `qh` is parked next to W4A16. Attention-side compute (fusion, GEMV shape, HVX occupancy) becomes the workstream |
| both mid-band | both mid-band | **Mixed regime** | Do not guess a ladder. Re-derive it from P1's per-op table before building anything further |
| **both < +5%** | | **Neither model holds** | Something outside both is binding — per-op dispatch (~220 µs RPC × op count), DVFS, or whatever drives the `pastkv2g` variance. P1's profile becomes the only admissible next input, and "is `llm_decode_burst` actually the maximum HTP/DDR clock?" escalates to the platform team |
| qh **negative** | — | Consistent with `REFERENCE.md` §8.1 — the 151 MB never reaches the device (only 12.5 MB left the ctx-bin) | Not a wasted build: it is the byte model's refutation on this arm. Record and park `qh` permanently |

## 2. P5 — the null test

| `p5_hvx8` vs `p5_ctrl` | Meaning |
|---|---|
| **> +5%** | **The byte model is falsified outright** — zero DDR bytes changed. Adopt `hvx_threads: 8` in every build immediately; it is free. Also re-read P1's cycle count at 8 threads, not 4 |
| within the rep spread | The build-time 4 was not binding, or 8 units were already in use. Interpret P1 at 8 threads. Keep 4; changing it buys nothing |
| **negative beyond the spread** | Thread oversubscription on a single HTP. Keep 4 and record — this would be the first evidence that HVX scheduling is contended |

## 3. P1 — the cycle profile

| Aggregate cycles | Meaning |
|---|---|
| **~88M, zero `Expand` ops** | The fix is confirmed structurally. **Then** combine with P5: at 4 threads this is ~99% DSP-busy (compute-bound); at 8 it is ~50% idle (byte-bound) |
| ~350M | The fix did **not** reach this binary. That is a build defect on our side, not a disproved hypothesis — send the bin's `qnn-context-binary-utility` dump and stop |
| between | Partial — some graphs kept the old attention. Check `lint_gqa_ops.py` on each DLC |
| still cannot run | Report **which** of the four environmental checks in runsheet §P1 failed. That is the deliverable; do not spend more than ~15 min on it |

**The new top-20 is as valuable as the total.** With replication gone, whatever
now sits at the top is the next round's target.

## 4. P6 — the cheap knobs

| Observation | Action |
|---|---|
| any knob **≥ +5%** vs `p5_ctrl` | Adopt immediately — zero build cost, zero risk, no quality gate needed |
| `p6_udma` ≥ +5% | Additionally: this is a **v81-and-above** feature that has never been enabled in any build we have shipped. Fold into every future build and note it for the 4B work |
| `p6_dlbc` / `p6_wpack` flat | Expected. Both produced byte-identical binaries offline. Record as tried, move on — do not re-test |
| `p6_socmodel72` ≥ +5% | Adopt as the default `soc_model` for all builds, including the ViT and the 4B tower |

## 5. P7 — fusion

| Observation | Action |
|---|---|
| **≥ +5%** | Fold into the ship base and stack with the P4 winner |
| flat | Fusion's pre-fix +15% was access-pattern recovery that the GQA fix already collected. **Retire it as a lever** — same reasoning that parked LADE |
| negative | Report. Fused GEMMs are larger; if the regime is compute-bound this is possible and interesting |

## 6. P2 — the variance question

| Observation | Action |
|---|---|
| spread tracks temperature | Thermal. Every future protocol needs a mandatory cool-down, and 2026-08-15's numbers need re-reading with that in mind |
| spread does **not** track temperature | More serious: something in the 2-graph layout or the scheduler is non-deterministic. Escalate — it bounds the resolution of every measurement we can take |
| spread ≤ 5% across 8 reps | The 2026-08-15 spread was transient. Note it and move on; keep 5 reps anyway |

---

## What is NOT being decided this session

Stated so nobody spends time on it:

- **LADE.** Parked by measurement. Post-fix break-even is 2.30 accepted
  tokens/call against ~1.6 measured. It comes back only if a learned draft head
  clears that bar, which is a build-side question.
- **W4A16 / INT4.** v81 has zero INT4 input datatypes for MatMul and the
  converter folds s4 → f16. Dead on kernels and on accuracy.
- **Multi-core.** Genie creates a single-core device; a 2-core bin measured
  slower; two processes split bandwidth linearly. The ×4 upside is a hypervisor
  allocation question for the platform team, not a build one.
- **The hybrid prefill / TTFT product.** Its bundle produced degenerate output
  last session and the mechanism is not yet reproduced device-free. It is also
  blended by construction, so it can never report a comparable decode rate.
