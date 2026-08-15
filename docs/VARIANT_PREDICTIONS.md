# Variant predictions ledger — what each build should do, before the device says

**The A2 gate of `MAX_TPS_QWEN3_0.6B_V4.md`.** Every variant records **two**
falsifiable predictions before it is allowed near the device: what the byte model
says, and what the compute model says. When they disagree, that variant is
*evidence*. When they agree, it is a *ship candidate*. When neither moves, the
build did not do what it claims and must not ship.

Started 2026-08-16. Baseline throughout: `gqafix_ladekv` basic, **44.707 ±
0.030 tok/s = 22.37 ms/step**, decode `read_total_bytes` **961,130,496**,
residual DSP cycles **88,225,159** (2026-08-13 category table minus the 261.8M
replication ops).

Percentages are **model-internal** — `1/(1 − Δ/baseline) − 1` within each model —
so the two columns are directly comparable to each other and to a measured delta
against a re-baselined control.

---

## 1. Byte side — measured, not predicted

`read_total_bytes` from each build's own `qnn-context-binary-generator` DDR
summary. Three graphs per bin.

| Build | prefill (AR=128) | **decode (AR=1)** | verify32 (AR=32) | Δ decode vs baseline |
|---|---:|---:|---:|---:|
| shipped `gqafix_ladekv` (baseline) | — | **961,130,496** | — | — |
| `ctrl` (control) | 1,013,235,712 | **961,130,496** | 955,887,616 | **0** |
| `hvx8` | 1,013,235,712 | **961,130,496** | 955,887,616 | **0** |
| `udma` | 1,013,235,712 | **961,130,496** | 955,887,616 | **0** |
| `dlbc` | 1,013,235,712 | **961,130,496** | 955,887,616 | **0** |
| `wpack` | 1,013,235,712 | **961,130,496** | 955,887,616 | **0** |
| **`qh_ladekv`** (W8 head) | 867,133,440 | **815,028,224** | 809,785,344 | **−146,102,272** |

**The qh arm's byte gate passes, with a caveat worth carrying.** The measured
saving is 146,102,272 B — **94% of the 155,582,464 ideal** (151936 × 1024). The
~9.5 MB shortfall is unexplained; per-channel scales account for only ~0.6 MB.
So the byte model's prediction for this arm is **+17.9%** measured, not the
+19.3% derived from the ideal saving. The bytes moved, which is what A2 gates
on — but note this is *converter accounting*, and `REFERENCE.md` §8.1's open
question is precisely whether the saving survives to the device: the DLC shrank
151.3 MB while the ctx-bin shrank only 12.5 MB on the 2026-08-12 build.

Head dtype verified `sFxp_8` on **all three** graphs (`decode`, `verify32`,
`prefill`), against a `Float_16` baseline — see `BUILD_GUIDE.md` §5.7 for why the
obvious grep gives the wrong answer here.

Two things worth keeping:

- **The control reproduces the shipped baseline's decode figure exactly**, which
  validates `ctxbin_variant.sh` as an equivalent build path. That matters because
  every A3a variant is compared against this control rather than against the
  shipped bin.
- **All five ctx-bin-only knobs move zero bytes.** For `hvx8` this is the whole
  point — it makes the arm a clean null test for the byte model. For `dlbc` and
  `wpack` it is a caution: they change neither the byte accounting *nor* the
  binary (§2), so they have earned nothing on offline evidence.

`verify32` reads ~5.2 MB less than `decode` because at AR=32 the past dim is
1120 rather than 1151 (−3.56 MB of KV) plus smaller activation terms.

## 2. Did the knob reach the artifact?

HTP backend-extension keys are silently ignored when wrong — no error, no
warning, exit 0 (`NOTES-htp-config-keys.md`; this repo has been bitten three
times). The only offline signal is whether the built artifact changed.

| Build | ctx-bin bytes | Δ vs control | Consumed? |
|---|---:|---:|---|
| `ctrl` | 1,086,570,496 | — | — |
| **`hvx8`** | 1,088,847,872 | **+2,277,376** | ✅ **yes** |
| **`udma`** | 1,086,783,488 | **+212,992** | ✅ **yes** |
| **`socmodel72`** | 1,086,820,352 | **+249,856** | ✅ **yes** |
| `dlbc` | 1,086,570,496 | 0 | ⚠️ identical — unproven |
| `wpack` | 1,086,570,496 | 0 | ⚠️ identical — unproven |

`socmodel72` sets `soc_model` **and** `soc_id` to 72 in the `devices` block —
`REFERENCE.md` §8.4's never-run A/B, which the device team's own verified config
(HTP doc §8.4) sets and ours left at the generic 0. It changes the artifact, so
whatever extra O=3 algorithms Qualcomm document behind naming the SoC are being
selected. `ctxbin_variant.sh` needed a `__devices` override key to build it.

`hvx_threads` was known inert at **runtime** (08-13 Test 5) and is documented
build-time-only. **This is the first evidence the build-time value is actually
consumed**, which is what qualifies it as a null test rather than a no-op.

`extended_udma` had sat in the `"memory"` section — which is `extra="forbid"`
with exactly one field — in every build this project has ever shipped, so it had
never applied. This is its first real build, on a v81-and-above feature on a v81
part.

**"Identical" does not prove a no-op** — some keys are runtime hints that do not
alter layout. It proves only that nothing observable offline changed, which is
enough to rank `dlbc` and `wpack` last.

### 2b. ⚠️ The qh ctx-bin independently reproduces `REFERENCE.md` §8.1

| Build | ctx-bin bytes | Δ vs control | spill |
|---|---:|---:|---:|
| `ctrl` | 1,086,570,496 | — | 0 |
| `qh_ladekv` | 1,078,185,984 | **−8,384,512** | 0 |
| `cl512_ladekv` | 1,084,981,248 | −1,589,248 | 0 |

**The qh DLCs are ~151 MB smaller each, and the converter's decode
`read_total_bytes` drops 146,102,272 — but the shipped ctx-bin shrinks only
8.4 MB.** That is the same signature the 2026-08-12 qh build showed (DLC
−151.3 MB, ctx-bin −12.5 MB, `REFERENCE.md` §6.4/§8.1), now reproduced on an
independent build with a different topology.

The hypothesis it supports is that **HTP re-materializes the INT8 head to 16
bits when preparing the context blob**, because the `FullyConnected`'s input and
output are both `Float_16`. If so, the converter's DDR estimate — a graph-level
model — does not describe what the device streams, and the qh arm's real device
result is **~0%, not +17.9%**.

This gives the qh arm a **third** prediction, and it sharpens the experiment
rather than muddying it:

| qh device result | Reading |
|---|---|
| ≈ +18% | converter accounting is right, byte-bound, §8.1 refuted |
| ≈ +4% | compute-bound |
| **≈ 0%** | **§8.1 confirmed — the saving never reaches the device**, and the byte model is untestable on this arm. `cl512` then carries the whole discrimination |

Either way it is not a wasted build. But it is a reason to weight `cl512` — whose
saving is activation traffic and cannot be re-materialized away — as the more
trustworthy half of the pair.

Weight sharing is healthy on both new bins (~1.08 GB, not the ~1.5 GB an
unshared build produces) and spill is 0 across all graphs, so neither carries
the bertcache weight-dup of V3 §10b.

## 3. Compute side — predicted

Cycle shares renormalized over the 88,225,159 residual: attention GEMV 44.6%,
weight GEMMs 35.4%, `lm_head` 6.9%, softmax/RMSNorm/cast/elementwise 9.8%,
shape ops 3.2%.

| Variant | Byte model | Compute model | Role |
|---|---:|---:|---|
| **W8 `lm_head`** (−155,582,464 B; 6.9% of cycles) | **+19.3%** | **+3.6%** — or **0%** if `REFERENCE.md` §8.1 holds and the head is re-materialized to FP16 at prepare time | ⭐ discriminator |
| **`cl512`** (−58,720,256 B of KV; GEMV+softmax scale with CL) | **+6.5%** | **+26.0%** (−20.6% cycles) | ⭐ discriminator, opposite ordering |
| **`hvx_threads: 8`** | **0.0%, by construction** (measured, §1) | up to large if build-time 4 was binding | ⭐ null test |
| QKV + Gate-Up fusion | +9.1% (−80 MB, device-team measured) | ~+11% (~10% of cycles) | ship candidate, not evidence |
| KV signed-INT8 (−66,002,944 B) | +7.4% | small — GEMV byte side only | build, kernel confirmed |
| `udma` / `socmodel72` | 0.0% | unknown | cheap lottery ticket |
| `dlbc` / `wpack` | 0.0% (measured) | unknown, artifact unchanged | lowest priority |

**The W8-head and CL=512 rows predict opposite orderings.** That pair settles the
regime with no assumption about clock or thread count, which is why it is
priority 1 of the device session and why both must be run.

## 4. Predicted `read_total_bytes` for the builds not yet complete

To be checked against each build's own DDR summary when it lands — a variant
whose bytes did not move did not do what it claims.

**M** = measured from the build's own DDR summary. **e** = estimated (see the
calibration note below).

| Variant | `context.size` | ctx-bin CL | decode past | decode `read_total_bytes` | Δ | byte % | cycle % |
|---|---:|---:|---:|---:|---:|---:|---:|
| baseline | 1024 | 1152 | 1151 | **961,130,496** ᴹ | — | — | — |
| **W8 head** | 1024 | 1152 | 1151 | **815,028,224** ᴹ | −146,102,272 | **+17.9%** | +3.6% |
| `cl768` | 768 | 896 | 895 | 917,090,304 ᵉ | −44,040,192 | +4.8% | +11.5% |
| **`cl512`** | 512 | **640** | **639** | **873,048,064** ᴹ | −88,082,432 | **+10.1%** | **+26.0%** |
| `cl384` | 384 | 512 | 511 | 851,030,016 ᵉ | −110,100,480 | +12.9% | +34.8% |
| `cl256` | 256 | 384 | 383 | 829,009,920 ᵉ | −132,120,576 | +15.9% | +44.9% |
| KV signed-INT8 @ CL=1152 | 1024 | 1152 | 1151 | 895,127,552 ᵉ | −66,002,944 | +7.4% | small |

> ### ⚠️ Two things this table corrects, one of them my own over-correction
>
> **1. The geometry.** "cl512" means `context.size = 512`, so the ctx-bin CL is
> 512 + 128 = **640** and the decode past dim is **639** — verified against the
> built bin, whose decode graph is `AR=1 CL=640`. V3 §3-3.3's *"CL=512: KV read
> 132 → 59 MB"* uses past = 511, which is `cl384` geometry, so that figure is
> mis-derived.
>
> **2. But V4 rev 1's `873,048,064` was RIGHT, and my re-attribution of it to
> `cl256` was wrong.** It is the genuine measured `read_total_bytes` of the
> cl512 decode graph, confirmed twice on 2026-08-16 (the lade intermediate and
> the ladekv bin both report it). Its Δ of **−88,082,432** is exact. What was
> wrong was *my* arithmetic: a KV-only model predicts −58,720,256 and
> **under-predicts the real saving by a factor of 1.50**.
>
> **Calibration.** The measured ratio is 88,082,432 / 58,720,256 = **1.50003** —
> clean enough to use. It is consistent with the concatenated **K** tensor being
> re-read after the `Concat`: past K+V is 1.0×, and re-reading the concatenated K
> alone adds 0.5×. The `ᵉ` rows apply that 1.50 factor to their KV term; they are
> calibrated on one measured point, so **record the converter's own figure when
> each is built** rather than trusting them.
>
> The discriminator is unaffected in kind: `cl512` is **+10.1% byte vs +26.0%
> compute**, still the opposite ordering to the W8 head's **+17.9% / +3.6%**.

## 5. How to add a row

```sh
# byte side
grep read_total_bytes <ctxbin build log>          # 3 lines: prefill, decode, verify32

# did the knob reach the artifact
scripts/validate/ctxbin_diff.py --control <ctrl.info.json> <variant.info.json>

# topology -- a blended bin cannot be compared against 44.707 at all
scripts/validate/lint_bundle_topology.py --require-pure <variant.bin>

# and for a bitwidth variant, verify the dtype rather than trusting the flag
qairt-dlc-info -i <dlc> | grep lm_head.weight     # sFxp_8, never Float_16
```
