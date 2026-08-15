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
| `dlbc` | 1,086,570,496 | 0 | ⚠️ identical — unproven |
| `wpack` | 1,086,570,496 | 0 | ⚠️ identical — unproven |

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

## 3. Compute side — predicted

Cycle shares renormalized over the 88,225,159 residual: attention GEMV 44.6%,
weight GEMMs 35.4%, `lm_head` 6.9%, softmax/RMSNorm/cast/elementwise 9.8%,
shape ops 3.2%.

| Variant | Byte model | Compute model | Role |
|---|---:|---:|---|
| **W8 `lm_head`** (−155,582,464 B; 6.9% of cycles) | **+19.3%** | **+3.6%** — or **0%** if `REFERENCE.md` §8.1 holds and the head is re-materialized to FP16 at prepare time | ⭐ discriminator |
| **CL=512** (−73,400,320 B of KV; GEMV+softmax scale with CL) | **+8.3%** | **+34.7%** (−25.8% cycles) | ⭐ discriminator, opposite ordering |
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

| Variant | Predicted decode `read_total_bytes` | Δ |
|---|---:|---:|
| W8 head | 805,548,032 | −155,582,464 |
| CL=512 (KV term only — see note) | 887,730,176 | −73,400,320 |
| KV signed-INT8 @ CL=1152 | 895,127,552 | −66,002,944 |
| W8 head + KV INT8 | 739,545,088 | −221,585,408 |

*KV read = 2 × 28 layers × 8 heads × 128 dim × past × 2 B; past = 1151 →
132,005,888, past = 511 → 58,605,568. The CL=512 figure counts only the KV term;
mask and activation terms also shrink, so **record the converter's own number**
rather than asserting this one.*

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
