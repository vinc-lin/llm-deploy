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

| Variant | `context.size` | ctx-bin CL | decode past | Predicted decode `read_total_bytes` | Δ | byte % | cycle % |
|---|---:|---:|---:|---:|---:|---:|---:|
| baseline | 1024 | 1152 | 1151 | 961,130,496 | — | — | — |
| **W8 head** | 1024 | 1152 | 1151 | **805,548,032** | −155,582,464 | **+19.3%** | +3.6% |
| `cl768` | 768 | 896 | 895 | 931,770,368 | −29,360,128 | +3.2% | +11.5% |
| **`cl512`** | 512 | **640** | **639** | **902,410,240** | −58,720,256 | +6.5% | **+26.0%** |
| `cl384` | 384 | 512 | 511 | 887,730,176 | −73,400,320 | +8.3% | +34.8% |
| `cl256` | 256 | 384 | 383 | 873,050,112 | −88,080,384 | +10.1% | +44.9% |
| KV signed-INT8 @ CL=1152 | 1024 | 1152 | 1151 | 895,127,552 | −66,002,944 | +7.4% | small |
| W8 head + KV INT8 | 1024 | 1152 | 1151 | 739,545,088 | −221,585,408 | +29.9% | +3.6% |

> ### ⚠️ A naming trap this table exists to close
>
> **"cl512" means `context.size = 512`, so the ctx-bin CL is 512 + 128 = 640 and
> the decode graph's past dim is 639** — verified against the built
> `gqafix_cl512` bin, whose decode graph is `AR=1 CL=640`.
>
> Three documents inherited the wrong geometry before this was checked:
> V3 §3-3.3's *"CL=512: KV read 132 → 59 MB"* is the **`cl384`** row; V4 rev 1's
> A2 entry *"CL=512 (known from V3) = 873,048,064"* is the **`cl256`** row; and
> V4 rev 2's original **+8.3% / +34.7%** is also `cl384`. All corrected here.
>
> The discriminator is unaffected in kind — `cl512` is still **+6.5% byte vs
> +26.0% compute**, a 4× gap in the opposite direction to the W8 head — but the
> numbers changed, and this is exactly the class of inherited error the A2 gate
> exists to catch. `cl384` would be a marginally sharper discriminator; `cl512`
> is built instead because it is the product-meaningful context size.

*KV read = 2 × 28 layers × 8 heads × 128 dim × past × 2 B. These count only the
KV term; mask and activation terms also shrink with CL, so **record the
converter's own `read_total_bytes`** rather than asserting the prediction.*

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
