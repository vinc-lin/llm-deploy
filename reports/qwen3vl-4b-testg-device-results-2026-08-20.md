# Qwen3-VL-4B — Test G device results (realistic prompt boundary)

**Source:** device team, received 2026-08-20, pasted into the build session.
Body below is **verbatim**; the annotations at the end are ours and are clearly
marked.

---

## Executive Summary

The 1.39× boundary gain measured in Tests B, E, and F is a probe artifact caused
by feeding bare word tokens at position 0 with an empty cache — a synthetic
attention-sink condition no real prompt ever produces. On realistic
chat-templated inputs, the device boundary is clean.

* **Worst |gain−1| across all rows, all cases: 4.4%** (vs 39% on the synthetic probe)
* **Real attention sink (row 1, RMS 220.3):** gain 0.989, cosine 1.000000
* **Decode with real cache (r3_decodectx):** gain 0.991 — clean
* **All logits argmax match**, both chained and isolated

**Conclusion:** Shard 0's W8A16 quantization is faithful on production input. The
4B text garbage under Genie is **downstream of shard 0**.

## Boundary Gain Results

### r0_text — EVAL text prompt (13 real rows)

| Row | ref RMS | gain | residual | cosine |
|---|---:|---:|---:|---:|
| 0 | 2.072 | 1.0441 | 7.36% | 0.9973 |
| 1 (real sink) | 220.330 | 0.9889 | 0.06% | 1.000000 |
| 2 | 1.117 | 0.9988 | 8.11% | 0.9967 |
| 3 | 1.782 | 1.0325 | 4.65% | 0.9989 |
| 12 (last real) | 0.954 | 0.9780 | 7.46% | 0.9972 |

Logits chained: argmax 5/5, worst cos 0.997 · isolated: 5/5, worst cos 0.99997

### r1_image — EVAL image+text prompt (113 real rows)

| Row | ref RMS | gain | residual | cosine |
|---|---:|---:|---:|---:|
| 0 | 2.072 | 1.0441 | 7.36% | 0.9973 |
| 1 (real sink) | 220.330 | 0.9889 | 0.06% | 1.000000 |
| 2 | 1.117 | 0.9988 | 8.11% | 0.9967 |
| 3 | 1.428 | 0.9756 | 6.23% | 0.9981 |
| 112 (last real) | 0.823 | 0.9900 | 9.61% | 0.9954 |

Logits chained: argmax 5/5, worst cos 0.985 · isolated: 5/5, worst cos 0.9999

### r2_chunk0 — Full 128-row image chunk

| Row | ref RMS | gain | residual | cosine |
|---|---:|---:|---:|---:|
| 0 | 2.072 | 1.0441 | 7.36% | 0.9973 |
| 1 (real sink) | 220.330 | 0.9889 | 0.06% | 1.000000 |
| 2 | 1.117 | 0.9988 | 8.11% | 0.9967 |
| 3 | 1.428 | 0.9756 | 6.23% | 0.9981 |
| 127 (last real) | 0.839 | 0.9820 | 12.97% | 0.9916 |

Logits chained: argmax 5/5, worst cos 0.958 · isolated: 5/5, worst cos 0.9999

### r3_decodectx — Decode step with real cache (cache_len=13)

| Row | ref RMS | gain | residual | cosine |
|---|---:|---:|---:|---:|
| 0 | 0.936 | 0.9908 | 7.25% | 0.9974 |

Logits chained: argmax 1/1, cos 0.99994 · isolated: 1/1, cos 0.99998

## Still viable suspects for 4B garbage under Genie

1. Genie LUT embedding path — LUT declared `float32` while graph expects `uFxp_16`
2. Dialog / config issues — rope-theta, m-ROPE lengths
3. Genie's multi-shard orchestration
4. KV cache handling under Genie
5. Shard 1 (unlikely)

---

## Annotations (build side, 2026-08-20)

**The result is accepted and matches the host prediction.** Predicted worst
|gain−1| 3.5%, measured 4.4%; predicted `r0_text` row 3 at 1.0347, measured
1.0325. Shard 0 is exonerated on production input.

**Internal consistency check, unprompted by the report and worth recording.**
Rows 0, 1 and 2 show *identical* gains (1.0441 / 0.9889 / 0.9988) across all
three prefill cases. That is exactly right — those three rows are the shared
chat-template preamble, so they are the same tokens in every window, and the
host reference RMS agrees (2.072 / 220.330 / 1.117 in all three). Row 3 is where
the windows diverge, and the gains diverge there too. The device is
deterministic and the measurement is trustworthy.

**Why the residuals are HIGH and that is good news.** Rows here show 5–13%
residual after removing the best-fit gain, against **0.447%** on the synthetic
probe. High residual means the error is *not* a uniform scale — it is ordinary
quantization noise spread across channels, which is what healthy W8A16 looks
like. The probe's near-zero residual was the signature of a single systematic
gain; its absence here is the positive evidence that no such gain exists. Note
also that residual is largest on the *smallest* rows (RMS ~1) and 0.06% on the
sink row (RMS 220) — absolute quantization error is roughly constant, so it is
relatively larger on small-magnitude vectors. Expected, not a defect.

**Calibration note for future predictions.** The clamp simulation predicted 3.5%
and the device measured 4.4%. The simulation applies `Clip` to the calibrated
range but does **not** round to the quantization grid, so it models clipping and
omits rounding noise; accumulated over 180 tensors and 18 layers, a ~1 point gap
is the expected consequence. **Read `sim_activation_clipping.py` predictions as
lower bounds on deviation, not point estimates.**

**Suspect 1 is largely already answered.** The probe fed `inputs_embeds` in
exactly the encoding Genie produces — `uFxp_16`, scale 3.5394e-04, offset
−32927, the same grid Genie's `quantizeInput` writes when it reads the fp32 LUT
— and the boundary came out clean. So the *precision* of that path, including
the fact that the grafted range spans ±11.6 to cover the ViT's `image_features`
while text embeddings only span ±0.24, is not the problem. What Test G does
**not** cover is whether Genie fetches the *right LUT row* (stride/offset), only
that the dtype and grid are fine. The 0.6B LUT probe (`02_probe_06b_u16in`)
exercised the lookup itself and was coherent, which weakens this suspect further
without eliminating it.

**A sixth suspect the report omits, and it is the strongest.** Multi-chunk
prefill. A 273-token prompt is three AR=128 calls, and Genie must carry KV
forward between them, so chunks 2 and 3 run against a *partially populated*
cache. Every probe to date — Test G included — has run a **single chunk against
an empty cache**. That path is unprobed on device, and it is precisely where the
FLOAT_16 padding bug lived (`variant > n_process`, the last partial chunk); the
dtype was fixed but the chunking path itself was never independently verified.
It also sits squarely inside suspects 3 and 4.

**Watch item.** `r2_chunk0`'s last row has the worst boundary residual (12.97%,
cos 0.9916) and its chained logits fall to cos 0.958 — still argmax-correct, and
isolated is 0.9999, so shard 1 is fine given good input. But the last row of the
last chunk is the row that produces the first generated token in real use, so it
is the tightest margin in this data set and worth re-checking if a later test
sees a wrong first token.

**Repo state:** `docs/REFERENCE.md` §0 updated; the boundary line of enquiry is
closed. `docs/ROOTCAUSE_qwen3vl_4b_boundary_gain.md` already carries the
retraction this result confirms.
