# The Qwen3-VL-4B boundary gain — explained, and NOT the production defect

**Status:** the probe defect is root-caused; **the production text garbage is
not** · **Date:** 2026-08-20
**This document was rewritten the same day it was written.** Its first version
claimed activation clipping was the production root cause and proposed a
requantize. That claim was falsified by measurement before the requantize was
run. The falsifying evidence is §4, and it is the most important section here.

---

## 1. What is established

**Test F's verdict stands.** The trigger is the attention-sink condition, not the
row index: `f1_row0ctx` (sink removed from row 0) is clean at 0.96×, `f2_shift4`
(sink moved to row 4) is amplified at 1.39×.

**The 1.39× gain is fully explained — for the probe.** The activations are INT16
with per-tensor calibrated ranges (`bw:16, dtype:INT`, asymmetric), not fp16, and
the probe's sink row lands outside those ranges:

| tensor | probe sink row | calibrated | over |
|---|---:|---:|---:|
| `layers.0/mlp/down_proj/MatMul_output_0` | **9.00** | 5.482 | **1.64×** |
| `layers.1/mlp/Mul_1_output_0` | 61.40 | 59.272 | 1.04× |

Clamping to those ranges in the host fp32 graph reproduces the device:

| variant | row-0 gain | residual | cosine |
|---|---:|---:|---:|
| `none` (control) | 1.0013 | 0.002% | 1.000000 |
| `layers.0/mlp/down_proj` alone | 0.9295 | 0.169% | 0.999999 |
| `layers.1/mlp/Mul_1` alone | 1.0613 | 0.089% | 1.000000 |
| **both** | **1.3914** | 0.416% | 0.999991 |
| **device, measured (Test E)** | **1.38959** | **0.447%** | **0.999990** |

Neither tensor alone does it — 0.93× and 1.06× — but together they give 1.39×,
because clipping layer 0's `down_proj` changes the residual entering layer 1,
which changes what layer 1's SwiGLU does. The clamp simulation is therefore a
faithful stand-in for the device, accurate to 0.3%, and needs no hardware.

---

## 2. Two claims that did NOT survive

### 2.1 "fp16 saturation in `input_layernorm`" — wrong

Proposed with the Test F result. These activations are not fp16, so 65504 never
enters the path. The arithmetic does not close either: saturating the sum of
squares gives a denominator 21× too small and dropping `c4²` entirely 3.9×,
neither near 1.39×. And a per-layer 1.39× cannot compound over 18 layers and
still be 1.39× at the boundary — in a pre-norm transformer a residual gain does
not compound at all, because RMSNorm is scale-invariant.

### 2.2 "`tf_enhanced` discards the outlier" — also wrong

This was **my** claim, in the first version of this document, and it is false.
`post_training_tf_enhanced` *is* an MSE-optimal range search that can discard
outliers, but measurement shows it did not do so here. The calibrated ranges sit
essentially **on** the observed calibration maximum:

| tensor | calibration max | calibrated range | |
|---|---:|---:|---|
| `layers.0/mlp/down_proj` | 5.4805 | 5.4819 | range ≈ max |
| `layers.0/mlp/gate_proj` | 3.0771 | 3.0768 | range ≈ max |
| `layers.1/mlp/Mul_1` | 59.2846 | 59.2717 | range ≈ max |

Nothing was thrown away. **Min-max would produce the same ranges**, so the
`--act-range-scheme minmax` requantize that this document originally proposed
could not have changed anything. It was not run.

---

## 3. What the calibration set actually covers

Measured by running the real shard-0 prefill graph over all 26 windows
(`calib` and held-out `eval` splits):

* the ranges **generalise**: the worst over-range on a held-out eval window is
  **1.082×**, and most tensors sit below 1.0;
* **row 0 never clips on a realistic window** — row-0 ratios run 0.08–0.99×;
* the calibration set already contains short padded turns (19, 25, 29 tokens),
  image chunks, and chat-templated text, so "add short prompts" was not missing.

---

## 4. The falsification — the probe is not production

The probe's row 0 is a **plain word token at position 0** (`3838`, "What"). A
production prompt never looks like that: it is chat-templated and begins
`<|im_start|>`. So the probe manufactures a sink that real inputs do not have.

Running the same clamp simulation on **real chat-templated windows** — every
encoded activation clamped, which is the device's activation path:

| window | row 0 gain | row 1 gain | worst rows 0–3 |
|---|---:|---:|---:|
| `EVAL img100 'What is happening in…'` | 0.9990 | 1.0000 | 0.001 |
| `EVAL img101 '这张图片里有几个人?'` | 0.9990 | 1.0000 | 0.001 |
| `EVAL text 'The capital of France is'` | 0.9990 | 1.0000 | 0.035 |
| `calib img0-chunk0[0:128]` | 0.9990 | 1.0000 | 0.001 |

**Worst deviation across six realistic windows, rows 0–3: 0.0347 (3.5%).**
The device measured **0.38959 (39%)** on the probe.

Note also where the real attention sink is: on a chat-templated prompt it sits at
**row 1, RMS 220.3** — larger than the probe's synthetic row-0 sink (107.2) — and
it comes through at gain **1.0000**. The calibration covers the model's genuine
massive activations. It does not cover the probe's artificial one.

**Conclusion: activation clipping does not fire on production inputs, and does
not explain the production text garbage.**

---

## 5. What this costs us

The probe-based chain — Tests B, C, E and F — has been measuring a defect
manufactured by the probe's own synthetic token sequence. The measurements are
real and internally consistent; their **relevance to production is not
established**. Specifically:

* the 1.39× boundary gain is an artifact of feeding a bare word token at
  position 0 with an empty cache;
* `decode1tok` (one token, empty and fully-masked cache) is the same artifact —
  a real decode step attends to a populated cache and is not a sink at all;
* so the production symptom (garbled text, EOS-collapsed captions on real
  photographs) is **back to unexplained**.

What *is* newly known and worth keeping: the activation path is clean on
realistic inputs to within 3.5%, which **rules the quantized activation ranges
out** as the production cause. That is a real narrowing, just not the one
claimed this morning.

---

## 6. The next measurement

Build the probe kit from **chat-templated prompts** instead of bare token ids —
the calibration/eval windows are exactly this and can be reused directly, so the
kit is a rebuild of `build_text_probe_kit.py` inputs, not new modelling. Then:

1. host: confirm the predicted boundary is clean (expected, from §4);
2. device: run it. If the device now shows a **clean** boundary on a realistic
   prompt, the fault is downstream of shard 0 and the whole boundary line of
   enquiry closes. If it shows a **dirty** boundary where the host predicts
   clean, the fault is in the ctx-bin/conversion — which the clamp simulation
   cannot see, because it models the encodings, not the converter.

That second branch is the one that would finally separate "our numbers" from
"the toolchain's numbers", and no test so far has done it on a realistic input.

---

## 7. Reproduce

```bash
ssh tank && cd ~/llm-deploy && source scripts/env.sh
E=$LLMDEPLOY_DATA/work/quant/qwen3vl-4b-w8a16-gqa-split-enc/chunk0.encodings
K=$LLMDEPLOY_DATA/work/text_probe_f
C=$LLMDEPLOY_DATA/work/quant/qwen3vl-4b-calib-ar128.npz

# the probe defect, and which tensors cause it
$PY_DEPLOY scripts/validate/scan_activation_clipping.py $K $E fp_ctrl_pre
$PY_DEPLOY scripts/validate/sim_activation_clipping.py  $K $E fp_ctrl_pre overrange

# the falsification: same simulation, realistic chat-templated prompts
$PY_DEPLOY scripts/validate/scan_clipping_realistic.py $C $E     # §3
$PY_DEPLOY scripts/validate/sim_clipping_realistic.py  $C $E 6   # §4
```

Tank only (~10 GB RSS; the fp32 shard does not fit locally).

**A method note worth carrying out of this.** The probe was built to be the
simplest thing that could fail, and that is exactly what made it lie: the
simplest input is not a *typical* input, and a defect that only a synthetic
input triggers will still reproduce perfectly on hardware. Every probe in this
project should now be checked against a realistic input before its result is
generalised.
