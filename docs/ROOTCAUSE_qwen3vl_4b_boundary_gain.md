# Root cause — the Qwen3-VL-4B boundary gain is activation-range clipping

**Status:** root-caused and reproduced on the host · **Date:** 2026-08-20
**Supersedes** the "fp16 saturation in `input_layernorm`" reading of the Test F
result. Fix is device-free and has not been built yet.

---

## 1. What Test F established, and what it did not

Test F's **verdict is correct and stands**: the trigger is the attention-sink
condition, not the row index.

| case | device | meaning |
|---|---|---|
| `f1_row0ctx` — sink removed from row 0 | **0.96×** clean | the gain needs the sink |
| `f2_shift4` — sink moved to row 4 | **1.39×** amplified | the gain follows the sink, not the index |
| layer scan | diverges from **layer 1** on sink rows; clean on non-sink rows | the fault starts in the first block |

What did **not** survive is the mechanism first proposed alongside it — "c4² =
2.75e7 overflows fp16 by 420× → `input_layernorm`'s denominator saturates → 1.39×
→ compounds across 18 layers". Three things are wrong with it:

1. **These activations are not fp16.** `chunk0.encodings` records every one of
   the 181 activation tensors as `bw: 16, dtype: INT, PER_TENSOR`, asymmetric,
   with a calibrated `scale`/`offset`. The fp16 limit of 65504 is not in this
   path at all.
2. **The arithmetic does not close.** Full saturation of the sum of squares at
   65504 would give a denominator 21× too small, and dropping c4² entirely 3.9×
   — not 1.39×. Nothing about a saturating reduction lands near the measured
   value.
3. **A per-layer 1.39× cannot "compound across 18 layers"** and still be 1.39×
   at the boundary; 1.39¹⁸ is ~10⁶. In a pre-norm transformer a residual gain
   does not compound at all — RMSNorm is scale-invariant, so each block's
   contribution is computed from a normalised input and added unscaled.

The real mechanism is one layer over, and it is **ours, not the hardware's**.

---

## 2. The mechanism

`quantize_aimet.py:475` hard-codes `QuantScheme.post_training_tf_enhanced`.
That is an **MSE-optimal range search**: it sweeps candidate ranges and keeps the
one minimising quantization error over the observed distribution. On a
heavy-tailed activation it deliberately **discards the tail**, because
sacrificing one extreme value buys finer resolution for the other 2559 — usually
the right trade, and exactly the wrong one on an attention-sink row.

So the calibrated ranges come out short, and only on sink rows:

| tensor | sink row (0) | calibrated max | over | normal row (1) |
|---|---:|---:|---:|---:|
| `layers.0/mlp/down_proj/MatMul_output_0` | **9.00** | 5.482 | **1.64×** | 1.325 |
| `layers.1/mlp/Mul_1_output_0` | **61.40** | 59.272 | 1.04× | 4.033 |
| `layers.0/mlp/gate_proj/MatMul_output_0` | 3.13 | 3.077 | 1.02× | 2.834 |
| `layers.2/mlp/up_proj/MatMul_output_0` | 35.64 | 35.40 | 1.01× | — |
| `layers.2/mlp/gate_proj/MatMul_output_0` | 37.61 | 37.42 | 1.01× | — |

Layer 0's block output is the residual entering layer 1 — which is precisely
where the device's layer scan saw divergence begin.

---

## 3. Host reproduction — this is what makes it the root cause

Clamping those tensors to their calibrated ranges inside the real fp32 graph,
and measuring the boundary against the untouched reference
(`scripts/validate/sim_activation_clipping.py`):

| variant | row-0 gain | residual | cosine |
|---|---:|---:|---:|
| `none` (control) | 1.0013 | 0.002% | 1.000000 |
| `layers.0/mlp/down_proj` alone | 0.9295 | 0.169% | 0.999999 |
| `layers.1/mlp/Mul_1` alone | 1.0613 | 0.089% | 1.000000 |
| **both** | **1.3914** | 0.416% | 0.999991 |
| all five over-range tensors | 1.3937 | 0.417% | 0.999991 |
| all 180 encoded tensors | 1.3505 | 0.384% | 0.999993 |
| **device, measured (Test E)** | **1.38959** | **0.447%** | **0.999990** |

Rows 1–3 stay within 0.6% in every variant, matching the device's clean rows.
Gain, residual and cosine all agree to three decimal places.

**Note the interaction.** Neither tensor alone reproduces it — 0.93× and 1.06×
multiply to ~0.99 — yet together they give 1.39×. Clipping layer 0's `down_proj`
changes the residual entering layer 1, which changes what layer 1's SwiGLU
product does. No single tensor's clipping ratio predicts the boundary error,
which is why hunting for one culprit would have missed this.

---

## 4. Why it destroys shard 1

The boundary is a **raw residual**, handed between chunks. Layers 18–35 add
contributions computed through RMSNorm, so those are scale-invariant and arrive
*unscaled* — the final hidden state is `1.39 × (residual at 17) + (unscaled
18–35 contributions)`. The final norm before `lm_head` removes the overall
magnitude but not that **ratio**, so the direction is wrong and the logits move.
Row 0's logits are the flattest (least contextualised), so it flips first — which
is exactly the row pattern Test E measured and the argmax 105196 it reproduced.

---

## 5. The fix

Device-free, and it is a build change, not a workaround:

1. Make the activation range scheme selectable in `quantize_aimet.py` (it is
   hard-coded today) and requantize the text tower with **min/max
   (`post_training_tf`)** or a high-percentile scheme for activations. Weights
   are unaffected — they are per-channel symmetric INT8 and not the problem.
2. Re-run `scan_activation_clipping.py` and require **nothing on a sink row over
   its range**, with headroom rather than a bare pass.
3. Re-run `sim_activation_clipping.py --variant overrange` and require the row-0
   gain back to **~1.00**.
4. Rebuild the two ctx-bins, re-run the existing gates, then one device session.

Steps 2 and 3 are the new gate: this defect class was invisible to every gate we
had, because cosine cannot see a scale and the host parity harness never
executes the quantized activation path.

### Worth checking, not yet checked

* **Does the 0.6B have it too?** It ships coherent at 44.707 tok/s, so if it
  clips at all it is not fatal there — plausibly because it is unsplit, so no
  raw residual is ever handed across a boundary. The same scan answers it; run
  it before assuming the 0.6B is clean.
* **The ViT tower** is quantized by the same code path and has not been scanned.

---

## 6. Reproduce

```bash
ssh tank && cd ~/llm-deploy && source scripts/env.sh
E=$LLMDEPLOY_DATA/work/quant/qwen3vl-4b-w8a16-gqa-split-enc/chunk0.encodings
K=$LLMDEPLOY_DATA/work/text_probe_f

$PY_DEPLOY scripts/validate/scan_activation_clipping.py $K $E fp_ctrl_pre
$PY_DEPLOY scripts/validate/sim_activation_clipping.py  $K $E fp_ctrl_pre overrange
```

Needs ~10 GB RSS and a few minutes per run; tank only (the fp32 shard does not
fit locally).
