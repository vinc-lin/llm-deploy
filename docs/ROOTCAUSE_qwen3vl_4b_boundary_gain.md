# Root cause — the Qwen3-VL-4B text tower is fed a prompt it was never calibrated for

**Status:** root-caused, device confirmation pending (Test I) · **Date:** 2026-08-21
**This document has been wrong twice and is on its third version.** Both earlier
versions are summarised in §6 so the corrections are not lost, because each was
wrong in an instructive way. Read §1–§3 for the mechanism; §5 for what to do.

---

## 1. One paragraph

The 4B activation ranges are calibrated **exclusively on chat-templated
prompts** — `vl_calib_build.py` builds every window with
`apply_chat_template(..., add_generation_prompt=True)` — so position 0 is always
`<|im_start|>` and the attention sink sits at row 1. Every 4B text test this
project has run feeds a **raw, untemplated** prompt: the v5 guide's Test 3a is
`genie-t2t-run -p "What is 2+2? Answer with one number."`, and
`genie_dialog_qwen3vl_4b.json` sets `bos-token: -1` and carries no prompt
template, so nothing prepends anything. A raw prompt puts a bare content word at
position 0, which forms an attention sink there instead — a sink the calibration
never saw. Its activations land outside the calibrated INT16 ranges, clip, and
corrupt position 0's hidden state and KV in both shards. Every later token
attends heavily to that poisoned sink, so generation degenerates. **The 4B has
never been asked a question in the form it was built for.**

---

## 2. The mechanism, measured

Activations are `bw:16, dtype:INT, PER_TENSOR` asymmetric with calibrated
ranges. On a raw prompt the sink row exceeds them:

| tensor | raw-prompt sink row | calibrated | over |
|---|---:|---:|---:|
| `layers.0/mlp/down_proj/MatMul_output_0` | **9.00** | 5.482 | **1.64×** |
| `layers.1/mlp/Mul_1_output_0` | 61.40 | 59.272 | 1.04× |

Clamping just those two in the host fp32 graph reproduces the device exactly:

| | row-0 gain | residual | cosine |
|---|---:|---:|---:|
| host, both clamped | **1.3914** | 0.416% | 0.999991 |
| **device, measured (Test E)** | **1.38959** | **0.447%** | **0.999990** |

Neither tensor alone does it (0.93× and 1.06×); the interaction does, because
clipping layer 0's `down_proj` changes the residual entering layer 1 and so
changes what layer 1's SwiGLU does. That agreement to three decimals is what
makes the clamp simulation a trustworthy device stand-in.

On **templated** input the same simulation gives row-0 gain **0.9990**, and the
device agrees: Test G measured worst |gain−1| **4.4%**, the real sink (row 1,
RMS 220.3) at **0.989 / cos 1.000000**, all logits argmax matching.

---

## 3. Why this explains everything, including the parts that looked contradictory

* **Test C's ladder.** n=1 generates from the poisoned row 0 → wrong. n≥4's
  *first* token comes from the last prompt row, which is not the sink → right.
  Every later decode step attends to position 0's corrupted KV → garbage. And
  Genie's n=1 output was the *exact token* the chained qnn-net-run probe
  produced, which independently shows Genie's feed is numerically faithful.
* **The layer scan** (Test F) showing taps diverge from layer 1 on sink rows is
  this clipping, one layer downstream of where it starts.
* **Photos collapsing to EOS while the synthetic sample degenerated into
  repetition**: different poisoned row-0 states select different degenerate
  first tokens.
* **Host parity 20/20.** `parity_e2e_vl.py` feeds windows built the same
  templated way as calibration, so it exercises the in-distribution path and
  cannot see this.
* **The 0.6B is coherent on the same raw prompts** because *its* calibration
  used raw prompts (`CALIB_PROMPTS` in `quantize_aimet.py`). Same code, opposite
  input contract. This is why the working 0.6B never pointed at the problem —
  and why "the 0.6B works, so the pipeline is fine" was misleading all along.

There is no longer anything unexplained, and no evidence of a second defect:
**no 4B Genie run with a correctly templated prompt has ever been attempted.**

---

## 4. What is ruled out, with evidence

| hypothesis | why it is dead |
|---|---|
| fp16 saturation in `input_layernorm` | activations are INT16 with calibrated ranges; and the arithmetic misses by 15× |
| `tf_enhanced` discarding outliers | the ranges sit **on** the calibration max (5.4819 vs 5.4805 observed) — nothing was discarded, and min-max would produce identical ranges. The requantize was cancelled before it ran |
| QDQ double-quantization | 0 Q/DQ ops in either ONNX |
| shard 0 numerics on in-distribution input | Test G, on device |
| shard 1 | isolated logits correct on device |
| LUT *precision* | Test G fed the identical `uFxp_16` grid Genie writes |
| VL deepstack injection | all six inputs are exactly zero |
| single-channel (c4) fault | predicts cos 0.9973; device measured 0.999990 |

---

## 5. The fix

**Primary, and free: send the model the prompt format it was calibrated for.**
Apply the ChatML template. `bos-token: -1` is already correct — the template
supplies `<|im_start|>` itself, and setting both would double it.

```
<|im_start|>user
What is 2+2? Answer with one number.<|im_end|>
<|im_start|>assistant
```

(Real newlines. `genie-app` script strings never unescape, so the pipeline path
must use `node set textFile`.) `docs/TEST_I_templated_prompt.md` is the
one-minute device check.

**Secondary, only if raw prompts must also work.** Add untemplated windows to
`vl_calib_build.py` and requantize — this is the case where
`--act-range-scheme` finally matters, since a calibration set containing both
distributions has genuine outliers that `tf_enhanced` might then discard. Not
needed for a chat application, which always templates.

**Not a fix:** rebuilding on any of §4.

---

## 6. The two earlier versions, and why they were wrong

**Version 1 — "`tf_enhanced` discards the sink outlier."** False: measurement
showed the ranges sit on the observed calibration maximum, so nothing was
discarded. Caught by a ten-minute pre-flight before the requantize ran.

**Version 2 — "the clipping is a probe artifact; production is unexplained."**
The measurement was right (templated input is clean) but the *inference* was
wrong. It assumed "production" means a chat-templated prompt. Our device tests
do not template — `genie-t2t-run -p` passes text verbatim and the config adds
nothing. The probe's tokens `[3838, 374, 264, 1273]` are literally "What is a
test", i.e. Test C's `prompt_4.txt`. **The probe was a faithful miniature of what
we actually run.** So the dichotomy was never "probe vs production"; it was
"untemplated vs templated", and every observed garbage run sits on the
untemplated side.

That version also produced two follow-on errors: it ranked **multi-chunk
prefill** (Test H) as the strongest suspect, when the failing prompts are ~15
tokens and therefore single-chunk — H cannot explain them; and it explained Test
E's n=129 result as "the last real token lands on its chunk's flat first row",
when chunk 2's row 0 attends to a populated past and is no sink. The real
mechanism is the poisoned position-0 KV carried forward in the cache. Same
prediction, different cause — which is how the error survived.

**The method lesson, corrected.** Version 2's lesson was "the simplest input is
not a typical input." The sharper and more useful one: **check a probe against
what your tests actually execute, not against an imagined production.** The
probe modelled our tests perfectly; our tests did not model the contract the
calibration assumed, and nobody had checked which prompt form the calibration
used until 2026-08-21.

---

## 7. Reproduce

```bash
ssh tank && cd ~/llm-deploy && source scripts/env.sh
E=$LLMDEPLOY_DATA/work/quant/qwen3vl-4b-w8a16-gqa-split-enc/chunk0.encodings
K=$LLMDEPLOY_DATA/work/text_probe_i     # --suite i: same question, raw vs templated

$PY_DEPLOY scripts/validate/scan_activation_clipping.py $K $E i0_raw
$PY_DEPLOY scripts/validate/scan_activation_clipping.py $K $E i1_templated
$PY_DEPLOY scripts/validate/sim_activation_clipping.py  $K $E i0_raw overrange
$PY_DEPLOY scripts/validate/sim_activation_clipping.py  $K $E i1_templated overrange
```

Tank only (~10 GB RSS). Measured results are in
`docs/TEST_I_templated_prompt.md` §3.
