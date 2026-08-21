# Test K session — results

**Date:** ____________  **Board / build:** ____________  **Operator:** ________

Procedure: `docs/TEST_K_lut_vs_split.md`. Fill this in as you go, not afterwards.

---

## 0. Preconditions (fill these first)

| check | required | what you saw |
|---|---|---|
| `qwen3-0.6b-w8a16-lutprobe-ladekv_ctx.bin` md5 | `880a6abdec4a64b67b275ec817c054ca` | |
| `embedding_float32_lut.bin` md5 | `2836420ce26c84f4ab7b217a22b2e125` | |
| lutprobe config has `"max-num-tokens": 64` | yes | |
| control bundle used | `qwen3_06b_w8a16_gqafix_ladekv` + `genie_dialog_basic.json` | |
| `qwen3vl-4b-w8a16_1_of_2.bin` md5 *(K2 only)* | `f031e3a7563bf16f2d5ca98a71b357f6` | |

**A failed precondition is itself a result** — report it and stop rather than
measuring the wrong bytes.

---

## Stage K1 — LUT probe vs control

### K1a — probe, short prompt (`What is 2+2? Answer with one number.`)

```
[paste k1a_short.txt verbatim -- if long, the first 200 characters, and say the rest repeats]


```

### K1b — control, short prompt (same prompt, working 0.6B)

```
[paste k1b_short.txt verbatim]


```

### K1a — probe, long prompt (mountain weather)

```
[paste k1a_long.txt verbatim, first 200 characters is fine]


```

### K1b — control, long prompt

```
[paste k1b_long.txt verbatim, first 200 characters is fine]


```

### The measurement

Put probe and control side by side, per prompt, and answer:

| | short prompt | long prompt |
|---|---|---|
| **how many tokens match before they diverge?** | | |
| first token identical? | | |
| the first token where they differ — probe said / control said | | |
| does the probe repeat? does the control also repeat? | | |

> ⚠ "Both repeated" is **not** a divergence. The control repeats too — it has no
> chat template. Only the **match length** discriminates.

### Verdict — tick one

- [ ] **matches the control for tens of tokens** → the LUT feed is sound ⇒ **the split is the fault**
- [ ] **first token right, then diverges immediately** → **the LUT feed is the fault** (the 4B's exact signature)
- [ ] **wrong from token 0** → a contract bug in the probe itself, not the decode defect
- [ ] **failed to load** → paste the exact error and `adb logcat -d` below

```
[load failure output, if any]


```

---

## Stage K2 — image path, one-word prompt

Confirm the swap actually happened — if `prompt_seg2.txt` was not replaced, every
row below is a repeat of Test J and says nothing new:

| | |
|---|---|
| `wc -c prompt_seg2.txt` before the sample run | ______ (expect **116**) |
| `wc -c prompt_seg2.txt` before the photo loop | ______ (expect **103**) |

**Judge the FIRST WORD only.** Generation is expected to degenerate after it —
that is the known decode defect, not a new finding.

| image | first word, verbatim | RELEVANT / WRONG / DEGENERATE / INCONCLUSIVE | does it match the picture? |
|---|---|---|---|
| `sample_image` (red circle + blue square) | | | |
| `wx_clear` | | | |
| `wx_clear2` | | | |
| `wx_clear_snow` | | | |
| `wx_fog_overcast_rain` | | | |
| `wx_snow` | | | |
| `wx_snow2` | | | |

A word that would fit *any* image (`A`, `The`, `This`) is `INCONCLUSIVE`, not
`RELEVANT` — it means the prompt still is not forcing an answer.

---

## Stage K3 — timing

| metric | cold | warm |
|---|---|---|
| init (ms) | | |
| TTFT (ms) | | |
| decode (tok/s) | | |
| K2a pipeline wall-clock (s) | | |

Do not average cold and warm. Say which run was which.

---

## Anything else

Three rules make this usable:

1. **Verbatim beats summary.** `ention ably ance` and `aged aged aged` are
   different findings.
2. **Say what you actually ran** — skipped stages, changed paths, retries,
   commands that failed. An unexplained gap costs a whole round-trip.
3. **Separate observation from interpretation.** "Probe and control diverge at
   token 2" is an observation. "The LUT feed is broken" is an interpretation.
   Both are welcome — label which is which.

```
[notes, surprises, anything that did not fit above]


```
