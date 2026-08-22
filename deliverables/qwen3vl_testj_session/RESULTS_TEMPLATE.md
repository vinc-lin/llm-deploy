# Test J session — results

**Date:** ____________  **Board / build:** ____________  **Operator:** ________

## 0. Preconditions (fill these first)

| check | required | what you saw |
|---|---|---|
| `qwen3vl-4b-w8a16_1_of_2.bin` md5 | `f031e3a7563bf16f2d5ca98a71b357f6` | |
| `qwen3vl-4b-w8a16_2_of_2.bin` md5 | `0f1c86e89752b499eec09e9e10a73014` | |
| `sample_image_fp32.raw` size | `6295552` | |

If either md5 differs, **stop and report** — everything below would be measured
on the wrong bytes.

---

## Stage A1 — decode-step probe

| case | expected argmax | **argmax you got** | match? |
|---|---:|---|---|
| `j0_2plus2_s1` | 151645 | | |
| `j1_weather_s1` | 9104 | | |
| `j2_weather_s2` | 4344 | | |

Paste the analyzer's `logits chained` lines:

```
```

## Stage A2 — cross-chunk prefill

| case | row-0 gain | row-1 | last row | within ±5%? |
|---|---|---|---|---|
| `c0_chunk0` | | | | |
| `c1_chunk1` | | | | |
| `c2_chunk2` | | | | |

---

## Stage B — Genie text

### B1 templated 2+2 — correct answer is `4` then stop (two tokens)

Prompt token count from profile (**must be 20**): ______

Generated text, **verbatim**:

```
```

### B2 templated weather — correct is "Mountain weather changes quickly because …"

Generated text, **verbatim**:

```
```

### B3 raw control — expected wrong from the first token

Generated text, **verbatim**:

```
```

---

## Stage C — image pipeline

**Judge the FIRST WORD only.** Generation is expected to degenerate afterwards
until the decode defect is fixed.

| image | first word of caption | RELEVANT / WRONG / DEGENERATE |
|---|---|---|
| `sample_image` | | |
| `wx_clear` | | |
| `wx_clear2` | | |
| `wx_clear_snow` | | |
| `wx_fog_overcast_rain` | | |
| `wx_snow` | | |
| `wx_snow2` | | |

Any crash, and the exact message:

```
```

---

## Stage D — timing

| metric | cold | warm |
|---|---|---|
| init (ms) | | |
| TTFT (ms) | | |
| decode (tok/s) | | |
| C1 pipeline wall-clock (s) | | |

---

## Anything else

Stages skipped, commands changed, retries, surprises — anything that would make
a number mean something different than it looks:

```
```

**Observation vs interpretation:** please label which is which if you add
analysis. Both are welcome; conflating them costs a round-trip.
