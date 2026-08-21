# Test L session — results

**Date:** ____________  **Board / build:** ____________  **Operator:** ________

Procedure: `TEST_L_ctxbin_vs_genie.md`. Fill this in as you go.

---

## 0. Preconditions — fill these first

| check | required | what you saw |
|---|---|---|
| `qwen3-0.6b-w8a16-lutprobe-ladekv_ctx.bin` md5 | **`9720e46e62f59d08f56301418cccc8c1`** | |
| (if it reads `880a6abdec4a64b67b275ec817c054ca` you have the **old** Test K bundle) | — | |
| `embedding_float32_lut.bin` md5 | `2836420ce26c84f4ab7b217a22b2e125` | |
| config has `"max-num-tokens": 64` | yes | |
| `past_kv.tar.gz` md5 check passed, tarball extracted | yes | |

**A failed precondition is a result.** Report it and stop.

---

## Stage L0 — Genie on the corrected bundle

### L0 short — `What is 2+2? Answer with one number.`

```
[paste l0_short.txt verbatim]


```

### L0 long — `Describe, in three or four sentences, what makes mountain weather change quickly.`

```
[paste l0_long.txt verbatim -- first 200 characters is fine, say if the rest repeats]


```

### Score it against the reference, not against intuition

The correct output **repeats**. That is what HF fp32 does on a raw prompt.

| | reference (HF fp32, greedy) | what you got |
|---|---|---|
| short | `' 2+2=4. 2+2=4. 2+2=4. …'` (first ids `[220,17,10,17,28,19]`) | |
| long | `' Also, explain why it is important to have a good understanding of the weather…'` | |
| does the output start with `</think>`? | **no** | |
| is the long answer about **weather** (not New York)? | **yes** | |

**Verdict — tick one:**

- [ ] **matches the reference** ⇒ the FLOAT_16 bin was the whole story; **Genie's LUT feed is sound**; the split is again the sole suspect for the 4B
- [ ] **still `</think>`-prefixed / still off-topic** ⇒ the fp16 bin was not the whole story; Genie's LUT feed really is implicated
- [ ] **something else** (describe below)

---

## Stage L1 / L2 — the same bin under `qnn-net-run`

Paste the analyzer's output:

```
[paste the output of analyze_lutprobe_kit.py]


```

Or fill the table by hand:

| case | graph | expect | **got** | match? |
|---|---|---:|---|---|
| `l1a_2plus2` | prefill | **220** | | |
| `l1b_paris` | prefill | **12095** | | |
| `l1c_boils` | prefill | **220** | | |
| `l2a_decode_s1` | decode | **17** | | |
| `l2b_decode_s2` | decode | **10** | | |

**Verdict — tick one:**

- [ ] **all five match** ⇒ the ctx-bin is faithful to its ONNX, which is gated 3/3 against HF. Combined with L0 this pins the remaining fault precisely
- [ ] **a prefill case mismatches** ⇒ the bin does not reproduce its own ONNX ⇒ **converter defect**
- [ ] **`l2a` matches, `l2b` does not** ⇒ reading back a decode-written KV row is broken
- [ ] **did not run** — say why (missing KV files? runner stopped? paste the message)

---

## Anything else

Skipped stages, changed paths, retries, commands that failed. Observation and
interpretation both welcome — **label which is which**.

```


```
