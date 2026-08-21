# Test N session — results

**Date:** ____________  **Board / build:** ____________  **Operator:** ________

Procedure: `DEVICE_SESSION_PROTOCOL_N.md`. Fill in as you go, not afterwards.

---

## 0. Preconditions (fill these first)

| check | required | what you saw |
|---|---|---|
| `qwen3vl-4b-w8a16_1_of_2.bin` md5 | `f031e3a7563bf16f2d5ca98a71b357f6` | |
| `qwen3vl-4b-w8a16_2_of_2.bin` md5 | `0f1c86e89752b499eec09e9e10a73014` | |
| 4B config has `"max-num-tokens": 64` | yes | |
| `testn/n1_2plus2_p21.tok` md5 | `6f6050fd27555c7bb14c1e815aa1972f` | |
| fingerprint / date | — | |

A failed check here **is** a result — report it and stop.

---

## N1a — token ladder

| run | prompt tokens (profile) | expect first | **first generated (verbatim)** | match? |
|---|---:|---:|---|---|
| `n1_2plus2_p20` | expect 20: ___ | 19 `4` | | |
| **`n1_2plus2_p21`** | expect 21: ___ | **151645 — EOS immediately, empty output = PASS** | | |
| `n1_weather_p18` | expect 18: ___ | 91169 `Mountain` | | |
| `n1_weather_p19` | expect 19: ___ | 9104 ` weather` | | |
| `n1_weather_p20` | expect 20: ___ | 4344 ` changes` | | |
| `n1_weather_p21` | expect 21: ___ | 6157 ` quickly` | | |

Degeneration *after* the first token is expected everywhere except P21 — judge
the first token, paste everything.

```
[paste the six transcripts verbatim -- first 200 chars each is fine, say if the rest repeats]


```

## N1b — `-e` embedding run

```
[paste n1b_emb.txt verbatim -- including any error message]


```

Same first token and same garbage pattern as the text P20 run?  YES / NO / ERROR

## N1c — state dump

| | size (bytes) | notes |
|---|---:|---|
| `state_p20.bin` | | |
| `state_p21.bin` | | |
| any error from `--save`? | | |

Reference sizes: ≈302 MB → prefill-width KV; ≈321 MB → decode-width KV;
a few KB → metadata only. Both files (or `xxd -l 64` of each) sent back? ____

Config restored to `"max-num-tokens": 64`?  YES / NO

---

## N2 — Test L (0.6B LUT probe)

lutprobe bin md5 (`9720e46e…` required): ______________________

L0 verbatim + the L1/L2 analyzer output, pasted whole:

```


```

Verdict per `TEST_L_ctxbin_vs_genie.md`: ______________________________________

## N3 — Test M (2-shard 0.6B)

Shard md5s (`1f4dcd44…` / `11cabce4…`): ______________ / ______________

```
[m_short.txt and m_long.txt verbatim]


```

Reference: short → `' 2+2=4. 2+2=4. …'` (repetition IS correct); long →
`' Also, explain why it is important…'`. Matches?  YES / NO / partially: ______

---

## N4 — image first-word grid

`wc -c prompt_seg2.txt` before sample run (expect 116): ___ · before photo loop
(expect 103): ___ · restored to 83 after: ___

| image | first word (verbatim) | RELEVANT / WRONG / DEGENERATE / INCONCLUSIVE | matches picture? |
|---|---|---|---|
| `sample_image` (red circle + blue square) | | | |
| `wx_clear` | | | |
| `wx_clear2` | | | |
| `wx_clear_snow` | | | |
| `wx_fog_overcast_rain` | | | |
| `wx_snow` | | | |
| `wx_snow2` | | | |

`S` + fragment on the clear photos = **correct first token** (`Sunny` is two
tokens); `A`/`The`/`This` = INCONCLUSIVE, not RELEVANT.

## N5 — timing

| metric | cold (fresh boot) | warm (immediate re-run) |
|---|---:|---:|
| init (ms) | | |
| TTFT (ms) | | |
| decode (tok/s) | | |
| pipeline wall-clock (s) | | |

Never average cold and warm. Which run was which: __________________________

---

## Anything else

Skipped stages, changed paths, retries, failures — and label observation vs
interpretation.

```


```
