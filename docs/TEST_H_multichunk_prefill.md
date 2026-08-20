# Test H — the cross-chunk prefill path

**Status:** ready to run · **Opened:** 2026-08-20 · **Needs:** no rebuild, no new
ctx-bins, ~2 minutes of device time.
Audience: device team + build side. Self-contained; no prior thread needed.

---

## 1. Why this is the path that matters now

Test G closed the boundary question: on realistic chat-templated prompts the
device boundary is clean (worst |gain−1| **4.4%**, the real attention sink at
0.989 / cos 1.000000, all logits argmax matching). **Shard 0 is exonerated.**

But every probe so far — Tests B, C, E, F **and G** — has run a **single chunk
against an empty cache**. A real prompt does not work that way. At AR = 128, a
273-token turn is *three* prefill calls, and qualla must carry KV forward
between them, so chunks 2 and 3 run against a **partially populated cache**.

That path has never executed on device under any probe. It is also precisely
where the FLOAT_16 padding bug lived:

> `quantizeInput` advances its destination by `tensorOffset` **elements** for
> `UFIXED_8/16` and `FLOAT_32` but by **bytes** for `FLOAT_16`
> (`nsp-model.cpp:3144`), while `setupInputEmbeddings` passes an element count
> when padding a partially-filled prefill chunk (`:1813`). It fires **only when
> `variant > n_process`** — the last, partial chunk.

The dtype was fixed and confirmed at 0.6B (the 127/128/129 triad). The
**chunking path itself** was never independently verified at 4B.

---

## 2. What Test H runs

One real image turn, walked end to end, instrumenting every chunk:

| case | rows | cache going in | why |
|---|---:|---:|---|
| `c0_chunk0` | 128 | **0** (empty) | anchor — must reproduce Test G's `r2_chunk0` |
| `c1_chunk1` | 128 | **128** | the first cross-chunk call, never before run on device |
| `c2_chunk2` | **21** | **256** | the **last and partial** chunk: 21 real rows in a 128-wide window — the exact `variant > n_process` condition |

`c2_chunk2` is the case this test exists for.

The feed is `parity_e2e_vl.PrefillKV`, not a re-derivation: row *i* sees the
valid past span `[0, nv)` plus the causal new span `[PAST, PAST+i]`; rows past
*n* stay fully masked and their KV is never committed.

**The anchor is a real check, not decoration.** `c0_chunk0` was replayed through
the same code as the other two and came out with argmax `3175` and row RMS
`2.072 / 220.330 / 1.117 / 1.428 / 0.839` — identical to Test G's `r2_chunk0`. If
the device disagrees between those two, the replay is not the computation Test G
measured and nothing below can be read.

**Nothing is rebuilt.** Same ctx-bins already on your device from v5:

| file | bytes | md5 |
|---|---:|---|
| `qwen3vl-4b-w8a16_1_of_2.bin` | 1850793984 | `f031e3a7563bf16f2d5ca98a71b357f6` |
| `qwen3vl-4b-w8a16_2_of_2.bin` | 2631094272 | `0f1c86e89752b499eec09e9e10a73014` |

---

## 3. The host prediction

Every reference row, with all 180 calibrated activation ranges clamped:

| case | row 0 | row 1 | row 2 | row 3 | last row |
|---|---:|---:|---:|---:|---:|
| `c0_chunk0` | 1.0001 | 1.0002 | 1.0002 | 1.0002 | 1.0000 |
| `c1_chunk1` | 1.0001 | 0.9992 | 1.0000 | 1.0000 | 0.9996 |
| `c2_chunk2` | 1.0000 | 1.0001 | 1.0001 | 1.0002 | 1.0000 |

**Prediction: every row clean, worst deviation 0.0008.**

⚠ **Read that as a lower bound.** The simulation clamps to the calibrated ranges
but does **not** round to the quantization grid, so it models clipping and omits
rounding noise. On Test G it predicted 3.5% and the device measured 4.4%.
**Expect the device within ±5%, not within 0.1%.** A result in the 1–5% band is a
PASS; the thing that would be a finding is a row well outside it, and especially
one that only appears on `c1`/`c2` and not on the `c0` anchor.

---

## 4. Running it

The kit merges into the v5 folder you already have.

```sh
cat testh/past_kv.tar.gz.part-* > testh/past_kv.tar.gz   # REQUIRED: reassemble
md5sum -c testh/past_kv.tar.gz.md5                      # verify before trusting it
tar xzf testh/past_kv.tar.gz -C testh/                  # 53 MB -> 594 MB of KV
adb push testh/. /data/local/tmp/v5/
adb shell
cd /data/local/tmp/v5 && export LD_LIBRARY_PATH=. && chmod +x qnn-net-run
sh run_text_probe.sh 2>&1 | tee text_probe_h.log
exit
adb pull /data/local/tmp/v5/text_probe_out ./text_probe_out_h
```

None of those three lines is optional — `c1` and `c2` feed real caches, and the
runner size-checks every file and stops with this hint if they are missing. `c0`
needs none (its cache is all zeros, so it uses the runner's shared zero file).

**Why it ships in parts.** Measured against this proxy on 2026-08-20: the 231
kit files commit in 105 s, a single 53 MB stream fails every time (520 s, 290 s,
224 s), and even 8 MB parts fail. So the tarball travels at the size the rest of
the kit already proves works — 2 MB parts. `md5sum -c` catches a bad
concatenation; without it a truncated cache would be fed to the graph and read
as a device defect.

Expect ~2 minutes, a push of roughly **700 MB** and a pull of roughly **150 MB**.

Host side:

```bash
$PY_DEPLOY scripts/validate/analyze_realistic_probe.py \
    --kit host_refs --results ./text_probe_out_h
```

---

## 5. Reading the result

| outcome | meaning |
|---|---|
| **all three clean** | the cross-chunk path is sound. Prefill is then fully exonerated end to end — empty cache, cross-chunk, and partial last chunk — and the fault is in Genie's orchestration above the graphs, or in the dialog/config, or in the LUT *lookup* (not its precision, which Test G already covered) |
| **`c0` clean, `c1`/`c2` off** | **the cross-chunk cache path is the defect.** The anchor proves the replay is right, so the fault is in how a populated cache is fed or read |
| **`c2` off, `c0`/`c1` clean** | the **partial last chunk** specifically — the `variant > n_process` condition. The FLOAT_16 padding bug's exact signature, in a build where the dtype is supposed to be fixed |
| **`c0` off** | stop. The anchor disagrees with Test G's `r2_chunk0` on the same bins, so the device is not in the state Test G measured and nothing else can be read |

The third row is the one to watch. It would mean the `uFxp_16` graft fixed the
dtype but something else on the padding path is still wrong at 4B.

---

## 6. What to send back

1. The analyzer's full output, or `text_probe_out_h/` (~150 MB).
2. `text_probe_h.log`, including the `shard0 out:` lines.
3. The md5s of the two ctx-bins you ran (§2).

---

## 7. What Test H still does not cover

It walks one turn of **three** chunks with the cache reaching 256 of 2048 slots.
It does not cover a long generation, where decode runs hundreds of steps and the
cache grows far past that. `r3_decodectx` in Test G covered a *single* decode
step on a 13-position cache; a multi-step decode with a growing cache is the
next thing after this, and the machinery now exists for it.

It also says nothing about output quality — that stays unmeasured until the
tower produces coherent text on device.
