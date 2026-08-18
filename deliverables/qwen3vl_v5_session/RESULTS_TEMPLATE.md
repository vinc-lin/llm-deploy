# v5 device session — results

**Board / date:** …
**Operator:** …
**Bundle md5s match `MANIFEST.md`:** yes / no (if no, say which differ)

---

## Test 1 — chunk-boundary triad, fp16-in 0.6B (the broken control)

| Prompt | Predicted | Observed | Generated text (verbatim, first ~200 chars) |
|---|---|---|---|
| 127 tok | garbled | | |
| 128 tok | **clean** | | |
| 129 tok | garbled | | |

**Verdict** (circle one):
- [ ] 127 garbled, 128 clean, 129 garbled → **theory confirmed**
- [ ] all three garbled
- [ ] all three clean
- [ ] other — describe:

Attach: `test1_fp16in.log`, `prof_fp16in_{127,128,129}.json`

---

## Test 2 — same triad, uFxp_16-in 0.6B (the fix)

| Prompt | Predicted | Observed | Generated text (verbatim) |
|---|---|---|---|
| 127 tok | clean | | |
| 128 tok | clean | | |
| 129 tok | clean | | |
| "What is 2+2?" | `4` | | |

**Verdict:**
- [ ] all coherent → **fix works**
- [ ] still garbled
- [ ] failed to load — paste the exact error and `adb logcat -d`:

Attach: `test2_u16in.log`, `prof_u16in_*.json`

---

## Test 3 — Qwen3-VL-4B with the fix

**3a text-only**

| Prompt | Generated text (verbatim) |
|---|---|
| "What is 2+2? Answer with one number." | |
| "Describe in three sentences why mountain weather changes quickly." | |

**3b image pipeline** (only if 3a was sensible)

| Image | Caption (verbatim) | Wall time |
|---|---|---|
| sample | | |

**Verdict:**
- [ ] text coherent AND caption coherent → **deployment works**
- [ ] text coherent, caption garbage/empty
- [ ] text still garbage → run Test 5

Attach: `v5_t2t_1.json`, `v5_t2t_2.json`, `v5_pipeline.log`

---

## Test 4 — timing

Report cold and warm **separately — do not average.**

| Run | Init (ms) | TTFT (ms) | Decode (tok/s) | Total (s) |
|---|---|---|---|---|
| cold | | | | |
| warm 1 | | | | |
| warm 2 | | | | |

| Pipeline run | Wall time (s) |
|---|---|
| 1 (cold) | |
| 2 | |
| 3 | |

Attach the three `v5_*.json` **verbatim** and `v5_timing.log`.

---

## Test 5 — qnn-net-run text probe (only if Test 3a was garbage)

- [ ] ran / [ ] skipped (Test 3a was fine)

Attach `text_probe.log` and the whole `text_probe_out/` directory.
If `qnn-net-run` rejected a flag, paste `./qnn-net-run --help` output here:

---

## Anything else

Crashes, tombstones, thermal throttling, anything that looked odd:
