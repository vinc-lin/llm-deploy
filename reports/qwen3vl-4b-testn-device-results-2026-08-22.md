# Test N device results — the final e2e diagnostic session

**Device session:** 2026-08-22. **Analysis:** 2026-08-22, build side.
§1–§5 summarise their report; §6 onward is annotation and is clearly marked.

Preconditions all passed: shard 0 `f031e3a7…`, shard 1 `0f1c86e8…`.

> **Headline:** N1a is the best result this project has produced — 6/6, on the
> real 4B, and it settles the shape of the defect. **N1c is a far bigger win
> than reported**: the KV dump is complete raw KV, not opaque, and it puts a
> direct decode-vs-prefill measurement within reach. **The combined verdict
> ("the split is the root cause") is not supported by these results** — it
> contradicts N1a.

---

## 1. N1a — token ladder: 6/6 ✅

| run | prompt | expected | got | |
|---|---:|---:|---:|---|
| `n1_2plus2_p20` | 20 | 19 | **19** | `4` |
| **`n1_2plus2_p21`** | 21 | 151645 | **EOS, empty output** | ✅ the decisive run |
| `n1_weather_p18` | 18 | 91169 | **91169** | `Mountain` |
| `n1_weather_p19` | 19 | 9104 | **9104** | ` weather` |
| `n1_weather_p20` | 20 | 4344 | **4344** | ` changes` |
| `n1_weather_p21` | 21 | 6157 | **6157** | ` quickly` |

Every profile's prompt-token count equalled its file's. All six degenerate after
token 1 (except P21) into the same `… 681 2897 804 …` loop.

## 2. N1b — `-e` embedding bypass ✅

First token `4`, then the identical degeneration. The prefill feed path is not
the defect.

## 3. N1c — `--save` state dump

`--save` writes a **directory**: `dialog.json`, `kv-cache.primary.qnn-htp`,
`sampler.primary.rng`, `unprocessed-data`.

| | P20 | P21 |
|---|---:|---:|
| kv-cache bytes | 3,097,168 | 3,244,624 |
| `n-past` | 21 | 22 |
| `last-tok` | 19 (`4`) | 151645 (EOS) |

Measured growth: **147,456 B/position**, delta exactly one position.

## 4. N2 / N3 — Tests L and M, carried forward

Reported from a prior session (⚠ build side has **not** received those two
reports; this is second-hand): **L** 5/5 pass, LUT feed correct at 0.6B.
**M** — the 2-shard 0.6B — short prompt degraded **from token 0**, long prompt
from token 2–3. Shard md5s quoted match the shipped build.

## 5. N4 / N5

Image grid: 5 RELEVANT, 1 INCONCLUSIVE, 0 WRONG. Timing: init 965–2116 ms
(warm-dependent), TTFT stable 511–512 ms, decode ~6.5 tok/s, pipeline
wall-clock 103.87 s.

---

# Annotations (build side, 2026-08-22)

## 6. N1a and N1b stand, and they are the strongest result yet

Six rungs, two unrelated prompts, four prompt lengths, exact token ids —
**prefill produces the correct first token every time on the real 4B**,
including the P21 case where the correct answer is immediate EOS. The `-e` run
adds that this holds with the tokenizer and the prefill LUT lookup both removed.

⇒ **The defect is decode-step machinery and is content-independent.** Not the
LUT range, not prompt length, not the token values. Accepted without reservation.

## 7. N1c — the KV cache is NOT opaque. The arithmetic used the wrong head count

The report calls the payload "~35× smaller than expected" and offers INT4
compression / a layer subset / an opaque handle. All three are unnecessary:

> `36 layers × 2 (K+V) × **28 heads** × 128 × 2 B = 516,096 B` uses the **query**
> head count. This is a **GQA** model — the cache is sized by the **8 KV heads**.
>
> `36 × 2 × **8** × 128 × 2 =` **`147,456 B/position`** — **exactly the measured
> value**, to the byte.

Read back from the shipped bin to be sure: `past_key_0_in [1,8,128,2048]`,
18 K-tensors per shard, 36 total. The dump is the **full, raw, uncompressed KV
cache** for the committed positions.

### The layout follows from the two file sizes, exactly

```
592 B header (16 B file header + 72 tensor descriptors × 8 B)
then 72 tensors (36 layers × {K,V}), each n_pos × 2048 B
2048 B = n_kv(8) × head_dim(128) × 2 B

  P20: 592 + 72×21×2048 = 3,097,168  == observed
  P21: 592 + 72×22×2048 = 3,244,624  == observed
```

Both to the byte, and the header the team transcribed corroborates it —
`0x48` = 72 tensors at offset 0, then `0xc0de`, then `uint16` fields
`{_, 8, 128, n_past}` = `{_, n_kv, head_dim, 21 or 22}`. Their own observation
that byte 14 increments 0x15→0x16 with `n-past` is that last field.

### What this unlocks — available now, no device time

P21's prompt is P20's plus the token `4` (id 19). Therefore:

* in **P20**, position 20 holds the KV for `4` **written by the DECODE graph**;
* in **P21**, position 20 holds the KV for `4` **written by the PREFILL graph**.

Same token, same position, same model, the two different paths. Positions 0–19
are prefill-written in both and must be byte-identical.

**Diffing those two files is a direct measurement of the decode step's KV
write** — the observation every probe so far has had to reconstruct.
`scripts/validate/parse_genie_kv_dump.py --diff` does it, and was validated
against synthesized dumps that reproduce both reported sizes exactly.

**⇒ Please send `state_p20.bin/kv-cache.primary.qnn-htp` and
`state_p21.bin/kv-cache.primary.qnn-htp` (≈3 MB each).** They are the highest-value
artefacts of the session and `collect_n.sh` deliberately excludes them.

## 8. ⛔ The combined verdict is not supported — it contradicts N1a

The report concludes "the 2-shard split is the root cause", from
N1a-all-correct × M-broken. That inference does not hold:

**Prefill traverses both shards too.** The 4B's prefill path is
`prefill_0 → last_hidden_states → prefill_1`; the boundary tensor crosses the
same ctx-bin split that decode uses. **N1a proves that handoff is correct on the
4B at six different prompt lengths.** If the split were broken, N1a would have
failed.

Meanwhile Test M's 0.6B split fails **at token 0** — which is prefill. So:

| | 4B split | 0.6B split (Test M) |
|---|---|---|
| prefill through both shards | ✅ correct (N1a, 6/6) | ❌ wrong from token 0 |
| decode | ❌ the open defect | ❌ wrong |

**The two failures are not the same failure**, and Test M did not reproduce the
4B's signature. The 4B's signature is *first token right, then degradation*;
Test M's is *wrong from the start*.

⚠ `TEST_M_split_reproduction.md` §2 assigns that outcome explicitly:

> **wrong from token 0** → *a prefill/feed problem, **not the split*** → check
> the shard-0 md5 first; then compare against Test L on the same board.

That is the row the session landed on. It was read as the row above it.

**This does not exonerate the split for the 4B.** Decode uses a different graph
pair (`decode_0/decode_1`) and a different boundary shape (`[1,1,2560]` vs
`[1,128,2560]`), so a decode-only handoff bug remains entirely viable — and is
still the leading hypothesis. It simply is not *established*, and Test M as run
is not the evidence for it.

## 9. Test M has a four-knob confound

`qwen3_06b_lutsplit`'s dialog config inherited the single-bin LUT probe's engine
block — deliberately, so Test M would read against Test L. But that block
differs from the 4B's in four places:

| `QnnHtp` key | 0.6B lutsplit | the real 4B |
|---|---|---|
| `enable-graph-switching` | **true** | *absent* |
| `allow-async-init` | **true** | false |
| `poll` | **true** | false |
| `mmap-budget` | **25** | 0 |

`enable-graph-switching: true` on a **two-ctx-bin** model is the standout: it is
exactly the machinery that routes execution between bins. Test M's failure may
be that knob rather than the split.

Recall also that the lutsplit's **prefill chain passes 3/3 on the host**
(220 / 12095 / 220, the unsplit answers) — so its token-0 failure on device is a
host-vs-device divergence in a path we gated, which is the signature of a
runtime/config problem rather than a bad build.

**Next step, config-only, no rebuild:** re-run Test M with the 4B's `QnnHtp`
block. If the 0.6B split then produces *first token right, then garbage* — the
4B's signature — that is the reproduction we wanted. If it produces correct
output, Test M's failure was the knobs and the split is clean at 0.6B.

## 10. N4 — two scoring notes

* **`wx_clear2` and `wx_clear_snow` produced the same output (`Ser` + garbage)
  and were scored differently** — RELEVANT and INCONCLUSIVE. Defensible on the
  scenes (for a clear sky `Sunny` is right; for clear-sky-plus-snow the host
  said `cold`), but it should be stated as a judgement about the *scene*, not
  read as two different model behaviours.
* **No token ids were reported for N4**, so `Ser` cannot be resolved between
  `S`(50)+`er` — the correct first token of `Sunny` — and the single token
  `Ser`(31745), which would be wrong. Five of seven are unambiguous either way,
  so the conclusion holds; the grid would just be firmer with ids.

## 11. N5 — the wall-clock is not comparable to Test K3's

103.87 s vs K3's "8–12 s" is not a regression: K3's figure was **first-token
latency** and this one is a **full run to the context limit**, inflated by the
decode defect generating ~512 tokens of garbage. The report's own decomposition
(ViT + ~7 s prefill + 0.5 s TTFT ≈ 8–10 s to first token) is the like-for-like
number and agrees with K3.

The cold/warm init question K3 raised is now answered differently than expected:
init is **monotonically warm-dependent across the session** (2116 → 1440 → 1378
→ 965 ms), so "cold vs warm" is not a two-state variable and K3's warm > cold
anomaly was most likely ordering, not a defect. Worth closing as such.

---

## 12. Status after Test N

| subsystem | status |
|---|---|
| image input · ViT · splice · prefill → first token | ✅ device-verified, 7/7 images |
| text prefill on the real 4B, 4 lengths × 2 prompts | ✅ **device-verified 6/6, incl. EOS** |
| prefill feed path (tokenizer, prefill LUT lookup) | ✅ exonerated on the 4B (N1b) |
| decode graphs incl. recurrence | ✅ device-verified (Test J A1) |
| **Genie's decode-step machinery** | ❌ **the defect** — content-independent |
| the two-ctx-bin split as its cause | ❓ **leading hypothesis, not established** (§8) |
| Test M's 0.6B split failure | ❓ confounded by 4 engine knobs (§9) |
| 4B timing | ✅ complete |

**Next, in order:**

1. **Send the two KV dumps** — `parse_genie_kv_dump.py --diff` gives a direct
   decode-vs-prefill KV measurement with no further device time (§7).
2. **Re-run Test M with the 4B's `QnnHtp` block** — config-only (§9).
3. Forward the L and M reports to the build side; they were analysed here
   second-hand.
