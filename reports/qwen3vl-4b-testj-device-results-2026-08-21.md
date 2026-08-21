# Qwen3-VL-4B — Test J device results (decode step + cross-chunk prefill + image pipeline)

**Source:** device team, received 2026-08-21. §1–§6 are their report. §7 onward is
build-side annotation and is clearly marked.

**Board:** SA8797P (qc8797_poc, userdebug BP2A.250605.031.A3.240707fa).
Device clock reads 2026-08-05; session date is 2026-08-21.
**Bundle:** `qwen3vl_v5_session/03_vl4b_v5/`.

---

## 1. Preconditions — all pass

| check | required | observed |
|---|---|---|
| shard 0 md5 | `f031e3a7563bf16f2d5ca98a71b357f6` | ✅ match |
| shard 1 md5 | `0f1c86e89752b499eec09e9e10a73014` | ✅ match |
| `sample_image_fp32.raw` | 6,295,552 B | ✅ match |

The correct (post-`uFxp_16`) shard 0 was used, not the stale `065056ba…`.

## 2. Stage A1 — decode-step probe (`qnn-net-run`)

| case | prompt | step | expected argmax | chained | isolated | worst cos (chained) |
|---|---|---:|---:|---|---|---|
| `j0_2plus2_s1` | templated 2+2 | 1 | 151645 | **1/1 ✅** | 1/1 ✅ | 0.999893 |
| `j1_weather_s1` | templated weather | 1 | 9104 | **1/1 ✅** | 1/1 ✅ | 0.999900 |
| `j2_weather_s2` | templated weather | 2 | 4344 | **1/1 ✅** | 1/1 ✅ | 0.999801 |

Shard-0 boundary: worst `|gain−1| = 0.0061` across all rows.

## 3. Stage A2 — cross-chunk prefill (`qnn-net-run`)

| case | rows | cache in | s0 boundary | logits chained | logits isolated |
|---|---:|---:|---|---|---|
| `c0_chunk0` | 128 | 0 | clean (row 1 gain 0.9889, cos 1.000000) | 5/5 ✅ | 5/5 ✅ |
| `c1_chunk1` | 128 | 128 | **row 2 gain 0.9226, cos 0.960** | **3/5 ❌** | 5/5 ✅ |
| `c2_chunk2` | 21 | 256 | row 0 gain 0.9846, cos 0.990 | 5/5 ✅ | 5/5 ✅ |

## 4. Stage B — Genie text

`--max-num-tokens` is **not accepted** by this `genie-t2t-run` binary. Runs hit
the context limit and aborted, so no profile file was flushed.

| run | expected | actual |
|---|---|---|
| B1 templated 2+2 | `4` then stop | `[BEGIN]: 4entionablyanceablyanceablyanceanceanceanceabilityabilityabilityarded…` |
| B2 templated weather | `Mountain weather changes quickly because…` | `[BEGIN]: Mountainagedagedagedagedagedaged…` |
| B3 raw 2+2 (control) | garbage from token 0 | `[BEGIN]:  0÷÷÷÷÷÷÷÷÷÷÷÷÷÷÷÷÷÷ the equation. Assistant:1.00000000000…` |

First token correct on both templated runs; wrong from token 0 on the raw control.

## 5. Stage C — image pipeline (`genie-app`), seven images

| image | first tokens | team's verdict |
|---|---|---|
| `sample_image` | `A` + `0000…` + `uring` | DEGENERATE |
| `wx_clear` | `A` + `0000…` + `ishapur的` | DEGENERATE |
| `wx_clear2` | `A` + `0000…` + `ish…ature` | DEGENERATE |
| `wx_clear_snow` | `A` + `0000…` + `ishation` | DEGENERATE |
| `wx_fog_overcast_rain` | `A` + `0000…` + `ish…ight` | DEGENERATE |
| `wx_snow` | `A` + `0000…` + `ish…arded` | DEGENERATE |
| `wx_snow2` | `isher…arded` | DEGENERATE |

## 6. Stage D — timing

Skipped; the team judged timing on degenerate output not worth collecting.

---

# Annotations (build side, 2026-08-21)

## 7. What is settled

**Stage A1 lands exactly on the predicted row of the Test J decision table.**
The decode graph is correct end to end **including the recurrence** — `j2` fed a
cache with one row written by the decode graph itself and still hit 4344. Taken
with `parity_e2e_vl` at 20/20, `host_generate_check.py`, and Test G's
`r3_decodectx`, the decode **graphs** are now verified on host and on silicon.

**Stage B confirms it from the Genie side.** Prefill produces the right first
token; the very first decode call is wrong. **Stage B3 re-confirms Test I's root
cause** — raw prompts are wrong from token 0, templated ones are not.

⇒ **Defect #1 is confirmed and localised: Genie's decode-step feed.** Not the
graphs, not the encodings, not the prompt form.

## 8. Three corrections to the report's reasoning

### 8.1 "Error partially recovers in the final chunk" — no; c2 never saw the error

`build_chunk_suite` is a **sequential host replay**: each chunk's incoming cache
is the *host's* committed KV from the preceding chunks
(`scripts/validate/build_text_probe_kit.py:1118-1124`). So `c2` was fed a clean,
host-computed 256-position cache — never the device's degraded chunk-1 output.
Nothing recovered because nothing propagated.

The corollary matters more than the correction: **this probe cannot measure
propagation at all.** It measures each chunk against a correct past. Whatever
`c1` is, its effect on a real 3-chunk prefill is unmeasured.

### 8.2 Stage C's "all seven DEGENERATE" misapplies the rubric — and the rubric was mine

The protocol defines `DEGENERATE` as *"not a word — punctuation, a fragment, or
empty."* Six of seven first words are **`A`**, which is a word, and the single
most common caption opener in any captioning corpus. Only `wx_snow2` (`isher`)
meets the definition.

The honest verdict for the other six is **INCONCLUSIVE**, and that is a defect in
the test I wrote, not in the team's reporting. `A` carries no information about
the image, so a first-token test on an open-ended caption prompt **cannot
discriminate** — it was never going to, for any of the seven. Fix in §11.

### 8.3 "Cross-chunk prefill most likely caused the image first-token failure" — unsupported

Two independent reasons:

1. There is no demonstrated first-token failure (§8.2).
2. The degeneration *after* token 0 is **fully explained by defect #1**, which
   reproduces on the **20-token, single-chunk** templated 2+2 prompt where no
   cross-chunk path exists at all. Adding a second defect explains nothing the
   first does not.

## 9. On `c1` — real signal, premature label

The deviation is real. `|gain−1| = 0.077` exceeds the kit's own `GAIN_TOL = 0.05`
(`analyze_realistic_probe.py:45`), and cos 0.960 sits far below the 0.997–1.000000
every clean case has produced. `chained 3/5` against `isolated 5/5` does localise
it to shard 0, with shard 1 merely propagating.

But **"the cross-chunk KV cache path is broken" is one hypothesis, and this data
does not select it** over at least three others:

* **`c2` contradicts it.** A larger cache (256 vs 128) came back *cleaner*. A
  broken cache-read path should get worse with depth, not better.
* **The sampled rows are confounded.** The kit saves rows `0,1,2,3,n−1`. For `c0`
  those are absolute 0–3 — `<|im_start|>`, `user`, `\n`, `<|vision_start|>`, all
  **text**. For `c1` they are absolute 128–131 and 255 — **all image-feature**
  rows, whose activations span ±11.6 against text's ±0.24. The clean rows and the
  dirty rows differ in content as well as in cache depth. (`c0`'s row 127 is an
  image row and was clean — one data point, not a control.)
* **The reference is fp32, the device is fed a quantized cache.** The host
  computed `hid` with an fp32 past; the device gets the same cache written through
  the graph's cache encodings. The gain metric charges that difference to the
  device.
* **These logits are discarded in production.** Only the last row of the last
  chunk yields a token. What would matter is the *KV* chunk 1 writes — which §8.1
  says this probe does not measure.

⇒ Recorded as **open**, not as defect #2. It should not gate anything until a
probe that chains device KV says otherwise.

## 10. Two suspects eliminated today, device-free

The Test I annotations ranked three candidates for defect #1. **Two are now dead.**

### 10.1 ✝ The prefill→decode KV width handoff — dead

Ranked #1 in `reports/qwen3vl-4b-testi-device-results-2026-08-21.md`. It is wrong,
and the counter-example is our own shipping model.

| | prefill past | decode past | key layout |
|---|---|---|---|
| **0.6B `gqafix-ladekv`** (works, 44.707 tok/s) | 1024 | **1151** | `[1,8,128,PAST]` |
| **VL-4B shard 0** (decode broken) | 2048 | **2175** | `[1,8,128,PAST]` |

Same width change across the same handoff, same `PAST`-fastest key layout, same
libGenie 1.19 — and the 0.6B generates correct, coherent, multi-hundred-token
output in **basic** mode (`reports/qwen3-0.6b-w8a16-ladekv-test-report.md` §5:
token-exact against the non-speculative run for the first 60 tokens).

Genie demonstrably re-lays a `[1,NKV,D,PAST]` key cache across a width change
correctly on this runtime. Withdrawn.

### 10.2 ✝ MRoPE decode advance — dead for text-only

`nsp-model.cpp:3803` gates the Qwen3-VL branch on `m_visionParam.size() > 0`. A
text-only `genie-t2t-run` never sets a vision param, so it falls through to plain
rope (`freqs[i][j] = i * inv_freq[j]`). **Stage B was a plain-rope run.** MRoPE
cannot be what broke it.

Two related hazards checked while in that code, both clean:

* the rope table is filled for the whole `m_ctx_size`, not just the prompt — so
  "decode indexes past the filled region" is dead as well;
* `setVisionParam` sets `m_ropeInitialized = false` (`:4309`) when the param
  changes, so the cached table **is** rebuilt. The standing "MRoPE never engages"
  hazard does not apply to a changed vision param.

## 11. What is left, and the test that decides it — already built, already shipped

Two structural differences remain between the 0.6B that works and the 4B that
does not:

| | 0.6B (works) | VL-4B (decode broken) |
|---|---|---|
| embedding feed | in-graph from token ids | **external float32 LUT → `inputs_embeds`** |
| ctx-bins | **one** | **two** — shard 0 → shard 1 every decode step, `[1,1,2560]` |

`qwen3_06b_lutprobe` isolates the first: a 0.6B on the **shipping ladekv recipe**,
**one** ctx-bin, LUT-fed, one variable changed. Built 2026-08-18, uploaded, and
**never run** — 17 files already at
`vinccniv/sa8797p-qwen3-w8a16-bundles/qwen3_06b_lutprobe`.

| outcome | conclusion |
|---|---|
| tracks the working 0.6B token-for-token | LUT feed exonerated ⇒ **the split** is the fault |
| first token right, then degenerates | **the LUT feed** is the fault — and it now reproduces in a 1.4 GB bundle instead of 4.5 GB |

⚠ Sharpened pass criterion: the 0.6B has no chat template and *will* repeat
eventually under greedy decode — the control does too. The measurement is
**how many tokens match the working bundle**, run back to back in the same
session, not "is it coherent".

## 12. Protocol corrections (mine, not theirs)

* **`--max-num-tokens` is not a `genie-t2t-run` flag.** The team is right. It is a
  **dialog-config key**, `dialog.max-num-tokens`, whitelisted at
  `Dialog.cpp:2493` and read at `:2888`; the SDK's own example configs set it
  there. Now set to 64 in `configs/genie_dialog_qwen3vl_4b.json`. On device it is
  a one-line JSON edit, no rebuild. Cost of the error: no profile flushed, so no
  prompt-token count and no timing this session.
* **`--profile` refuses an existing output file** (`main.cpp:528-533`), so a
  re-run with the same path fails with `Invalid --profile argument`. Use a fresh
  name per run.
* **`-tok` / `--tokens_file` exists.** Feeding explicit token ids bypasses the
  tokenizer and settles the "did Genie split `<|im_start|>`" question by
  construction, rather than inferring it from a prompt-token count.
* **Stage C needs a prompt that forces a content word.** With an open-ended
  caption prompt the first token is `A` regardless of the image. Ask for one word
  — *"Answer with one word: what is the dominant weather in this photo?"* — so the
  first token is `Snow` / `Fog` / `Clear` and the test discriminates. Prompt-file
  change only.

## 13. Where this leaves end-to-end

| subsystem | status |
|---|---|
| image input (fp32 blob contract) | ✅ device-verified |
| image analysis (ViT W8A16) | ✅ device-verified |
| text prefill numerics | ✅ device-verified (Test G, I, and A1/A2 anchors) |
| prompt form | ✅ root cause found, confirmed twice |
| decode **graphs**, incl. recurrence | ✅ device-verified (A1, this session) |
| **Genie decode-step feed** | ❌ **the one confirmed defect** — now down to two candidates |
| cross-chunk prefill | ❓ one anomalous row, four candidate explanations (§9) |
| image → caption, end to end | ❓ untested by a discriminating prompt (§8.2) |

**One defect, two candidates, and the probe that separates them is already on the
Hub.** Nothing here needs a rebuild.
