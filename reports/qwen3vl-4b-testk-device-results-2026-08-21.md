# Test K device results — LUT vs split, image first tokens, 4B timing

**Device session:** 2026-08-21 (device team). **Analysis:** 2026-08-22, build side.
§1–§4 are their report. §5 onward is annotation and is clearly marked.

> **Both headline conclusions change under host reference.** K1's verdict inverts:
> the probe is the deviant, not the control, and the LUT feed is **not**
> exonerated. K2's result improves sharply: the image path is right on **6–7 of 7**
> first tokens, not 3.

> ## ⛔ §5 IS VOID — corrected 2026-08-22
>
> **K1 measured our own packaging bug, not Genie.** The bundle published for
> Test K shipped the **pre-graft ctx-bin**, whose `inputs_embeds` is
> `QNN_DATATYPE_FLOAT_16`. It fails `lint_embedding_dtype.py`. The corrected
> build (`9720e46e…`, `UFIXED_POINT_16`) passes.
>
> FLOAT_16 is the dtype for which `quantizeInput` advances by `tensorOffset`
> **bytes** while `setupInputEmbeddings` passes **elements**
> (`nsp-model.cpp:3144` vs `:1813`), so the pad write for a partially-filled
> prefill chunk **overwrites the back half of the real prompt**. It fires
> whenever `variant > n_process` — and the probe's prompts are 12/5/6 tokens in
> an **AR=128** graph, so it fired on every one. That is why the front of `2+2`
> survived, the long prompt lost its topic, and decode (AR=1, `variant ==
> n_process`) was untouched.
>
> ⇒ **Genie's LUT feed is neither exonerated nor implicated** — it was never
> tested. §5.7's "unified hypothesis" is withdrawn. §5.1–5.6's *measurements*
> stand (the control does match HF; the probe did diverge at token 0); only the
> attribution was wrong. §5.4's line "fp16 `inputs_embeds` — ✝ eliminated" is the
> specific error: it was read from `work/ctxbin/…/info.json`, the **rebuilt** bin,
> while the device ran the **staged** copy.
>
> **§6 (image path 7/7), §7 (timing) and §5.5 (correction #38) are unaffected.**
> Re-run as **Test L** (`docs/TEST_L_ctxbin_vs_genie.md`). `REFERENCE.md`
> correction #39.

---

## 1. Preconditions — all pass

lutprobe ctx-bin `880a6abd…`, LUT `2836420c…`, `max-num-tokens: 64` present,
v5 shard 0 `f031e3a7…`, K2 prompt files 116 B / 103 B, original restored. ✅

## 2. Stage K1 — probe vs control (device)

| run | output |
|---|---|
| probe, short | `[BEGIN]:\n</think>\n\n2 + 2 = 4.[END]` |
| control, short | `[BEGIN]:  2+2=4. 2+2=4. 2+2=4. …[END]` |
| probe, long | `[BEGIN]:\n</think>\n\nThe city of New York is a dynamic hub of culture, innovation, and global influence…` |
| control, long | `[BEGIN]:  Also, explain why it is important to be aware of the weather in the future. What are the main factors…` |

**Team's verdict:** probe is clean and terminates, control repeats ⇒ the LUT feed
is sound ⇒ the split is the fault.

## 3. Stage K2 — image, one-word prompt

`sample_image` → `Red`; `wx_clear` → `S`+`000…`; `wx_clear2` → `Sally…`;
`wx_clear_snow` → `Ser…`; `wx_fog_overcast_rain` → `Rain`; `wx_snow` → `Snow`;
`wx_snow2` → `Snow`+`er…`. **Team's verdict:** 3/7 relevant; the 3 failures
attributed to the cross-chunk prefill defect.

## 4. Stage K3 — timing

4B two-shard W8A16, 18-token text prompt: **init 1375 ms cold / 1725 ms warm,
TTFT 510 ms, prefill 35.3 tok/s, decode 6.5 tok/s.**
0.6B reference: probe 43.2–51.3 tok/s decode, control 42.9–43.3.
Image pipeline wall-clock 8–12 s per image (untimed, qualitative).

---

# Annotations (build side, 2026-08-22)

## 5. K1 — the verdict inverts

The report reads probe-vs-control without a reference, and concludes the probe
looks healthier. With a reference, it is the probe that is wrong.

### 5.1 Ground truth: the CONTROL is correct, token for token

HF Qwen3-0.6B fp32, same **raw** prompts, greedy, on this box:

| prompt | HF fp32 reference | control (device) |
|---|---|---|
| short | ` 2+2=4. 2+2=4. 2+2=4. …` — first ids `[220, 17, 10, 17, 28, 19]` | ` 2+2=4. 2+2=4. …` ✅ **match** |
| long | ` Also, explain why it is important to have a good understanding of the weather…` | ` Also, explain why it is important to be aware of the weather in the future…` ✅ **match** |

**The control's repetition is the correct behaviour of this model on a raw
prompt** — HF fp32 does exactly the same thing. It is not a deficiency, and the
probe's clean terminated answer is not a virtue. The report's central comparison
is inverted.

The probe diverges at **token 0** on both prompts: it emits `</think>` (151668)
where the reference emits `Ġ` (220). On the long prompt it then writes 64 fluent
tokens about **New York** in answer to a question about **mountain weather** —
coherent, and unconditioned on the prompt.

### 5.2 The report's stated reason for dismissing the comparison is false

> *"different tokenizer states (the probe has a `</think>` starter)… presumably
> different calibration data… cannot be compared token-by-token"*

Both bundles ship **byte-identical** `tokenizer.json`
(`6423133b9cc1a2077b57822c30c211aa`). The `</think>` cannot come from
tokenization. The probe is the shipping ladekv recipe with one flag changed;
these two models are comparable by construction, which was the entire point of
building it.

### 5.3 The probe's graph and LUT are correct — verified on the host

`scripts/validate/parity_lutprobe.py`, run against the shipped probe's own
past-KV prefill export and the same LUT the runtime reads, at the runtime's byte
offsets:

```
OK  argmax graph=  220 hf=  220  ' '      <- 'What is 2+2? Answer with one number.'
OK  argmax graph=12095 hf=12095  ' Paris' <- 'The capital of France is'
OK  argmax graph=  220 hf=  220  ' '      <- 'Water boils at a temperature of'
last-token argmax agreement: 3/3 = 100%
PASS
```

That gate exists precisely for this moment. Its own docstring: *"Passing here
means the graph and the LUT are correct together on the host. Only then does a
device failure implicate the runtime."*

**The graph says 220. The device says `</think>`. Same bin, same prompt, same LUT.**

### 5.4 The other benign explanations are eliminated

| candidate | checked | result |
|---|---|---|
| fp16 `inputs_embeds` (the 4B's old defect) | `info.json` | ✝ probe is `UFIXED_POINT_16`, grafted — same as the working 4B |
| `graph_names` mismatch → backend defaults | ext config vs bin | ✝ both graphs listed and covered; `vtcmSize 16`, `spillFillBufferSize 0` read back from the bin. Identical ext config to the control |
| wrong `inputs_embeds` tensor name | export | ✝ literal `inputs_embeds`, which is what selects `InputType::EMBEDDINGS` |
| different tokenizer | md5 | ✝ byte-identical |
| quantization noise | control | ✝ the control is also W8A16 and reproduces HF exactly |

### 5.5 ⚠ My error: the control they ran is not the shape-matched one

`TEST_K_lut_vs_split.md` named `qwen3_06b_w8a16_gqafix_ladekv` as the control.
That is the **CTX-1152** build. The probe is **CTX-640**:

| | prefill mask / past | decode mask / past | first input |
|---|---|---|---|
| probe `lutprobe-ladekv` | `[1,128,640]` / 512 | `[1,1,640]` / 639 | **`inputs_embeds`** |
| control they ran `gqafix-ladekv` | `[1,128,1152]` / 1024 | `[1,1,1152]` / 1151 | `input_ids` |
| **shape-matched `gqafix-cl512-ladekv`** | `[1,128,640]` / 512 | `[1,1,640]` / 639 | `input_ids` |

`gqafix-cl512-ladekv` is identical to the probe in every shape, with `input_ids`
instead of `inputs_embeds` as the only difference — the true one-variable
control. It is on the Hub at
`2026-08-16-regime/bundles/qwen3_06b_w8a16_gqafix_cl512_ladekv.tar.gz`.

This also explains a K3 inference the report drew: the probe's ~2× prefill rate
over the control is **not** "LUT-fed prefill skips the embedding lookup" (a
Gather is negligible). The control was attending over a **1152-wide** mask
against the probe's **640** — roughly twice the attention work.

The confound does not affect the text comparison (mask width does not change
greedy output), so §5.1–5.3 stand.

### 5.6 What K1 actually established

**Two things, and they point opposite ways.**

1. **The probe's decode is healthy.** 64 fluent, grammatical tokens; clean EOS on
   the short prompt. It does *not* reproduce the 4B's `entionably…` collapse. On
   the narrow question the probe was built for — *does LUT feeding break
   decode?* — the answer is no.
2. **The probe's prefill is wrong on device**, from token 0, with graph and LUT
   proven correct on the host. That is new, and it implicates **Genie's LUT feed**.

### 5.7 The unified hypothesis — and why the 4B does not contradict it

The obvious objection: the 4B's prefill is correct on device, so how can Genie's
LUT feed be broken?

Because the 4B's prefill was verified two different ways, and only the weaker one
involves Genie:

| evidence | path | strength |
|---|---|---|
| `i1_templated` 5/5 chained argmax | **`qnn-net-run`, host-supplied files** — bypasses Genie's LUT entirely | strong, but says nothing about the LUT feed |
| Genie emits `4` / `Mountain` as the first token | Genie's LUT feed | **one token**, and a robust one |

So the 4B's Genie-level prefill evidence is a single token that a partially
corrupted context would still get right. That admits one hypothesis covering both
models:

> **Genie's LUT feed delivers a degraded `inputs_embeds`.** The 4B keeps enough
> of the prompt to get a robust first token and then collapses in decode; the
> 0.6B probe keeps enough of `2+2` to say `2 + 2 = 4` after a wrong opener, and
> loses the long prompt entirely.

⚠ **This is a hypothesis, not a finding.** It is consistent with everything
measured, but the probe's failure signature (wrong token 0, healthy decode) is
not the 4B's (right token 0, broken decode), and that difference is unexplained.

### 5.8 The test that decides it — small, and device-side only

Run the **probe's own ctx-bin under `qnn-net-run`** with host-built
`inputs_embeds` for the same 12-token prompt, and read the argmax.

| result | meaning |
|---|---|
| **220** | the ctx-bin is fine and only Genie's feed differs ⇒ **Genie's LUT feed is the defect**, root cause found, reproducible in a 1.4 GB bundle |
| **`</think>` (151668)** | the ctx-bin conversion diverges from its own ONNX ⇒ a converter defect, and a different investigation |

`build_text_probe_kit.py` already builds exactly this shape; only the tokens and
the target bin change.

---

## 6. K2 — a much better result than reported

The report scored 3/7 by reading rendered strings. Scored on **first tokens**
against a host reference, it is 6–7 of 7.

Host reference (HF Qwen3-VL-4B fp32, greedy, same one-word prompts, images at
512×512 → 256 image tokens, `scripts/validate/host_first_token_vl.py`). Prompt
lengths came back **281 / 279 tokens**, exactly as predicted when the segments
were generated — so the shipped prompt files are correct:

| image | host reference | device token 0 | reading |
|---|---|---|---|
| `sample_image` | `blue` (12203) | `Red` (6033) | **both defensible — see below** |
| `wx_clear` | `sunny` = `['s','unny']` | `S` (50) + garbage | ✅ `Sunny` = `['S','unny']` — **token 0 correct** |
| `wx_clear2` | `sunny` | `Sally` = `['S','ally']` | ✅ **token 0 correct**, token 1 wrong = the decode defect, caught red-handed |
| `wx_clear_snow` | `cold` (87072) | `Ser…` | ambiguous — clear sky *and* snow; both answers defensible |
| `wx_fog_overcast_rain` | `rainy` = `['rain','y']` | `Rain` (59039) | ✅ |
| `wx_snow` | `snowy` = `['snow','y']` | `Snow` (62285) | ✅ |
| `wx_snow2` | `snowy` | `Snow` + `er…` | ✅ **token 0 correct** |

### 6.1 Why three images "failed" — tokenization, not prefill

`Snow`, `Rain` and `Red` are **single tokens**. `Sunny` is **two**: `['S','unny']`.

* a one-token answer is complete before decode runs — it survives;
* a two-token answer needs **decode step 1**, which is the broken path — so it
  comes out `S` + garbage.

Every image whose correct answer fits in one token was answered correctly. Every
image whose answer needed two shows **the correct first token followed by the
known decode defect**. There is no image where the vision path failed.

### 6.2 ⇒ the cross-chunk attribution is wrong

The report assigns the three failures to Test J's `c1` cross-chunk anomaly. That
cannot be right: **all seven prompts are 279–281 tokens — three AR=128 chunks
each.** The successes and the failures went through identical chunking. The
splitting variable is answer *tokenization*, not chunk count. `c1` remains
unproven and is not implicated here.

### 6.3 The sample image: the device may be more right than HF

The scene is a red circle in bbox (96,96)–(288,288) and a blue square
(300,300)–(460,460).

| shape | area |
|---|---|
| red circle | π·96² ≈ **28,953 px²** |
| blue square | 160² = **25,600 px²** |

The red circle is larger — by area *and* by bounding box (36,864 vs 25,600). The
device said `Red`; HF fp32 said `blue`. Treat neither as ground truth on a
near-tie; what it proves is that the device **saw the image and discriminated
between two shapes and two colours**.

### 6.4 What K2 establishes

> **The image → ViT → splice → 3-chunk prefill → lm_head path is functionally
> correct on device, on all seven images.**

That is the multimodal capability question answered, on hardware. Everything
after token 0 is the one known decode defect.

---

## 7. K3 — the numbers, and one anomaly

First timing for a 4B two-shard W8A16 tower on this silicon: **init ~1.4 s,
TTFT 510 ms (18 prompt tokens), prefill 35.3 tok/s, decode 6.5 tok/s.**

* Internally consistent: 18 ÷ 35.3 = 510 ms = the reported TTFT. ✅
* The report's "6.5 vs 43 tok/s ≈ the 6.67× parameter ratio, so decode is
  compute-bound" is sound and agrees with the 0.6B GQA-fix finding (`REFERENCE.md`
  correction #27).
* **Project-level consequence the report does not draw:** at 35.3 tok/s prefill, a
  279-token image prompt costs **~7.9 s** before the first token — which is what
  the 8–12 s per-image wall-clock is made of. Image TTFT is prefill-dominated.
* ⚠ **Warm init (1725 ms) exceeding cold (1375 ms) is unexplained** and the
  report's guess is hand-waving. Worth one clean cold/warm pair next session
  rather than a theory.

---

## 8. Status after Test K

| subsystem | status |
|---|---|
| image input, ViT, splice, prefill → first token | ✅ **device-verified on 7/7 images** (§6) |
| text prefill numerics, templated | ✅ device-verified (`qnn-net-run`) |
| decode graphs incl. recurrence | ✅ device-verified (Test J A1) |
| 4B timing baseline | ✅ first numbers (§7) |
| **Genie's decode-step feed** | ❌ the confirmed defect |
| **Genie's LUT feed at prefill** | ❌ **NEW — the probe is wrong from token 0 with graph and LUT proven correct on the host** (§5.3) |
| the two-ctx-bin split | ❓ still a live suspect, no longer the sole one |
| cross-chunk prefill `c1` | ❓ unproven, and **not** implicated by K2 (§6.2) |

**Next, in order:**

1. **`qnn-net-run` on the probe's ctx-bin** (§5.8) — separates Genie's LUT feed
   from the ctx-bin conversion. Small kit, no rebuild.
2. **Re-run K1 against `gqafix_cl512_ladekv`** (§5.5) — removes the CTX confound,
   ~5 minutes, makes the comparison what it was meant to be.
3. **The 2-shard LUT-fed 0.6B** — reproduces the 4B's structure at 0.6B scale,
   ~30 min local build. Still the right testbed for the split, and now also for
   the feed.
