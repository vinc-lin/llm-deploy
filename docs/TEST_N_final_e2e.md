# Test N — everything still standing between us and end-to-end, on the real 4B

**Status:** ready to run · **Opened:** 2026-08-22 · **Needs:** no rebuild — every
artefact is on the Hub, and the primary stage runs on the bundle already on the
board. **Board time ≈ 60 min for N1–N5;** N6 is the post-fix confirmation.

Execution and result-recording detail: **`DEVICE_SESSION_PROTOCOL_N.md`**
(separate document — read it alongside this one). Registry:
`DEVICE_TEST_INDEX.md`.

---

## 0. Where we are, in one screen

| subsystem | status |
|---|---|
| image input (fp32 blob contract) | ✅ device-verified |
| image analysis (ViT W8A16) | ✅ device-verified |
| image → ViT → splice → prefill → first token | ✅ device-verified, **7/7 images** (Test K2, scored on token ids) |
| text prefill numerics, templated | ✅ device-verified |
| prompt form (templated vs raw) | ✅ root cause confirmed twice on device |
| decode **graphs**, recurrence included | ✅ device-verified (Test J A1) |
| 4B timing baseline | ✅ init ~1.4 s, TTFT 510 ms, prefill 35.3 tok/s, decode 6.5 tok/s |
| **Genie's decode-step feed** | ❌ **the one open defect** — prefill right, decode step 1 wrong |
| which of {LUT feed, split} is at fault | ❓ Test K's attempt was void (it ran a FLOAT_16 bin, correction #39) |
| cross-chunk `c1` anomaly | ❓ open, unproven, not implicated by K2 |
| image caption beyond token 0 · full e2e | blocked on the decode fix |

**Why this test is shaped differently from L and M.** Tests L and M probe the
defect on 0.6B *proxies* — fast to build, but one proxy session has already
measured our own packaging bug instead of the 4B, and a clean proxy result
cannot clear the 4B (it controls neither scale nor width). **Stage N1 therefore
probes the real 4B directly**, using two `genie-t2t-run` capabilities that need
nothing built: `-tok` (feed exact token ids) and `--save` (dump dialog state).
L and M remain in the session as supporting stages — their value is now
*locating* the fault once N1 confirms its shape, not establishing it.

---

## Stage N1 — the decode defect, bisected on the REAL 4B (~15 min) — PRIMARY

### N1a — the token ladder: generation by repeated prefill

The defect in one line: prefill of 20 tokens produces the right token; the
decode step that follows produces the wrong one. The ladder asks the sharpest
possible next question:

> **If we hand Genie the prompt PLUS the token it just generated, as a fresh
> prefill, does it produce the right NEXT token?**

Each `testn/n1_*.tok` file is the previous prompt extended by the known-correct
continuation, so every run's first generated token comes from **prefill** and
has a single expected id. The expectations are the committed 2026-08-21 host
generation, and the weather steps 1–2 are additionally **device-verified**
(Test J A1 measured 9104 and 4344 on this silicon):

| run | file | prompt | **expect first generated id** | |
|---|---|---:|---:|---|
| P20 | `n1_2plus2_p20.tok` | 20 | **19** | `4` — reproduces the known state |
| **P21** | `n1_2plus2_p21.tok` | 21 | **151645** | `<|im_end|>` — **the decisive run** |
| W18 | `n1_weather_p18.tok` | 18 | **91169** | `Mountain` |
| W19 | `n1_weather_p19.tok` | 19 | **9104** | ` weather` |
| W20 | `n1_weather_p20.tok` | 20 | **4344** | ` changes` |
| W21 | `n1_weather_p21.tok` | 21 | **6157** | ` quickly` |

`-tok` bypasses the tokenizer by construction, which also settles the
`<|im_start|>`-splitting question for free: the profile's prompt-token count
must equal the file's token count exactly.

**How to read it:**

| P21/W19–W21 | meaning |
|---|---|
| **all correct** | Genie's prefill handles the *exact content* its decode step fails on. The defect is **decode-step machinery, content-independent** — it is not the LUT range, not the extended prompt, not the tokens themselves. Combined with N2/N3 below, this pins it |
| **wrong from P21 on** | the failure follows the *content* (a generated token in the sequence), not the step type — a different defect class than everything measured so far, and N1b's result becomes the next discriminator |
| P20/W18 wrong | the board is not in the Test I/J state — stop and re-check preconditions |

Expect every run except P21 to degenerate *after* its first token — that is the
known defect, not a finding. P21's correct output is EOS immediately: `[BEGIN]:`
followed by nothing.

### N1b — bypass the LUT at prefill (`-e`), one run

`n1b_2plus2_p20_emb.raw` is the P20 prompt's 20 LUT rows as raw fp32
(20 × 2560 × 4 = 204,800 B). `-e` feeds it as the query directly — **no
tokenizer, no prefill-time LUT lookup on the real 4B**:

* first token `4`, then the same degeneration → the prefill feed path is
  irrelevant to the defect (expected);
* anything else — including a load error — is new information; capture it
  verbatim. ⚠ Decode-time LUT lookup of *generated* tokens still happens either
  way; this run cannot exonerate that half.

### N1c — dump Genie's own state (`--save`)

`GenieDialog_save` serializes the dialog to a file. **Its contents are
undocumented** — this stage is exploratory, and its first deliverable is the
file itself. With `"max-num-tokens": 1` (config edit, §N1c of the protocol),
run P20 and P21 each with `--save`, pull both files, and report their **sizes**
before anything else. Size arithmetic that would identify a raw KV payload:

| if the file is ≈ | it likely holds |
|---|---|
| 36·2·8·128·**2048**·2 B ≈ **302 MB** | the prefill-width KV cache |
| 36·2·8·128·**2175**·2 B ≈ **321 MB** | the decode-width KV cache |
| a few KB | metadata only — the KV stays on-device, and this lever is dead |

If it is KV-sized, the two files differ by exactly one committed position, and
we can diff Genie's own cache against the host's — the direct observation of
the decode-step feed that every probe so far has had to reconstruct.

---

## Stage N2 — Test L: the 0.6B LUT probe, corrected bin (~15 min)

Unchanged from `TEST_L_ctxbin_vs_genie.md` — run it as written there (L0 Genie
re-run scored against the HF strings, then the L1/L2 `qnn-net-run` kit).
Bundle `qwen3_06b_lutprobe/`, bin **`9720e46e…`** — if the md5 reads
`880a6abd…` you have the FLOAT_16 bundle that voided Test K; stop.

**Role in this session:** clears or convicts the LUT feed *mechanism* at 0.6B.

## Stage N3 — Test M: the 2-shard 0.6B (~10 min)

Unchanged from `TEST_M_split_reproduction.md`. Bundle `qwen3_06b_lutsplit/`
(shards `1f4dcd44…` / `11cabce4…`); copy the LUT/tokenizer/runner/libs across
from the Test L bundle — that sharing is what makes L and M comparable.

**Role:** the only test that can *reproduce* the split defect somewhere cheap.
If it fires, the bisect moves to the host at 0.6B scale. If it stays clean, it
does **not** clear the 4B (scale and width are uncontrolled) — that is exactly
why N1 leads this session.

### The combined verdict — N1a × N3

| N1a ladder | N3 (0.6B split) | conclusion |
|---|---|---|
| all correct | first token right, then garbage | **the split is the defect**, reproduced at 0.6B — bisect host-side, fix, then N6 |
| all correct | clean | decode-step machinery is at fault but only on the 4B ⇒ a **scale/width-dependent** Genie defect. The evidence package (J + N1 + L + M) is complete enough to escalate to Qualcomm; N1c's state dump is our remaining lever |
| wrong from P21 | — | content-dependent failure — re-read N1b, and the LUT feed returns as prime suspect |

## Stage N4 — the image first-word grid, done right (~10 min)

Test K2's one-word runs are already scored 7/7 on **token ids**, but the human
grid was never completed with a working prompt. Re-run the seven images with
the one-word segments (`qwen3vl_4b_testk_session/prompts/`, swap procedure in
the protocol) and fill the RELEVANT/WRONG/DEGENERATE/INCONCLUSIVE grid.

**Judge the first word only.** Reference answers (HF fp32, same prompts,
`host_first_token_vl.py`): sample → `blue`/`Red` both defensible (the areas
differ by 12%); `wx_clear`/`wx_clear2` → `sunny` — so an `S`+fragment is a
**correct first token** plus the decode defect, not a failure;
`wx_clear_snow` → `cold` (ambiguous scene); the rest → `rainy`/`snowy`.

## Stage N5 — timing completion (~10 min)

Two numbers Test K left open:

1. **A clean cold/warm init pair.** K3's warm init (1725 ms) exceeded cold
   (1375 ms), unexplained. One fresh-boot cold run, then an immediate warm
   re-run, same command, both profiles kept.
2. **Image pipeline wall-clock, measured** — `time ./genie-app -s …` on the
   sample image, cold and warm. K3's 8–12 s was an eyeball estimate; the
   prefill arithmetic (279 tokens ÷ 35.3 tok/s ≈ 7.9 s) predicts most of it.

## Stage N6 — the end-to-end confirmation (post-fix ONLY)

**This is the definition of done, and it cannot pass until the decode fix from
N1–N3 lands.** It is `RUNBOOK_e2e_qwen3vl_4b.md` §2–§5 verbatim:

| gate | pass |
|---|---|
| text | templated 2+2 → **`4` then stop** (two tokens, EOS honoured) |
| sample image | a caption that describes the picture — a full sentence, not just token 0 |
| six photographs | six captions, each recognisably about its own photo (human judgement, one line each) |
| timing | init / TTFT / decode tok/s + pipeline wall-clock, cold and warm separately |

When those four rows pass, the model runs fully end-to-end on the device —
image input, image analysis, text generation — and the remaining work is speed,
not correctness.

---

## Priority order, if the session runs short

1. **N1a P21** — one command, and it is the sharpest single measurement ever
   available on this defect
2. the rest of N1a, then **N1c** (two runs + two pulls)
3. **N3** (Test M) — the reproduction attempt
4. **N2** (Test L) — the LUT-mechanism check
5. N4, N5
6. N6 runs in its own session after the fix

Stopping after N1a alone is a genuinely useful session.
