# Qwen3-VL-4B — Test I device results (templated vs raw prompt)

**Source:** device team, received 2026-08-21. Body is their report; the
annotations at the end are ours and are clearly marked.

---

## Genie-level (end-to-end generation)

**A. Templated 2+2** (20 tokens, ChatML) → first token **`4` — correct**, then a
repetition loop (`ance` / `ably` / `ability`) until `Context Size was exceeded`.

**B. Raw 2+2** (12 tokens) → ` 0÷÷÷÷…` — first token already wrong. Matches the
previous v5 Test 3a behaviour.

**C. Templated weather** → first token **`Mountain` — correct topic**, then the
same repetition loop.

## Graph-level (`qnn-net-run`)

| case | row 0 | row 1 | last row | chained argmax | isolated argmax |
|---|---|---|---|---|---|
| `i0_raw` | gain **1.390**, cos 0.999990 | 0.932 | 0.998 | 4/5 | 5/5 |
| `i1_templated` | 1.044, cos 0.9973 | **0.989, cos 1.000000** | 0.988 | **5/5** | **5/5** |

---

## Annotations (build side, 2026-08-21)

### The root cause is CONFIRMED, at both levels

`i0_raw` row 0 measured **1.390** against a host prediction of 1.3505 (and Test
E's 1.38959 on the older probe). `i1_templated` is clean with 5/5 chained argmax.
And the Genie-level first token flips from wrong (raw) to **correct** (templated)
on the same bins, same session. The calibration/prompt-form mismatch is settled.

### The repetition is a DEFECT, not greedy-without-EOS

This needed checking rather than assuming, because
`docs/ISSUE_qwen3vl_4b_text_numerics.md` §2.2 previously recorded "2+2 → 4 then
repeats is expected: greedy sampling with no EOS runs to the context limit."
**That note is false and is hereby withdrawn.** `genie_dialog_qwen3vl_4b.json`
configures `eos-token: [151645, 151643]` with greedy sampling, and the same is
true of the 0.6B lutprobe config the note was written about.

Measured on the host, the real split fp32 graphs, same 20 templated tokens, full
KV recurrence exactly as `parity_e2e_vl.Decoder` does it
(`scripts/validate/host_generate_check.py`):

```
prefill 20 tokens -> first generated id 19
  step 1: EOS (151645) -- generation stops here
generated ids: [19, 151645]        # '4' + '<|im_end|>'  ->  "4<|im_end|>"
```

**The correct output is two tokens.** The device's continuation past `4` is
wrong, full stop.

### The failure is at DECODE STEP 1, not "after the first token"

The first generated token comes from **prefill** (argmax of the last prompt
row), not from decode. So:

* prefill on device → `4` → **correct**, and `i1_templated`'s 5/5 chained argmax
  says so independently;
* decode step 1 on device → should be `151645 <|im_end|>`, produced `2939
  'ention'` → **the very first decode call is already wrong**.

That is a much tighter localisation than "decode fails after token 1".

### ⚠ Correction: shard 1 decode HAS been tested

The report's primary suspect is "Shard 1 Decode — never been tested
independently." That is not correct. Test G's `r3_decodectx` ran a decode step
with a real 13-position cache through **both** shards, in both configurations:

| | argmax | cosine |
|---|---|---|
| logits chained (fed the device's own shard-0 output) | 1/1 | 0.99994 |
| logits isolated (fed the host reference boundary) | 1/1 | 0.99998 |

So the decode **graphs** produce the right answer on this silicon when the
inputs are supplied by the kit. Combined with the host run above (graphs + KV
recurrence correct) and `parity_e2e_vl` at 20/20 token-identical against HF, the
decode path is verified everywhere except one place.

### What is actually left

The difference between `r3_decodectx` (device, correct) and Genie's decode step 1
(device, wrong) is **not the graph**. It is who supplies the decode step's
inputs:

| input | r3_decodectx | Genie |
|---|---|---|
| KV cache | built on the host, pushed as files | Genie's own prefill → decode handoff |
| `inputs_embeds` | kit file (LUT row of the next token) | Genie's LUT lookup of the generated token |
| mask / positions | kit files | Genie's own advance |

**So the defect is in Genie's decode-step feed**, and the three candidates above
are the whole space. Ranked by what the evidence supports:

1. **The prefill → decode KV handoff.** The prefill graph's cache is
   `[1,8,D,2048]`-wide and the decode graph's is `[1,8,D,2175]`; Genie must
   re-lay 20 committed positions across that change. Nothing has ever tested it.
2. **The LUT lookup of a *generated* token** at decode time. Note the prompt
   tokens are looked up correctly (prefill is right), so this would have to be
   specific to the decode path.
3. **mask / position advance** for the first decode step.

The report's suspects 2–4 are these, restated; suspect 1 (shard 1) is dead.

### The next test

`r3_decodectx` with **this** prompt: prefill the 20 templated tokens, seed the
decode cache from it, run one decode step under `qnn-net-run`. Expected argmax
**151645**. If the device returns 151645, Genie's plumbing is proven at fault and
the bisect moves inside Genie; if it returns 2939, the ctx-bin decode path is at
fault despite `r3_decodectx` passing, and the difference is the cache contents.

It is a small build — the `--suite r` decode-with-context machinery already does
exactly this and only needs the token list swapped.
