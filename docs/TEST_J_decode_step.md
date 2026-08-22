# Test J — the final session: everything still standing between us and end-to-end

**Status: RUN AND CLOSED, 2026-08-21.** Results, corrections and the re-ranked
suspect list are in
**`reports/qwen3vl-4b-testj-device-results-2026-08-21.md`** — read that, not the
predictions below.

Outcome in one line: **Stage A1 landed on the predicted row — the decode graph is
correct including the recurrence, so the fault is Genie's decode-step feed.** Of
the three candidates this document named, **two are now dead** (§10 of the
report), and the probe that separates the surviving two is
`docs/DEVICE_TEST_qwen3_06b_lut.md`, already built and already on the Hub.

What follows is the plan as written before the session, kept for provenance.
Two things in it are known wrong: the Stage C first-word test could not
discriminate with an open-ended caption prompt, and `--max-num-tokens` is a
config key rather than a CLI flag.

---

**Opened:** 2026-08-21 · **Needs:** no rebuild, no new ctx-bins.
**Board time ≈ 30 minutes**, in four stages.
Execution and result-recording detail: **`DEVICE_SESSION_PROTOCOL.md`** (separate
document — read it alongside this one).

---

## 0. Where we are, in one screen

| subsystem | status |
|---|---|
| image input (fp32 blob contract) | ✅ device-verified |
| image analysis (ViT W8A16) | ✅ device-verified, under `qnn-net-run` and Genie |
| text: prefill numerics | ✅ device-verified (Test G, Test I `i1_templated` 5/5 argmax) |
| text: prompt form | ✅ **root cause found and confirmed** — the 4B is calibrated on chat-templated prompts; every earlier test sent raw text. Templated flips the first token to correct |
| text: **decode step 1** | ❌ **the one open defect.** Genie's first decode call is wrong |
| cross-chunk prefill | ❓ never run on device — and the 273-token image prompt needs it |
| the image pipeline, post-fix | ❓ never run since the `uFxp_16` fix, because the old guide gated it on a test that used a raw prompt |

**One defect and three unrun paths.** This session closes all four.

### The decode defect, precisely

The first generated token comes from **prefill**, not decode — it is the argmax
of the last prompt row. So on a templated prompt prefill is right and the **very
first decode call** is already wrong:

| prompt | correct next token | Genie produced |
|---|---|---|
| `What is 2+2? …` | **151645** `<\|im_end\|>` | 2939 `ention` |
| `Describe … mountain weather …` | **9104** ` weather` | 3279 `aged` |

Ground truth, from the real split fp32 graphs with the full KV recurrence
(`scripts/validate/host_generate_check.py`):

```
2+2      -> [19, 151645]                       = "4<|im_end|>"    # TWO tokens
weather  -> [91169, 9104, 4344, 6157, 1576, …] = "Mountain weather changes
                                                  quickly because elevation
                                                  causes rapid shifts in
                                                  temperature and air pressure…"
```

⚠ A prior note said *"4 then repeats is expected: greedy with no EOS."* **False,
and withdrawn.** `eos-token: [151645, 151643]` is configured and the graphs emit
it immediately. The repetition loop is a defect.

**The graphs are not the suspect.** The host does the whole recurrence correctly;
`parity_e2e_vl.py` is 20/20 token-identical against HF; and Test G's
`r3_decodectx` already ran a decode step on device through **both** shards
(chained argmax 1/1 cos 0.99994, isolated 1/1 cos 0.99998). ⚠ So *"shard 1
decode has never been tested"* is not correct — it has, and it passed. What has
never been tested is **who supplies the decode step's inputs**.

---

## Stage A — `qnn-net-run` probes (~8 min, no Genie)

Two kits, same runner. Neither needs a rebuild.

### A1 — the decode step (`testj/`)

Each case hands the decode graph a **host-built version of exactly the state
Genie should have produced**, and asks whether the graph then gets it right.

| case | prompt | step | cache going in | **expected argmax** |
|---|---|---:|---|---:|
| `j0_2plus2_s1` | templated 2+2 | 1 | 20 positions, all from prefill | **151645** |
| `j1_weather_s1` | templated weather | 1 | 18 positions, all from prefill | **9104** |
| `j2_weather_s2` | templated weather | 2 | 19 positions — **one written by the DECODE graph** | **4344** |

`j2` is the recurrence, and it is the only part of the decode path that has never
run on device under any probe.

### A2 — cross-chunk prefill (`testh/`, already on HF)

A real image prompt is 273 tokens = **three** AR=128 prefill calls, so chunks 2
and 3 run against a partially populated cache. Every probe so far ran a single
chunk against an empty one. This is **on the critical path for Stage C**, not
optional.

| case | rows | cache in | note |
|---|---:|---:|---|
| `c0_chunk0` | 128 | 0 | anchor — must reproduce Test G's `r2_chunk0` |
| `c1_chunk1` | 128 | 128 | first cross-chunk call |
| `c2_chunk2` | **21** | **256** | last and **partial** — the `variant > n_process` condition |

**Headline number for A1 is the logits argmax; for A2 it is the boundary gain.**
The analyzer prints both.

---

## Stage B — Genie text (~5 min)

Re-run the Test I prompts and **record the generated text verbatim**. Two
purposes: confirm A's result against what Genie actually does, and give us the
exact divergence point.

Also run the **raw** control. The contrast is the evidence.

If Stage A says the graph is fine, Stage B's text is what we bisect Genie
against — so verbatim matters more than usual here. Do not summarise, do not
truncate to "garbage".

---

## Stage C — the image pipeline (~12 min)

**Run this even though decode is broken.** Here is why it is worth the time:

> The pipeline's **first generated token also comes from prefill**. So the first
> token — or the first word of the caption — validates the entire
> **image → ViT → splice → prefill** path *independently of the decode defect*.
> If the first word is about the picture, the vision half works. That is a real
> result and we can get it today.

Expect generation to degenerate after the first token, exactly as in Stage B.
That is the known defect, not a new one.

⚠ **Two traps, both of which have already cost a session:**

1. **Use `03_vl4b_v5/`, not `qwen3vl_4b_e2e_pipeline_v5/`.** The latter shipped a
   **stale shard 0** (`065056ba…`, the pre-`uFxp_16` bin). It has been replaced
   on the Hub, but check anyway: `md5sum qwen3vl-4b-w8a16_1_of_2.bin` must be
   **`f031e3a7563bf16f2d5ca98a71b357f6`**.
2. **Never feed a `*_u16.raw` image.** Genie stages the file as float32
   regardless of tensor dtype, so a UFixed16 blob is read at 2× its size — the
   `SIGSEGV (SEGV_ACCERR)`. Every image must be `*_fp32.raw` and **6,295,552
   bytes**.

C1 = the sample image. C2 = the six photographs.

---

## Stage D — timing (~5 min)

Run whatever the text looks like: decode rate is the same compute whether the
tokens are right or wrong, and there is still **no init / TTFT / decode number**
for a 4B two-shard W8A16 tower on this silicon.

---

## What each Stage A outcome means

| A1 j0/j1 | A1 j2 | meaning | next |
|---|---|---|---|
| **match** ← *observed* | **match** ← *observed* | the decode graph is right end to end, **including the recurrence**. The fault is entirely in **Genie's decode-step feed** | ~~bisect inside Genie: (1) the prefill→decode KV handoff across the 2048→2175 width change, (2) the LUT lookup of a *generated* token, (3) the mask/position advance~~ — **(1) and (3) were eliminated device-free on 2026-08-21**; see below |
| match | mismatch | the graph is right on a prefill-built cache and wrong on one it wrote itself | a **KV write-back/read-back** defect in the ctx-bin decode path |
| mismatch | — | the decode ctx-bin is wrong on this cache although `r3_decodectx` passed on a 13-position one | bisect on cache length |

The first row is the expected outcome.

| A2 | meaning |
|---|---|
| all three clean | cross-chunk prefill is sound; the 273-token image prompt is safe |
| `c0` clean, `c1`/`c2` off | the cross-chunk cache path is a **second** defect, and it would also affect Stage C |
| `c2` alone off | the **partial last chunk** specifically — the padding-bug condition, in a build where the dtype is supposed to be fixed |

---

## After this session — what actually happened

Stage A came back as expected: **graph fine, Genie at fault.** Stage C was
inconclusive rather than negative, because the caption prompt let every image
answer `A`.

**The candidate list is now two, not three:**

| ✝ | prefill→decode KV width handoff | dead — the shipping 0.6B `gqafix-ladekv` makes the same 1024→1151 change with the same `[1,8,128,PAST]` key layout on the same runtime and generates correct text |
|---|---|---|
| ✝ | MRoPE decode advance | dead for text-only — `nsp-model.cpp:3803` gates the Qwen3-VL branch on `m_visionParam.size() > 0`, so Stage B was a plain-rope run |
| **?** | **the external-LUT `inputs_embeds` feed** | live |
| **?** | **the two-ctx-bin split** (shard 0 → shard 1 every decode step) | live |

**Next test: `docs/DEVICE_TEST_qwen3_06b_lut.md`.** A 0.6B on the shipping ladekv
recipe, **one** ctx-bin, LUT-fed — one variable — which separates the two
survivors in ~10 minutes. Built 2026-08-18, on the Hub, never run.

`RUNBOOK_e2e_qwen3vl_4b.md` is the procedure for the confirmation run once the
decode fix lands.
