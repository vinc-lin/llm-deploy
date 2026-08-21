# 0.6B LUT probe — testing the embedding feed against a working control

**THIS IS THE NEXT TEST.** After Test J (2026-08-21) it is the single experiment
that moves the Qwen3-VL-4B investigation forward, and it is already built and
already on the Hub.

**~10 minutes on device. It is not a product build and produces nothing
shippable.** Its only job is to answer one question the 4B cannot answer from
inside itself:

> Does Genie's **external-LUT embedding feed** work at all on this runtime?

## 1. Why this exists

> ⚠ **Premise updated 2026-08-21.** This document was written when the 4B was
> believed to produce garbage outright. It does not. Test I and Test J showed the
> 4B's **prefill is correct** — on a chat-templated prompt the first token is
> right — and that **decode step 1** is the first thing that is wrong. So the
> question this probe asks is unchanged, but the answer to watch for is sharper:
> not "is it coherent" but **"does decode survive past the first token"**.

The 0.6B runs correctly on device — 44.707 tok/s, same silicon, same
libGenie 1.19, same SDK, same converter, same AIMET path. Everything else has
been eliminated:

* no QDQ ops in either ONNX; the 0.6B uses the *same* `quantize_aimet.py` path,
  so "AIMET double-quantization" cannot be what separates them
* the 4B DLC already carries per-channel symmetric `sFxp_8` weights
* the 4B embedding LUT is bit-exact against its checkpoint
* the shard boundary is `FLOAT_16` on both sides — nothing to mismatch
* the 4B's **decode graphs are device-verified**, recurrence included
  (Test J Stage A1: `j0`/`j1`/`j2` all hit their expected argmax)
* the **prefill→decode KV width handoff** is device-proven — the shipping 0.6B
  `gqafix-ladekv` makes the same 1024→1151 change with the same `[1,8,128,PAST]`
  key layout and generates correct text
* **MRoPE** is not involved in a text-only run — `nsp-model.cpp:3803` gates the
  Qwen3-VL branch on `m_visionParam.size() > 0`

That leaves exactly **two** structural differences between the working model and
the broken one:

1. **the embedding feed** — the 0.6B does the lookup *in-graph* from token ids;
   the 4B receives pre-computed hidden states from an external float32 LUT
2. **the split** — the 0.6B is one ctx-bin, the 4B is two, and shard 0 hands
   `[1,1,2560]` to shard 1 on *every decode step*

Neither has ever been exercised on device by a model known to be good. This probe
isolates the first, which decides the second by elimination. The host-side variant
ranking (`feed_variants.json`, v5) points the same way: `emb_fp32_as_fp16` is the
top candidate and it degenerates into repetition — the observed symptom.

## 2. What the build is

`qwen3-0.6b-w8a16-lutprobe-ladekv` is built on the **shipping model's own
recipe** — `full_build.sh ... 128 512 --grouped-gqa` followed by
`ladekv_build.sh ... 128 512 128`, the same past-KV-prefill chain behind the
device-proven 44.707 tok/s build — with exactly one thing added:
`--input-embeds`.

Measured against the shipping ctx-bin, every graph shape matches:

| | probe | shipping |
|---|---|---|
| prefill mask / past | `[1,128,640]` / `[1,8,128,512]` | `[1,128,640]` / `[1,8,128,512]` |
| decode mask / past | `[1,1,640]` / `[1,8,128,639]` | `[1,1,640]` / `[1,8,128,639]` |
| first input | **`inputs_embeds [1,1,128,1024]`** | `input_ids [1,128]` |

The first row is the only difference. An earlier version of this probe used the
`full_build.sh` **bertcache** prefill (`mask [1,128,128]`), which would have let
a garbage result be blamed on the prefill type rather than the feed; that is why
it was rebuilt on the ladekv recipe. `verify32` is omitted — it exists for LADE,
which is parked as a 30% regression and unused in basic mode.

`--input-embeds` changes three coupled things, which is why the flag is detected
in `full_build.sh` / `ladekv_build.sh` rather than only passed to the quantizer:

* the exported tower takes `inputs_embeds [1,1,AR,1024]` instead of
  `input_ids [1,AR]`, and drops the embedding table from the graph
* calibration observes real hidden states rather than token ids, so the input
  quantizer's ranges describe the tensor that will actually arrive
* the I/O rename uses `--vl-text --n-deepstack 0`, because **qualla selects
  `InputType::EMBEDDINGS` by matching the literal input name `inputs_embeds`**
  (`nsp-model.cpp:668`). Rename it anything else and the runtime drives the
  graph as token ids.

Everything else — GQA fix, CL 128, CTX 512, W8A16, HTP config, rope theta 1e6 —
is identical to the model that works. **That is the point: one variable.**

## 3. Run it

Download from `vinccniv/sa8797p-qwen3-w8a16-bundles`, folder
`qwen3_06b_lutprobe/` (17 files, ~1.4 GB — the 622 MB LUT and the 768 MB
ctx-bin are most of it).

```bash
adb push qwen3_06b_lutprobe /data/local/tmp/lutprobe
adb shell
cd /data/local/tmp/lutprobe && chmod +x genie-t2t-run
export LD_LIBRARY_PATH=.

./genie-t2t-run -c genie_dialog_qwen3_0.6b_lutprobe.json \
    -p "What is 2+2? Answer with one number." \
    --profile lutprobe_short.json 2>&1 | tee lutprobe_short.txt

./genie-t2t-run -c genie_dialog_qwen3_0.6b_lutprobe.json \
    -p "Describe, in three or four sentences, what makes mountain weather change quickly." \
    --profile lutprobe_long.json 2>&1 | tee lutprobe_long.txt
```

**Then run the same two prompts against the working 0.6B bundle
(`qwen3_06b_w8a16_gqafix_ladekv`, `genie_dialog_basic.json`) in the same session**,
on the same board, back to back:

```bash
cd /data/local/tmp/gqafix && export LD_LIBRARY_PATH=.
./genie-t2t-run -c genie_dialog_basic.json \
    -p "What is 2+2? Answer with one number." \
    --profile control_short.json 2>&1 | tee control_short.txt
./genie-t2t-run -c genie_dialog_basic.json \
    -p "Describe, in three or four sentences, what makes mountain weather change quickly." \
    --profile control_long.json 2>&1 | tee control_long.txt
```

**The comparison is the measurement**; an absolute result on its own is much
weaker. Four transcripts, two pairs.

Two flag facts, both learned the hard way in the Test J session:

* generation is capped by the config key **`dialog.max-num-tokens`** (set to 64
  in the shipped `genie_dialog_qwen3_0.6b_lutprobe.json`). There is no
  `--max-num-tokens` flag; passing one aborts the run. Without a cap the query
  runs to `Context Size was exceeded` and **no profile is written**.
* **`--profile` refuses an output file that already exists.** Use a fresh name on
  every re-run, or delete the old one first.

## 4. Reading the result

⚠ **"Coherent" is the wrong pass criterion, and using it will produce a false
negative.** The 0.6B has no chat template and *will* fall into a greedy repetition
loop eventually — the **working control does too** (`reports/qwen3-0.6b-w8a16-ladekv-test-report.md`
§5: R1 answers `The capital of France is Paris.` and then repeats the sentence).
So "it repeated" proves nothing on its own.

**The measurement is the diff against the control**, same prompts, same board,
same session, back to back:

> **How many tokens does the probe match the working bundle for?**

| Outcome | Meaning | What it does to the 4B investigation |
|---|---|---|
| **Matches the control for tens of tokens** | the external-LUT feed is **sound** on this runtime | the feed is exonerated ⇒ **the split is the fault** by elimination. That is a positive result, not a dead end: it points at the shard 0 → shard 1 `[1,1,2560]` decode handoff, which is a much smaller search space than "somewhere in Genie" |
| **First token right, then diverges immediately** | the LUT feed is **broken for everyone**, not just the 4B — and it is the *4B's exact signature* | root cause found, on a model that rebuilds in ~30 min locally instead of hours on tank. Work `feed_variants.json` from the top: `emb_fp32_as_fp16` |
| **Wrong from token 0** | not the decode defect — a prefill/feed contract bug in the probe itself | check the `inputs_embeds` naming and the LUT dtype before drawing any 4B conclusion |
| **Fails to load** | the config or the `inputs_embeds` naming is wrong | capture the exact error; it is a contract bug in the probe, not a finding about the 4B |

Row 2 would be the **strongest possible outcome**: it converts an intractable 4B
problem into a fast local reproduction.

⚠ The prompts here are deliberately **raw, not chat-templated** — unlike the 4B's.
That is correct and must not be "fixed": the 0.6B was calibrated on raw prompts
(`CALIB_PROMPTS`), so raw *is* its in-distribution form. The templated-prompt
requirement is a 4B calibration fact, not a Genie one.

## 5. What this probe cannot tell you

It does not test the split (one ctx-bin), MRoPE (plain rope), deepstack (none),
or image handling. A clean result narrows the 4B fault to those; it does not
clear the 4B.

It also no longer tests the prefill *type*: the past-KV prefill is now known to
be device-proven at 0.6B (the shipping 44.707 tok/s build uses `mask [1,128,640]`,
`past [1,8,128,512]` — the same class as the 4B's `[1,128,2176]`). That variable
is eliminated, which is precisely why this probe was rebuilt to match it rather
than to vary it.

It also shares no code path with the v5 `qnn-net-run` probe: that one asks "is
the 4B ctx-bin right?", this one asks "does the LUT feed work at all?". They
attack the question from opposite ends and neither substitutes for the other.

## 6. What to send back

1. Both `lutprobe_*.json` profiles and the generated text, verbatim, garbage
   included.
2. The same two prompts' output from the **working** 0.6B bundle, same session.
3. Full stdout/stderr on any load failure, plus `adb logcat -d`.
