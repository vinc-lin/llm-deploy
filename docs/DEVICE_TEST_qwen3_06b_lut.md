# 0.6B LUT probe — testing the embedding feed against a working control

**~10 minutes on device. It is not a product build and produces nothing
shippable.** Its only job is to answer one question that the Qwen3-VL-4B
investigation cannot answer from inside the 4B:

> Does Genie's **external-LUT embedding feed** work at all on this runtime?

## 1. Why this exists

The 0.6B runs correctly on device — 44.707 tok/s, same silicon, same
libGenie 1.19, same SDK, same converter, same AIMET path. The Qwen3-VL-4B text
tower produces garbage. Everything else has been eliminated on the host:

* no QDQ ops in either ONNX; the 0.6B uses the *same* `quantize_aimet.py` path,
  so "AIMET double-quantization" cannot be what separates them
* the 4B DLC already carries per-channel symmetric `sFxp_8` weights
* the 4B embedding LUT is bit-exact against its checkpoint
* the shard boundary is `FLOAT_16` on both sides — nothing to mismatch

That leaves **two** structural differences between the working model and the
broken one:

1. **the embedding feed** — the 0.6B does the lookup *in-graph* from token ids;
   the 4B receives pre-computed hidden states from an external float32 LUT
2. **the split** — the 0.6B is one ctx-bin, the 4B is two

Neither has ever been exercised on device by a model known to be good. This
probe isolates the first one, and the host-side variant ranking
(`feed_variants.json`, v5) already points there: `emb_fp32_as_fp16` is the top
candidate and it degenerates into repetition, matching the observed symptom.

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

```bash
adb push qwen3_06b_lutprobe /data/local/tmp/lutprobe
adb shell
cd /data/local/tmp/lutprobe && chmod +x genie-t2t-run
export LD_LIBRARY_PATH=.

./genie-t2t-run -c genie_dialog_qwen3_0.6b_lutprobe.json \
    -p "What is 2+2? Answer with one number." \
    --profile lutprobe_short.json

./genie-t2t-run -c genie_dialog_qwen3_0.6b_lutprobe.json \
    -p "Describe, in three or four sentences, what makes mountain weather change quickly." \
    --profile lutprobe_long.json
```

**Run the same two prompts against the working 0.6B bundle in the same
session**, on the same board, back to back. The comparison is the measurement;
an absolute result on its own is much weaker.

## 4. Reading the result

| Outcome | Meaning | What it does to the 4B investigation |
|---|---|---|
| **Coherent** ("4", a sensible paragraph) | the external-LUT feed path is **sound** on this runtime | the feed is exonerated. Next is **not** a split-only probe: v5 probe A already covers the split's numerics, and device slots are scarcer than host builds. Build the *maximally-4B-like* 0.6B — LUT **and** 2-way split together — so one slot either reproduces the bug (then bisect on the host, free) or clears the whole structural class |
| **Garbage / repetition** | the LUT feed path is **broken for everyone**, not just the 4B | root cause found, on a model that rebuilds in ~30 min locally instead of hours on tank. Work `feed_variants.json` from the top: `emb_fp32_as_fp16` |
| **Fails to load** | the config or the `inputs_embeds` naming is wrong | capture the exact error; it is a contract bug in the probe, not a finding about the 4B |

A garbage result here would be the **strongest possible outcome**: it converts
an intractable 4B problem into a fast local reproduction.

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
