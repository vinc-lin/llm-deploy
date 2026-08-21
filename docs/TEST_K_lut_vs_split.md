# Test K — which of the two remaining suspects breaks decode

**Status:** ready to run · **Opened:** 2026-08-21 · **Needs:** no rebuild, no new
ctx-bins — every artefact is already on the Hub. **Board time ≈ 30 minutes.**

Self-contained on purpose: what to run, why, and how to record it are all in this
file. Test J needed a separate protocol document because it was five stages of
probe kits; K1 is four `genie-t2t-run` invocations, and splitting four commands
across two documents would cost more than it saves.

Results template: `deliverables/qwen3vl_testk_session/RESULTS_TEMPLATE.md`.

---

## 0. Where we are, in one screen

| subsystem | status |
|---|---|
| image input (fp32 blob contract) | ✅ device-verified |
| image analysis (ViT W8A16) | ✅ device-verified |
| text prefill numerics, templated | ✅ device-verified |
| prompt form (templated vs raw) | ✅ root cause confirmed twice on device |
| decode **graphs**, recurrence included | ✅ device-verified — Test J Stage A1 |
| **Genie's decode-step feed** | ❌ **the one confirmed defect** |
| cross-chunk prefill | ❓ one anomalous row, four candidate explanations |
| image → caption, semantically | ❓ never tested by a prompt that could discriminate |

The first generated token comes from **prefill** and is correct. **Decode step 1**
is already wrong: `2+2` should emit `151645 <\|im_end\|>` and Genie emits
`2939 'ention'`. Test J proved the decode *graph* gets this right when the kit
supplies its inputs, so the fault is in **who supplies them**.

Three candidates were on the table after Test I. Two are dead:

| ✝ | prefill→decode KV width handoff | our shipping 0.6B `gqafix-ladekv` makes the same 1024→1151 change with the same `[1,8,128,PAST]` key layout on the same libGenie and generates correct text |
|---|---|---|
| ✝ | MRoPE decode advance | `nsp-model.cpp:3803` gates the Qwen3-VL branch on `m_visionParam.size() > 0`, so a text-only run is plain rope |

**Two survive**, and they are the only structural differences left between the
0.6B that works and the 4B that does not:

| | 0.6B (works) | VL-4B (decode broken) |
|---|---|---|
| embedding feed | in-graph, from token ids | **external float32 LUT → `inputs_embeds`** |
| ctx-bins | **one** | **two** — shard 0 hands `[1,1,2560]` to shard 1 every decode step |

---

## Stage K1 — the LUT probe against its control (~12 min)

**This is the whole point of the session.** Everything else is a free rider.

`qwen3_06b_lutprobe` is a 0.6B built on the **shipping ladekv recipe** — the same
`full_build.sh … 128 512 --grouped-gqa` + `ladekv_build.sh` chain behind the
device-proven 44.707 tok/s build — with exactly one thing changed:
`--input-embeds`. One ctx-bin, LUT-fed. **One variable.**

So it isolates the feed. And because it isolates the feed, it decides the split
too, by elimination.

### K1a — the probe

Download `qwen3_06b_lutprobe/` (17 files, ~1.4 GB) from
`vinccniv/sa8797p-qwen3-w8a16-bundles`.

```sh
adb push qwen3_06b_lutprobe /data/local/tmp/lutprobe
adb shell
cd /data/local/tmp/lutprobe && chmod +x genie-t2t-run
export LD_LIBRARY_PATH=.

md5sum qwen3-0.6b-w8a16-lutprobe-ladekv_ctx.bin | cut -c1-32   # 880a6abdec4a64b67b275ec817c054ca
md5sum embedding_float32_lut.bin | cut -c1-32                  # 2836420ce26c84f4ab7b217a22b2e125
grep max-num-tokens genie_dialog_qwen3_0.6b_lutprobe.json      # "max-num-tokens": 64,

./genie-t2t-run -c genie_dialog_qwen3_0.6b_lutprobe.json \
    -p "What is 2+2? Answer with one number." \
    --profile k1a_short.json 2>&1 | tee k1a_short.txt

./genie-t2t-run -c genie_dialog_qwen3_0.6b_lutprobe.json \
    -p "Describe, in three or four sentences, what makes mountain weather change quickly." \
    --profile k1a_long.json 2>&1 | tee k1a_long.txt
```

### K1b — the control, same session, same board

`2026-08-14-gqafix/bundles/qwen3_06b_w8a16_gqafix_ladekv.tar.gz` in the same
repo. This is the device-proven 44.707 tok/s bundle.

```sh
# host: tar xzf qwen3_06b_w8a16_gqafix_ladekv.tar.gz
#       adb push qwen3_06b_w8a16_gqafix_ladekv /data/local/tmp/gqafix
cd /data/local/tmp/gqafix && chmod +x genie-t2t-run && export LD_LIBRARY_PATH=.

./genie-t2t-run -c genie_dialog_basic.json \
    -p "What is 2+2? Answer with one number." \
    --profile k1b_short.json 2>&1 | tee k1b_short.txt

./genie-t2t-run -c genie_dialog_basic.json \
    -p "Describe, in three or four sentences, what makes mountain weather change quickly." \
    --profile k1b_long.json 2>&1 | tee k1b_long.txt
```

`genie_dialog_basic.json`, **not** `genie_dialog.json` — basic mode is prefill +
decode with no speculation, which is exactly the mode the 4B runs in. Comparing
against a LADE run would introduce a second variable.

⚠ **The control is half the experiment, not a formality.** Skipping it makes the
probe uninterpretable — see §K1c.

### K1c — how to read it

> ⚠ **"Is it coherent?" is the wrong question and will produce a false negative.**
> The 0.6B has no chat template, so under greedy decode **the control repeats
> too** — `reports/qwen3-0.6b-w8a16-ladekv-test-report.md` §5 records R1 answering
> `The capital of France is Paris.` and then repeating the sentence. "It repeated"
> proves nothing on its own.

**The measurement is the diff.** Put the two transcripts side by side and count:

> **How many tokens does the probe match the control for?**

| what you see | meaning | what happens next |
|---|---|---|
| **matches the control for tens of tokens** | the external-LUT feed is **sound** on this runtime | the feed is exonerated ⇒ **the split is the fault** by elimination. Not a dead end — it points at the shard 0 → shard 1 `[1,1,2560]` decode handoff, a far smaller search space than "somewhere in Genie". Next build is a 0.6B that is LUT-fed **and** 2-way split, ~30 min locally, to reproduce it on the host |
| **first token right, then diverges immediately** | the LUT feed is **broken for everyone**, not just the 4B — the 4B's exact signature | root cause found, and it now reproduces in a 1.4 GB bundle that rebuilds locally in ~30 min instead of hours on tank. Work `feed_variants.json` from the top: `emb_fp32_as_fp16` |
| **wrong from token 0** | not the decode defect — a prefill/feed contract bug in the probe itself | check the `inputs_embeds` naming and the LUT dtype before drawing any 4B conclusion |
| **fails to load** | the config or the `inputs_embeds` naming is wrong | capture the exact error and `adb logcat -d`; it is a bug in the probe, not a finding about the 4B |

⚠ **The prompts here are deliberately raw, not chat-templated** — unlike the 4B's.
That is correct and must not be "fixed". The 0.6B was calibrated on raw prompts
(`CALIB_PROMPTS`), so raw *is* its in-distribution form. The templating
requirement is a 4B calibration fact, not a Genie one.

---

## Stage K2 — the image path's first token, finally discriminating (~12 min)

**Free rider, and it closes a question Test J could not.** Test J asked whether
the first word of the caption was about the picture. All seven images answered
**`A`** — a real word, a perfectly good caption opener, and completely
uninformative about any image. That was a flaw in the prompt, not a device
failure: an open-ended caption prompt could never have discriminated.

The fix is a **prompt-file swap, no rebuild**. Two replacement segments ship in
`deliverables/qwen3vl_testk_session/prompts/`:

| file | question | expected first token |
|---|---|---|
| `prompt_seg2_oneword_weather.txt` | *Answer with one word: what is the weather in this photo?* | `Snow` / `Fog` / `Clear` / `Rain` |
| `prompt_seg2_oneword_shape.txt` | *Answer with one word: what colour is the largest shape in this image?* | `Red` (the sample image is a red circle + a blue square) |

Both keep the ChatML frame byte-exact — they start `<|vision_end|>` and end
`<|im_end|>\n<|im_start|>assistant\n`, the same invariants
`vl_pipeline_bundle.sh` asserts on the generated segments. Token totals go
273 → 279 / 281, i.e. **still three AR=128 prefill chunks**, so the cross-chunk
path is exercised under exactly the same conditions as before.

```sh
cd /data/local/tmp/v5 && export LD_LIBRARY_PATH=.
md5sum qwen3vl-4b-w8a16_1_of_2.bin | cut -c1-32     # f031e3a7563bf16f2d5ca98a71b357f6
cp prompt_seg2.txt prompt_seg2_ORIGINAL.txt         # keep the shipped one

# K2a -- synthetic sample image
cp prompt_seg2_oneword_shape.txt prompt_seg2.txt
./genie-app -s genie_pipeline_qwen3vl.script 2>&1 | tee k2a_sample.log

# K2b -- the six photographs
cp prompt_seg2_oneword_weather.txt prompt_seg2.txt
for f in wx_*.script; do
  s=${f%.script}
  echo "===== $s ====="
  ./genie-app -s "$f" 2>&1 | tee "k2b_$s.log"
done
```

Every pipeline script reads `prompt_seg2.txt` by that exact name, which is why
the swap works and why the copy order matters.

**Expect generation to degenerate after the first token** — that is defect #1,
already known, not a new finding. **Judge the first word only**, against the
`.png`/`.jpg` open on screen:

| verdict | meaning |
|---|---|
| `RELEVANT` | the first word describes that image |
| `WRONG` | a confident word about something not in the image |
| `DEGENERATE` | not a word — punctuation, a fragment, or empty |
| `INCONCLUSIVE` | a word that would fit any image (`A`, `The`, `This`) — means the prompt still isn't forcing an answer |

`WRONG` and `DEGENERATE` are different diagnoses. Do not merge them.

**What a `RELEVANT` sweep would buy:** the entire **image → ViT → splice →
prefill** path validated end to end on hardware, independently of the decode
defect. That is most of "does the multimodal model work", available today.

⚠ Restore the original afterwards: `cp prompt_seg2_ORIGINAL.txt prompt_seg2.txt`.

---

## Stage K3 — timing (~5 min)

There is still **no init / TTFT / decode number** for a 4B two-shard W8A16 tower
on this silicon. Decode rate is the same compute whether the tokens are right or
wrong, so this is collectable now.

Set `"max-num-tokens": 128` in `genie_dialog_qwen3vl_4b.json` (see §Traps), then:

```sh
P=$(cat /data/local/tmp/testi/prompt_weather_templated.txt; printf x); P=${P%x}
./genie-t2t-run -c genie_dialog_qwen3vl_4b.json -p "$P" --profile k3_timing.json
```

Report **init**, **TTFT**, **decode tok/s**, plus the wall-clock of K2a from its
log. **Do not average cold and warm runs** — report them separately and say which
was which.

---

## Things that will waste your time

* **`--max-num-tokens` is not a flag.** It is the dialog-config key
  `dialog.max-num-tokens` (`Dialog.cpp:2493`). Passing it as a flag aborts the
  run. It is already set to 64 in the lutprobe config and in
  `configs/genie_dialog_qwen3vl_4b.json`; if the on-device 4B config predates
  that, add `"max-num-tokens": 64,` right after `"type": "basic",`. Without a cap
  the query ends in `Context Size was exceeded`, which frees the handle and
  **writes no profile** — that cost all 4 profiles in the Test J session.
* **`--profile` refuses an output file that already exists** and aborts with
  `Invalid --profile argument`. Fresh name on every re-run.
* **Do not use `genie_dialog.json` for the control.** That is the LADE config;
  `genie_dialog_basic.json` is the like-for-like one.
* **Do not "fix" the LUT probe's prompts by templating them.** Raw is correct for
  the 0.6B (§K1c).
* **Do not edit the JSON configs to enable debugging.** `debug-tensors` and
  friends are compiled into `libGenie.so`, but `Engine.cpp` validates against a
  strict whitelist and throws `Unknown QnnHtp config key`.
* **Do not feed a `*_u16.raw` image.** Genie stages the file as float32
  regardless of tensor dtype; every image must be `*_fp32.raw` and **6,295,552
  bytes**.
* **Do not re-push the ViT bin or the LUT** if they are already on the board.

---

## What to send back

Fill in `RESULTS_TEMPLATE.md` and attach:

1. **All four K1 transcripts, verbatim** — `k1a_short.txt`, `k1a_long.txt`,
   `k1b_short.txt`, `k1b_long.txt`. Exact characters are the measurement; the
   whole result is a diff between two of these files, so a paraphrase destroys it.
2. The K1 profiles.
3. `k2a_sample.log` and the six `k2b_wx_*.log`, plus **one line of human
   judgement per image** — the only measurement here a script cannot make.
4. `k3_timing.json`.
5. The `md5sum`s you actually ran (§K1a, §K2). One line each, and they make every
   other number interpretable.

### If time runs short

1. **K1a + K1b short prompt** — two commands, and they answer the session's
   question on their own
2. K1a + K1b long prompt — more tokens to diff, so a sharper divergence point
3. K2b — the six photographs
4. K2a, K3

**Stopping after the two short-prompt runs is a complete result.** Stopping with
K1a alone is not — without the control it cannot be read (§K1c).
