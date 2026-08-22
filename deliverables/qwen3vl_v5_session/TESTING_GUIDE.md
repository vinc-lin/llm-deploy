# Qwen3-VL-4B v5 — device testing guide

> ## 🔴 Revised 2026-08-21 — Test 3 changed. If you ran this guide before, re-run Test 3.
>
> **Test 3 was sending the 4B a prompt form it was never calibrated for.** The
> 4B is calibrated exclusively on chat-templated prompts; this guide told you to
> send raw text. That single mismatch puts the attention sink at row 0, out of
> the calibrated range, and corrupts position 0 in both shards — which is very
> likely the whole of "the 4B text tower produces garbage".
>
> Three changes:
> 1. **Test 3a now sends a chat-templated prompt**, with the raw command kept
>    beside it as a control. The contrast is the result.
> 2. **Test 3b is no longer gated on 3a.** That gate is why the image path has
>    not been tested since the `uFxp_16` fix — and the image path was *always*
>    correctly templated, so it may simply work now.
> 3. **Tests 1 and 2 (0.6B) stay raw on purpose** — opposite calibration
>    contract. Do not "fix" them to match Test 3.
>
> Everything else is unchanged. Background:
> `ROOTCAUSE_qwen3vl_4b_boundary_gain.md`; standalone one-minute version:
> `TEST_I_templated_prompt.md`.

**Read this file top to bottom and run the tests in the order given.** Each test
states what to run, what result would mean what, and exactly what to send back.
Total board time ≈ **35 minutes**. Nothing here needs a host toolchain.

Everything you need is in this folder. Paths in the commands are relative to the
folder you push to the device.

**Also in this folder:** `ISSUE_qwen3vl_4b_text_numerics.md` — what the previous
session settled, what it did not, and a free re-analysis of the data you already
have. Read it if you ran the v5 session.

---

## 0. What changed since v4, in one paragraph

v4 fixed the image crash (confirmed on your side: 3 runs, no SIGSEGV, 6
photographs). The remaining defect is the **text tower producing garbage**. We
believe we have found the mechanism, in Genie's own embedding-staging code
rather than in the model, and v5 is built to *prove or kill that theory in the
first five minutes* and then test the fix.

**The theory, stated plainly.** When Genie fills the `inputs_embeds` tensor from
the float32 embedding LUT it calls `QnnNspModel::quantizeInput`. That function
takes a `tensorOffset` argument which it treats as an **element** offset for
`UFIXED_8`, `UFIXED_16` and `FLOAT_32`, but as a **byte** offset for
`FLOAT_16`:

```cpp
case QNN_DATATYPE_UFIXED_POINT_16:  reinterpret_cast<uint16_t*>(buf) + tensorOffset  // elements
case QNN_DATATYPE_FLOAT_32:         reinterpret_cast<float*>(buf)    + tensorOffset  // elements
case QNN_DATATYPE_FLOAT_16:         reinterpret_cast<uint8_t*>(buf)  + tensorOffset  // BYTES  <-- 
```

`setupInputEmbeddings` passes an **element** count (`i * m_embd_size`) when it
pads a partially-filled prefill chunk with the pad/EOS embedding. Our
`inputs_embeds` is `FLOAT_16`. So the padding write starts at byte
`n_process × n_embd` instead of `n_process × n_embd × 2` — that is **halfway
into the real prompt** — and overwrites the back half of it with pad vectors.

**The fix** is to stop that tensor being `FLOAT_16`. We give it a 16-bit INT
activation encoding so the converter types it `UFIXED_POINT_16`, which lands on
the correct element-offset branch and is native to the HTP backend. (We first
tried forcing it to `FLOAT_32`; the converter then inserts a Convert op that
graph-prepare cannot create — `could not create op: q::QNN_Convert`, both
ctx-bins failed to finalize. `UFIXED_POINT_16` is also how the ViT gets its I/O,
and that path is already device-proven.)

This predicts something very specific, which is what Test 1 measures.

> **Caveat, stated honestly.** This is read from the SDK's `qualla` sources,
> which are *not* the source of the shipped `libGenie.so`. It is a strong
> hypothesis, not proof. Test 1 is what settles it, and it costs one bundle you
> already have on the device.

---

## Test 1 — the chunk-boundary triad (5 min) ⭐ **most important test**

**Bundle:** `01_probe_06b_fp16in/` — a Qwen3-0.6B built on the device-proven
44.707 tok/s recipe with exactly one thing changed: it is fed from an external
embedding LUT instead of token ids. It is **expected to be broken.** We are not
testing whether it works; we are testing *when* it breaks.

The prefill graph has AR = 128. The padding bug only runs when a chunk is
**partially** filled (`variant > n_process`). So:

| Prompt length | Chunking | Padding write? | Prediction if the theory is right |
|---:|---|---|---|
| **127 tokens** | one chunk, 127 of 128 | yes | **garbled** |
| **128 tokens** | one chunk, exactly full | **no** | **clean and coherent** |
| **129 tokens** | 128 + 1 | yes, on chunk 2 | **garbled** |

A model that is broken by *quantization*, *rope*, *the split*, or *encodings*
cannot produce "broken, clean, broken" as the prompt grows by one token. Nothing
else we know of produces that signature.

> ### ⚠ Tests 1 and 2 use RAW prompts, deliberately — do not template them
>
> Test 3 was changed to send the 4B a chat-templated prompt because the **4B**
> is calibrated on templated windows. **The 0.6B is the opposite**: its
> calibration used raw prompts (`CALIB_PROMPTS` in `quantize_aimet.py`), so raw
> is its in-distribution form and templating it would push it *out* of
> distribution — the exact mistake, mirrored.
>
> It also has to stay raw for the triad to mean anything: the prompts are
> length-controlled at exactly 127/128/129 tokens, and a template would add ~8
> tokens and destroy the boundary this test is built to find.
>
> Same reason the working 0.6B never pointed at the 4B's problem: same code,
> opposite input contract.

### Run it

```sh
adb push 01_probe_06b_fp16in /data/local/tmp/p16
adb push prompts /data/local/tmp/prompts
adb shell
cd /data/local/tmp/p16 && chmod +x genie-t2t-run
export LD_LIBRARY_PATH=.

for n in 127 128 129; do
  echo "===== fp16in prompt$n ====="
  ./genie-t2t-run -c genie_dialog_qwen3_0.6b_lutprobe.json \
      -p "$(cat /data/local/tmp/prompts/prompt$n.txt)" \
      --profile prof_fp16in_$n.json
done 2>&1 | tee /data/local/tmp/test1_fp16in.log
```

The three prompts are the **same sentence** trimmed/extended by one token, so
any difference in output quality is caused by length alone, not content.

### Reading it

| What you see | Meaning | Next |
|---|---|---|
| **127 garbled, 128 clean, 129 garbled** | theory **confirmed** | Test 2 confirms the fix; this is the good outcome |
| all three garbled | the padding bug is not the (only) cause | still run Tests 2–4; Test 5 becomes important |
| all three clean | the 0.6B LUT feed is fine; the 4B fault is elsewhere | skip to Test 3 |

**Send back:** `test1_fp16in.log`, the three `prof_fp16in_*.json`, and the
generated text for each prompt **verbatim, garbage included**. The exact garbage
characters matter — do not summarise them as "garbage".

---

## Test 2 — the same triad on the fixed build (5 min)

**Bundle:** `02_probe_06b_u16in/` — byte-for-byte the same model, same
recipe, same LUT. The **only** difference is that its `inputs_embeds` graph
input is declared `UFIXED_POINT_16` instead of `FLOAT_16`, which moves Genie
onto the element-offset branch shown above.

```sh
adb push 02_probe_06b_u16in /data/local/tmp/p32
adb shell
cd /data/local/tmp/p32 && chmod +x genie-t2t-run
export LD_LIBRARY_PATH=.

for n in 127 128 129; do
  echo "===== u16in prompt$n ====="
  ./genie-t2t-run -c genie_dialog_qwen3_0.6b_lutprobe_u16in.json \
      -p "$(cat /data/local/tmp/prompts/prompt$n.txt)" \
      --profile prof_u16in_$n.json
done 2>&1 | tee /data/local/tmp/test2_u16in.log

./genie-t2t-run -c genie_dialog_qwen3_0.6b_lutprobe_u16in.json \
    -p "What is 2+2? Answer with one number." --profile prof_u16in_short.json
```

### Reading it

| What you see | Meaning |
|---|---|
| **all three coherent** | the fix works. Combined with Test 1 this is a complete, proven story |
| still garbled | the uFxp_16 input is not sufficient; report and stop — do not try config edits |
| fails to load | capture the exact error and `adb logcat -d`; it is a contract bug in the probe |

**Send back:** `test2_u16in.log`, all four profiles, all generated text.

> Run Tests 1 and 2 **back to back in the same session on the same board.** The
> comparison is the measurement; either result alone is much weaker.

---

## Test 3 — the 4B text tower with the fix (10 min)

**Bundle:** `03_vl4b_v5/`. The two **text** ctx-bins are rebuilt with the same
one-tensor change as Test 2. The **ViT ctx-bin and the LUT are byte-identical to
v3/v4** — do not re-push them if you still have them (md5s in `MANIFEST.md`).

Run text-only first, then the image pipeline **regardless of what text-only
produced** — see the box after the commands. The old advice ("if text-only is
broken there is no point spending time on captions") is why the image path has
not been tested since the `uFxp_16` fix, and it was wrong: the two paths send
different prompt forms, so one failing says nothing about the other.

```sh
adb push 03_vl4b_v5 /data/local/tmp/v5
adb shell
cd /data/local/tmp/v5 && chmod +x genie-app genie-t2t-run qnn-net-run
export LD_LIBRARY_PATH=.

# 3a - text only, CHAT-TEMPLATED.  See the box below: this is not cosmetic.
#      The trailing newline after <|im_start|>assistant is part of the template
#      and a bare $(cat file) strips it, hence the `printf x` form.
P=$(cat /data/local/tmp/testi/prompt_2plus2_templated.txt; printf x); P=${P%x}
./genie-t2t-run -c genie_dialog_qwen3vl_4b.json -p "$P" --profile v5_t2t_1.json

P=$(cat /data/local/tmp/testi/prompt_weather_templated.txt; printf x); P=${P%x}
./genie-t2t-run -c genie_dialog_qwen3vl_4b.json -p "$P" --profile v5_t2t_2.json

# 3a-control - the SAME question raw, i.e. the command this guide used to give.
#              Expected to be garbage. Run it: the contrast is the evidence.
./genie-t2t-run -c genie_dialog_qwen3vl_4b.json \
    -p "What is 2+2? Answer with one number." --profile v5_t2t_raw.json

# 3b - full image pipeline.  RUN THIS REGARDLESS of what 3a produced.
./genie-app -s genie_pipeline_qwen3vl.script 2>&1 | tee v5_pipeline.log
```

> ### ⚠ Why 3a is templated now, and why 3b is no longer gated on it
>
> **The 4B is calibrated exclusively on chat-templated prompts.**
> `vl_calib_build.py` builds every calibration window with
> `apply_chat_template(..., add_generation_prompt=True)`, so the tower's
> activation ranges assume position 0 is `<|im_start|>` and the attention sink
> sits at row 1. The command this guide used to give — `-p "What is 2+2? …"` —
> is **raw**: `genie-t2t-run` passes `-p` verbatim and
> `genie_dialog_qwen3vl_4b.json` sets `bos-token: -1` with no prompt template,
> so nothing prepends anything. A raw prompt puts the attention sink at row 0,
> where calibration never saw one; it clips, and position 0's hidden state and
> KV are corrupted in both shards.
>
> Measured on the host, same question, template the only variable:
>
> | | sink | worst overshoot vs range | boundary row-0 gain |
> |---|---|---|---:|
> | raw | row 0 | `layers.0/mlp/down_proj` **1.64×** | **1.3505** |
> | templated | row 1 | 1.01× | **1.0001** |
>
> **So every "the 4B text tower is garbage" result in this project was measured
> on a prompt the model was never built for.** Full account:
> `ROOTCAUSE_qwen3vl_4b_boundary_gain.md`; standalone check:
> `TEST_I_templated_prompt.md`.
>
> **3b was previously gated on 3a, and that gate is why the image path has not
> been tested since the `uFxp_16` fix.** It should never have been: the image
> pipeline has *always* been correctly templated — `prompt_seg1.txt` begins
> `<|im_start|>user\n<|vision_start|>` with a real newline, and the segments are
> verified token-exact against `processor(text, image).input_ids`. Its only
> failures (garbage caption, EOS collapse) were measured in **v4, before** the
> `uFxp_16` fix, when the FLOAT_16 padding bug was live and corrupted the last
> partial chunk of the 273-token prompt. **The templated image path may simply
> work now. Run it.**
>
> **Check the token count in each profile before reading the text.** The
> templated 2+2 prompt must tokenize to **20** tokens. Much larger means Genie's
> tokenizer split `<|im_start|>` into pieces instead of matching the added
> token — that is itself the finding.
>
> The prompt files ship in `qwen3vl_4b_testi_session/testi/`; push them with
> `adb push testi /data/local/tmp/testi`.

### Reading it

Read 3a (templated) against 3a-control (raw). **The contrast is the result**, not
either run alone.

| 3a templated | 3a-control raw | 3b caption | Meaning |
|---|---|---|---|
| **coherent** | garbage | coherent | **the deployment works.** The whole text defect was an input-contract violation. Go to Test 4 and report |
| **coherent** | garbage | garbage/empty | text confirmed fixed; the image splice is a separate, now-isolatable defect — valuable, and only visible once text is right |
| coherent | coherent | — | also fine, but something changed since v5 — send both profiles |
| garbage | garbage | — | the templating theory is wrong. Send both profiles **and the token counts**, then run **Test 5** |
| garbage | coherent | — | would invert everything known. Report immediately |

Expected: **templated coherent (`4` for the 2+2 prompt), raw garbage.**

**Send back:** both profiles, all generated text, `v5_pipeline.log`, and the
caption verbatim.

---

## Test 4 — timing decomposition (5 min)

Run this **whatever** Test 3 produced — decode rate is the same compute whether
the tokens are right or wrong, and we still have no init/TTFT/decode numbers for
a 4B two-shard W8A16 tower on this silicon.

**Always pass `--max-num-tokens`.** `genie-t2t-run` only flushes its profile JSON
when a query completes cleanly; if generation runs into
`Context Size was exceeded / Failed to query`, the handle is freed and the file
is never written. That cost 7 of ~8 profiles in the previous session, and it was
our guide's fault for not saying so.

```sh
cd /data/local/tmp/v5
for tag in cold warm1 warm2; do
  ./genie-t2t-run -c genie_dialog_qwen3vl_4b.json \
      -p "What is 2+2? Answer with one number." \
      --max-num-tokens 64 --profile v5_$tag.json
done

for i in 1 2 3; do echo "== pipeline run $i"; time ./genie-app -s genie_pipeline_qwen3vl.script; done 2>&1 | tee v5_timing.log
```

**Send the profile JSONs verbatim — do not summarise them.** Our parser reads
every numeric leaf; guessing key names is how a harness silently reports zeros.
Report **cold and warm separately, never averaged.**

---

## Test 5 — `qnn-net-run` text probe (10 min) — *only if Test 3a TEMPLATED was garbage*

> **The probe kit in this bundle was rebuilt on 2026-08-19.** The previous one
> was built for the older `FLOAT_16` ctx-bins and fed the current ones IEEE fp16
> where they expect `UFIXED_POINT_16` — the same byte count, so nothing errored,
> and every cosine collapsed toward zero. If you still have the previous
> session's `v5_probe_out.tar.gz`, **re-analysing it is free and comes first**:
> see `ISSUE_qwen3vl_4b_text_numerics.md` §4. The comparator now refuses to print
> a verdict when the kit and the ctx-bin disagree.

This runs the shipped text ctx-bins with **no Genie involved**, on inputs we
control, against references computed on the host. It answers "is the ctx-bin
itself right?" — which is the question left over if the Genie-side fix did not
help.

```sh
cd /data/local/tmp/v5
sh run_text_probe.sh 2>&1 | tee text_probe.log
```

Do **not** judge it on the device. Pull the whole `text_probe_out/` directory
back (it is a few MB); the host compares it:

```bash
$PY_DEPLOY scripts/validate/compare_text_probe.py \
    --kit <bundle>/03_vl4b_v5 --results text_probe_out \
    --ctxbin-info-0 <bundle>/03_vl4b_v5/qwen3vl-4b-w8a16_1_of_2.info.json \
    --ctxbin-info-1 <bundle>/03_vl4b_v5/qwen3vl-4b-w8a16_2_of_2.info.json
```

It prints `kit/bin encoding cross-check: OK` before any numbers. If it does not,
stop and report — the numbers below it would be meaningless.

**Report `shard1-isolated` separately from `shard1-chained`.** They answer
different questions and only the pair localises a fault.

If `qnn-net-run` rejects a flag name, run `./qnn-net-run --help` and report what
it calls its native input/output file options. **Do not** fall back to feeding
the blobs as float32.

**Send back:** `text_probe.log` and the entire `text_probe_out/` directory.

---

## Collecting everything

```sh
cd /data/local/tmp
sh /data/local/tmp/v5/collect.sh          # writes /data/local/tmp/v5_results.tar.gz
exit
adb pull /data/local/tmp/v5_results.tar.gz .
```

Then fill in `RESULTS_TEMPLATE.md` and send it with the tarball.

If anything crashes at any point, also send `adb logcat -d` from around the
crash and the tombstone from `/data/tombstones/` if one was produced.

---

## Quick reference — what each bundle is

| Folder | Model | `inputs_embeds` | Expected |
|---|---|---|---|
| `01_probe_06b_fp16in/` | Qwen3-0.6B, LUT-fed | `FLOAT_16` | **broken** (that is the point) |
| `02_probe_06b_u16in/` | Qwen3-0.6B, LUT-fed | `UFIXED_POINT_16` | coherent |
| `03_vl4b_v5/` | Qwen3-VL-4B, 2-shard | `UFIXED_POINT_16` | coherent text + caption |

The two 0.6B bundles differ **only** in that one declared dtype. Same weights,
same encodings, same LUT, same graph shapes, same HTP config. Verify with
`MANIFEST.md` if you want to confirm it before trusting the comparison.

## Things that will waste your time

* **Do not edit the JSON configs to enable debugging.** `debug-tensors` and
  friends are compiled into `libGenie.so` but `Engine.cpp` validates the public
  config against a strict whitelist and throws `Unknown QnnHtp config key`. It
  will not load and you will have burned a slot.
* **Do not re-push the ViT ctx-bin or the LUT** for Test 3 — they are unchanged
  since v3, and they are most of the download.
* **Do not average cold and warm timings.**
* **Do not paraphrase generated garbage.** The exact characters are evidence.
* **Do not "fix" Tests 1/2 by templating their prompts.** The 0.6B is
  calibrated on raw prompts and its triad is length-controlled to the token —
  templating would break both. Only Test 3 (the 4B) templates. See the box in
  Test 1.
* **Do not use a bare `$(cat prompt.txt)`** for the templated prompts. Command
  substitution strips the trailing newline that the template ends with; use the
  `printf x` form given in Test 3 (verified: 86 bytes vs 85).
* **Do not skip 3b because 3a failed.** That gate is removed, and it is why the
  templated image path has never been run post-fix.
