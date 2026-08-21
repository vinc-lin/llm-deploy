# Runbook — the complete Qwen3-VL-4B on SA8797P, end to end

**Image input → image analysis → text generation.** Run this **after Test I's
templated prompt comes back coherent** (`docs/TEST_I_templated_prompt.md`). If
Test I is garbage, stop: this runbook will not work and the failure is upstream.

**Board time ≈ 25 minutes.** No host toolchain, no rebuild.

---

## 0. Two traps that will cost you the session — check both first

### 0a. Use the right folder. `qwen3vl_4b_e2e_pipeline_v5/` shipped a STALE shard 0.

Shard 0 (`_1_of_2`) is the bin carrying `inputs_embeds` — the one tensor the
`uFxp_16` fix changed. The e2e-pipeline folder shipped the **pre-fix** copy.

```sh
md5sum qwen3vl-4b-w8a16_1_of_2.bin qwen3vl-4b-w8a16_2_of_2.bin
```

| bin | REQUIRED md5 | the stale one |
|---|---|---|
| `qwen3vl-4b-w8a16_1_of_2.bin` | `f031e3a7563bf16f2d5ca98a71b357f6` | ❌ `065056baf6db142aa318ec0cc5662d42` |
| `qwen3vl-4b-w8a16_2_of_2.bin` | `0f1c86e89752b499eec09e9e10a73014` | — (this one was always right) |

**Use `qwen3vl_v5_session/03_vl4b_v5/`.** It is a complete pipeline bundle — ViT
bin, `genie-app`, the templated segment files, the fp32 image blobs, the weather
kit, LUT, tokenizer, libs — *and* it carries the fixed shard 0. If you already
pushed it for Test 3, it is already on the board.

If your shard 0 is `065056ba…`, everything below will reproduce the old garbage
and tell you nothing.

### 0b. Never feed a `*_u16.raw` image.

Genie stages the image file as **float32** regardless of the tensor's dtype, so a
UFixed16 blob is read at 2× its size — a ~3 MB over-read that lands in the guard
page as `SIGSEGV (SEGV_ACCERR)`. The `*_u16.raw` files exist for `qnn-net-run`
triage only.

Every image you feed `genie-app` must be `*_fp32.raw` and **6,295,552 bytes**:

```sh
ls -l *_fp32.raw | awk '{print $5, $9}'      # every one must read 6295552
```

---

## 1. Install

```sh
adb push qwen3vl_v5_session/03_vl4b_v5 /data/local/tmp/v5
adb shell
cd /data/local/tmp/v5
chmod +x genie-app genie-t2t-run qnn-net-run
export LD_LIBRARY_PATH=.

md5sum qwen3vl-4b-w8a16_1_of_2.bin | cut -c1-32     # must be f031e3a7...
ls -l sample_image_fp32.raw | awk '{print $5}'      # must be 6295552
```

~2.6 GB of it is the embedding LUT and ~4.5 GB the two text bins, so the push
takes a few minutes. Do not re-push if it is already there.

---

## 2. Step 1 — text generation alone (2 min)

Prove the tower before adding the image. **Templated**, per Test I:

```sh
adb push qwen3vl_4b_testi_session/testi /data/local/tmp/testi     # host side, once

P=$(cat /data/local/tmp/testi/prompt_2plus2_templated.txt; printf x); P=${P%x}
./genie-t2t-run -c genie_dialog_qwen3vl_4b.json -p "$P" \
    --max-num-tokens 64 --profile e2e_text.json
```

**Pass:** the answer is `4` (or a short sentence containing it), and the profile
reports **20** prompt tokens.

If the token count is much larger, Genie's tokenizer split `<|im_start|>` instead
of matching the added token — stop and report that; it is a tokenizer/config
issue, not a model one.

**Always pass `--max-num-tokens`.** `genie-t2t-run` only flushes its profile when
a query completes cleanly; a run that hits `Context Size was exceeded` frees the
handle and writes nothing. That cost 7 of 8 profiles in an earlier session.

---

## 3. Step 2 — the full pipeline: image in, caption out (5 min)

This is the end-to-end path: **ViT encodes the image → its features are spliced
onto the image-token positions → the text tower generates.**

```sh
./genie-app -s genie_pipeline_qwen3vl.script 2>&1 | tee e2e_pipeline.log
```

The script needs no prompt argument: it already sets `prompt_seg1.txt` and
`prompt_seg2.txt` via `node set textFile`, which is **already chat-templated** —
`prompt_seg1.txt` begins `<|im_start|>user\n<|vision_start|>` with a real
newline, and the two segments are verified token-exact against
`processor(text, image).input_ids` (273 tokens, 256 `<|image_pad|>`). Nothing
about the image path needs changing for Test I's fix.

**Pass:** a caption that describes `sample_image.png`. Open that PNG and judge it
yourself — a fluent sentence about the wrong picture is a *failure*, and only a
human looking at both can tell.

**Expect ~11 s** for the run (the v4 session measured 3 cold/warm runs at
~11.1 s). First run after boot is slower.

---

## 4. Step 3 — six real photographs (10 min)

The sample image is synthetic. This is the first test on real camera data.

```sh
for f in wx_*.script; do
  s=${f%.script}
  echo "===== $s ====="
  ./genie-app -s "$f" 2>&1 | tee "e2e_$s.log"
done
```

Globbed rather than listed, so it runs whatever the bundle actually ships. As of
v5 that is six: `wx_clear`, `wx_clear2`, `wx_clear_snow`,
`wx_fog_overcast_rain`, `wx_snow`, `wx_snow2`.

Each script is the same pipeline with a different `*_fp32.raw`. **Pass:** six
captions, each recognisably about its own photograph. Compare against the `.jpg`
files by eye.

This is the step that finally answers *"can it actually describe a photo?"* —
which no run so far has been able to reach, because every previous attempt was
behind one of the defects now fixed.

---

## 5. Step 4 — timing (5 min)

Run this whatever the captions look like: decode rate is the same compute whether
the tokens are right or wrong, and there is still no init/TTFT/decode number for
a 4B two-shard W8A16 tower on this silicon.

```sh
P=$(cat /data/local/tmp/testi/prompt_weather_templated.txt; printf x); P=${P%x}
./genie-t2t-run -c genie_dialog_qwen3vl_4b.json -p "$P" \
    --max-num-tokens 128 --profile e2e_timing.json
```

From the profile, report **init**, **TTFT**, and **decode tok/s**, plus the
pipeline wall-clock from §3. Do not average cold and warm runs.

---

## 6. Pass / fail, and what each outcome means

| Step 1 text | Step 2 caption | Step 3 photos | Meaning |
|---|---|---|---|
| ✅ | ✅ | ✅ | **The deployment works.** Report everything; the remaining work is speed, not correctness |
| ✅ | ✅ | mixed | text + splice work; caption *quality* on real photos is the open question — a W8A16 4B result, not a bug per se |
| ✅ | garbage/empty | — | text is fixed and the **image splice** is a separate, now-isolatable defect. This is a good outcome: it was invisible until text worked |
| ✅ | SIGSEGV | — | you fed a `*_u16.raw`, or a blob that is not 6,295,552 bytes. See §0b |
| garbage | — | — | you are not running the fixed shard 0, or Test I did not actually pass. Re-check §0a |

---

## 7. What to send back

1. All generated text — **verbatim, garbage included**. Exact characters are
   evidence; do not paraphrase or summarise as "garbage".
2. The three profile JSONs (`e2e_text`, `e2e_timing`, plus any others).
3. `e2e_pipeline.log` and the six `e2e_wx_*.log`.
4. The `md5sum` of the shard-0 bin you actually ran (§0a). One line, and it makes
   every other number interpretable.
5. For Step 3, a one-line human judgement per photo: *does the caption match the
   picture?* That is the only measurement here a script cannot make.

---

## 8. Things that will waste your time

* **Do not edit the JSON configs to enable debugging.** `debug-tensors` and
  friends are compiled into `libGenie.so`, but `Engine.cpp` validates the public
  config against a strict whitelist and throws `Unknown QnnHtp config key`. It
  will not load.
* **Do not use `node set text`** for a prompt that needs a newline. `genie-app`
  script strings never unescape: `"\n"` yields the two characters `\` and `n`.
  The shipped scripts already use `textFile`; keep it that way.
* **Do not use a bare `$(cat prompt.txt)`** for the templated prompts —
  substitution strips the trailing newline the template ends with. Use the
  `printf x` form above (verified: 86 bytes vs 85).
* **Do not re-push the ViT bin or the LUT** if they are already on the board;
  they are most of the transfer and have not changed since v3.
* **Do not run the `_decodeonly` pipeline script** unless asked. It exists to
  bypass prefill and is untested on device; TTFT would be ~30 s instead of ~3–4 s.
