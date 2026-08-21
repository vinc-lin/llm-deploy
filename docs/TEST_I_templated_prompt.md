# Test I — ask the 4B the question in the form it was built for

**Status:** ready to run · **Opened:** 2026-08-21 · **Needs:** no rebuild, no new
ctx-bins. **The core check is one minute of device time.**
Audience: device team + build side. Self-contained; no prior thread needed.

---

## 1. What we got wrong, and what it means

Every 4B text test this project has run — including the one behind "4B text-only
output remains unintelligible" — feeds a **raw, untemplated** prompt:

```sh
./genie-t2t-run -c genie_dialog_qwen3vl_4b.json \
    -p "What is 2+2? Answer with one number."          # v5 guide, Test 3a
```

`genie-t2t-run` passes `-p` verbatim, and `genie_dialog_qwen3vl_4b.json` sets
`bos-token: -1` and carries **no prompt template**, so nothing prepends
anything. The model sees a bare content word at position 0.

But the 4B was **calibrated exclusively on chat-templated prompts** —
`vl_calib_build.py` builds every window with `apply_chat_template(...,
add_generation_prompt=True)`. So its activation ranges are fitted to a
distribution where position 0 is always `<|im_start|>` and the attention sink
sits at **row 1**. Feed it raw and the sink forms at **row 0** instead, in a
place the calibration never saw. It clips, and position 0's hidden state and KV
are corrupted in both shards — after which every later token attends to a
poisoned sink.

**The 4B has never been asked a question in the form it was built for.**

The 0.6B is coherent on the very same raw prompts because *its* calibration used
raw prompts (`CALIB_PROMPTS`). Same code, opposite input contract — which is why
"the 0.6B works" never pointed at this.

---

## 2. The controlled measurement (host, already done)

Same question, `"What is 2+2? Answer with one number."`, template the only
variable:

| | row 0 | row 1 | where the sink is |
|---|---:|---:|---|
| **raw** (12 tokens, row 0 = `3838 'What'`) | RMS **107.202** | 2.256 | **row 0** |
| **templated** (20 tokens, row 0 = `151644 '<\|im_start\|>'`) | 2.072 | RMS **220.330** | row 1 |

Tensors exceeding their calibrated range on the sink row:

| | worst overshoot |
|---|---|
| **raw** | `layers.0/mlp/down_proj` **9.00 vs 5.482 = 1.64×** (plus four more) |
| **templated** | **1.01×** — nothing meaningfully out of range |

Resulting shard-0 boundary, with all 180 calibrated ranges clamped:

| | row-0 gain | verdict |
|---|---:|---|
| **raw** | **1.3505** | inside [1.25, 3.0], the band that reproduces the device's wrong argmax 105196 |
| **templated** | **1.0001** | clean; every row ≤ 1.0011 |

This is the whole defect, with one variable, measured.

---

## 3. The device check — one minute

The prompt must contain **real newlines**. `$(cat file)` strips trailing
newlines and the template ends with one, so use the `printf x` trick:

```sh
adb push testi /data/local/tmp/testi
adb shell
cd /data/local/tmp/v5 && export LD_LIBRARY_PATH=. && chmod +x genie-t2t-run

# A — the templated prompt (THE test)
P=$(cat /data/local/tmp/testi/prompt_2plus2_templated.txt; printf x); P=${P%x}
./genie-t2t-run -c genie_dialog_qwen3vl_4b.json -p "$P" --profile testi_templated.json

# B — the same question raw, as a control (this is the v5 Test 3a command)
./genie-t2t-run -c genie_dialog_qwen3vl_4b.json \
    -p "What is 2+2? Answer with one number." --profile testi_raw.json

# C — a second templated prompt, longer answer
P=$(cat /data/local/tmp/testi/prompt_weather_templated.txt; printf x); P=${P%x}
./genie-t2t-run -c genie_dialog_qwen3vl_4b.json -p "$P" --profile testi_weather.json
```

**Check the token count in the profile before reading the text.** The templated
2+2 prompt must tokenize to **20 tokens**. If it comes out much larger, Genie's
tokenizer split `<|im_start|>` into pieces instead of matching the added token —
that is itself the finding, and the fix is a tokenizer/config issue rather than
anything numeric.

---

## 4. Reading it

| A (templated) | B (raw) | meaning |
|---|---|---|
| **coherent** | garbage | **confirmed.** The defect was an input-contract violation. Fix is templating; no rebuild. Go straight to the image pipeline |
| coherent | coherent | also good news, but then something else changed since v5 — report both profiles |
| garbage | garbage | the theory is wrong. Send both profiles and the token counts; the next instruments (Test H, and a Genie-vs-probe differential) are already built |
| garbage | coherent | would invert everything known — report immediately |

Expected: **A coherent, B garbage.** For 2+2 the answer should simply be `4`.

---

## 5. Optional, same session — the graph-level confirmation

`testi/` also ships a `qnn-net-run` kit with the two cases above, so the device
can reproduce §2's numbers directly rather than trusting the host:

```sh
adb push testi/kit/. /data/local/tmp/v5/
adb shell
cd /data/local/tmp/v5 && export LD_LIBRARY_PATH=. && chmod +x qnn-net-run
sh run_text_probe.sh 2>&1 | tee text_probe_i.log
exit
adb pull /data/local/tmp/v5/text_probe_out ./text_probe_out_i
```

Then on the host:

```bash
$PY_DEPLOY scripts/validate/analyze_realistic_probe.py \
    --kit host_refs --results ./text_probe_out_i
```

Expect `i1_templated` clean within ±5% and `i0_raw` off by roughly 35% on row 0.
⚠ Note the analyzer's headline verdict is written for all-clean kits, so it will
report "BOUNDARY IS OFF" here — **that is the expected result for `i0_raw`**.
Read the per-case rows, not the summary line.

---

## 6. What to send back

1. The generated text for A, B and C, **verbatim, garbage included**.
2. The three profile JSONs (for the token counts).
3. If you ran §5: `text_probe_out_i/` and `text_probe_i.log`.

---

## 7. If A is coherent, what happens next

The fix is a prompt-format change, not a rebuild:

* `genie-t2t-run` callers template the prompt (as in §3);
* the pipeline script already needs `node set textFile` for the image path
  because `genie-app` script strings never unescape `\n` — that file must carry
  the templated prompt with real newlines;
* optionally, if raw prompts must *also* work, add untemplated windows to
  `vl_calib_build.py` and requantize. That is the one case where
  `--act-range-scheme` matters. Not needed for a chat application.

Then the remaining end-to-end run is: templated text-only → sample image →
the 6-photo weather kit.
