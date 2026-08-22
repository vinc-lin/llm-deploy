# Device session protocol — how to run each test, and how to record it

Companion to **`TEST_J_decode_step.md`**, which says *what* the tests are and
*why*. This document is the *how*: exact commands in order, what to capture from
each, and how to write it down so it is usable.

> **Revised 2026-08-21 after the Test J session.** Three things in the first
> version were wrong and are fixed below: `--max-num-tokens` is not a CLI flag
> (§3), `--profile` refuses an existing file (§3), and Stage C's first-word test
> could not discriminate with the prompt it used (§4). Test J itself is complete
> — results and analysis in `reports/qwen3vl-4b-testj-device-results-2026-08-21.md`.

**Total ≈ 30 minutes.** Nothing here needs a host toolchain. Every command is
copy-paste; every step names the artefact it produces.

---

## 0. Before you start — five minutes that protect the whole session

```sh
adb push qwen3vl_v5_session/03_vl4b_v5 /data/local/tmp/v5     # if not already there
adb push qwen3vl_4b_testi_session/testi /data/local/tmp/testi
adb shell
cd /data/local/tmp/v5
chmod +x genie-app genie-t2t-run qnn-net-run
export LD_LIBRARY_PATH=.
```

Then run these four checks and **write the output into your results file before
anything else.** Every later number is uninterpretable without them.

```sh
md5sum qwen3vl-4b-w8a16_1_of_2.bin | cut -c1-32   # MUST be f031e3a7563bf16f2d5ca98a71b357f6
md5sum qwen3vl-4b-w8a16_2_of_2.bin | cut -c1-32   # MUST be 0f1c86e89752b499eec09e9e10a73014
ls -l sample_image_fp32.raw | awk '{print $5}'    # MUST be 6295552
getprop ro.build.fingerprint 2>/dev/null; date
```

| check | why it matters |
|---|---|
| shard-0 md5 | `qwen3vl_4b_e2e_pipeline_v5/` shipped a **stale, pre-fix shard 0**. If yours is `065056ba…` everything below reproduces old garbage |
| shard-1 md5 | confirms the pair |
| image size | a `*_u16.raw` (or any other size) is a guaranteed `SIGSEGV`, not a model result |
| date / build | so a later session can tell two runs apart |

**If a check fails, stop and report that.** A failed precondition *is* a result,
and it is worth more than a run made on the wrong bytes.

---

## 1. Stage A1 — decode-step probe

```sh
# host side, once:
cat testj/past_kv.tar.gz.part-* > testj/past_kv.tar.gz
md5sum -c testj/past_kv.tar.gz.md5
tar xzf testj/past_kv.tar.gz -C testj/
adb push testj/. /data/local/tmp/v5/

# device:
cd /data/local/tmp/v5 && export LD_LIBRARY_PATH=.
sh run_text_probe.sh 2>&1 | tee text_probe_j.log

# host:
adb pull /data/local/tmp/v5/text_probe_out ./text_probe_out_j
```

**Capture:** `text_probe_j.log`, the whole `text_probe_out_j/` directory.

**Record:** for each of `j0_2plus2_s1`, `j1_weather_s1`, `j2_weather_s2`, the
`logits chained` **argmax match** line. Expected `151645`, `9104`, `4344`.

> ⚠ The three `cat`/`md5sum`/`tar` lines are not optional — all three cases feed
> a real KV cache. The runner size-checks every file and stops with a clear hint
> if one is missing. **If it stops, that message is the result.**

---

## 2. Stage A2 — cross-chunk prefill

Same shape, different kit. **Run it in its own directory** so the two probes'
outputs do not overwrite each other.

```sh
# host:
cat testh/past_kv.tar.gz.part-* > testh/past_kv.tar.gz
md5sum -c testh/past_kv.tar.gz.md5
tar xzf testh/past_kv.tar.gz -C testh/
adb push testh/. /data/local/tmp/v5h/          # NOTE: v5h, not v5

# device: the kit needs the bins and libs, so link or copy them in
cd /data/local/tmp/v5h && export LD_LIBRARY_PATH=/data/local/tmp/v5
cp /data/local/tmp/v5/qnn-net-run /data/local/tmp/v5/netrun_htp_config.json .
cp /data/local/tmp/v5/qwen3vl-4b-w8a16_*.bin .
chmod +x qnn-net-run
sh run_text_probe.sh 2>&1 | tee text_probe_h.log

# host:
adb pull /data/local/tmp/v5h/text_probe_out ./text_probe_out_h
```

If disk is tight, run A1 fully, pull it, `rm -rf text_probe_out`, then A2 in the
same directory instead — just do not mix the outputs.

**Capture:** `text_probe_h.log`, `text_probe_out_h/`.
**Record:** per-row **gain** for `c0_chunk0`, `c1_chunk1`, `c2_chunk2`.

---

## 3. Stage B — Genie text

**First, cap generation — it is a config key, not a flag.** Confirm the shipped
config has it, and add it if not:

```sh
grep max-num-tokens genie_dialog_qwen3vl_4b.json    # want: "max-num-tokens": 64,
```

If missing, insert the line `"max-num-tokens": 64,` immediately after
`"type": "basic",` inside the `"dialog"` block. No rebuild — Genie reads the JSON
at load.

```sh
cd /data/local/tmp/v5 && export LD_LIBRARY_PATH=.

# B1 templated 2+2  -- the trailing newline matters, hence `printf x`
P=$(cat /data/local/tmp/testi/prompt_2plus2_templated.txt; printf x); P=${P%x}
./genie-t2t-run -c genie_dialog_qwen3vl_4b.json -p "$P" \
    --profile b1_templated.json 2>&1 | tee b1_templated.txt

# B2 templated weather
P=$(cat /data/local/tmp/testi/prompt_weather_templated.txt; printf x); P=${P%x}
./genie-t2t-run -c genie_dialog_qwen3vl_4b.json -p "$P" \
    --profile b2_weather.json 2>&1 | tee b2_weather.txt

# B3 raw control -- expected to be wrong from the first token
./genie-t2t-run -c genie_dialog_qwen3vl_4b.json \
    -p "What is 2+2? Answer with one number." \
    --profile b3_raw.json 2>&1 | tee b3_raw.txt
```

**Capture:** the three `.txt` transcripts and three `.json` profiles.

**Record, per run:**
* the generated text **verbatim** — every character, including garbage;
* the **prompt token count** from the profile (B1 must be **20**; if it is much
  larger, Genie's tokenizer split `<|im_start|>` and *that* is the finding);
* where the output first diverges from expected — for B1 the correct answer is
  `4` then stop; for B2 it is `Mountain weather changes quickly because …`.

> **`--max-num-tokens` is NOT a flag.** An earlier version of this document said
> to pass it; `genie-t2t-run` rejects it. It is the dialog-config key
> `dialog.max-num-tokens` (`Dialog.cpp:2493`, read at `:2888`). Capping matters
> because the profile is only flushed when a query completes cleanly — a run that
> hits `Context Size was exceeded` frees the handle and writes nothing. That cost
> 7 of 8 profiles in one session and all 4 in the Test J session.

> **`--profile` refuses a file that already exists** (`main.cpp:528-533`) and
> aborts with `Invalid --profile argument`. Use a fresh name on every re-run, or
> delete the old one first.

> **`-tok` / `--tokens_file` exists** and takes explicit token ids. Prefer it when
> the question is *"did Genie tokenize this the way we think?"* — it removes the
> tokenizer from the experiment instead of inferring its behaviour from a
> prompt-token count.

> **Do not paraphrase the garbage.** `ention ably ance` and `aged aged aged` are
> different findings. The exact characters are the measurement.

---

## 4. Stage C — the image pipeline

```sh
cd /data/local/tmp/v5 && export LD_LIBRARY_PATH=.

# C1 sample image
./genie-app -s genie_pipeline_qwen3vl.script 2>&1 | tee c1_pipeline.log

# C2 six photographs
for f in wx_*.script; do
  s=${f%.script}
  echo "===== $s ====="
  ./genie-app -s "$f" 2>&1 | tee "c2_$s.log"
done
```

**Capture:** `c1_pipeline.log` and every `c2_wx_*.log`.

**Record, per image — and this is the part only a human can do:**

> **Is the FIRST WORD of the caption about that picture?**

Generation is expected to degenerate after the first token until the decode
defect is fixed. That does not make this useless: the first token comes from
**prefill**, so a first word that matches the image proves the whole
**image → ViT → splice → prefill** path. Open the `.png`/`.jpg` and judge.

⚠ **The prompt must force a content word, or this test cannot discriminate.**
In the Test J session all seven images answered `A` — a perfectly good caption
opener that says nothing about any picture, so the run was uninterpretable either
way. The segment file must ask for **one word**:

> *Answer with one word: what is the dominant weather in this photo?*

Then the first token is `Snow` / `Fog` / `Clear` / `Rain` and it either matches
the photograph or it does not. Edit `prompt_seg2.txt` — prompt-file change only,
no rebuild. Keep the real trailing newline (`node set textFile`, never
`node set text`).

Use three values, and nothing vaguer:

| verdict | meaning |
|---|---|
| `RELEVANT` | the first word plausibly describes that image |
| `WRONG` | it is a confident word about something not in the image |
| `DEGENERATE` | it is not a word — punctuation, a fragment, or empty |

`WRONG` and `DEGENERATE` are different diagnoses. Do not merge them. And a
generic word that fits any image (`A`, `The`, `This`) is **none of the three** —
record it as `INCONCLUSIVE` and say the prompt needs fixing.

---

## 5. Stage D — timing

Set `"max-num-tokens": 128` in the dialog config for this run (§3), then:

```sh
P=$(cat /data/local/tmp/testi/prompt_weather_templated.txt; printf x); P=${P%x}
./genie-t2t-run -c genie_dialog_qwen3vl_4b.json -p "$P" --profile d1_timing.json
```

**Record:** init time, TTFT, decode tok/s, plus the wall-clock of C1 from its
log. **Do not average cold and warm runs** — report them separately and say
which was which.

---

## 6. Collect everything into one tarball

```sh
sh /data/local/tmp/v5/collect_j.sh
# host:
adb pull /data/local/tmp/testj_results.tar.gz .
```

The collector is deliberately tolerant: a stage you skipped simply contributes
nothing rather than failing the run. A half-full tarball beats a clean exit code
and no data.

---

## 7. How to write the results up

Fill in **`RESULTS_TEMPLATE.md`** and send it with the tarball. It is short on
purpose. Three rules make a report usable:

1. **Verbatim beats summary.** Paste generated text exactly. If it is long,
   paste the first 200 characters and say the rest repeats — but paste
   *something* literal.
2. **Say what you actually ran.** If you skipped a stage, changed a path, or a
   command failed and you retried differently, write that down. An unexplained
   gap costs a whole round-trip to resolve.
3. **Separate observation from interpretation.** "First token `4`, then
   `entionably…` repeating to context limit" is an observation. "Decode is
   broken" is an interpretation. Both are welcome — label which is which.

### What we can do with a partial session

If time runs out, the value order is:

1. **Stage A1** — decides graph vs Genie, and it is the whole point of this round
2. **Stage A2** — unblocks the image path
3. **Stage B1** (templated 2+2) — one command, confirms A against Genie
4. **Stage C1** — first-token check on the vision path
5. Stage C2, D

Stopping after A1 + B1 is a genuinely useful session. Stopping after the
preconditions in §0 with a failed md5 is *also* useful.
