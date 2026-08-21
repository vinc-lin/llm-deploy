# Test N session protocol — how to run each stage, and how to record it

Companion to **`TEST_N_final_e2e.md`**, which says *what* the stages are and
*why*. This document is the *how*: exact commands in order, the artefact each
produces, and how to write the results down so they are usable.

**Total ≈ 60 minutes for N1–N5.** Fill in `RESULTS_TEMPLATE.md` as you go, not
afterwards, and collect everything with `collect_n.sh` at the end.

Three rules make a report usable — they have each earned their place:

1. **Verbatim beats summary.** Paste generated text exactly, garbage included.
   `ention ably ance` and `aged aged aged` are different findings.
2. **Say what you actually ran** — skipped stages, changed paths, retries,
   commands that failed. An unexplained gap costs a whole round-trip.
3. **Score against the printed reference, never against intuition.** Twice a
   correct result has been recorded as a failure: greedy repetition on a raw
   0.6B prompt is what HF fp32 does, and `S`+garbage is the correct first token
   of `Sunny` plus a known decode bug. Every stage below prints its expected
   values — use them.

Runtime facts that keep biting:

| | |
|---|---|
| `--max-num-tokens` | **not a flag** — the config key `dialog.max-num-tokens`. Without it, a run that hits the context limit writes **no profile** |
| `--profile` | **refuses an output file that already exists.** Fresh name per run |
| `-tok` | takes a text file of whitespace-separated integer ids |
| prompts with newlines | `$(cat f; printf x)`-style capture; `node set textFile` in genie-app scripts |
| images | `*_fp32.raw`, exactly **6,295,552 B** — anything else is a SIGSEGV |

---

## 0. Preconditions — five minutes that protect the whole session

```sh
adb push qwen3vl_4b_testn_session/testn /data/local/tmp/v5/testn   # the N kit
adb shell
cd /data/local/tmp/v5 && export LD_LIBRARY_PATH=.
```

Write these into the results file **before anything else**:

```sh
md5sum qwen3vl-4b-w8a16_1_of_2.bin | cut -c1-32   # MUST be f031e3a7563bf16f2d5ca98a71b357f6
md5sum qwen3vl-4b-w8a16_2_of_2.bin | cut -c1-32   # MUST be 0f1c86e89752b499eec09e9e10a73014
grep max-num-tokens genie_dialog_qwen3vl_4b.json  # want "max-num-tokens": 64
md5sum testn/n1_2plus2_p21.tok | cut -c1-32       # MUST be 6f6050fd27555c7bb14c1e815aa1972f
getprop ro.build.fingerprint; date
```

**If a check fails, stop and report that.** A failed precondition *is* a result,
and three sessions have already been damaged by running the wrong bytes.

---

## 1. Stage N1a — the token ladder (~8 min)

Six runs, one per `.tok` file. Same command shape each time; only the file and
the output names change:

```sh
cd /data/local/tmp/v5 && export LD_LIBRARY_PATH=.
for t in n1_2plus2_p20 n1_2plus2_p21 n1_weather_p18 n1_weather_p19 \
         n1_weather_p20 n1_weather_p21; do
  echo "===== $t ====="
  ./genie-t2t-run -c genie_dialog_qwen3vl_4b.json \
      -tok "testn/$t.tok" --profile "$t.json" 2>&1 | tee "$t.txt"
done
```

**Capture:** six `.txt` transcripts and six `.json` profiles.

**Record, per run:** the **first generated token** (id if visible, text
otherwise, verbatim), and the **prompt-token count from the profile** — it must
equal the file's count (20/21/18/19/20/21).

| run | expect first | as text |
|---|---:|---|
| `n1_2plus2_p20` | 19 | `4` |
| **`n1_2plus2_p21`** | **151645** | **nothing — EOS immediately.** `[BEGIN]:` then `[END]` **is the PASS** |
| `n1_weather_p18` | 91169 | `Mountain` |
| `n1_weather_p19` | 9104 | ` weather` |
| `n1_weather_p20` | 4344 | ` changes` |
| `n1_weather_p21` | 6157 | ` quickly` |

> Every run except P21 is **expected to degenerate after its first word** —
> that is the known decode defect, not a finding. Judge the first token only,
> but paste the whole transcript anyway.

## 2. Stage N1b — the `-e` embedding run (~2 min)

```sh
./genie-t2t-run -c genie_dialog_qwen3vl_4b.json \
    -e testn/n1b_2plus2_p20_emb.raw --profile n1b_emb.json 2>&1 | tee n1b_emb.txt
```

**Record:** the transcript verbatim. Expected: `4` then the usual degeneration
(same as the text run). Anything else — a different first token, a different
garbage pattern, or a load/param error — paste it exactly; an error message here
is a finding about the embedding-query contract, not a botched run.

## 3. Stage N1c — the state dump (~5 min)

Edit `genie_dialog_qwen3vl_4b.json`: change `"max-num-tokens": 64` → `1`.
Then:

```sh
./genie-t2t-run -c genie_dialog_qwen3vl_4b.json \
    -tok testn/n1_2plus2_p20.tok --save state_p20.bin --profile n1c_p20.json \
    2>&1 | tee n1c_p20.txt
./genie-t2t-run -c genie_dialog_qwen3vl_4b.json \
    -tok testn/n1_2plus2_p21.tok --save state_p21.bin --profile n1c_p21.json \
    2>&1 | tee n1c_p21.txt
ls -l state_p20.bin state_p21.bin
```

**Restore the config to `64` afterwards.**

**Record:** the two file **sizes** (the single most important number in this
stage), and whether the save call printed any error. Then, host side:

```sh
adb pull /data/local/tmp/v5/state_p20.bin .
adb pull /data/local/tmp/v5/state_p21.bin .
```

Send both files back with the results — their format is undocumented and the
analysis happens on the build side. If `--save` fails or writes nothing, that
verbatim error is the entire deliverable of this stage.

## 4. Stage N2 — Test L (~15 min)

Follow **`TEST_L_ctxbin_vs_genie.md`** exactly as written — it is
self-contained (bundle `qwen3_06b_lutprobe/`, bin md5 **`9720e46e…`**, L0
Genie runs scored against the printed HF strings, then the L1/L2
`qnn-net-run` kit with its own runner and analyzer).

**Record:** into Test L's own section of the results template — the two L0
transcripts verbatim, and the L1/L2 analyzer output pasted whole.

## 5. Stage N3 — Test M (~10 min)

Follow **`TEST_M_split_reproduction.md`** — bundle `qwen3_06b_lutsplit/`
(shards `1f4dcd44…` / `11cabce4…`). ⚠ That folder holds only what is new: copy
the LUT, tokenizer, `genie-t2t-run` and `lib*.so` across from the Test L bundle
first — the doc shows the exact `cp`. Two runs, scored against the same HF
strings as L0.

**Record:** both transcripts verbatim + both shard md5s.

## 6. Stage N4 — the image first-word grid (~10 min)

The one-word prompt segments are in `qwen3vl_4b_testk_session/prompts/`.

```sh
cd /data/local/tmp/v5 && export LD_LIBRARY_PATH=.
cp prompt_seg2.txt prompt_seg2_ORIGINAL.txt          # keep the shipped one

# sample image, "what colour is the largest shape"
cp prompt_seg2_oneword_shape.txt prompt_seg2.txt
./genie-app -s genie_pipeline_qwen3vl.script 2>&1 | tee n4_sample.log

# six photographs, "what is the weather"
cp prompt_seg2_oneword_weather.txt prompt_seg2.txt
for f in wx_*.script; do
  s=${f%.script}; echo "===== $s ====="
  ./genie-app -s "$f" 2>&1 | tee "n4_$s.log"
done

cp prompt_seg2_ORIGINAL.txt prompt_seg2.txt          # restore
```

**Record, per image — first word only, one row each:**

| verdict | meaning |
|---|---|
| `RELEVANT` | the first word describes that image |
| `WRONG` | a confident word about something not in the image |
| `DEGENERATE` | not a word — punctuation, fragment, empty |
| `INCONCLUSIVE` | a word that fits any image (`A`, `The`, `This`) |

Reference answers (HF fp32, same prompts): sample → `blue`, though `Red` is
equally defensible (the two shapes' areas differ by 12%); `wx_clear`,
`wx_clear2` → `sunny` — **`S` + fragment counts as a correct first token**
(`Sunny` = `['S','unny']` and token 2 is the known decode defect);
`wx_clear_snow` → `cold` (ambiguous scene — snow *and* clear sky);
`wx_fog_overcast_rain` → `rainy`; `wx_snow`, `wx_snow2` → `snowy`.
Generation **will** degenerate after the first word until the decode fix lands.

## 7. Stage N5 — timing completion (~10 min)

1. **Reboot the board.** Then, cold and immediately again warm:

```sh
P=$(cat /data/local/tmp/testi/prompt_weather_templated.txt; printf x); P=${P%x}
./genie-t2t-run -c genie_dialog_qwen3vl_4b.json -p "$P" --profile n5_cold.json
./genie-t2t-run -c genie_dialog_qwen3vl_4b.json -p "$P" --profile n5_warm.json
```

2. **Pipeline wall-clock, measured** (any `time` builtin, or `date +%s` before
   and after):

```sh
time ./genie-app -s genie_pipeline_qwen3vl.script > /dev/null 2>&1   # warm
```

**Record:** init / TTFT / decode tok/s from both profiles, labelled cold/warm —
**never averaged** — and the pipeline wall-clock with its warm/cold state.

## 8. Stage N6 — post-fix confirmation (separate session)

Do not run in this session unless a decode fix has landed. When it has:
follow `RUNBOOK_e2e_qwen3vl_4b.md` §2–§5, and judge against the four gates in
`TEST_N_final_e2e.md` §N6. That session's pass is the project's definition of
done.

---

## 9. Collect and send back

```sh
sh /data/local/tmp/v5/collect_n.sh
# host:
adb pull /data/local/tmp/testn_results.tar.gz .
```

The collector is tolerant — a skipped stage contributes nothing rather than
failing the run. Send: the tarball, the two `state_*.bin` files (they are large;
if transfer is a problem, their **sizes and first 64 bytes** — `xxd -l 64` —
are the minimum), and `RESULTS_TEMPLATE.md` filled in.

### If time runs short

1. **N1a P21** — one command, the sharpest single measurement available
2. rest of N1a, then N1c
3. N3, then N2
4. N4, N5

Stopping after N1a alone is a genuinely useful session. Stopping at a failed
precondition in §0 is *also* useful — say which check failed and what it read.
