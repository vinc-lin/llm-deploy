# Test O session protocol — execution and recording

Companion to **`TEST_O_variable_matrix.md`** (the *what* and *why*; read it
first — especially §3, the decision tree). This is the *how*: commands in
order, the artefact each produces, what to write down.

**≈ 45 min for O1–O5; +25 min for O6 if a knob passes.** Fill in
`RESULTS_TEMPLATE.md` as you go. The three recording rules, as always:
**verbatim beats summary · say what you actually ran · score against the
printed reference, never intuition.**

---

## 0. Preconditions (5 min)

```sh
adb push qwen3vl_4b_testo_session/configs /data/local/tmp/v5/testo
adb push qwen3vl_4b_testo_session/run_o2_sweep.sh /data/local/tmp/v5/
adb shell
cd /data/local/tmp/v5 && export LD_LIBRARY_PATH=.

md5sum qwen3vl-4b-w8a16_1_of_2.bin | cut -c1-32   # MUST be f031e3a7563bf16f2d5ca98a71b357f6
md5sum qwen3vl-4b-w8a16_2_of_2.bin | cut -c1-32   # MUST be 0f1c86e89752b499eec09e9e10a73014
ls testn/n1_2plus2_p20.tok testn/n1_2plus2_p21.tok \
   testn/n1_weather_p18.tok testn/n1_weather_p19.tok      # all four must exist
ls testo/ | wc -l                                          # 10 configs
df /data/local/tmp | tail -1                               # want >= 20 MB free
getprop ro.build.fingerprint; date
```

**Record all of it before anything else.** A failed check is a result — stop
and report it.

---

## 1. Stage O1 — the four KV dumps (10 min)

The `o1_save` config has `max-num-tokens: 1` baked in — **no JSON editing**.

```sh
for t in n1_2plus2_p20 n1_2plus2_p21 n1_weather_p18 n1_weather_p19; do
  s=$(echo "$t" | sed 's/n1_2plus2_p/state_p/; s/n1_weather_p/state_w/')
  echo "===== $t -> $s ====="
  ./genie-t2t-run -c testo/genie_dialog_qwen3vl_4b_o1_save.json \
      -tok "testn/$t.tok" --save "$s" --profile "o1_$t.json" 2>&1 | tee "o1_$t.txt"
  ls -l "$s"/
done
```

**Record:** for each dump — the directory listing (file names + sizes) and the
`dialog.json` contents (`cat state_*/dialog.json`). Expected `kv-cache` sizes:

| dump | expected bytes |
|---|---:|
| `state_p20` | 3,097,168 |
| `state_p21` | 3,244,624 |
| `state_w18` | 2,802,256 |
| `state_w19` | 2,949,712 |

(Formula: 592 + 72 × n_past × 2048.) A different size is itself a finding —
report it and continue.

**Host side, immediately:**

```sh
for d in state_p20 state_p21 state_w18 state_w19; do
  adb pull /data/local/tmp/v5/$d ./$d
done
tar czf testo_state_dumps.tar.gz state_p20 state_p21 state_w18 state_w19
```

**Send `testo_state_dumps.tar.gz` (~12 MB) to the build side NOW, before
continuing** — the analysis (`parse_genie_kv_dump.py --diff`) runs while you do
O2–O5, and its result may change what O4 asks of you. Do **not** delete the
on-device state directories.

## 2. Stage O2 — the knob sweep (15 min)

```sh
cd /data/local/tmp/v5 && export LD_LIBRARY_PATH=.
sh run_o2_sweep.sh 2>&1 | tee o2_sweep.log
```

The script runs seven configs against the same 20 exact token ids and prints a
verdict per run:

* **PASS** = output `4` then stop (EOS honoured, 2 tokens).
* **FAIL** = `4` then the repetition loop.
* **`o2a_ctrl` must FAIL.** If it passes, stop — the board is not in the known
  state and the sweep means nothing.

**Record:** the verdict line per run (the transcripts are captured by the
script as `o2_*.txt`).

**If any run PASSED:** confirm it —

```sh
./genie-t2t-run -c testo/genie_dialog_qwen3vl_4b_<passing>.json \
    -tok testn/n1_weather_p18.tok --profile o2_confirm.json 2>&1 | tee o2_confirm.txt
# expect: "Mountain weather changes quickly because ..." -- a real sentence
```

— then **skip to Stage O6.** O3–O5 become optional documentation.

## 3. Stage O3 — Test M deconfounded (5 min)

On the lutsplit install (LUT/tokenizer/runner/libs copied from the Test L
bundle as before):

```sh
adb push qwen3vl_4b_testo_session/configs/genie_dialog_qwen3_06b_lutsplit_o3a_4bknobs.json \
         qwen3vl_4b_testo_session/configs/genie_dialog_qwen3_06b_lutsplit_o3b_nogswitch.json \
         /data/local/tmp/lutsplit/
adb shell
cd /data/local/tmp/lutsplit && export LD_LIBRARY_PATH=.
for c in o3a_4bknobs o3b_nogswitch; do
  echo "===== $c ====="
  ./genie-t2t-run -c "genie_dialog_qwen3_06b_lutsplit_${c}.json" \
      -p "What is 2+2? Answer with one number." \
      --profile "o3_${c}.json" 2>&1 | tee "o3_${c}.txt"
done
```

**Score against the HF reference:** correct output is
`' 2+2=4. 2+2=4. …'` — **the repetition IS correct** for a raw 0.6B prompt.

| what you see (o3a) | record as |
|---|---|
| ` 2+2=4. 2+2=4. …` | **CORRECT** — Test M's failure was the knobs |
| `4`-ish start, then fragments | **4B-SIGNATURE** — the split reproduced at 0.6B |
| garbage from token 0 (as before) | **UNCHANGED** — knob-independent |

## 4. Stage O4 — restore probes (5 min)

```sh
cd /data/local/tmp/v5 && export LD_LIBRARY_PATH=.
./genie-t2t-run -c testo/genie_dialog_qwen3vl_4b_o2a_ctrl.json \
    --restore state_w19 -p "" --profile o4_restore.json 2>&1 | tee o4_restore.txt
```

**Record verbatim** — including any error. Expected if restore+continue works:
the next tokens continue the weather answer (` weather changes …` would mean
decode-after-restore is *correct*, itself a major finding). If `-p ""` is
rejected, retry with `--tokens_file /dev/null`, and if that fails too, record
both errors — the attempt is the data.

Keep all four state dirs on device: if the build side returns a **patched**
state (`state_p20_patched`), the follow-up is
`--restore state_p20_patched` and one continued generation.

## 5. Stage O5 — evidence completion (5 min)

```sh
adb logcat -c    # clear, then one failing run:
./genie-t2t-run -c testo/genie_dialog_qwen3vl_4b_o2a_ctrl.json \
    -tok testn/n1_2plus2_p20.tok --profile o5_fail.json > o5_fail.txt 2>&1
adb logcat -d > /data/local/tmp/v5/o5_logcat.txt
```

**Record:** attach `o5_logcat.txt` whole. Do not filter it.

## 6. Stage O6 — the end-to-end run (25 min, ONLY with a passing O2 config)

`RUNBOOK_e2e_qwen3vl_4b.md` §2–§5, substituting the passing config filename
everywhere `genie_dialog_qwen3vl_4b.json` appears. The four gates:

1. text: templated 2+2 → **`4` then stop**
2. sample image → a caption describing the picture (full sentence)
3. six photographs → six captions, one-line human judgement each
4. timing: init / TTFT / decode tok/s, cold and warm separately

**Four passes = done.** Send everything regardless of outcome.

---

## 7. Collect

```sh
sh /data/local/tmp/v5/collect_o.sh
# host:
adb pull /data/local/tmp/testo_results.tar.gz .
```

Send: the tarball, `testo_state_dumps.tar.gz` (if not already sent at O1), and
the filled `RESULTS_TEMPLATE.md`.

### Value order if time runs short

1. **O1** — the dumps decide the diagnosis even if nothing else runs
2. **O2** — the only branch that can finish the project today
3. O3, O4, O5
4. O6 only ever runs on a passing O2

Stopping after O1 + O2 is a complete session.
