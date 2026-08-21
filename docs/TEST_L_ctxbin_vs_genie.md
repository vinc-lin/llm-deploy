# Test L — is it the ctx-bin or is it Genie?

**Status:** ready to run · **Opened:** 2026-08-22 · **Needs:** one corrected
bundle, already on the Hub. **Board time ≈ 15 minutes.**

Results template: `RESULTS_TEMPLATE.md`. Registry of every test: `DEVICE_TEST_INDEX.md`.

---

## 0. Read this first — Test K's K1 was measuring our bug, not Genie's

Test K ran the 0.6B LUT probe through Genie and it was wrong from the very first
token: `</think>` (151668) where the reference is `' '` (220). The device report
concluded the LUT feed was sound; the build side then concluded the opposite,
that **Genie's LUT feed was implicated**.

**Both were wrong, and the cause is ours.** The bundle published for Test K
shipped the **pre-graft ctx-bin**:

| | `inputs_embeds` | `lint_embedding_dtype.py` |
|---|---|---|
| bin Test K ran (`880a6abd…`) | **`QNN_DATATYPE_FLOAT_16`** | **FAIL** |
| corrected bin (`9720e46e…`) | `QNN_DATATYPE_UFIXED_POINT_16`, scale 8.300073e-06, offset −38356 | **PASS** |

An embeddings-fed `inputs_embeds` must never be `FLOAT_16`. `quantizeInput`
advances its destination by `tensorOffset` **elements** for UFIXED and FLOAT_32
but by **bytes** for FLOAT_16 (`nsp-model.cpp:3144`), while
`setupInputEmbeddings` passes an element count when padding a partially-filled
prefill chunk (`:1813`). The pad write therefore lands inside the real prompt and
**overwrites its back half**.

It fires only when `variant > n_process` — a partial prefill chunk. The probe's
prompts are 12, 5 and 6 tokens in an **AR=128** graph, so it fires on *every*
prompt, every time. And the symptom matches exactly:

* the **front** of the prompt survives → the 12-token `2+2` question still
  produces `2 + 2 = 4` somewhere in the output;
* the **back** is destroyed → the 15-token mountain-weather prompt loses its
  topic entirely and the model writes 64 fluent tokens about New York;
* **decode is untouched**, because at AR=1 `variant == n_process == 1` and the
  pad path never runs → 64 coherent tokens, clean EOS.

That is the whole of K1. It is the same defect class as the 2026-08-15 Qwen3-VL
text garbage, already documented as a hard contract, already fixed for the 4B —
and the probe simply shipped from the wrong build directory. `REFERENCE.md`
correction #39.

**So the LUT feed is neither exonerated nor implicated. K1 is void and is
re-run here.**

---

## Stage L0 — re-run K1 on the corrected bundle (~4 min)

The fastest and highest-value measurement in this session. If the corrected bin
now tracks the control, the fp16 bin was the entire story.

```sh
adb push qwen3_06b_lutprobe /data/local/tmp/lutprobe
adb shell
cd /data/local/tmp/lutprobe && chmod +x genie-t2t-run && export LD_LIBRARY_PATH=.

md5sum qwen3-0.6b-w8a16-lutprobe-ladekv_ctx.bin | cut -c1-32
#   MUST be 9720e46e62f59d08f56301418cccc8c1
#   if it is 880a6abdec4a64b67b275ec817c054ca you have the OLD bundle -- stop

./genie-t2t-run -c genie_dialog_qwen3_0.6b_lutprobe.json \
    -p "What is 2+2? Answer with one number." \
    --profile l0_short.json 2>&1 | tee l0_short.txt

./genie-t2t-run -c genie_dialog_qwen3_0.6b_lutprobe.json \
    -p "Describe, in three or four sentences, what makes mountain weather change quickly." \
    --profile l0_long.json 2>&1 | tee l0_long.txt
```

**Expected, now that the reference is known.** HF Qwen3-0.6B fp32, greedy, these
exact raw prompts:

| prompt | correct output | Test K's fp16 bin gave |
|---|---|---|
| short | `' 2+2=4. 2+2=4. 2+2=4. …'` — first ids `[220,17,10,17,28,19]` | `</think>\n\n2 + 2 = 4.` |
| long | `' Also, explain why it is important to have a good understanding of the weather…'` | `</think>\n\nThe city of New York…` |

> ⚠ **The repetition is correct.** HF fp32 does exactly the same thing on a raw
> prompt, and so does the shipping 0.6B. Test K read the control's repetition as a
> deficiency and the probe's clean answer as a virtue; it is the other way round.
> **Do not score this as "coherent vs garbage" — score it against the two strings
> above.**

| what you see | meaning |
|---|---|
| **matches the reference** (` 2+2=4. …` / ` Also, explain why…`) | the fp16 bin was the whole story ⇒ **Genie's LUT feed is sound** ⇒ the two-ctx-bin **split** is again the sole suspect for the 4B's decode defect |
| still `</think>`-prefixed or off-topic | the fp16 bin was *not* the whole story ⇒ Genie's LUT feed really is implicated, and L1/L2 below localise it |

---

## Stage L1 / L2 — the same bin under `qnn-net-run` (~8 min)

Runs the **same ctx-bin** with its inputs supplied as files instead of by Genie.
Worth running whatever L0 says: if L0 is clean this is the graph-level
confirmation, and if L0 is still wrong this is what separates "the bin is wrong"
from "Genie's feed is wrong".

Five cases. Expected argmax comes from the probe's own ONNX, which
`parity_lutprobe.py` independently gates **3/3 against HuggingFace** — so a match
chains all the way back to HF.

| case | graph | prompt / step | **expect argmax** | |
|---|---|---|---:|---|
| `l1a_2plus2` | prefill | `What is 2+2? Answer with one number.` | **220** | `' '` |
| `l1b_paris` | prefill | `The capital of France is` | **12095** | `' Paris'` |
| `l1c_boils` | prefill | `Water boils at a temperature of` | **220** | `' '` |
| `l2a_decode_s1` | decode | step 1, cache = l1a's prefill KV | **17** | `'2'` |
| `l2b_decode_s2` | decode | step 2, cache holds a **decode-written** row | **10** | `'+'` |

`l2b` is the recurrence — the same thing Test J's `j2` established for the 4B,
and the only part of a decode path a single step never exercises.

```sh
# host, once: the decode cases carry a real KV cache
cd testl
md5sum -c past_kv.tar.gz.md5
tar xzf past_kv.tar.gz
cd .. && adb push testl /data/local/tmp/lutprobe/testl

# device
cd /data/local/tmp/lutprobe/testl && export LD_LIBRARY_PATH=/data/local/tmp/lutprobe
cp ../qwen3-0.6b-w8a16-lutprobe-ladekv_ctx.bin ../qnn-net-run .
chmod +x qnn-net-run
sh run_lutprobe_kit.sh 2>&1 | tee lutprobe_kit.log

# host
adb pull /data/local/tmp/lutprobe/testl/probe_out ./probe_out_l
$PY_DEPLOY scripts/validate/analyze_lutprobe_kit.py --kit testl --out probe_out_l
```

The analyzer prints per-case MATCH/MISMATCH and the verdict. **Do not eyeball the
raw files** — the logits row is 151,936 wide.

> ⚠ The three `md5sum -c` / `tar` lines are not optional: both decode cases feed
> a real cache. The runner size-checks every file and stops with a clear message
> if one is missing. **If it stops, that message is the result.**

### Reading L1/L2

| L1 | L2 | meaning | next |
|---|---|---|---|
| all match | both match | the bin is faithful to its ONNX, which is faithful to HF. If L0 also disagreed with the reference, the fault is **Genie's LUT feed** and nothing else | bisect inside Genie |
| any mismatch | — | the bin does **not** reproduce its own ONNX ⇒ a **converter** defect | capture the converter version and the case |
| all match | `l2a` ok, `l2b` fails | reading back a row the decode graph itself wrote is broken | a KV write-back defect, at 0.6B, cheap to chase |

---

## What to send back

1. `l0_short.txt`, `l0_long.txt` — **verbatim**, and the `md5sum` you actually ran.
2. `lutprobe_kit.log` and the whole `probe_out_l/`.
3. The analyzer's output, pasted.
4. `RESULTS_TEMPLATE.md`, filled in.

### If time runs short

**L0 short prompt alone is a complete result** — one command, and it decides
whether Genie's LUT feed is in the picture at all. L1/L2 are the backstop.

---

## Things that will waste your time

* **Check the md5 before anything else.** The whole reason this test exists is
  that a bundle shipped from the wrong build directory. `9720e46e…`, not
  `880a6abd…`.
* **`--max-num-tokens` is not a flag.** It is `dialog.max-num-tokens`, already 64
  in the shipped config.
* **`--profile` refuses an output file that already exists.** Fresh name per run.
* **Do not chat-template these prompts.** Raw is correct for the 0.6B — it was
  calibrated on raw prompts. Templating is a 4B calibration fact, not a Genie one.
* **Do not compare against `qwen3_06b_w8a16_gqafix_ladekv`** if you want a
  shape-matched control: that build is CTX-1152 and the probe is CTX-640. The
  matched one is `2026-08-16-regime/bundles/qwen3_06b_w8a16_gqafix_cl512_ladekv.tar.gz`
  (`REFERENCE.md` correction #38). For L0 you do not need a control at all — the
  expected strings above are the reference.
