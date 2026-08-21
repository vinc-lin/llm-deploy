# Test J — the decode step Genie gets wrong

**Status:** ready to run · **Opened:** 2026-08-21 · **Needs:** no rebuild, no new
ctx-bins, ~2 minutes of device time.
Audience: device team + build side. Self-contained; no prior thread needed.

---

## 1. Where this comes from

Test I settled the root cause: the 4B is calibrated on chat-templated prompts,
every earlier test sent raw text, and templating flips the first generated token
from wrong to **correct**. That is confirmed at both levels (graph probe
`i0_raw` row 0 = 1.390, `i1_templated` clean at 5/5 argmax; Genie's first token
`4`).

It also exposed a **second, separate defect**, and this test is aimed at it.

**The first generated token comes from PREFILL**, not decode — it is the argmax
of the last prompt row. So on a templated prompt:

* prefill is right (device says `4`; the graph probe agrees at 5/5);
* the **very first decode call** is already wrong.

| prompt | correct next token | Genie produced |
|---|---|---|
| `What is 2+2? …` | **151645** `<\|im_end\|>` | 2939 `ention` |
| `Describe … mountain weather …` | **9104** ` weather` | 3279 `aged` |

### The correct output is short, and that matters

Running the real split fp32 graphs with the full KV recurrence
(`scripts/validate/host_generate_check.py`):

```
2+2      -> [19, 151645]                       = "4<|im_end|>"      # TWO tokens
weather  -> [91169, 9104, 4344, 6157, 1576, …] = "Mountain weather changes
                                                  quickly because elevation
                                                  causes rapid shifts in
                                                  temperature and air pressure…"
```

⚠ A prior note in `ISSUE_qwen3vl_4b_text_numerics.md` said *"4 then repeats is
expected: greedy with no EOS."* **That is false and is withdrawn.**
`eos-token: [151645, 151643]` is configured, and the graphs emit it immediately.
The repetition loop is a defect.

### The graphs are not the suspect

* the host run above does the whole recurrence correctly;
* `parity_e2e_vl.py` is 20/20 token-identical against HF;
* Test G's `r3_decodectx` already ran a decode step on device through **both**
  shards — chained argmax 1/1 (cos 0.99994), isolated 1/1 (cos 0.99998).

⚠ So **"shard 1 decode has never been tested" is not correct** — it has, and it
passed. What has never been tested is **who supplies the decode step's inputs**:

| input | `r3_decodectx` (passed) | Genie (fails) |
|---|---|---|
| KV cache | host-built, pushed as files | Genie's own prefill → decode handoff |
| `inputs_embeds` | kit file | Genie's LUT lookup of a **generated** token |
| mask / positions | kit files | Genie's own advance |

---

## 2. What Test J runs

Each case hands the decode graph a **host-built version of exactly the state
Genie should have produced** at that step, and asks whether the graph then gets
the right answer.

| case | prompt | step | cache going in | **expected argmax** |
|---|---|---:|---|---:|
| `j0_2plus2_s1` | templated 2+2 | 1 | 20 positions, all from prefill | **151645** `<\|im_end\|>` |
| `j1_weather_s1` | templated weather | 1 | 18 positions, all from prefill | **9104** ` weather` |
| `j2_weather_s2` | templated weather | 2 | 19 positions — **one written by the DECODE graph** | **4344** ` changes` |

`j2` is the recurrence: its cache contains a row the decode graph produced, not
one prefill produced. That is the only part of the decode path that has never
run on device, under any probe.

The KV contract is `parity_e2e_vl.Decoder`, not a re-derivation: cache
left-aligned at `[0, cache_len)`, mask open on `[0, cache_len)` plus the new
token's own slot at index `PAST`, key `[1,8,D,PAST]` and value `[1,8,PAST,D]`.

**Nothing is rebuilt.** Same ctx-bins already on your device:

| file | bytes | md5 |
|---|---:|---|
| `qwen3vl-4b-w8a16_1_of_2.bin` | 1850793984 | `f031e3a7563bf16f2d5ca98a71b357f6` |
| `qwen3vl-4b-w8a16_2_of_2.bin` | 2631094272 | `0f1c86e89752b499eec09e9e10a73014` |

---

## 3. Running it

```sh
cat testj/past_kv.tar.gz.part-* > testj/past_kv.tar.gz   # REQUIRED: reassemble
md5sum -c testj/past_kv.tar.gz.md5                       # verify before trusting it
tar xzf testj/past_kv.tar.gz -C testj/                   # 10 MB -> 962 MB of KV
adb push testj/. /data/local/tmp/v5/
adb shell
cd /data/local/tmp/v5 && export LD_LIBRARY_PATH=. && chmod +x qnn-net-run
sh run_text_probe.sh 2>&1 | tee text_probe_j.log
exit
adb pull /data/local/tmp/v5/text_probe_out ./text_probe_out_j
```

None of the first three lines is optional — all three cases feed a real cache,
and the runner size-checks every file and stops with this hint if any is
missing. The KV ships as 2 MB parts because this proxy drops larger single
transfers.

Host side:

```bash
$PY_DEPLOY scripts/validate/analyze_realistic_probe.py \
    --kit host_refs --results ./text_probe_out_j
```

**The headline number here is the logits argmax, not the boundary gain.** The
analyzer prints both; read the `logits chained` line for each case.

---

## 4. Reading the result

| j0 / j1 argmax | j2 argmax | meaning |
|---|---|---|
| **match** (151645 / 9104) | **match** (4344) | **the decode graph is fine end to end, including the recurrence.** The fault is entirely in **Genie's decode-step feed** — its cache handoff, its LUT lookup of a generated token, or its position advance. Bisect moves inside Genie |
| **match** | mismatch | the graph is right on a prefill-built cache but wrong on one it wrote itself — a **KV write-back/read-back** defect in the ctx-bin decode path |
| mismatch | — | the decode **ctx-bin** is wrong on this cache, even though `r3_decodectx` passed on a 13-position one. The difference is cache contents/length — send the results and we bisect on cache length |

The first row is the expected outcome, given everything above.

### If j0/j1/j2 all match — the three Genie candidates, in order

1. **The prefill → decode KV handoff.** Prefill's cache is `[1,8,D,2048]` and
   decode's is `[1,8,D,2175]`; Genie must re-lay the committed positions across
   that width change. Never tested.
2. **The LUT lookup of a generated token.** Prompt-token lookup is proven good
   (prefill is right), so this would have to be specific to the decode path.
3. **Mask / position advance** on the first decode step.

---

## 5. What to send back

1. The analyzer's full output — especially the `logits chained` argmax per case.
2. `text_probe_j.log`, including the `shard0 out:` lines.
3. The md5s of the two ctx-bins you ran (§2).

If you also still have Test I's Genie profiles, send the **generated token ids**
rather than the rendered text where you can — for this test the ids are the
measurement and the text is just their rendering.
