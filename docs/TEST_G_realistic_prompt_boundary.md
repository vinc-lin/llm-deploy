# Test G — does the boundary hold on a REALISTIC prompt?

**Status:** ready to run · **Opened:** 2026-08-20 · **Needs:** no rebuild, no new
ctx-bins, ~2 minutes of device time.
Audience: device team + build side. Self-contained; no prior thread needed.

---

## 1. Read this first — the previous four tests measured an artifact

Tests B, C, E and F all fed **bare token ids**. `decode1tok` is token `3838`
("What") alone at position 0; `prefill4tok` is four content words. A production
prompt never looks like that — it is chat-templated and begins `<|im_start|>`.

That difference turns out to be the entire result. Measured on the host
2026-08-20, clamping every activation to its calibrated range (which reproduces
the device's boundary defect to **0.3%**, so the simulation is trustworthy):

| input | row-0 gain | worst over rows 0–3 |
|---|---:|---:|
| the Test F probe | **1.39** | **0.390** |
| six realistic chat-templated windows | **0.9990** | **0.035** |

And the model's **real** attention sink is not where the probe put it: on a
chat-templated prompt it sits at **row 1, RMS 220.3** — *larger* than the probe's
synthetic row-0 sink at 107.2 — and comes through at gain **1.0000**.

So the 1.39× that Tests B/C/E/F chased is a defect the probe manufactured by
placing a bare content word at position 0. The measurements were real and
self-consistent; their relevance to production was never established. **The
production text garbage is currently unexplained.**

What that ruled out is still worth having: the quantized activation ranges are
clean on realistic input to within 3.5%, so the encodings are not the cause.

Test G is the first probe with production-shaped input.

---

## 2. What Test G runs

Three cases, built from the multimodal calibration/eval windows — chat-templated
turns with **real ViT features** spliced onto the image-token positions, i.e.
exactly what qualla feeds. Deepstack is zero-filled, matching the shipped tower.

| case | window | real rows | what it adds |
|---|---|---:|---|
| `r0_text` | *held-out* `EVAL text 'The capital of France is'` | 13 | text-only, short + padded |
| `r1_image` | *held-out* `EVAL img100 'What is happening in this image?'` | 113 | the real image+text path |
| `r2_chunk0` | `img0-chunk0[0:128]` | 128 | a **completely full** AR window, no padding at all |

Two of the three are from the **held-out eval split**, so they were never seen by
calibration.

**Nothing is rebuilt.** The ctx-bins are the ones already on your device from the
v5 session:

| file | bytes | md5 |
|---|---:|---|
| `qwen3vl-4b-w8a16_1_of_2.bin` | 1850793984 | `f031e3a7563bf16f2d5ca98a71b357f6` |
| `qwen3vl-4b-w8a16_2_of_2.bin` | 2631094272 | `0f1c86e89752b499eec09e9e10a73014` |

Check those first. If they differ, stop and say so.

---

## 3. The host prediction — this is falsifiable, on purpose

Every reference row, with the full activation clamp applied (the device's own
activation path):

| case | row 0 | row 1 (the real sink) | row 2 | row 3 | last row |
|---|---:|---:|---:|---:|---:|
| `r0_text` | 1.0001 | 1.0002 | 1.0002 | **1.0347** | 0.9999 |
| `r1_image` | 1.0001 | 1.0002 | 1.0002 | 1.0002 | 1.0002 |
| `r2_chunk0` | 1.0001 | 1.0002 | 1.0002 | 1.0002 | 1.0000 |

**Prediction: every row within ±5%, worst 1.0347.** Cosine ≥ 0.9996 everywhere.

If the device disagrees with this table, the disagreement is the finding.

---

## 4. Running it

The kit merges into the v5 folder you already have — it needs that folder's
`qnn-net-run`, libraries, `netrun_htp_config.json` and the two ctx-bins.

```sh
adb push testg/. /data/local/tmp/v5/
adb shell
cd /data/local/tmp/v5 && export LD_LIBRARY_PATH=. && chmod +x qnn-net-run
sh run_text_probe.sh 2>&1 | tee text_probe_g.log
exit
adb pull /data/local/tmp/v5/text_probe_out ./text_probe_out_g
```

`testg/` ships its own `probe_cases.txt`, so this runs Test G and not the
earlier cases. Expect ~2 minutes and a pull of roughly **120 MB**.

The runner stops hard if shard 0's output is the wrong size or ambiguous. **If it
stops, that message is the result** — send it.

Host side:

```bash
$PY_DEPLOY scripts/validate/analyze_realistic_probe.py \
    --kit host_refs --results ./text_probe_out_g
```

---

## 5. Reading the result

| outcome | meaning | what happens next |
|---|---|---|
| **clean** — every row within ±5%, matching §3 | shard 0 is faithful on production input | the whole boundary line of enquiry **closes**. The fault is downstream: Genie's feed, the dialog/config path, or shard 1. Stop measuring shard 0 |
| **dirty** — any row off, where §3 predicts clean | the fault is in the **ctx-bin or the converter** | that is the one stage the host simulation cannot model — it models the encodings, not the conversion. No test so far has separated "our numbers" from "the toolchain's numbers" on a realistic input. This would be the first |

Both outcomes are decisive, which is why this is worth a device session.

The analyzer also reports **logits**, chained vs isolated, per case: chained feeds
shard 1 the device's own shard-0 output, isolated feeds it the host reference.
Chained wrong + isolated right localises the fault to shard 0; both wrong points
at shard 1.

---

## 6. What to send back

1. The analyzer's full output, or `text_probe_out_g/` itself (~120 MB).
2. `text_probe_g.log`, including the `shard0 out:` lines.
3. The md5s of the two ctx-bins you ran (§2).

---

## 7. What Test G cannot tell us

It tests the **prefill** path with an empty cache. A real decode step attends to
a populated KV cache, and this kit does not exercise that — shipping real past-KV
would add ~300 MB per case to express what is otherwise a zero file. If Test G
comes back clean and the device still garbles real generation, the decode-with-
context path is the next thing to instrument.

It also says nothing about output quality. Every 4B run so far has been behind
this defect, so how well a W8A16 Qwen3-VL-4B actually captions a photograph
remains unmeasured.
