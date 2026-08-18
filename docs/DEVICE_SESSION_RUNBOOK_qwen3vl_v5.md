# Device session runbook — Qwen3-VL-4B v5 (text-tower diagnosis)

**Bundle:** `qwen3vl_4b_e2e_pipeline_v5`
**Written:** 2026-08-18, after the v4 session
**Board time:** ~20 min
**Companion:** `OPERATOR_GUIDE.md` (metric definitions, install, triage)

## 0. What this session is for

v4 settled the image path: the pipeline runs, no SIGSEGV, six photographs
executed. **That defect is closed and is not re-tested here.**

One defect is left: the text tower produces garbage on device while scoring
20/20 token-identical against HuggingFace on the host. v5 does **not** try to
fix it — it is a diagnostic bundle whose only job is to say *which stage is at
fault*, because every remaining hypothesis lives in the one hop no host gate
covers: ONNX → DLC → ctx-bin → Genie's feed.

**Expect no caption from this session.** The deliverable is a verdict.

### What was already eliminated on the host

So the session is not spent re-testing them:

| Ruled out | How |
|---|---|
| "AIMET QDQ double-quantization" | **0** `QuantizeLinear`/`DequantizeLinear` ops in either ONNX; the working 0.6B (44.707 tok/s) uses the *same* AIMET path, and `qairt-quantizer` appears nowhere in this project |
| Wrong weight format | The 4B DLC is already `sFxp_8`, per-channel `axis-quant`, offset 0 — i.e. the per-channel symmetric INT8 the "fix" would produce |
| Corrupt embedding LUT | Bit-exact against the checkpoint at the runtime's own byte offsets, `worst 0.000e+00`, vision/pad markers included |
| Shard-boundary encoding mismatch | `last_hidden_states` is `FLOAT_16` on **both** sides — there is no encoding to mismatch |
| `bos-token` | Fixed in v4; the device confirmed no change |

## 1. Setup

```bash
adb push qwen3vl_4b_e2e_pipeline_v5 /data/local/tmp/qwen3vl_v5
adb shell
cd /data/local/tmp/qwen3vl_v5
chmod +x genie-app genie-t2t-run qnn-net-run
export LD_LIBRARY_PATH=.
ls -l qwen3vl-4b-w8a16_1_of_2.bin      # the ctx-bins are unchanged from v3/v4
```

The three ctx-bins and the LUT are **byte-identical to v3 and v4** (md5s in
`V4_CHANGES.md` §4), so copy them from the existing deployment rather than
re-downloading ~6 GB.

## 2. P-A — the text-graph probe (the decisive test, ~10 min)

Runs the shipped text ctx-bins under `qnn-net-run` with **no Genie involved**,
on inputs we control, against references computed from the same per-shard ONNX
the DLCs were converted from.

```bash
sh run_text_probe.sh 2>&1 | tee text_probe.log
```

It generates its own zero past-KV file (one 4,454,400-byte file reused for all
72 past inputs — the cache is empty *and* fully masked, so its contents cannot
affect the result), then per case runs three inferences:

| Run | What it feeds | What it answers |
|---|---|---|
| `shard0` | our `inputs_embeds` | is shard 0's ctx-bin right? |
| `shard1-chained` | the **device's own** shard-0 output | what the real chain produces |
| `shard1-isolated` | the **host reference** boundary | is shard 1 right *given* good input? |

The last two exist as a pair on purpose: a single end-to-end number cannot
distinguish "shard 0 corrupted the boundary" from "shard 1 is broken".

Two cases run: `pos0`, where position 0 makes cos=1/sin=0 so rope is the
identity and a failure is *pure numerics*; and `pos7`, identical except rope is
active. `pos0` clean + `pos7` broken means the graph mishandles rope tables —
and since those tables came from us, that is a conversion fault, not Genie's.

**Collect:** `text_probe.log` and the **entire `text_probe_out/` directory**
(it is small — a few MB of raw tensors).

**Do not judge it on device.** Pull `text_probe_out/` back; the host runs
`compare_text_probe.py`, which prints the verdict and what it implies.

If `qnn-net-run` rejects a flag name, run `./qnn-net-run --help` and report
what it calls its native input/output file options — do not fall back to
feeding the blobs as float32.

## 3. P-C — the timing decomposition (~5 min)

v4 reported "~11.1 s" with no breakdown, so a 4B two-shard W8A16 tower still
has no init/TTFT/decode numbers on this silicon. **These are valid even though
the output is garbage** — decode rate is the same compute either way.

```bash
./genie-t2t-run -c genie_dialog_qwen3vl_4b.json \
    -p "What is 2+2? Answer with one number." --profile v5_t2t_cold.json
./genie-t2t-run -c genie_dialog_qwen3vl_4b.json \
    -p "What is 2+2? Answer with one number." --profile v5_t2t_warm1.json
./genie-t2t-run -c genie_dialog_qwen3vl_4b.json \
    -p "What is 2+2? Answer with one number." --profile v5_t2t_warm2.json

for i in 1 2 3; do echo "== pipeline run $i"; \
  time ./genie-app -s genie_pipeline_qwen3vl.script; done 2>&1 | tee v5_pipeline_timing.log
```

**Collect:** the three profile JSONs **verbatim** (do not summarise — the host
parser reads every numeric leaf, because guessing key names is how a harness
silently reports zeros) and `v5_pipeline_timing.log`. Report cold and warm
separately; never average.

Also report the generated text from each run, garbage included — it is data,
and a *change* in the garbage pattern between runs would itself be a finding.

## 3b. Already done on the host: which feed mistakes could even cause this

`probe_feed_variants.py` has already been run (results in
`feed_variants.json`). It corrupts the feed one hypothesis at a time on the
real tower and measures how far the logits move, so the session is not spent
guessing. Ranked, worst first:

| Hypothesis | worst cos | argmax | Could it cause garbage? |
|---|---:|---|---|
| `emb_fp32_as_fp16` | **−0.87** | 0/4 | **Yes — and it degenerates into repetition** |
| `mask_multiplicative` | −0.71 | 0/4 | yes |
| `mask_all_visible` | −0.10 | 1/4 | yes |
| `mask_row0_only` | +0.78 | 1/4 | yes |
| `rope_not_applied` | +0.997 | 3/4 | marginal |
| `rope_no_offset` | +0.997 | 3/4 | marginal |
| `rope_wrong_theta` | +1.000 | 4/4 | **no — eliminated** |
| `emb_halved` | +0.9996 | 4/4 | **no — eliminated** |

**The leading candidate is `emb_fp32_as_fp16`**, and not only because it scores
worst. Its output *degenerates into repetition* — `[188, 26610, 26610, 26610]`
— which is the shape of what you actually observed on device
("…abilityability…", "…uringuring…", running until the context was exceeded).
It is also the same class of defect as the v3 image crash: a dtype
misinterpretation in Genie's input staging, one layer over. `inputs_embeds` is
declared `FLOAT_16` while the accumulator carries float32, and the conversion
between them is exactly the step that would produce this.

Two hypotheses are **eliminated** and should not be re-tested: rope theta and
embedding scale barely move the logits at all.

This does not prove anything about the device yet — that is what probe A is
for. It tells us where to look the moment probe A exonerates the ctx-bin.

## 4. What is NOT in this session, and why

**There is no Genie debug-tensor dump.** It was planned and then dropped on
evidence: `debug-tensors` / `debug-path` are compiled into the shipped
`libGenie.so`, but `Engine.cpp` validates the public config against a strict
whitelist and throws `Unknown QnnHtp config key` on anything outside it. The
engine level is equally strict. Shipping that config would have produced a load
failure and burned a device slot — the same trap that broke the v3 fallback.
The equivalent evidence is instead produced **on the host** by
`probe_feed_variants.py`, which ranks which feed mistakes could produce garbage
at all, at zero device cost.

**Do not attempt config edits to enable debugging.** If a future SDK exposes
those keys, that changes; in this one it does not.

## 5. What to send back

1. `text_probe.log` and the whole `text_probe_out/` directory.
2. The three `v5_t2t_*.json` profiles, verbatim, plus each run's generated text.
3. `v5_pipeline_timing.log` and the three pipeline captions.
4. `adb logcat -d` around any crash, and the tombstone if one occurs.
5. `ls -l *.raw` and the `*.info.json` from the bundle you actually pushed.

Screen photographs are fine; they get transcribed and the Markdown becomes the
record.

## 6. What each outcome means for the next build

| P-A verdict | What it means | Next |
|---|---|---|
| all three runs correct | ctx-bin and converter **exonerated** | the fault is Genie's feed; the host variant ranking becomes the shortlist |
| shard0 wrong | fault is in shard 0's ctx-bin | rebuild justified, aimed at shard 0 |
| shard0 right, shard1-isolated wrong | fault isolated to shard 1 (it owns `lm_head`) | rebuild shard 1 only |
| both shards right alone, chained wrong | the boundary hand-off corrupts `last_hidden_states` | inspect the seam layout, not the weights |
| `pos0` right, `pos7` wrong | the graph mishandles rope | conversion fault, not Genie |

Every one of those is a different next build. That is the point of the session.
