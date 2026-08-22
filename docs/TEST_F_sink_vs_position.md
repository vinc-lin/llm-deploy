# Test F — does the row-0 gain follow the attention sink, or the row index?

**Status:** ready to run · **Opened:** 2026-08-20 · **Needs:** no rebuild, no new
ctx-bins, ~2 minutes of device time.
Audience: device team + build side. Self-contained; no prior thread needed.

---

## 1. Where this stands

Test E measured shard 0's boundary output against the host and found it **scaled**,
not misdirected:

| | |
|---|---:|
| reference RMS | 107.2226 |
| device RMS | 149.0009 |
| cosine | 0.999990 |
| **best-fit uniform gain** | **1.38959×** |
| residual after removing that gain | **0.447%** |

Your per-row follow-up on the prefill case then localised it: **row 0 at 1.39×,
rows 1–3 clean**. And `decode1tok` — a different graph — shows the same 1.3896×.

Those two cases share exactly one property: **the row attends to nothing but
itself.** That is the attention-sink condition, and under it this row carries
massive activations:

| row | RMS | max abs | top channels |
|---:|---:|---:|---|
| **0** | **107.22** | **5244** | c4 = 5244, c396 = −1357 |
| 1 | 2.26 | 15.8 | those same channels: 0.60, −9.45 |
| 2 | 1.26 | 32.8 | |
| 3 | 1.20 | 31.8 | |
| padding (4–127) | 0.6185 | | all identical |

c4 alone is **93.44%** of row 0's squared norm, and **c4² = 2.75e7 overflows
fp16's 65504 by 420×**.

### Already ruled out — do not re-test these

| hypothesis | why it is dead |
|---|---|
| VL deepstack injection at position 0 | all six `deepstack_visual_embed_*` inputs are **exactly zero** (0 nonzero elements) in both cases, identical host and device. Zeroing them is already the state under test |
| a single bad channel (c4) | scaling c4 alone by 1.413× matches the RMS ratio but gives cosine **0.997288**; the device measures **0.999990**. The whole row is scaled |
| KV bookkeeping | the probe feeds a zero, fully-masked cache |
| LUT requantization | the probe bypasses the LUT |
| Genie orchestration | it all reproduces under bare `qnn-net-run` |
| boundary dtype / encodings mismatch | FLOAT_16 on both sides; no conflicting seam entry in either chunk |

### What is left

Two causes, and they need **different fixes**:

* **CONDITION** — self-attention / massive activations saturate or clamp
  somewhere, so any such row is amplified wherever it sits.
* **INDEX** — something specific to element 0 of the AR window (a tile edge, an
  offset bug), and the sink is a coincidence.

Test F separates them.

---

## 2. What Test F changes — and what it does not

**Nothing is rebuilt.** The ctx-bins are the ones already on your device from the
v5 session:

| file | bytes | md5 |
|---|---:|---|
| `qwen3vl-4b-w8a16_1_of_2.bin` | 1850793984 | `f031e3a7563bf16f2d5ca98a71b357f6` |
| `qwen3vl-4b-w8a16_2_of_2.bin` | 2631094272 | `0f1c86e89752b499eec09e9e10a73014` |

Check those before starting. If they differ, stop and say so — everything below
compares against measurements made on these exact bytes.

The only things that change are the **attention mask** and **where the tokens sit
in the AR window**. Both are ordinary graph inputs, and the host reference for
each case is computed from the identical feed, so a difference can only come from
the device.

|  | sink at row 0 | no sink at row 0 |
|---|---|---|
| **row 0** | `fp_ctrl_pre` — the known 1.39× | `f1_row0ctx` |
| **row 4** | `f2_shift4` | — |

### The four cases

| case | graph | what it is | host row RMS |
|---|---|---|---|
| `f0_ctrl_dec` | decode | the Test E case, unchanged. **Anchor** | r0 = 107.202 |
| `fp_ctrl_pre` | prefill | the Test B/E prefill case, unchanged. **Anchor** | 107.202 / 2.256 / 1.262 / 1.199 |
| `f1_row0ctx` | prefill | row 0 made **non-causal** — it attends to all four tokens instead of itself alone. Row index still 0, sink gone | **0.995** / 4.891 / 0.937 / 0.998 |
| `f2_shift4` | prefill | the same four tokens moved to **rows 4–7** with rope positions 0–3; rows 0–3 are masked padding. Sink kept, index moved | 107.202 / 2.256 / 1.262 / 1.199 |

Two host-side properties make these decisive, and the kit builder **asserts both**
and refuses to ship if either fails:

1. **`f2_shift4`'s row 4 computes bit-for-bit what `fp_ctrl_pre`'s row 0
   computes** — same token, same self-only mask, same rope position 0. Measured
   `max|diff| = 0.000e+00`. So any device-side difference between them is the row
   index and nothing else.
2. **`f1_row0ctx` really does remove the sink** — row-0 RMS falls `107.20 → 1.00`,
   a 107× collapse. It is a genuine magnitude test, not just a mask change.

`f1_row0ctx` is not a valid language-model step (a token attending to its own
future). It does not need to be. It is a valid *graph execution*, and the host
reference is computed the same way.

---

## 3. Free first — the layer scan may already be in a capture you have

Shard 0 emits **36 per-layer KV tensors on every run**, and no probe session has
ever compared them. If you still have `testb_probe_out.tar.gz` or the Test E
capture, they are already in it:

```bash
tar tzf testb_probe_out.tar.gz | grep -c 'past_value_.*\.raw'
```

Any non-zero count means the layer scan can be run **with no device access at
all**. Send the capture (or just the `past_*_out` files from the `*_s0`
directories) and we will run it here.

**What the scan can and cannot see.** It does *not* localise a scale. Tracing the
graph, `v_proj` sits behind `input_layernorm` and `k_proj` behind
`input_layernorm` **and** `k_norm` — RMSNorm is scale-invariant, so a uniform
residual gain is normalised away before it reaches either tap. (An earlier draft
of this test claimed the taps would show the gain. That was wrong; the graph was
traced and the claim withdrawn.)

What they *do* read is each RMSNorm denominator — and a saturating denominator is
the leading mechanism here, since `input_layernorm` is the first place this row's
squares are summed and c4² overshoots fp16 by 420×:

| outcome | meaning |
|---|---|
| taps clean at all 18 layers | every denominator is intact; the block maths is right and only the final magnitude is wrong — the fault is on the residual/output path |
| taps diverge from layer *k* | the fault is inside the block maths from layer *k*, and the fp16-overflow story is live. That layer is the target |

Both outcomes are informative, which is why it is worth pulling.

---

## 4. Running Test F on device

The kit merges into the v5 folder you already have — it needs that folder's
`qnn-net-run`, libraries, `netrun_htp_config.json` and the two ctx-bins.

```sh
adb push testf/. /data/local/tmp/v5/
adb shell
cd /data/local/tmp/v5 && export LD_LIBRARY_PATH=. && chmod +x qnn-net-run
sh run_text_probe.sh 2>&1 | tee text_probe_f.log
exit
adb pull /data/local/tmp/v5/text_probe_out ./text_probe_out_f
```

`testf/` ships its own `probe_cases.txt` listing the four F cases, so this runs
Test F and not the old pair. Expect about **2 minutes** and a pull of roughly
**150 MB** — the per-layer taps hold only the new AR-wide slice (256 KB each on
prefill, 2 KB on decode), not the whole cache.

The runner stops hard if shard 0's output is the wrong size or ambiguous. **If it
stops, that message is the result** — send it.

Then, on the host:

```bash
$PY_DEPLOY scripts/validate/analyze_test_f.py \
    --kit <kit> --results ./text_probe_out_f
```

---

## 5. Reading the result

The analyzer prints per-row gains, then a verdict. `clean` is within ±5% of 1.0;
`AMPL*` is inside [1.25, 3.0], the band that reproduces the wrong argmax 105196.

**It checks the anchor first.** If `f0_ctrl_dec` does not reproduce ~1.39×, it
refuses to interpret anything else — that would mean the device is not in the
state Test E measured, and the cases below would be noise.

| `f1_row0ctx` row 0 | `f2_shift4` row 4 | verdict | what it means |
|---|---|---|---|
| clean | **amplified** | **CONDITION** | the gain followed the sink and vanished when row 0 got context. Triggered by massive activations, not by position. Chase the fp16 range on the sink row |
| **amplified** | clean | **INDEX** | a genuine sink at row 4 was fine while row 0 stayed broken with context. The sink is a coincidence; chase the output write / tiling |
| amplified | amplified | model incomplete | neither property alone explains it — send the full table |
| clean | clean | model incomplete | the gain needs the exact control configuration; a strong constraint in itself — send the full table |

The two outcomes point at genuinely different work, which is the point of running
it before spending a rebuild.

---

## 6. What to send back

1. The analyzer's full output (or `text_probe_out_f/` itself — it is ~150 MB).
2. `text_probe_f.log`, including the `shard0 out:` lines.
3. The md5s of the two ctx-bins you ran, from §2.
4. If you still have an older capture: whether it contains `past_*_out` files
   (§3). That answer costs nothing and may be worth a whole device session.

---

## 7. What Test F cannot decide

It locates the *trigger*, not the mechanism. Neither outcome tells us whether the
amplification is a clamp, a saturating reduction, or a requantization step — only
which of those is worth looking for, and where. Expect one more measurement after
this one.

It also says nothing about output quality. Every 4B run so far has been behind
this defect, so how well a W8A16 Qwen3-VL-4B actually captions a photograph is
still unmeasured, and will stay that way until the text tower is fixed.
