# Test E — capture and validate shard 0's boundary output

**Two paths. The first needs no device at all and takes about a minute.**
This is currently the highest-value single measurement in the investigation.

---

## 1. What we are measuring, and why it is one small file

Test B reported shard 0's boundary output at cosine **1.0000** against the host
reference, yet feeding that same file to shard 1 gave argmax **105196** where the
host reference gave **374**. Two inputs agreeing to 1.0000 cannot diverge by 2.8
logits — unless the difference is **scale**, because **cosine is scale-invariant**.

Measured on the host against `prefill_1.onnx` / `decode_1.onnx`:

| boundary scale | `decode1tok` argmax | `prefill4tok` rows 0–3 |
|---|---|---|
| 0.50 – 1.10 | 374 ✓ | 374, 279, 330, 315 — all ✓ |
| **1.25 – 3.00** | **105196** | **105196**, 279, 330, 315 |

A *uniform* scale error anywhere in that band reproduces the device result
**exactly, including the row pattern** — row 0 wrong, rows 1–3 right. Row 0 is
not specially corrupted; it is the least-contextualised row, so its logits are
flattest and it flips first. That single parameter also explains Test C: n=1 uses
only the flat row (wrong), n≥4's first token comes from the last prompt row
(right), every later decode step runs on shard-1 state built from a mis-scaled
boundary (garbage), and n=129's last real token lands on its chunk's first row
(wrong).

So the whole question reduces to one number: **the RMS of shard 0's boundary
output**. The file is `1 × 2560` fp16 = **5,120 bytes**.

---

## 2. Path A — from the capture you already have (no device)

`testb_probe_out.tar.gz` already contains it.

```bash
tar xzf testb_probe_out.tar.gz                 # -> text_probe_out/
find text_probe_out/decode1tok_s0 -name 'last_hidden_states*'
```

Expect exactly **one** file of **5,120 bytes**. Then:

```bash
$PY_DEPLOY scripts/validate/check_boundary_scale.py \
    --device <that file> \
    --ref    <bundle>/03_vl4b_v5/decode1tok/decode_1/last_hidden_states.raw
```

The reference ships in the bundle — it is exactly the host boundary the isolated
run was fed, so this is a like-for-like comparison.

**If more than one file matches**, do not pick one: report all of them with their
sizes. Two matches is itself a finding — the previous runner selected shard 0's
output with `find … | head -1` (directory order) while the host comparator scored
`sorted()[0]`, so the file *fed* to shard 1 need not have been the file *scored*.
That has been fixed, but a capture made before the fix may carry the ambiguity.

If only the number survived and not the file:

```bash
$PY_DEPLOY scripts/validate/check_boundary_scale.py --device-rms <number>
```

---

## 3. Path B — regenerate on device (~2 minutes)

Only if the capture is gone. This runs shard 0 alone; shard 1 is not needed.

```sh
adb push 03_vl4b_v5 /data/local/tmp/v5      # if not already there
adb shell
cd /data/local/tmp/v5 && export LD_LIBRARY_PATH=. && chmod +x qnn-net-run
sh run_text_probe.sh 2>&1 | tee text_probe_e.log
exit
adb pull /data/local/tmp/v5/text_probe_out ./text_probe_out_e
```

The current runner **asserts** the size of shard 0's output against
`HIDDEN_BYTES` and stops if it disagrees, and it writes the path it actually fed
to shard 1 into `<case>_s0_fed.txt`. If it stops, that message *is* the result —
send it.

Then run the same host command as Path A against `text_probe_out_e`.

---

## 4. Reading the result

The script prints a verdict and sets an exit code.

| ratio | exit | meaning | what happens next |
|---|---:|---|---|
| **within ±5% of 1.0** | 0 | boundary magnitude is **correct** | the scale hypothesis is **dead**. Do not rebuild on it. Re-run `compare_text_probe.py` (it now prints `mag_ratio` per row) and look downstream |
| **1.25 – 3.0** | 1 | **mis-scaled, and in the band that reproduces 105196** | this is the defect. Build side bisects ONNX → DLC → ctx-bin |
| other, ≠1 | 1 | magnitude wrong but outside the reproducing band | send the file; the story is incomplete |
| any inf/nan present | 2 | fp16 **overflow** at the seam | a *different* defect, and it takes precedence — the ratio understates it. Send the file, do not act on the ratio |

It also decides **uniform vs structural**, and the way it does so was corrected
on 2026-08-20. Read the `residual after removing it` line, not the percentiles:

* `best-fit gain` is least squares, `g = <dev,ref>/<ref,ref>`.
* `residual after removing it` is what is left once that gain is divided out.
  **< 2% means a single uniform gain**; a genuine per-channel distortion lands
  near 8%.
* The **per-element ratio percentiles are not the uniformity test.** These hidden
  states are extremely heavy-tailed — median |x| ≈ 1.06, p99 ≈ 8.5, max ≈ 5244 —
  so ~78% of elements sit near zero and their ratios are noise. The first version
  of this script computed percentiles over all of them and reported
  "NON-uniform" for a boundary that is in fact a clean uniform gain. The script
  now restricts that readout to elements above 2% of RMS and labels the cut.
* With more than one row it also prints a **per-row gain** spread. Same gain on
  every row = one scalar fault; varying by row = structural.

### Measured, 2026-08-15 (decode1tok)

| | |
|---|---:|
| reference RMS | 107.2226 |
| device RMS | 149.0009 |
| cosine | 0.999990 |
| **best-fit uniform gain** | **1.38959×** |
| residual after removing it | **0.447%** |

**Verdict: a uniform 1.3896× gain, accurate to 0.45%** — which is ordinary
W8A16-vs-fp32 error. Look for a scalar gain, *not* per-channel dequantization.

---

## 4b. Then do the same for the prefill boundary — also free

Test E measured `decode1tok` only. The same capture contains `prefill4tok`'s
shard-0 output, and the same command reads it (the script infers the row count):

```bash
$PY_DEPLOY check_boundary_scale.py \
    --device text_probe_out/prefill4tok_s0/**/last_hidden_states*.raw \
    --ref    03_vl4b_v5/prefill4tok/prefill_1/last_hidden_states.raw
```

This is the sharpest remaining question and costs nothing:

| prefill gain vs decode gain | meaning |
|---|---|
| **same ~1.39×** | one constant gain in shard 0, independent of graph — points at the shared weights or a conversion-wide factor |
| **different** | graph- or data-dependent — points at the per-graph conversion |
| **1.00× (prefill clean)** | only the AR=1 decode graph is affected, which narrows it enormously |

Also report the **per-row gain spread** the script prints for prefill: 128 rows
with the same gain is a scalar fault, a gain that varies by row is structural.

## 5. What to send back

1. The 5,120-byte file itself, or the script's full output.
2. If Path A: the `find` listing — **how many** files matched and their sizes.
3. The `shard0 out: …` lines from whichever probe log you have. The older runner
   printed `(N bytes, expect M)` without enforcing it, so a mismatch may be
   sitting in `testb_probe.log` already.

---

## 6. What this rules in and out

Whatever the answer, it is **not** about Genie: the whole chain reproduces under
`qnn-net-run` with no Genie involved. And three hypotheses from the 2026-08-15
report are already dead on that report's own data — the probe fed a **zero KV
cache** (so not KV bookkeeping), **bypassed the LUT** (so not requantization),
and ran **without Genie** (so not orchestration).

This measurement decides between the two that remain: a mis-scaled boundary in
the ctx-bins, or something downstream of a boundary that is actually fine.
