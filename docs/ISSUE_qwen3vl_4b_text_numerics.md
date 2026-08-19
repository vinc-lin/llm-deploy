# Qwen3-VL-4B text tower — what v5 settled, what it did not, and the test that decides

**Status:** open · **Opened:** 2026-08-19, after the v5 device session of 2026-08-15
**Audience:** device team + build side. Self-contained; no prior thread needed.

---

## 1. One-paragraph summary

The v5 session **confirmed** the Genie `FLOAT_16` embedding-padding bug and
**confirmed** that re-typing `inputs_embeds` to `UFIXED_POINT_16` fixes it — both
decisively, at 0.6B scale. It also reported that the Qwen3-VL-4B text ctx-bins are
"numerically incorrect independently of Genie", based on a `qnn-net-run` probe
whose outputs barely correlated with host references. **That last conclusion is
not safe**: the probe kit was built for the *previous* ctx-bins and fed the new
ones the wrong byte encoding. This document explains which findings stand, why the
fourth is in doubt, and gives the exact procedure to settle it — the first step of
which needs **no device time at all**.

---

## 2. What is settled

### 2.1 The FLOAT_16 padding bug is real — confirmed on hardware

`QnnNspModel::quantizeInput` advances its destination pointer by `tensorOffset`
**elements** for `UFIXED_8/16` and `FLOAT_32`, but by **bytes** for `FLOAT_16`:

```cpp
case QNN_DATATYPE_UFIXED_POINT_16:  reinterpret_cast<uint16_t*>(buf) + tensorOffset  // elements
case QNN_DATATYPE_FLOAT_32:         reinterpret_cast<float*>(buf)    + tensorOffset  // elements
case QNN_DATATYPE_FLOAT_16:         reinterpret_cast<uint8_t*>(buf)  + tensorOffset  // BYTES
```

`setupInputEmbeddings` passes an **element** count when padding a partially-filled
prefill chunk, so on an fp16 `inputs_embeds` the pad write lands halfway into the
real prompt and overwrites its back half. It fires only when `variant > n_process`,
i.e. the last *partial* chunk.

Device result, 0.6B control (`01_probe_06b_fp16in`), AR = 128:

| prompt | chunking | padding runs? | observed |
|---:|---|---|---|
| 127 tok | one chunk, 127/128 | yes | **garbled** |
| 128 tok | exactly full | **no** | **clean, coherent** |
| 129 tok | 128 + 1 | yes, chunk 2 | **garbled** |

Broken → clean → broken across a one-token change. Nothing else — quantization,
rope, the split, encodings — produces that. **This is no longer a hypothesis.**

### 2.2 The UFIXED_POINT_16 fix works — confirmed on hardware

`02_probe_06b_u16in` is the same model, same weights, same LUT, same graph shapes,
with `inputs_embeds` re-typed. All three prompt lengths produce coherent output.

*(The "2+2 → 4 then repeats" observation is expected: greedy sampling with no EOS
runs to the context limit. Not a defect.)*

### 2.3 The 4B text tower is still broken under Genie

With the `uFxp_16` fix applied, 4B text-only output remains unintelligible. This
observation does not depend on any probe and is accepted. Note it is also not new
— the 4B has produced garbage since v1. What is new is the *character*: the 0.6B
fp16 corruption still yielded recognizable English, the 4B yields token-level
noise.

---

## 3. What is NOT settled: the qnn-net-run result

The session reported near-zero agreement between device and host:

| measurement | reported |
|---|---:|
| `decode1tok` logits cosine | 0.372 |
| `decode1tok` top-10 overlap | 0 / 10 |
| `prefill4tok` hidden cos, rows 0–3 | 0.072 / −0.012 / −0.038 / −0.013 |

and concluded the ctx-bins are numerically wrong. **Those numbers are consistent
with a perfectly good ctx-bin fed the wrong bytes**, and that is what happened.

### 3.1 The defect in the probe kit

`run_text_probe.sh` invokes `qnn-net-run --use_native_input_files`, so every
`.raw` must contain the tensor's **native** bytes. The kit wrote IEEE float16
unconditionally:

```python
p.write_bytes(np.ascontiguousarray(a, dtype=np.float32).astype("<f2").tobytes())
```

That was correct while `inputs_embeds` was `FLOAT_16`. The v5 rebuild re-typed it
to `UFIXED_POINT_16` — **also 2 bytes per element**, so there was no size error, no
warning, and no failure. The graph simply decoded IEEE fp16 bit patterns as
quantized integers (`scale = 3.5394e-04`, `offset = −32927`):

| intended value | what the graph received |
|---:|---:|
| 0.02 | −8.29 |
| −0.02 | +3.31 |
| **0.0** | **−11.65** |

cosine(intended, received) ≈ **−0.72** over realistic embedding values. Worse,
`prefill4tok` zero-pads rows 4–127, so **124 of its 128 rows arrived as a constant
−11.65 instead of 0**.

A shard-0 fed that cannot produce anything resembling the reference. Cosines of
0.07 / −0.01 / −0.04 / −0.01 are the expected consequence.

Additionally, the probe inputs shipped in `03_vl4b_v5/` were copied verbatim from
the previous bundle when the session folder was assembled; only the `.bin` files
were swapped. So the kit in the operator's hands was stale by construction.

**This is a build-side defect, not an operator error.** It is fixed (§6).

### 3.2 What survives the defect

`run_text_probe.sh` runs shard 1 **twice** per case:

* `<case>_s1chain` — shard 1 fed the **device's own** shard-0 output
* `<case>_s1iso` — shard 1 fed the **host reference** boundary

`_s1iso`'s inputs are `last_hidden_states`, `attention_mask`, the rope tables and
a zero past cache. **None of those dtypes changed in the v5 rebuild.** So
`_s1iso` is unaffected by the encoding bug and remains a valid measurement.

The report quotes a single `decode1tok` logits figure without saying which run it
came from. That distinction is the whole question.

---

## 4. Test A — re-analyse the data you already have (no device time)

Everything needed is inside `v5_probe_out.tar.gz` (456 files, ~72 MB), already
captured. **Do this before anything else.**

```bash
tar xzf v5_probe_out.tar.gz            # -> text_probe_out/

$PY_DEPLOY scripts/validate/compare_text_probe.py \
    --kit     <bundle>/03_vl4b_v5 \
    --results text_probe_out \
    --ctxbin-info-0 <bundle>/03_vl4b_v5/qwen3vl-4b-w8a16_1_of_2.info.json \
    --ctxbin-info-1 <bundle>/03_vl4b_v5/qwen3vl-4b-w8a16_2_of_2.info.json
```

With the **old** kit this now **stops with exit 2** and prints the encoding
mismatch rather than a verdict — that is the fix working, and it confirms the
diagnosis in §3.1.

To get the number that matters out of the existing capture, read the
`shard1-isolated` rows specifically. The comparator prints all three runs per
case, labelled `shard0`, `shard1-chained`, `shard1-isolated`. If you would rather
report raw numbers, the relevant tensors are:

```
text_probe_out/decode1tok_s1iso/...     logits   <- the valid one
text_probe_out/decode1tok_s1chain/...   logits   <- inherits shard 0's bad input
text_probe_out/decode1tok_s0/...        last_hidden_states  <- fed wrong bytes
text_probe_out/prefill4tok_s1iso/...    logits   <- the valid one
```

compared against `03_vl4b_v5/<case>/ref/logits.npy`.

### Reading Test A

| `shard1-isolated` | meaning | next |
|---|---|---|
| **agrees with host** (cos ≳ 0.99, argmax matches) | the encoding artefact explains everything; **no ctx-bin defect is demonstrated** | run Test B to re-measure shard 0 properly |
| **disagrees** | shard 1's ctx-bin is genuinely wrong, independent of the encoding bug | the export/quantize/convert/split hunt is justified — start with §7 |

---

## 5. Test B — re-run the probe with a corrected kit (~10 min on device)

Only needed after Test A. The corrected kit encodes every input to the dtype the
ctx-bin actually declares.

```bash
adb push probe_kit_u16/. /data/local/tmp/v5/
adb shell
cd /data/local/tmp/v5 && export LD_LIBRARY_PATH=.
sh run_text_probe.sh 2>&1 | tee text_probe2.log
exit
adb pull /data/local/tmp/v5/text_probe_out ./text_probe_out2
```

Then on the host, the same `compare_text_probe.py` command as Test A but with
`--results text_probe_out2`. It will print the encoding cross-check as **OK**
before any numbers; if it does not, stop and report.

### Reading Test B

| shard0 | s1-isolated | s1-chained | meaning |
|---|---|---|---|
| ok | ok | ok | ctx-bins are **fine**; the 4B fault is in Genie's driving of them, not the graphs |
| **bad** | ok | bad | fault is in **shard 0** |
| ok | **bad** | bad | fault is in **shard 1** (it owns `lm_head`) |
| ok | ok | **bad** | the shard-0 → shard-1 **boundary hand-off** corrupts `last_hidden_states` |
| bad | bad | bad | fault spans both shards — suspect the shared encodings or the split |

`decode1tok` clean + `prefill4tok` broken would additionally point at rope /
cross-token attention rather than plain numerics: position 0 makes cos = 1, sin = 0,
so `decode1tok` exercises the arithmetic with rope as the identity.

---

## 6. What was fixed on the build side (2026-08-19)

| Fix | Detail |
|---|---|
| Probe kit is dtype-aware | `build_text_probe_kit.py` now takes `--ctxbin-info-0/-1`, reads each input's `dataType` and `quantizeParams.scaleOffset` from the bin that will execute, encodes to it, warns on out-of-range clipping, and records what it wrote in `cases.json`. An unhandled dtype is a hard error, never a silent fp16 fallback. |
| Comparator refuses stale data | `compare_text_probe.py` cross-checks those recorded dtypes against the bin actually run and **exits 2 instead of printing a verdict** when they differ. |
| `_comment` removed | Both 0.6B probe configs carried a top-level `_comment`; libGenie's whitelist rejected the whole document (`Unknown dialog config key: _comment`) and the operator had to strip it by hand. Notes moved to sibling `<name>.notes.md`. |
| Config whitelist gated | `lint_bundle_dialogs.py` now rejects any unknown top-level key in a `genie_*.json`, so an unloadable config cannot ship again. |

---

## 7. If a real ctx-bin defect is confirmed, bisect in this order

Cheapest and most likely first. Do **not** start here before Test A.

1. **The `inputs_embeds` encoding range (build-side, ours, cheapest).** For the
   VL tower the encoding must cover both text-LUT rows (±0.24) and spliced image
   features (±11.65), so it is fitted to ±11.65 with step 3.54e-04. Typical text
   embedding magnitudes (~±0.02) therefore get ~7–8 effective bits, versus ~10
   bits of *relative* precision under `FLOAT_16`. This is a conversion-only
   rebuild to test and it is a change **we** made. It cannot be the whole story —
   the 4B was broken before it — but rule it out first.
2. **Host-side ONNX bisect (free).** `parity_e2e_vl.py` already reproduces HF
   token-for-token from the same ONNX the DLCs were converted from. If the ONNX
   is right and the ctx-bin is wrong, the fault is in convert or ctx-bin
   generation, not export or quantization. This narrows §11 of the device report
   from four candidates to one or two without touching hardware.
3. **Split boundary.** Compare shard 0's `last_hidden_states` output encoding
   against shard 1's input encoding, and confirm the layer ranges (0–17 / 18–35)
   and global KV indices.
4. **Only then** export/quantization: per-channel axis, fused-QKV and gate-up
   layouts, rope-theta 5e6, MRoPE.

---

## 8. Operational notes carried forward from the session

* **Profiles are only written on clean completion.** When `genie-t2t-run` ends
  with `Context Size was exceeded / Failed to query`, the profile handle is freed
  without flushing and the JSON is lost — 7 of ~8 runs were lost this way. Use
  `--max-num-tokens` so the query completes. This was a defect in our guide, not
  in the tool.
* `QnnDevice_setConfig(enableHtpExtension) failed: 0x3e8` under `qnn-net-run` is
  benign; the process exits 0 and produces output tensors.
* `adb` USB drops on long sessions: `adb reconnect`, or
  `adb kill-server; adb start-server`.

---

## 9. Bottom line

Two defects were in play and they are not the same thing:

1. **Genie's `FLOAT_16` embedding padding** — confirmed, understood, fixed by
   `UFIXED_POINT_16`, verified on device at 0.6B.
2. **Whatever still breaks the 4B text tower** — real, but *not yet localised*.
   The evidence that pointed at the ctx-bins came from a probe that was fed the
   wrong bytes, and must be re-established before anyone spends days in the
   export/quantization pipeline.

**Test A costs nothing and decides which.**
