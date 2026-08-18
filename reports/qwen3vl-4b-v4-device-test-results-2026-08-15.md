# Qwen3-VL-4B v4 — Device Test Results for the Bundle Developer

**Source:** device team, received 2026-08-15, pasted into the build session.
**Committed:** 2026-08-19 — it had been acted on but never written down, and
`reports/` is supposed to be the record. Body below is **verbatim**; the
annotations at the end are ours and are clearly marked.

---

2026-08-15. Two defects, one fixed, one root-caused. No ctx-bin was rebuilt.

## 1. Image crash: fixed

**Problem (v3):** SIGSEGV (SEGV_ACCERR) at `GenieNode_setData+572`, pc `0x646e84`,
fault at the page-rounded end of the image buffer.

**Root cause:** `QnnNspImageModel::setupInput` branches on `embedding-datatype`,
which defaults to `QNN_DATATYPE_FLOAT_32` with no config route to change it for
an image-encoder node. It always does:

```cpp
float* embeddingSrc = reinterpret_cast<float*>(inputs.data());
quantizeInput(embeddingSrc, name, 0, numElements);  // reads numElements × 4 bytes
```

v3's UFixed16 blob was numElements × 2 bytes — a 2× (~3 MB) over-read past the
buffer into the Scudo guard page. Every v3 diagnostic (T2 probe, T3 mmap, T5
qnn-net-run) is explained by this mechanism.

**Fix:** ship the image as float32 (`*_fp32.raw`, 6,295,552 bytes). Genie
quantizes on-device against the ctx-bin's own encoding. The three ctx-bins and
LUT are byte-identical to v3 (md5s in `V4_CHANGES.md` §4).

**Device evidence:** 3 cold/warm runs of the sample-image pipeline at ~11.1 s, no
crash. 6 real photographs all loaded and executed without SIGSEGV.

## 2. Text tower: garbage output — root cause identified

**Problem (v3, unchanged in v4):** the text tower produces garbage on device
despite passing host parity (20/20 token-identical). The v4 `bos-token` fix made
no difference.

**Root cause:** the 4B text tower uses AIMET Quantsim QDQ-baked INT8, the same
quantization approach that already produced garbage for 0.6B.

| | Working 0.6B | Broken 4B text tower |
|---|---|---|
| Quantization | `qairt-quantizer` | AIMET Quantsim |
| Weight format in DLC | FP16 + per-channel symmetric encodings | Q/DQ ops baked into ONNX, collapsed to INT8 |
| DLC / ctx-bin size | 1.5 GB (0.6B × FP16) | 4.3 GB (4B × INT8) |
| Pipeline | `sa8797 convert --dtype w8a16` | GenAI notebook example1/example2 |

The failure mode is double quantization: AIMET bakes Q/DQ ops into the ONNX, AND
the converter applies `--quantization_overrides` on top of the already-quantized
graph. The host parity gate validates ONNX numerics — it never executes the
QDQ-baked DLC, so this class of defect is invisible to the gate.

**Device evidence:**

* **V1 text-only:** `Assistantairsabilityability...` → mixed Cyrillic/Latin
  repetition → context exceeded. Same pattern as v3.
* **V2 sample-image pipeline:** no crash, garbage caption (`fttyuringuring...`)
* **V3 weather kit (6 real photos):** all ran, all produced empty output (zero
  tokens). Different image embeddings select different degenerate first tokens in
  the broken logits: real photos collapse to EOS, synthetic sample to garbage.

## 3. What to change

Replace AIMET Quantsim with `qairt-quantizer` for the text tower — the same
pipeline that works for 0.6B and 1.7B:

```
FP16 ONNX (no AIMET Q/DQ)
→ qairt-converter → FP16 DLC
→ qairt-quantizer (per-channel symmetric, --float_bitwidth 16)
→ W8A16 DLC (FP16 weights + encodings, NOT QDQ-baked)
→ qnn-context-binary-generator → ctx-bin
```

The `sa8797_deploy_kit` pipeline (`sa8797 convert --model qwen3-vl-4b --dtype
w8a16`) already runs this flow. What's needed: export FP16 ONNX for the text
tower; generate calibration inputs for `qairt-quantizer`; run the standard
pipeline; re-verify host parity; bundle + test on device.

The ViT ctx-bin and LUT are unaffected.

## 4. Verification strategy

1. Host parity gate (same 7-chain harness, 20/20 expected)
2. `qnn-net-run` on the text decode graph (the method that vindicated the ViT in T5)
3. Text-only device test (`genie-t2t-run`, "What is 2+2?")
4. Full e2e pipeline with sample image
5. Weather kit (6 real photographs)

## 5. Fixed reference points

* QAIRT SDK 2.48.40.260702, QNN API v2.37.0, libGenie 1.19.0 (BuildId `f6899695c925325c`)
* ViT ctx-bin runs correctly under both `qnn-net-run` and Genie (v4)
* Image format: float32, 6,295,552 bytes (`*_fp32.raw`)
* Text tower: 36 layers, n_embd 2560, GQA (20 heads, 4 KV heads), rope-theta 5M,
  MRoPE, 2-shard ctx-bins (layers 0-17 / 18-35 + lm_head)
* The host parity gate is not a DLC gate. It validates ONNX numerics.

---

## Annotations (build side, added 2026-08-19)

**§1 is correct and was acted on.** v4 ships fp32 image blobs; confirmed here.
The mechanism generalises — see the note on §2 below.

**§2's proposed root cause is falsified.** Do not spend a rebuild on it:

* **0** `QuantizeLinear`/`DequantizeLinear` ops in either ONNX. AIMET's
  `sim.export()` writes a clean ONNX plus a *separate* `.encodings`; QDQ is baked
  only with `use_embedded_encodings=True`, which this project never sets. Nothing
  is QDQ-baked, so nothing can be double-quantized.
* The **working 0.6B** (44.707 tok/s on device) is built by the **same**
  `quantize_aimet.py` path as the 4B. `qairt-quantizer` appears nowhere in this
  repo, so it cannot be what makes the 0.6B work.
* The 4B DLC already carries `sFxp_8` per-channel `axis-quant` weights with
  `uFxp_16` activations — i.e. exactly the per-channel symmetric INT8 the
  proposed fix was meant to produce.
* The head counts in §5 are wrong for this checkpoint: it is **32 q / 8 kv**, not
  20/4.

**The actual mechanism is the same class of bug as §1, one layer over.**
`QnnNspModel::quantizeInput` advances its destination pointer by `tensorOffset`
**elements** for `UFIXED_8`/`UFIXED_16`/`FLOAT_32` but by **bytes** for
`FLOAT_16` (`nsp-model.cpp:3144`), while `setupInputEmbeddings` passes an element
count when padding a partially-filled prefill chunk (`:1813`). Our
`inputs_embeds` was `FLOAT_16`, so the pad write started halfway into the real
prompt and overwrote its back half. It fires only when `variant > n_process` —
the last, partial prefill chunk — which is why decode looks fine and why the
0.6B, which feeds `input_ids` and never calls `setupInputEmbeddings`, is
unaffected.

This also explains §2's own evidence that the QDQ theory does not: garbage from
**both** the dialog path (V1) and the pipeline path (V2), host parity 20/20, and
photos collapsing to EOS while the synthetic sample degenerates into repetition.

**Fix:** graft a 16-bit INT activation encoding onto `inputs_embeds` so the
converter types it `uFxp_16` (`scripts/quant/graft_input_encoding.py`). Verified
at DLC level 2026-08-19. `--preserve_io_datatype` is **not** the way — it pins
the input to float32 and graph-prepare then fails with `could not create op:
q::QNN_Convert`.

**Caveat:** the SDK's `qualla` tree is not the source of the shipped
`libGenie.so`. The v5 session's Test 1 (127/128/129-token triad) is what settles
it on hardware.
