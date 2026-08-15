# HTP v81 MatMul / FullyConnected kernel table — what dtype combinations exist

**Audited 2026-08-16** against QAIRT 2.48.40.260702. Source of truth:
`$QAIRT_SDK/lib/python/qti/aisw/converters/common/backend_aware_configs/htp_v2.json`,
the converter's own backend-awareness table. This is the same file and the same
method that settled the W4A16 kernel question.

## Why this file exists

Two of this repo's most expensive failures were dtype-combination failures that
the tooling reports only as an opaque validation error at ctx-bin generation
time (`validateOpConfig failed 3110`, `Failed to validate op … with error
0xc26`). Both were answerable in advance by reading this table. It is short
enough to reproduce in full, so there is no reason to guess again.

## `MatMul` — all 21 supported kernels on v81

Signature is `input0 × input1 × input2(bias, optional) -> output`.
`num_mandatory_inputs: 2`.

| input0 | input1 | bias | output |
|---|---|---|---|
| ufxp8 | ufxp8 | ufxp8 / sfxp32 | ufxp8 |
| ufxp8 | sfxp8 | ufxp8 / sfxp32 | ufxp8 |
| sfxp8 | ufxp8 | ufxp8 / sfxp32 | sfxp8 |
| sfxp8 | sfxp8 | ufxp8 / sfxp32 | sfxp8 |
| ufxp16 | ufxp8 | ufxp8 / sfxp32 | ufxp16 |
| ufxp16 | sfxp8 | ufxp8 / sfxp32 | ufxp16 |
| ufxp16 | ufxp16 | ufxp8 / sfxp32 | ufxp16 |
| ufxp16 | sfxp16 | ufxp8 / sfxp32 | ufxp16 |
| sfxp16 | sfxp8 | sfxp32 | sfxp16 |
| sfxp16 | sfxp16 | sfxp32 | sfxp16 |
| **f16** | **f16** | f16 | **f16** |
| **f16** | **sfxp8** | f16 | **f16** |
| f32 | f32 | f32 | f32 |

`FullyConnected` is the same list minus `sfxp16 × sfxp16` (20 kernels).

## The three facts that matter

### 1. There is exactly ONE mixed float×fixed kernel: `f16 × sfxp8 -> f16`

Not `f16 × ufxp8`. Not `f16 × ufxp16`. Not `f16 × sfxp16`. **Signed INT8 only,
and only in the input1 slot.**

This retro-explains the Qwen3-VL ViT failure exactly (`REFERENCE.md` §4): the
converter left V as `uFxp_16` and asked HTP for `FLOAT_16 x UFIXED_16 ->
FLOAT_16`, which is not in the table and never was. The fix (dropping that one
activation encoding per block so V converts to FP16) was the right one, and this
table says no converter flag could have produced a working alternative.

### 2. Quantized KV is reachable — but only as `sfxp8`

`f16 × sfxp8 -> f16` has **unconstrained** `input_quant_params` on input1, i.e.
per-tensor `SCALE_OFFSET` is allowed, not just the per-axis weight encodings.
That is what a *dynamic* tensor like a KV cache needs. So:

- **`REFERENCE.md` §4.1's "unconfirmed kernel path" for native KV INT8 is now
  confirmed to exist**, and HTP-doc open question 8 is answered on the kernel
  half. What remains is ONNX-level graph work, not a capability gap.
- **The KV cache must be signed INT8.** Unsigned INT8 (`f16 × ufxp8`) is not in
  the table; AIMET's default for activations is *unsigned* 16-bit
  (`uFxp_16`), so this requires a symmetric signed 8-bit quantizer on the
  K/V-projection outputs specifically.
- **An INT16 KV cache is NOT reachable while activations stay FP16.** There is
  no `f16 × ufxp16` or `f16 × sfxp16` kernel. The all-fixed-point 16-bit
  kernels (`ufxp16 × sfxp16 -> ufxp16`) require input0 — Q, and the softmax
  output — to be fixed-point too, i.e. quantizing the whole attention body.
  **This inverts the "INT16 first as the safe fallback" plan**: INT16 is the
  *harder* path here, not the easier one.

Corollary for the cross-graph rule (`NOTES-genie-io.md`): `validateModel`'s
Check 4 already compares `(scale, offset)` byte-for-byte on KV tensors across
graphs, and `checkShape` carries a per-tensor `bitwidth`. The runtime therefore
anticipates quantized KV; our `--export-decode` / `--adopt-encodings` pipeline
satisfies the identity requirement by construction.

### 3. The W4A16 kernel claim needs a precision fix (it does not revive W4A16)

`REFERENCE.md` correction #8 and §4.1 say *"**zero** INT4 MatMul/FC entries in
`htp_v2.json`"*. Precisely:

- **Zero INT4 input *datatypes*** — every `input_datatypes` entry is ≥ 8-bit.
  True as stated, and this is the load-bearing half.
- **But 31 MatMul and 21 FullyConnected `input_quant_params` entries declare
  `Bitwidth: "4"`, and 15/13 declare `"2"`**, under `BW_AXIS_SCALE_OFFSET` —
  i.e. LPBQ-style block-quantized sub-byte weights unpacked into an 8-bit
  container.

So v81 *does* have a 4-bit weight representation; it has no 4-bit *storage*
datatype for a MatMul input. W4A16 remains dead on both original grounds —
qairt-converter folds s4 → f16 (HTP doc §5.3/§10.1), and all three recipes
including LPBQ block-64 scored **0/4** on the argmax gate at 0.6B. The claim
should be restated so that someone who greps the file and finds `"Bitwidth":
["4"]` does not wrongly conclude the dead end was mis-diagnosed.

## Reproducing this

```python
import json
P = "$QAIRT_SDK/lib/python/qti/aisw/converters/common/backend_aware_configs/htp_v2.json"
d = json.load(open(P))
for k in d["MatMul"]["supported_kernels"]:
    print([ (i["bitwidth"], i["dtype"], i["is_signed"]) for i in k["input_datatypes"] ],
          "->", [ (o["bitwidth"], o["dtype"]) for o in k["output_datatypes"] ])
```

The table is per-arch: `htp_v2.json` is the v68/v73/v75/v79/v81 family file, and
`super_groups_v73` / `super_groups_v68` keys inside it show it carries
arch-specific fusion groups. Re-audit on any SDK bump — the W4A16 kernel gap was
identical in 2.43 and 2.48, but that is not a guarantee for 2.5x.
