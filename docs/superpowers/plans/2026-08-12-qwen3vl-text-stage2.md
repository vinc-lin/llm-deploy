# Qwen3-VL-4B Text Tower (Stage 2) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build and numerically validate a W8A16 QNN context binary of the
Qwen3-VL-4B-Instruct **text tower**, taking `inputs_embeds` (not `input_ids`)
with optional deepstack injection, plus the embedding LUT Genie needs.

**Architecture:** The Qwen3-VL text tower is architecturally identical to
Qwen3 text — same RMSNorm, per-head q_norm/k_norm, GQA, SwiGLU MLP — so the
existing `ExportQwen3` wrapper is extended in place rather than rewritten.
Three surgical changes: embeddings-in instead of tokens-in, additive deepstack
after layers 0/1/2, and an externalised embedding LUT. Every existing Genie
contract carries over untouched.

**Tech Stack:** PyTorch 2.13 / transformers 5.14.1 (`qwen3-deploy`), AIMET
W8A16, ONNX opset 17, QAIRT 2.48.40.260702, ONNX Runtime.

---

## Context an implementer needs

Read `docs/superpowers/specs/2026-08-12-qwen3-vl-4b-sa8797p-design.md` (esp. §5.1,
§7-RESOLVED, §9b) and `docs/NOTES-genie-io.md` before starting.

**Always `source scripts/env.sh` first.** Use `$PY_DEPLOY` / `$PY_QAIRT`, never
bare `python`. No pytest — validation is standalone argparse scripts in
`scripts/validate/` that assert and exit non-zero.

### Model dimensions (Qwen3-VL-4B-Instruct text_config)

| | |
|---|---|
| Layers | 36 |
| Hidden | 2560 |
| Heads / KV heads | 32 / 8 (head_dim 128) |
| Intermediate | 9728 |
| Vocab | 151936, `tie_word_embeddings: true` |
| rope_theta | **5e6** |
| mrope_section | `[24, 20, 20]` (sums to 64 = pos-id-dim) |

Checkpoint: `$LLMDEPLOY_DATA/models/Qwen3-VL-4B-Instruct`.
Text tower module: `hf.model.language_model` (vision is `hf.model.visual`).

### Build parameters (fixed)

CL=2048, prefill AR=128, decode AR=1 → `past_len = 2048 + 128 - 1 = 2175`,
total window 2176.

### The Genie contracts, verified in SDK source

**1. `inputs_embeds` — the tensor name is load-bearing.** `nsp-model.cpp:668`
looks for an input literally named `inputs_embeds`; finding it sets
`m_layerNames[LayerType::INPUT] = "inputs_embeds"` and
`m_inputType = InputType::EMBEDDINGS`. Documented shape is
`[1, 1, AR, embd_size]` (rank 4, `nsp-graph.cpp:92`). Validation is on element
*count*: `numElements == n_tokens * batch * m_embd_size`. Use rank-4 to match
the documented convention.

**2. MRoPE needs NO graph change.** This is the key simplification.
`nsp-model.cpp:3803` shows Genie computes the 3-D position IDs host-side —
including the vision-token layout from `m_visionParam` — interleaves them per
`mrope_section`, and feeds ordinary `POS_COS`/`POS_SIN` graph inputs. So the
graph keeps `position_ids_cos`/`position_ids_sin` of shape `[1, AR, 64]`
exactly as the text pipeline already does. Genie errors if
`pos_dim != sum(mrope_section)`; ours is 64 == 24+20+20. MRoPE is a **config**
change, not a graph change.

**3. Deepstack is additive after layers 0/1/2.** HF applies it *after* the
decoder layer runs (`modeling_qwen3_vl.py:835`), and `_deepstack_process` does
`hidden_states[visual_pos_masks, :] += visual_embeds`. For a static graph the
equivalent formulation is a **zero-padded full-width** tensor added
unconditionally: `x = x + deepstack_i`. Mathematically identical when the host
zero-pads non-visual positions, and it makes the input optional by
construction — all zeros is exactly HF-minus-deepstack.

**4. Everything else carries over unchanged.** All-position prefill logits
`[1,AR,vocab]`; past keys transposed `[1,n_kv,D,P]`; past values natural
`[1,n_kv,P,D]`; outputs carry only the new slice; rank-3 additive mask
`[1,AR,CTX]` with `-100` for masked; one encodings lineage across
prefill/decode.

### Stage 1 findings that apply here

- **The QNN CPU backend has no FP16/quantized execution path.** A converted-DLC
  gate must run against a separately converted FP32 DLC of the same ONNX.
- **`graph_names` is a name-keyed selector in BOTH the build-time and runtime
  HTP configs.** A graph absent from the list silently compiles at O=0 / 4 MB
  VTCM. Assert `optimizationLevel`, `vtcmSize`, `numHvxThreads` and `graphName`
  read back out of the finalised binary. See `docs/NOTES-vit-htp-config.md`.
- `qairt-converter` rewrites activation patterns (it swapped GELU variants on
  the ViT). Watch the conversion log for passes touching SiLU/SwiGLU.

### Known risk to surface, not solve

Genie cannot route deepstack through a stock `ImageEncoder` node (spec §7).
The deepstack graph inputs therefore exist for a custom driver. **Whether Genie
zero-initialises graph inputs it does not feed is unverified** — if it does,
a stock pipeline degrades gracefully to HF-minus-deepstack; if it feeds
garbage, output is wrong. Task 1 makes the inputs a build flag so both
artifacts are buildable; resolving the zero-init question is Stage 3's job.

---

## File Structure

| File | Responsibility |
|---|---|
| `scripts/export/modeling_export.py` (modify) | Add embeddings-in + deepstack to `ExportQwen3` |
| `scripts/export/export_qwen3vl_text.py` (create) | ONNX export CLI for prefill + decode |
| `scripts/export/extract_embed_lut.py` (create) | Embedding table → int8 LUT + quant params |
| `scripts/validate/parity_vl_text.py` (create) | ONNX vs HF, qualla feed pattern |
| `scripts/build/vl_text_build.sh` (create) | AIMET → convert → ctx-bin |
| `scripts/validate/lint_vl_text_contract.py` (create) | ctx-bin contract + HTP config lint |
| `configs/genie_dialog_qwen3vl_4b.json` (create) | Genie text-generator config |
| `configs/htp_backend_ext_config_vltext.json` (create) | Runtime HTP config scoped to our graph names |

---

## Task 1: Embeddings-in + deepstack in the export wrapper

**Files:**
- Modify: `scripts/export/modeling_export.py`
- Test: `scripts/validate/parity_vl_text_wrapper.py` (create)

`ExportQwen3` is shared with the working text pipeline. **Every change must be
backwards compatible** — the existing text builds must be bit-identical
afterwards. Both new features are off by default.

- [ ] **Step 1: Write the failing test**

Create `scripts/validate/parity_vl_text_wrapper.py`:

```python
#!/usr/bin/env python
"""Gate 1 (Stage 2): Qwen3-VL text wrapper vs HF, torch level.

Covers the two new modes together, because they are only ever used together:
embeddings-in (no in-graph embed lookup) and additive deepstack after layers
0/1/2. Deepstack is fed non-zero here on purpose -- an all-zero probe would
pass even if the wrapper ignored the inputs entirely.

Run:
  $PY_DEPLOY scripts/validate/parity_vl_text_wrapper.py \
      --model $LLMDEPLOY_DATA/models/Qwen3-VL-4B-Instruct
"""
import argparse
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "export"))
from modeling_export import ExportQwen3, causal_mask, rope_tables, rope_theta_of  # noqa: E402

TOL = 5e-3
S = 16


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    args = ap.parse_args()

    from transformers import Qwen3VLForConditionalGeneration
    hf = Qwen3VLForConditionalGeneration.from_pretrained(
        args.model, dtype=torch.float32, attn_implementation="eager"
    ).eval()
    lm = hf.model.language_model
    cfg = hf.config.text_config
    n_deep = 3

    torch.manual_seed(0)
    embeds = torch.randn(1, S, cfg.hidden_size) * 0.02
    deep = [torch.randn(1, S, cfg.hidden_size) * 0.02 for _ in range(n_deep)]
    mask_full = torch.ones(1, S, dtype=torch.long)

    # HF reference: deepstack applied at ALL positions, so visual_pos_masks is all-True.
    with torch.no_grad():
        ref = lm(
            inputs_embeds=embeds,
            attention_mask=mask_full,
            deepstack_visual_embeds=[d[0] for d in deep],
            visual_pos_masks=torch.ones(1, S, dtype=torch.bool),
        ).last_hidden_state
        ref_logits = hf.lm_head(ref)

    m = ExportQwen3.from_hf_vl_text(hf, use_past=False, n_deepstack=n_deep)
    mask = causal_mask(S, S)
    cos, sin = rope_tables(torch.arange(S), cfg.head_dim, rope_theta_of(cfg))
    with torch.no_grad():
        ours = m(embeds, mask, cos, sin, *deep)[0]

    assert torch.isfinite(ours).all(), "wrapper produced non-finite logits"
    assert torch.isfinite(ref_logits).all(), "HF reference produced non-finite logits"
    assert ours.shape == ref_logits.shape, f"{ours.shape} != {ref_logits.shape}"
    d = (ours - ref_logits).abs().max().item()
    print(f"  logits max|d| = {d:.3e}  shape={tuple(ours.shape)}")
    assert d < TOL, f"wrapper diverges from HF: {d:.3e} >= {TOL}"

    # Deepstack must actually be wired: zeroing it MUST change the output.
    with torch.no_grad():
        zeroed = m(embeds, mask, cos, sin, *[torch.zeros_like(x) for x in deep])[0]
    dz = (zeroed - ours).abs().max().item()
    print(f"  zero-deepstack delta = {dz:.3e} (must be >> 0)")
    assert dz > 1e-3, "deepstack inputs are ignored by the wrapper"

    print(f"PASS: VL text wrapper matches HF (max|d|={d:.3e})")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Run it, confirm it fails for the right reason**

```bash
source scripts/env.sh
$PY_DEPLOY scripts/validate/parity_vl_text_wrapper.py --model $LLMDEPLOY_DATA/models/Qwen3-VL-4B-Instruct
```
Expected: `AttributeError: type object 'ExportQwen3' has no attribute 'from_hf_vl_text'`.

- [ ] **Step 3: Extend `ExportQwen3` in `scripts/export/modeling_export.py`**

Three edits. Keep defaults so existing text builds are unaffected.

3a. `__init__` — add two keyword args and skip the embedding table when
embeddings-in:

```python
    def __init__(self, cfg, fuse_gate_up=False, fuse_qkv=False, use_past=True,
                 logits_last_only=False, input_embeds=False, n_deepstack=0):
        super().__init__()
        self.cfg = cfg
        self.use_past = use_past
        self.logits_last_only = logits_last_only
        self.input_embeds = input_embeds
        self.n_deepstack = n_deepstack
        # Embeddings-in: the runtime does the lookup from an external LUT so it
        # can splice visual features into the sequence, so the table must NOT
        # be in the graph -- it would add 389M params of dead weight.
        self.embed_tokens = None if input_embeds else nn.Embedding(cfg.vocab_size, cfg.hidden_size)
        ...  # layers / norm / lm_head unchanged
```

3b. `forward` — accept embeddings, and add deepstack after layers 0..n-1:

```python
    def forward(self, input_ids, attention_mask, position_ids_cos, position_ids_sin, *rest):
        # Embeddings-in mode: `input_ids` IS the embedding tensor. Deepstack
        # tensors come first in *rest, then past KV, so the ONNX input order
        # matches the declared input_names.
        deep = rest[: self.n_deepstack]
        past = rest[self.n_deepstack:]

        x = input_ids if self.input_embeds else self.embed_tokens(input_ids.to(torch.long))
        if self.input_embeds and x.dim() == 4:
            x = x.squeeze(1)  # Genie's documented [1,1,AR,H] -> [B,S,H]

        new_kv = []
        for i, layer in enumerate(self.layers):
            pk = past[2 * i] if self.use_past else None
            pv = past[2 * i + 1] if self.use_past else None
            x, nk, nv = layer(x, position_ids_cos, position_ids_sin, attention_mask, pk, pv)
            new_kv.extend([nk, nv])
            # HF applies deepstack AFTER the layer (modeling_qwen3_vl.py:835),
            # adding only at visual positions. The host zero-pads elsewhere, so
            # an unconditional add is equivalent -- and all-zeros is exactly
            # HF-minus-deepstack, which is the graceful-degradation path.
            if i < self.n_deepstack:
                d = deep[i]
                x = x + (d.squeeze(1) if d.dim() == 4 else d)

        x = self.norm(x)
        if self.logits_last_only:
            x = x[:, -1:, :]
        return (self.lm_head(x), *new_kv)
```

3c. Add the VL loader beside `from_hf`. It reuses `from_hf`'s weight mapping
but reads from the VL checkpoint's nested text tower and drops the embedding
table:

```python
    @staticmethod
    def from_hf_vl_text(hf_vl, fuse_gate_up=False, fuse_qkv=False, use_past=True,
                        logits_last_only=False, n_deepstack=3):
        """Build the export module from a Qwen3VLForConditionalGeneration.

        The VL checkpoint nests the text tower under `model.language_model.`
        and ties lm_head to `model.language_model.embed_tokens.weight`
        (tie_word_embeddings: true).
        """
        cfg = hf_vl.config.text_config
        m = ExportQwen3(cfg, fuse_gate_up, fuse_qkv, use_past, logits_last_only,
                        input_embeds=True, n_deepstack=n_deepstack)
        src = hf_vl.state_dict()
        p = "model.language_model."
        dst = {"norm.weight": src[p + "norm.weight"]}
        dst["lm_head.weight"] = src.get("lm_head.weight", src[p + "embed_tokens.weight"])
        for i in range(cfg.num_hidden_layers):
            s, d = f"{p}layers.{i}.", f"layers.{i}."
            dst[d + "input_layernorm.weight"] = src[s + "input_layernorm.weight"]
            dst[d + "post_attention_layernorm.weight"] = src[s + "post_attention_layernorm.weight"]
            dst[d + "self_attn.q_norm.weight"] = src[s + "self_attn.q_norm.weight"]
            dst[d + "self_attn.k_norm.weight"] = src[s + "self_attn.k_norm.weight"]
            dst[d + "self_attn.o_proj.weight"] = src[s + "self_attn.o_proj.weight"]
            if fuse_qkv:
                dst[d + "self_attn.qkv_proj.weight"] = torch.cat(
                    [src[s + "self_attn.q_proj.weight"],
                     src[s + "self_attn.k_proj.weight"],
                     src[s + "self_attn.v_proj.weight"]], dim=0)
            else:
                for w in ("q_proj", "k_proj", "v_proj"):
                    dst[d + f"self_attn.{w}.weight"] = src[s + f"self_attn.{w}.weight"]
            dst[d + "mlp.down_proj.weight"] = src[s + "mlp.down_proj.weight"]
            if fuse_gate_up:
                dst[d + "mlp.gate_up_proj.weight"] = torch.cat(
                    [src[s + "mlp.gate_proj.weight"], src[s + "mlp.up_proj.weight"]], dim=0)
            else:
                for w in ("gate_proj", "up_proj"):
                    dst[d + f"mlp.{w}.weight"] = src[s + f"mlp.{w}.weight"]
        m.load_state_dict(dst, strict=True)
        return m.eval()
```

If the real state-dict keys differ from `model.language_model.*`, print
`[k for k in src if 'layers.0.' in k][:5]` and adapt — do not guess.

- [ ] **Step 4: Run the gate until it passes**

```bash
$PY_DEPLOY scripts/validate/parity_vl_text_wrapper.py --model $LLMDEPLOY_DATA/models/Qwen3-VL-4B-Instruct
```
Expected: `logits max|d|` under 5e-3, a large `zero-deepstack delta`, then
`PASS`. Do not weaken `TOL`.

If HF's `language_model` forward rejects `visual_pos_masks`/
`deepstack_visual_embeds`, inspect its real signature in
`modeling_qwen3_vl.py` (`Qwen3VLTextModel.forward`, ~line 772) and adapt the
reference call — the wrapper side stays as specified.

- [ ] **Step 5: Prove the existing text pipeline is unaffected**

This is the regression that matters — `modeling_export.py` is shared code.

```bash
$PY_DEPLOY scripts/validate/parity_qualla_read.py --help
```
and re-run whichever existing gate exercises `ExportQwen3` with default args
against Qwen3-0.6B. Report what you ran and its output. If no cheap existing
gate exists, construct a minimal equivalence check: build `ExportQwen3.from_hf`
on Qwen3-0.6B before and after your change (via `git stash`) and assert the
logits are bit-identical.

- [ ] **Step 6: Commit**

```bash
git add scripts/export/modeling_export.py scripts/validate/parity_vl_text_wrapper.py
git commit -m "feat: embeddings-in + deepstack modes for Qwen3-VL text tower"
```

---

## Task 2: Embedding LUT extraction

**Files:**
- Create: `scripts/export/extract_embed_lut.py`

Genie does the token-embedding lookup host-side from an external LUT, because
the runtime must splice visual features into the sequence. Config shape (from
the SDK's `glm-4v.json`):

```json
"embedding": { "version": 1, "type": "lut", "lut-path": "embedding_int8_lut.bin",
               "size": 2560, "datatype": "ufixed8",
               "quant-param": { "scale": <float>, "offset": <int> } }
```

- [ ] **Step 1: Write the extractor**

Create `scripts/export/extract_embed_lut.py`. It must:
- load the VL checkpoint's `model.language_model.embed_tokens.weight`
  (`[151936, 2560]`, and note `tie_word_embeddings: true` means this is also
  the lm_head weight — do NOT modify it)
- quantize to **ufixed8** with a single scale/offset over the whole table
  (asymmetric: `scale = (max-min)/255`, `offset = round(-min/scale)` expressed
  in the sign convention the SDK example uses — glm-4v shows a negative
  offset, `-129`, so match that convention and say in your report which
  convention you concluded and why)
- write the raw uint8 table to `embedding_int8_lut.bin` (row-major, vocab-major)
- print the resulting scale/offset and the **max dequantization error**, plus
  the per-row cosine between original and dequantized rows (min across rows)
- assert the LUT file size is exactly `vocab_size * hidden_size` bytes

- [ ] **Step 2: Run it**

```bash
source scripts/env.sh
$PY_DEPLOY scripts/export/extract_embed_lut.py \
  --model $LLMDEPLOY_DATA/models/Qwen3-VL-4B-Instruct \
  --out $LLMDEPLOY_DATA/work/lut/qwen3vl-4b
```
Expected: a 389,003,840-byte file (151936 × 2560) and printed quant params.

Report the max dequant error. If it is large enough to worry you (say, worse
than 1% relative on typical rows), say so — an 8-bit whole-table scale is what
the SDK example uses, but our vocab is 74× larger than glm-4v's and the
dynamic range may not tolerate it. That is a finding, not something to hide.

- [ ] **Step 3: Commit**

```bash
git add scripts/export/extract_embed_lut.py
git commit -m "feat: embedding LUT extraction for Genie embeddings-in path"
```

---

## Task 3: ONNX export (prefill + decode)

**Files:**
- Create: `scripts/export/export_qwen3vl_text.py`

Model it directly on `scripts/export/export_qwen3.py`, which already handles
prefill/decode pairs, the all-position-logits contract, and the KV naming.

- [ ] **Step 1: Write the exporter**

Requirements:
- args: `--model`, `--out`, `--cl-prefill` (default 128), `--ctx` (default
  2048), `--n-deepstack` (default 3), `--parity-check`
- build with `ExportQwen3.from_hf_vl_text(...)`
- **input names, in this exact order:** `inputs_embeds`, `attention_mask`,
  `position_ids_cos`, `position_ids_sin`, `deepstack_visual_embed_0..2`, then
  `past_key_{i}_in` / `past_value_{i}_in` for the decode graph
- `inputs_embeds` shape **`[1, 1, AR, 2560]`** (rank 4, Genie's documented
  convention); deepstack the same rank for consistency
- output names: `logits`, then `past_key_{i}_out` / `past_value_{i}_out`
- **prefill MUST emit all-position logits `[1, 128, 151936]`** — `logits_last_only=False`
- decode: `past_len = ctx + cl_prefill - 1 = 2175`
- carry over Stage 1's static-shape guard: assert every graph I/O is fully
  static (positive int dims, known rank) and that input names match exactly
- run `onnx.checker.check_model`

- [ ] **Step 2: Export**

```bash
$PY_DEPLOY scripts/export/export_qwen3vl_text.py \
  --model $LLMDEPLOY_DATA/models/Qwen3-VL-4B-Instruct \
  --out $LLMDEPLOY_DATA/work/onnx/qwen3vl-4b-text --parity-check
```

The FP32 graphs are large (~16 GB of weights across both). **Expect ONNX to
write external data** — unlike the ViT, this will exceed the 2 GiB protobuf
limit. Report the exact on-disk layout, because Task 5's converter invocation
depends on it.

- [ ] **Step 3: Commit**

```bash
git add scripts/export/export_qwen3vl_text.py
git commit -m "feat: ONNX export for Qwen3-VL text tower (embeds-in, deepstack)"
```

---

## Task 4: ONNX parity gate with the qualla feed pattern

**Files:**
- Create: `scripts/validate/parity_vl_text.py`

This is the gate that would have caught the 2026-08-11 device bug. Model it on
the existing `scripts/validate/parity_qualla_read.py`, which already encodes
qualla's feed/read semantics.

- [ ] **Step 1: Write the gate**

It must, against ONNX Runtime and HF:
- feed the prompt **left-aligned** into the prefill graph and read logits at
  row `n_process - 1` (NOT the last row) — the exact pattern from
  `nsp-model.cpp:3295`
- then run several decode steps with past-KV, checking argmax agreement with
  HF at every step
- exercise deepstack with **non-zero** values at a contiguous span of positions
  and zeros elsewhere, mirroring how a real visual span appears
- assert finiteness of both sides before any reduction (Stage 1's NaN lesson:
  `max(0.0, nan)` is `0.0`, so NaN is silently swallowed)
- assert greedy argmax matches HF for every generated token

- [ ] **Step 2: Run until it passes**

Report the per-step argmax agreement. **Argmax must match exactly** — this is
a stricter bar than a float tolerance and it is the bar the text pipeline uses.

- [ ] **Step 3: Prove it is not vacuous**

Mutation-test as every Stage 1 gate was: e.g. read logits at the wrong row
(the original bug) and confirm the gate fails. Scratchpad only, commit nothing
extra.

- [ ] **Step 4: Commit**

```bash
git add scripts/validate/parity_vl_text.py
git commit -m "test: qualla-feed-pattern parity gate for Qwen3-VL text tower"
```

---

## Task 5: AIMET W8A16 quantization

**Files:**
- Create: `scripts/build/vl_text_build.sh` (quantization stages only for now)

**This is the long pole: a 4 B model quantized on CPU. Budget hours, not
minutes.** `QUANT_DEVICE=cpu` is mandatory (8 GB VRAM box).

Model the script on `scripts/build/full_build.sh`, which already encodes the
critical cross-graph rule: **prefill and decode DLCs must convert against the
SAME encodings lineage** (`--export-decode` / `--adopt-encodings`). Mixed
encodings are a fatal Genie load error because KV quant params must be
byte-identical.

- [ ] **Step 1: Calibration data**

Calibration must be **multimodal** — text sequences with real image embeddings
spliced in — because the activation ranges of a tower fed spliced visual
features differ from a text-only tower. Use the Stage 1 ViT (its ONNX at
`$LLMDEPLOY_DATA/work/onnx/qwen3vl-4b-vit/vit.onnx`) to produce real
`image_features`, and splice them at the image-token positions of a handful of
prompts built with the model's chat template.

Report how many calibration samples you used and why.

- [ ] **Step 2: Run quantization**

Follow `full_build.sh`'s two-call structure (prefill quantize, then decode
export adopting the prefill encodings), then the encodings filter and the I/O
rename — noting `--layers 36`, not 28.

- [ ] **Step 3: Report**

Wall-clock, peak RSS, and the `quantize_aimet.py --eval` result. The text
pipeline's reference bar is **3/4 last-token argmax agreement**.

- [ ] **Step 4: Commit**

```bash
git add scripts/build/vl_text_build.sh
git commit -m "feat: AIMET W8A16 quantization for Qwen3-VL text tower"
```

---

## Task 6: DLC conversion and ctx-bin

**Files:**
- Modify: `scripts/build/vl_text_build.sh`

- [ ] **Step 1: Convert both graphs**

Per `full_build.sh`: convert prefill and decode against the SAME
`model_filtered_renamed.encodings`, `--float_bitwidth 16 --target_backend HTP`,
with explicit `-d` dims for every input including the three deepstack tensors
and all 72 past-KV tensors (36 layers × 2).

- [ ] **Step 2: Generate the ctx-bin**

Two-graph, weight-shared. **Write a private HTP config with
`graph_names: ["prefill", "decode"]`** — do not reuse the shared
`configs/htp_config.json` (Stage 1 established it binds by name; see
`docs/NOTES-vit-htp-config.md`). Keep `context.weight_sharing_enabled: true`
here — unlike the ViT, this IS a multi-graph context and weight sharing is
what keeps the binary from doubling.

**Expect this to be large (~5-7 GB) and expect it may need splitting into
multiple parts** (the SDK's glm-4v uses `1_of_2`, llava `1_of_4`). If a single
binary fails to generate, that is the multi-part requirement arriving — report
it with the exact error before attempting a split.

- [ ] **Step 3: Assert the config bound**

Carry over Stage 1's hardening: read back `graphName`, `optimizationLevel`,
`vtcmSize`, `numHvxThreads` from the finalised binary and fail the build on
mismatch. Also assert both graphs are present and share weights
(`sharedWeightsSize > 0`).

- [ ] **Step 4: Commit**

```bash
git add scripts/build/vl_text_build.sh
git commit -m "feat: DLC conversion + weight-shared ctx-bin for Qwen3-VL text"
```

---

## Task 7: Converted-DLC parity gate

**Files:**
- Create: `scripts/validate/parity_vl_text_dlc.py`

Model it on `scripts/validate/parity_vit_dlc.py`.

**The QNN CPU backend cannot execute a quantized or FP16 DLC** — Stage 1
established it rejects them at graph composition (`OpConfig validation failed
for FullyConnected`). So convert the same ONNX to a separate FP32 DLC purely
for validation, put it in the scratchpad (never in `work/dlc/`), and run that.

State plainly in the docstring what this covers (the converter's translation
of the graph) and what it does not (W8A16 quantization error, FP16 rounding,
the shipped file's weights).

- [ ] **Step 1: Write it. Step 2: Run it. Step 3: Mutation-test it. Step 4: Commit.**

```bash
git add scripts/validate/parity_vl_text_dlc.py
git commit -m "test: converted-DLC parity gate for Qwen3-VL text tower"
```

---

## Task 8: Contract lint, Genie config, bundle

**Files:**
- Create: `scripts/validate/lint_vl_text_contract.py`
- Create: `configs/genie_dialog_qwen3vl_4b.json`
- Create: `configs/htp_backend_ext_config_vltext.json`
- Modify: `scripts/build/vl_text_build.sh` (bundle stage)

- [ ] **Step 1: Contract lint**

Assert, by name: `inputs_embeds` present on both graphs; prefill logits are
`[1, 128, 151936]` (all-position — the 2026-08-11 bug); past-key tensors
transposed `[1,8,128,P]` and past-values `[1,8,P,128]`; no unexpected tensors;
exactly two graphs. Report all violations at once.

- [ ] **Step 2: Genie config**

`configs/genie_dialog_qwen3vl_4b.json` — a `dialog` config carrying:
- the `embedding` LUT block from Task 2 (with the real scale/offset)
- `positional-encoding` with `rope-scaling.rope-type: "qwen3vl-mrope"`,
  `mrope_section: [24,20,20]`, `rope-theta: 5000000`, `rope-dim: 64`
- `context.size: 2048`, `n-vocab: 151936`, correct eos tokens
- `extensions` pointing at `htp_backend_ext_config_vltext.json`, whose
  `graph_names` must be `["prefill", "decode"]`

- [ ] **Step 3: Bundle + verify internal references resolve**

Same as Stage 1's `vit_bundle.sh`: flat layout, verify every filename
referenced inside the JSON exists in the bundle, and verify no ctx-bin graph
is left unbound by `graph_names`.

- [ ] **Step 4: Commit**

```bash
git add scripts/validate/lint_vl_text_contract.py configs/genie_dialog_qwen3vl_4b.json \
        configs/htp_backend_ext_config_vltext.json scripts/build/vl_text_build.sh
git commit -m "feat: contract lint, Genie config and bundle for Qwen3-VL text tower"
```

---

## Definition of done

- [ ] Wrapper matches HF and the existing text pipeline is provably unregressed
- [ ] Prefill emits all-position logits; qualla-feed gate argmax-matches HF
- [ ] Embedding LUT extracted with reported quantization error
- [ ] W8A16 ctx-bin built, HTP config asserted bound, weight sharing confirmed
- [ ] Converted-DLC gate passes, with its coverage stated honestly
- [ ] Every gate mutation-tested
- [ ] Memory footprint measured and reported against the unknown device budget
