# Qwen3-VL-4B Vision Tower (Stage 1) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build, numerically validate, and publish to Hugging Face an FP16 QNN
context binary of the Qwen3-VL-4B-Instruct vision tower for SA8797P.

**Architecture:** A static-shape export wrapper reuses the HF
`Qwen3VLVisionModel` submodules directly (blocks, mergers) but precomputes
every position-dependent tensor for one fixed image grid and folds the Conv3d
patch embed into a Linear. One ONNX graph → FP16 DLC → ctx-bin → Genie
image-encoder bundle. Four numerical gates run before publication, the
strongest being execution of the converted DLC on the QNN CPU backend.

**Tech Stack:** PyTorch 2.13 / transformers 5.14.1 (`qwen3-deploy` uv env),
ONNX opset 17, onnxruntime, QAIRT 2.48.40.260702 (`qairt-converter`,
`qnn-net-run`, `qnn-context-binary-generator`), Hugging Face Hub CLI.

---

## Context an implementer needs

Read `docs/superpowers/specs/2026-08-12-qwen3-vl-4b-sa8797p-design.md` first,
then `docs/NOTES-genie-io.md` §Genie graph I/O contract.

**Always `source scripts/env.sh` first.** It sets `$QAIRT_SDK`, `$PY_DEPLOY`,
`$PY_QAIRT`, `$LLMDEPLOY_ROOT`, `$LLMDEPLOY_DATA`, and the `LD_LIBRARY_PATH`
needed by the SDK binaries. Nothing works without it.

**This repo has no pytest.** Validation follows the existing convention:
standalone argparse scripts in `scripts/validate/` that assert and exit
non-zero on failure. Match the style of `scripts/validate/parity_verify.py`.

**Fixed build parameters** (from the spec, do not change):

| Thing | Value |
|---|---|
| Checkpoint | `$LLMDEPLOY_DATA/models/Qwen3-VL-4B-Instruct` |
| Image bucket | 512×512 → `grid_thw = [1, 32, 32]` |
| Patch tokens | 1024 |
| Visual tokens out | 256 |
| `pixel_values` | `[1024, 1536]` (1536 = 3 ch × 2 temporal × 16 × 16) |
| Outputs | `image_features [256, 2560]`, `deepstack_visual_embed_{0,1,2} [256, 2560]` |
| Precision | FP16 (no AIMET, no calibration) |

**Why the main output is named `image_features`:** Genie's image model
initialises `m_layerNames[LayerType::OUTPUT] = "image_features"` by default
(`nsp-image-model.hpp`), overriding it only for outputs literally named
`vision_embedding` or `cross_attention_states`. Naming ours `image_features`
hits the default path with no override logic involved.

**Why the deepstack outputs ride along on the same graph:** per spec §7 their
routing into the text tower is unresolved. Emitting them costs nothing now and
they are needed in Stage 3 under outcomes (a) and (b). Genie ignores extra
outputs it was not told to read.

---

## File Structure

| File | Responsibility |
|---|---|
| `scripts/export/modeling_vit_export.py` (create) | The static-shape ViT wrapper. Pure model code, no I/O, no argparse. |
| `scripts/export/export_qwen3vl_vit.py` (create) | ONNX export CLI + `--parity-check`. |
| `scripts/validate/parity_vit_wrapper.py` (create) | Gate 1: wrapper vs HF, torch level. |
| `scripts/validate/parity_vit.py` (create) | Gate 2: ONNX Runtime vs HF. |
| `scripts/validate/parity_vit_dlc.py` (create) | Gate 3: converted DLC on QNN CPU backend vs HF. |
| `scripts/validate/lint_vit_contract.py` (create) | Gate 4: ctx-bin graph I/O contract lint. |
| `scripts/build/vit_build.sh` (create) | ONNX → DLC → ctx-bin. |
| `scripts/build/vit_bundle.sh` (create) | ctx-bin → flat device bundle. |
| `configs/genie_image_encoder_qwen3vl.json` (create) | Genie image-encoder engine config. |

---

## Task 1: Static-shape vision export wrapper

**Files:**
- Create: `scripts/export/modeling_vit_export.py`
- Test: `scripts/validate/parity_vit_wrapper.py`

- [ ] **Step 1: Write the failing test**

Create `scripts/validate/parity_vit_wrapper.py`:

```python
#!/usr/bin/env python
"""Gate 1: static-shape ViT wrapper vs HF Qwen3VLVisionModel (torch level).

Proves the wrapper's constant-folding (position embeddings, rotary tables,
Conv3d->Linear patch embed) is mathematically identical to HF's dynamic path.

Run:
  $PY_DEPLOY scripts/validate/parity_vit_wrapper.py \
      --model $LLMDEPLOY_DATA/models/Qwen3-VL-4B-Instruct
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "export"))
from modeling_vit_export import ExportQwen3VLViT, make_pixel_values  # noqa: E402

TOL = 1e-4


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    args = ap.parse_args()

    from transformers import Qwen3VLForConditionalGeneration
    hf = Qwen3VLForConditionalGeneration.from_pretrained(
        args.model, dtype=torch.float32, attn_implementation="eager"
    ).eval()
    vision = hf.model.visual

    pixel_values, grid_thw = make_pixel_values(args.model, seed=0)
    assert tuple(grid_thw[0].tolist()) == (1, 32, 32), f"unexpected grid {grid_thw}"
    assert pixel_values.shape == (1024, 1536), f"unexpected pixels {pixel_values.shape}"

    with torch.no_grad():
        ref = vision(pixel_values, grid_thw)
        ref_main = ref.pooler_output
        ref_deep = ref.deepstack_features

        wrapper = ExportQwen3VLViT(vision, grid_thw).eval()
        got = wrapper(pixel_values)

    assert len(got) == 1 + len(ref_deep), f"expected {1+len(ref_deep)} outputs, got {len(got)}"

    worst = 0.0
    for name, a, b in [("image_features", got[0], ref_main)] + [
        (f"deepstack_visual_embed_{i}", got[1 + i], ref_deep[i]) for i in range(len(ref_deep))
    ]:
        assert a.shape == b.shape, f"{name}: shape {a.shape} != {b.shape}"
        d = (a - b).abs().max().item()
        cos = torch.nn.functional.cosine_similarity(
            a.flatten().unsqueeze(0), b.flatten().unsqueeze(0)
        ).item()
        print(f"  {name:28s} shape={tuple(a.shape)} max|d|={d:.3e} cos={cos:.8f}")
        worst = max(worst, d)

    assert worst < TOL, f"wrapper diverges from HF: max|d|={worst:.3e} >= {TOL}"
    print(f"PASS: wrapper matches HF (worst max|d|={worst:.3e})")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Run test to verify it fails**

```bash
source scripts/env.sh
$PY_DEPLOY scripts/validate/parity_vit_wrapper.py --model $LLMDEPLOY_DATA/models/Qwen3-VL-4B-Instruct
```

Expected: FAIL with `ModuleNotFoundError: No module named 'modeling_vit_export'`.

- [ ] **Step 3: Write the implementation**

Create `scripts/export/modeling_vit_export.py`:

```python
"""Export-friendly Qwen3-VL vision tower for QNN HTP / Genie deployment.

Static-shape: every position-dependent tensor is precomputed for ONE fixed
image grid and stored as a buffer, so the exported graph takes exactly one
input (pixel_values) and needs no attention mask.

Two properties of Qwen3-VL make this clean, and both are load-bearing:

1. Qwen3-VL DROPPED Qwen2.5-VL's windowed attention. For a single image
   cu_seqlens degenerates to [0, N], so attention is plain full attention and
   the per-window torch.split becomes a single no-op chunk that traces away.
2. Position handling depends only on grid_thw -- the bilinear interpolation of
   the 48x48 learned pos-embed (get_vision_bilinear_indices_and_weights) and
   the rotary tables (get_vision_position_ids). At a fixed grid both are
   constants.

The HF patch embed is a Conv3d whose kernel EQUALS its stride, i.e. every
patch is projected independently -- mathematically a Linear over the flattened
patch. We fold it into a Linear because MatMul support on HTP is far more
certain than Conv3d.

Patch ORDER is not rearranged here: the HF image processor already emits
pixel_values pre-shuffled so that each group of spatial_merge_size**2
consecutive patches is one 2x2 spatial block, which is what the mergers'
.view(-1, hidden*4) relies on.
"""
import torch
import torch.nn as nn
from transformers.vision_utils import (
    get_vision_bilinear_indices_and_weights,
    get_vision_position_ids,
)


class ExportQwen3VLViT(nn.Module):
    """Qwen3-VL vision tower specialised to one grid_thw.

    Args:
        vision: an HF ``Qwen3VLVisionModel`` (fp32, eager attention).
        grid_thw: ``[1, 3]`` int tensor, the (t, h, w) patch grid.

    Returns from forward: ``(image_features, *deepstack_visual_embed)``, each
    ``[num_patches // merge_unit, out_hidden_size]``.
    """

    def __init__(self, vision, grid_thw):
        super().__init__()
        cfg = vision.config
        self.blocks = vision.blocks
        self.merger = vision.merger
        self.deepstack_merger_list = vision.deepstack_merger_list
        self.deepstack_visual_indexes = list(cfg.deepstack_visual_indexes)

        # --- Conv3d -> Linear (kernel == stride, so patches are independent)
        conv = vision.patch_embed.proj
        patch_dim = (
            cfg.in_channels * cfg.temporal_patch_size * cfg.patch_size * cfg.patch_size
        )
        self.patch_embed = nn.Linear(patch_dim, cfg.hidden_size, bias=True)
        with torch.no_grad():
            self.patch_embed.weight.copy_(conv.weight.reshape(cfg.hidden_size, patch_dim))
            self.patch_embed.bias.copy_(conv.bias)

        # --- constant position embeddings for this grid
        with torch.no_grad():
            idx, w = get_vision_bilinear_indices_and_weights(
                grid_thw,
                num_grid_per_side=vision.num_grid_per_side,
                spatial_merge_size=cfg.spatial_merge_size,
            )
            pos_embeds = (vision.pos_embed(idx) * w[:, :, None]).sum(0)

            pos_ids = get_vision_position_ids(grid_thw, cfg.spatial_merge_size)
            rot = vision.rotary_pos_emb(pos_ids)
            seq_len = int(grid_thw[:, 0].sum() * grid_thw[0, 1] * grid_thw[0, 2])
            rot = rot.reshape(seq_len, -1)
            emb = torch.cat((rot, rot), dim=-1)

        self.register_buffer("pos_embeds", pos_embeds, persistent=False)
        self.register_buffer("rope_cos", emb.cos(), persistent=False)
        self.register_buffer("rope_sin", emb.sin(), persistent=False)
        self.register_buffer(
            "cu_seqlens", torch.tensor([0, seq_len], dtype=torch.int32), persistent=False
        )

    def forward(self, pixel_values):
        h = self.patch_embed(pixel_values) + self.pos_embeds
        pos = (self.rope_cos, self.rope_sin)

        deep = []
        for i, blk in enumerate(self.blocks):
            h = blk(h, cu_seqlens=self.cu_seqlens, position_embeddings=pos)
            if i in self.deepstack_visual_indexes:
                merger = self.deepstack_merger_list[self.deepstack_visual_indexes.index(i)]
                deep.append(merger(h))

        return (self.merger(h), *deep)


def output_names(n_deepstack=3):
    """Canonical graph output names. `image_features` matches the default in
    Genie's nsp-image-model.hpp, so no name-override logic is involved."""
    return ["image_features"] + [f"deepstack_visual_embed_{i}" for i in range(n_deepstack)]


def make_pixel_values(model_dir, seed=0, edge=512):
    """Deterministic synthetic image -> (pixel_values, grid_thw) via the real
    HF processor, so patch ordering and normalisation match production exactly.

    Hermetic on purpose: this runs in every gate, and the proxy is unreliable.
    Numerical parity compares two implementations of identical math, so image
    semantics are irrelevant -- only shape, ordering and value range matter.
    """
    import numpy as np
    from PIL import Image
    from transformers import AutoProcessor

    rng = np.random.default_rng(seed)
    img = Image.fromarray(rng.integers(0, 256, (edge, edge, 3), dtype=np.uint8))

    proc = AutoProcessor.from_pretrained(
        model_dir, min_pixels=edge * edge, max_pixels=edge * edge
    )
    out = proc.image_processor(images=img, return_tensors="pt")
    return out["pixel_values"].to(torch.float32), out["image_grid_thw"]
```

- [ ] **Step 4: Run test to verify it passes**

```bash
$PY_DEPLOY scripts/validate/parity_vit_wrapper.py --model $LLMDEPLOY_DATA/models/Qwen3-VL-4B-Instruct
```

Expected: four `max|d|` lines all below `1e-4`, then
`PASS: wrapper matches HF`.

If `make_pixel_values` returns a grid other than `[1,32,32]`, the processor's
`smart_resize` disagreed with the min/max pixel pinning — print
`out["image_grid_thw"]` and adjust `edge` so `edge` is a multiple of
patch_size × spatial_merge_size (32) before changing anything else.

- [ ] **Step 5: Commit**

```bash
git add scripts/export/modeling_vit_export.py scripts/validate/parity_vit_wrapper.py
git commit -m "feat: static-shape Qwen3-VL ViT export wrapper + torch parity gate"
```

---

## Task 2: ONNX export

**Files:**
- Create: `scripts/export/export_qwen3vl_vit.py`

- [ ] **Step 1: Write the exporter**

Create `scripts/export/export_qwen3vl_vit.py`:

```python
#!/usr/bin/env python
"""Export the Qwen3-VL vision tower to a static-shape ONNX graph.

Usage:
  $PY_DEPLOY scripts/export/export_qwen3vl_vit.py \
      --model $LLMDEPLOY_DATA/models/Qwen3-VL-4B-Instruct \
      --out   $LLMDEPLOY_DATA/work/onnx/qwen3vl-4b-vit [--parity-check]

Emits vit.onnx: pixel_values [1024,1536] -> image_features [256,2560]
plus deepstack_visual_embed_{0,1,2} [256,2560].

FP16 happens later at qairt-converter (--float_bitwidth 16); this graph is
all-FP32, matching the text pipeline's convention.
"""
import argparse
import sys
from pathlib import Path

import onnx
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from modeling_vit_export import (  # noqa: E402
    ExportQwen3VLViT,
    make_pixel_values,
    output_names,
)

OPSET = 17


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--parity-check", action="store_true",
                    help="compare wrapper outputs vs HF before exporting")
    args = ap.parse_args()

    from transformers import Qwen3VLForConditionalGeneration
    hf = Qwen3VLForConditionalGeneration.from_pretrained(
        args.model, dtype=torch.float32, attn_implementation="eager"
    ).eval()
    vision = hf.model.visual

    pixel_values, grid_thw = make_pixel_values(args.model, seed=0)
    print(f"grid_thw={grid_thw.tolist()} pixel_values={tuple(pixel_values.shape)}")

    model = ExportQwen3VLViT(vision, grid_thw).eval()

    if args.parity_check:
        with torch.no_grad():
            ref = vision(pixel_values, grid_thw)
            ours = model(pixel_values)
        d = (ours[0] - ref.pooler_output).abs().max().item()
        print(f"wrapper-vs-HF max|d image_features| = {d:.3e}")
        assert d < 1e-4, "wrapper does not match HF vision forward"

    names_out = output_names(len(vision.deepstack_visual_indexes))
    out_path = Path(args.out) / "vit.onnx"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with torch.no_grad():
        torch.onnx.export(
            model,
            (pixel_values,),
            str(out_path),
            input_names=["pixel_values"],
            output_names=names_out,
            opset_version=OPSET,
            do_constant_folding=True,
            dynamo=False,
        )

    onnx.checker.check_model(str(out_path))
    m = onnx.load(str(out_path), load_external_data=False)
    print(f"  {out_path}: {len(m.graph.node)} nodes, opset {OPSET}")
    for t in m.graph.input:
        dims = [d.dim_value for d in t.type.tensor_type.shape.dim]
        print(f"  IN  {t.name}: {dims}")
    for t in m.graph.output:
        dims = [d.dim_value for d in t.type.tensor_type.shape.dim]
        print(f"  OUT {t.name}: {dims}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Run the export**

```bash
source scripts/env.sh
$PY_DEPLOY scripts/export/export_qwen3vl_vit.py \
  --model $LLMDEPLOY_DATA/models/Qwen3-VL-4B-Instruct \
  --out $LLMDEPLOY_DATA/work/onnx/qwen3vl-4b-vit --parity-check
```

Expected output must include exactly:

```
  IN  pixel_values: [1024, 1536]
  OUT image_features: [256, 2560]
  OUT deepstack_visual_embed_0: [256, 2560]
  OUT deepstack_visual_embed_1: [256, 2560]
  OUT deepstack_visual_embed_2: [256, 2560]
```

If any shape has a `0` or symbolic dim, a constant did not fold — stop and fix
the wrapper rather than proceeding with a dynamic graph.

- [ ] **Step 3: Commit**

```bash
git add scripts/export/export_qwen3vl_vit.py
git commit -m "feat: ONNX export for Qwen3-VL vision tower"
```

---

## Task 3: ONNX Runtime parity gate

**Files:**
- Create: `scripts/validate/parity_vit.py`

- [ ] **Step 1: Write the gate**

Create `scripts/validate/parity_vit.py`:

```python
#!/usr/bin/env python
"""Gate 2: exported ONNX vs HF Qwen3VLVisionModel, via ONNX Runtime.

Catches breakage introduced by the torch->ONNX trace itself (constant folding,
op lowering) that Gate 1 cannot see.

Run:
  $PY_DEPLOY scripts/validate/parity_vit.py \
      --model $LLMDEPLOY_DATA/models/Qwen3-VL-4B-Instruct \
      --onnx  $LLMDEPLOY_DATA/work/onnx/qwen3vl-4b-vit/vit.onnx
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import onnxruntime as ort
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "export"))
from modeling_vit_export import make_pixel_values, output_names  # noqa: E402

TOL = 2e-3
COS_MIN = 0.9999


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--onnx", required=True)
    args = ap.parse_args()

    from transformers import Qwen3VLForConditionalGeneration
    hf = Qwen3VLForConditionalGeneration.from_pretrained(
        args.model, dtype=torch.float32, attn_implementation="eager"
    ).eval()
    vision = hf.model.visual

    pixel_values, grid_thw = make_pixel_values(args.model, seed=1)
    with torch.no_grad():
        ref = vision(pixel_values, grid_thw)
    refs = [ref.pooler_output.numpy()] + [d.numpy() for d in ref.deepstack_features]

    sess = ort.InferenceSession(args.onnx, providers=["CPUExecutionProvider"])
    got = sess.run(None, {"pixel_values": pixel_values.numpy()})

    names = output_names(len(ref.deepstack_features))
    assert len(got) == len(refs), f"expected {len(refs)} outputs, got {len(got)}"

    worst_d, worst_cos = 0.0, 1.0
    for name, a, b in zip(names, got, refs):
        assert a.shape == b.shape, f"{name}: shape {a.shape} != {b.shape}"
        d = float(np.abs(a - b).max())
        cos = float(
            np.dot(a.ravel(), b.ravel())
            / (np.linalg.norm(a.ravel()) * np.linalg.norm(b.ravel()))
        )
        print(f"  {name:28s} max|d|={d:.3e} cos={cos:.8f}")
        worst_d, worst_cos = max(worst_d, d), min(worst_cos, cos)

    assert worst_d < TOL, f"ONNX diverges from HF: max|d|={worst_d:.3e} >= {TOL}"
    assert worst_cos > COS_MIN, f"ONNX cosine {worst_cos:.8f} <= {COS_MIN}"
    print(f"PASS: ONNX matches HF (max|d|={worst_d:.3e}, min cos={worst_cos:.8f})")


if __name__ == "__main__":
    main()
```

Note the gate deliberately uses `seed=1` while the exporter traced with
`seed=0`: a graph that accidentally baked the traced input as a constant
passes at seed 0 and fails here.

- [ ] **Step 2: Run the gate**

```bash
$PY_DEPLOY scripts/validate/parity_vit.py \
  --model $LLMDEPLOY_DATA/models/Qwen3-VL-4B-Instruct \
  --onnx  $LLMDEPLOY_DATA/work/onnx/qwen3vl-4b-vit/vit.onnx
```

Expected: `PASS: ONNX matches HF`.

- [ ] **Step 3: Commit**

```bash
git add scripts/validate/parity_vit.py
git commit -m "test: ONNX Runtime parity gate for Qwen3-VL vision tower"
```

---

## Task 4: DLC conversion and ctx-bin build

**Files:**
- Create: `scripts/build/vit_build.sh`

- [ ] **Step 1: Write the build script**

Create `scripts/build/vit_build.sh`:

```bash
#!/usr/bin/env bash
# Qwen3-VL vision tower: ONNX -> FP16 DLC -> single-graph ctx-bin.
#
# No AIMET stage: the ViT ships FP16 (spec section 4), so there are no
# encodings and no calibration set. FP16 is requested at conversion time via
# --float_bitwidth 16, the same flag the text pipeline uses for its
# non-quantised tensors.
#
# Usage: vit_build.sh [name]
set -euo pipefail
source "$(dirname "$0")/../env.sh"

NAME=${1:-qwen3vl-4b-vit-fp16}
ONNX=$LLMDEPLOY_DATA/work/onnx/qwen3vl-4b-vit/vit.onnx
DLC=$LLMDEPLOY_DATA/work/dlc/$NAME
CTXBIN=$LLMDEPLOY_DATA/work/ctxbin/$NAME
CONVERTER="$QAIRT_SDK/bin/x86_64-linux-clang/qairt-converter"

[ -f "$ONNX" ] || { echo "missing $ONNX -- run export_qwen3vl_vit.py first"; exit 1; }

echo "== [1/3] convert ViT -> FP16 DLC =="
mkdir -p "$DLC"
$PY_QAIRT "$CONVERTER" --input_network "$ONNX" \
    --output_path "$DLC/vit.dlc" \
    --float_bitwidth 16 --target_backend HTP \
    -d pixel_values "1024,1536"

echo "== [2/3] single-graph ctx-bin (vtcm 16, unsigned PD, v81) =="
cd "$LLMDEPLOY_ROOT/configs"
mkdir -p "$CTXBIN"
qnn-context-binary-generator \
    --model "$QAIRT_SDK/lib/x86_64-linux-clang/libQnnModelDlc.so" \
    --dlc_path "$DLC/vit.dlc" \
    --backend "$QAIRT_SDK/lib/x86_64-linux-clang/libQnnHtp.so" \
    --output_dir "$CTXBIN" --binary_file "${NAME}_ctx" \
    --config_file htp_config.json

echo "== [3/3] dump graph info =="
qnn-context-binary-utility --context_binary "$CTXBIN/${NAME}_ctx.bin" \
    --json_file "$CTXBIN/info.json"
$PY_DEPLOY - <<PYEOF
import json
d = json.load(open("$CTXBIN/info.json"))
for g in d["info"]["graphs"]:
    print("GRAPH:", g["info"]["graphName"])
PYEOF
ls -lh "$CTXBIN"
echo "VIT BUILD COMPLETE: $CTXBIN/${NAME}_ctx.bin"
```

- [ ] **Step 2: Run the build**

```bash
chmod +x scripts/build/vit_build.sh
./scripts/build/vit_build.sh
```

Expected: a `GRAPH:` line and `VIT BUILD COMPLETE`, with a ctx-bin of roughly
700 MB (350 M params × 2 bytes).

If `qairt-converter` rejects a rank-2 input spec, retry with the equivalent
`-d pixel_values "1,1024,1536"` and add a leading `unsqueeze` in the wrapper's
forward — but only after confirming the rank-2 form actually failed.

- [ ] **Step 3: Commit**

```bash
git add scripts/build/vit_build.sh
git commit -m "feat: FP16 DLC + ctx-bin build for Qwen3-VL vision tower"
```

---

## Task 5: DLC execution parity on the QNN CPU backend

This is the strongest gate available without a device: it executes the
**converted** graph, so it catches conversion bugs that every earlier gate is
blind to. It runs on the CPU backend because HTP context binaries cannot
execute on x86; the graph topology is identical, only the backend differs.

**Files:**
- Create: `scripts/validate/parity_vit_dlc.py`

- [ ] **Step 1: Write the gate**

Create `scripts/validate/parity_vit_dlc.py`:

```python
#!/usr/bin/env python
"""Gate 3: converted DLC executed by qnn-net-run (CPU backend) vs HF.

Earlier gates validate torch and ONNX. This one validates the artifact that
actually ships -- the DLC produced by qairt-converter. A conversion bug that
survives to here would otherwise only surface as garbage on device, which is
exactly the failure mode that cost this project a device cycle on 2026-08-11.

The CPU backend runs the graph in FP32, so this proves TOPOLOGY and WEIGHTS,
not FP16 rounding. FP16 error is bounded separately by the tolerance below.

Run:
  $PY_DEPLOY scripts/validate/parity_vit_dlc.py \
      --model $LLMDEPLOY_DATA/models/Qwen3-VL-4B-Instruct \
      --dlc   $LLMDEPLOY_DATA/work/dlc/qwen3vl-4b-vit-fp16/vit.dlc
"""
import argparse
import os
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "export"))
from modeling_vit_export import make_pixel_values, output_names  # noqa: E402

TOL = 5e-3
COS_MIN = 0.9995


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--dlc", required=True)
    args = ap.parse_args()

    sdk = os.environ["QAIRT_SDK"]
    net_run = f"{sdk}/bin/x86_64-linux-clang/qnn-net-run"
    backend = f"{sdk}/lib/x86_64-linux-clang/libQnnCpu.so"

    from transformers import Qwen3VLForConditionalGeneration
    hf = Qwen3VLForConditionalGeneration.from_pretrained(
        args.model, dtype=torch.float32, attn_implementation="eager"
    ).eval()
    vision = hf.model.visual

    pixel_values, grid_thw = make_pixel_values(args.model, seed=2)
    with torch.no_grad():
        ref = vision(pixel_values, grid_thw)
    refs = [ref.pooler_output.numpy()] + [d.numpy() for d in ref.deepstack_features]
    names = output_names(len(ref.deepstack_features))

    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        raw = td / "pixel_values.raw"
        pixel_values.numpy().astype(np.float32).tofile(raw)
        (td / "input_list.txt").write_text(f"pixel_values:={raw}\n")

        cmd = [
            net_run,
            "--backend", backend,
            "--dlc_path", args.dlc,
            "--input_list", str(td / "input_list.txt"),
            "--output_dir", str(td / "out"),
        ]
        r = subprocess.run(cmd, capture_output=True, text=True)
        if r.returncode != 0:
            print(r.stdout[-4000:]); print(r.stderr[-4000:])
            raise SystemExit(f"qnn-net-run failed ({r.returncode})")

        result_dir = td / "out" / "Result_0"
        worst_d, worst_cos = 0.0, 1.0
        for name, b in zip(names, refs):
            f = result_dir / f"{name}.raw"
            assert f.exists(), f"missing output {f}; got {sorted(p.name for p in result_dir.iterdir())}"
            a = np.fromfile(f, dtype=np.float32).reshape(b.shape)
            d = float(np.abs(a - b).max())
            cos = float(
                np.dot(a.ravel(), b.ravel())
                / (np.linalg.norm(a.ravel()) * np.linalg.norm(b.ravel()))
            )
            print(f"  {name:28s} max|d|={d:.3e} cos={cos:.8f}")
            worst_d, worst_cos = max(worst_d, d), min(worst_cos, cos)

    assert worst_d < TOL, f"DLC diverges from HF: max|d|={worst_d:.3e} >= {TOL}"
    assert worst_cos > COS_MIN, f"DLC cosine {worst_cos:.8f} <= {COS_MIN}"
    print(f"PASS: DLC matches HF (max|d|={worst_d:.3e}, min cos={worst_cos:.8f})")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Run the gate**

```bash
$PY_DEPLOY scripts/validate/parity_vit_dlc.py \
  --model $LLMDEPLOY_DATA/models/Qwen3-VL-4B-Instruct \
  --dlc   $LLMDEPLOY_DATA/work/dlc/qwen3vl-4b-vit-fp16/vit.dlc
```

Expected: `PASS: DLC matches HF`.

`qnn-net-run`'s output-file naming and `--input_list` syntax vary across QAIRT
releases. If it errors on the input list, run `qnn-net-run --help` and adapt
the two lines that build `input_list.txt` and read `Result_0` — do not weaken
the tolerances to make the gate pass.

- [ ] **Step 3: Commit**

```bash
git add scripts/validate/parity_vit_dlc.py
git commit -m "test: converted-DLC parity gate via qnn-net-run CPU backend"
```

---

## Task 6: Contract lint on the ctx-bin

**Files:**
- Create: `scripts/validate/lint_vit_contract.py`

- [ ] **Step 1: Write the lint**

Create `scripts/validate/lint_vit_contract.py`:

```python
#!/usr/bin/env python
"""Gate 4: static contract lint of the ViT ctx-bin against Genie's expectations.

Genie's image model (nsp-image-model.hpp) keys off exact tensor NAMES:
  input  "pixel_values"   -> LayerType::INPUT
  output "image_features" -> LayerType::OUTPUT (the built-in default)
A name mismatch is not a load error -- it is silent wrong behaviour on device,
so it is checked here rather than discovered later.

Run:
  $PY_DEPLOY scripts/validate/lint_vit_contract.py \
      --info $LLMDEPLOY_DATA/work/ctxbin/qwen3vl-4b-vit-fp16/info.json
"""
import argparse
import json

EXPECT_IN = {"pixel_values": [1024, 1536]}
EXPECT_OUT = {
    "image_features": [256, 2560],
    "deepstack_visual_embed_0": [256, 2560],
    "deepstack_visual_embed_1": [256, 2560],
    "deepstack_visual_embed_2": [256, 2560],
}


def tensors(graph, key):
    return {t["info"]["name"]: t["info"]["dimensions"] for t in graph["info"].get(key, [])}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--info", required=True)
    args = ap.parse_args()

    d = json.load(open(args.info))
    graphs = d["info"]["graphs"]
    assert len(graphs) == 1, f"expected exactly 1 graph, got {len(graphs)}"
    g = graphs[0]
    print(f"graph: {g['info']['graphName']}")

    ins = tensors(g, "graphInputs")
    outs = tensors(g, "graphOutputs")
    problems = []

    for name, dims in EXPECT_IN.items():
        if name not in ins:
            problems.append(f"missing input {name!r} (have {sorted(ins)})")
        elif list(ins[name]) != dims:
            problems.append(f"input {name}: dims {ins[name]} != {dims}")

    for name, dims in EXPECT_OUT.items():
        if name not in outs:
            problems.append(f"missing output {name!r} (have {sorted(outs)})")
        elif list(outs[name]) != dims:
            problems.append(f"output {name}: dims {outs[name]} != {dims}")

    for name, dims in sorted(ins.items()):
        print(f"  IN  {name}: {dims}")
    for name, dims in sorted(outs.items()):
        print(f"  OUT {name}: {dims}")

    if problems:
        for p in problems:
            print(f"FAIL: {p}")
        raise SystemExit(1)
    print("PASS: ctx-bin matches the Genie image-encoder contract")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Run the lint**

```bash
$PY_DEPLOY scripts/validate/lint_vit_contract.py \
  --info $LLMDEPLOY_DATA/work/ctxbin/qwen3vl-4b-vit-fp16/info.json
```

Expected: `PASS: ctx-bin matches the Genie image-encoder contract`.

The JSON key names (`graphInputs`/`graphOutputs`, `dimensions`) come from
`qnn-context-binary-utility`. If a KeyError fires, inspect `info.json` and fix
the `tensors()` accessor to match the actual schema.

- [ ] **Step 3: Commit**

```bash
git add scripts/validate/lint_vit_contract.py
git commit -m "test: Genie image-encoder contract lint for ViT ctx-bin"
```

---

## Task 7: Genie image-encoder config and device bundle

**Files:**
- Create: `configs/genie_image_encoder_qwen3vl.json`
- Create: `scripts/build/vit_bundle.sh`

- [ ] **Step 1: Write the Genie config**

Create `configs/genie_image_encoder_qwen3vl.json`, modelled on the SDK's
`examples/Genie/configs/glm-4v/siglip.json`:

```json
{
  "image-encoder": {
    "version": 1,
    "engine": {
      "version": 1,
      "mode": "image",
      "backend": {
        "version": 1,
        "type": "QnnHtp",
        "QnnHtp": {
          "version": 1,
          "spill-fill-bufsize": 0,
          "use-mmap": true,
          "mmap-budget": 0,
          "poll": false,
          "cpu-mask": "0xe0",
          "allow-async-init": false
        },
        "extensions": "htp_backend_ext_config.json"
      },
      "model": {
        "version": 1,
        "type": "binary",
        "binary": {
          "version": 1,
          "ctx-bins": [
            "qwen3vl-4b-vit-fp16_ctx.bin"
          ]
        }
      }
    }
  }
}
```

- [ ] **Step 2: Write the bundle script**

Create `scripts/build/vit_bundle.sh`:

```bash
#!/usr/bin/env bash
# Flat, push-ready device bundle for the Qwen3-VL vision tower.
#
# Flat by design: Genie's loader resolves the runtime .so files from the
# bundle root, not from a lib/ subdirectory (see docs/BUILD_GUIDE.md).
#
# Usage: vit_bundle.sh [bundlename] [ctxbin_name]
set -euo pipefail
source "$(dirname "$0")/../env.sh"

BUNDLE=${1:-qwen3vl_4b_vit_fp16}
NAME=${2:-qwen3vl-4b-vit-fp16}
CTXBIN=$LLMDEPLOY_DATA/work/ctxbin/$NAME/${NAME}_ctx.bin
OUT=$LLMDEPLOY_DATA/bundles/$BUNDLE

[ -f "$CTXBIN" ] || { echo "missing $CTXBIN -- run vit_build.sh first"; exit 1; }

rm -rf "$OUT"; mkdir -p "$OUT"
cp "$CTXBIN" "$OUT/"
cp "$LLMDEPLOY_ROOT/configs/genie_image_encoder_qwen3vl.json" "$OUT/image_encoder.json"
cp "$LLMDEPLOY_ROOT/configs/htp_backend_ext_config.json" "$OUT/"

for f in libGenie.so libQnnHtp.so libQnnHtpV81Stub.so libQnnHtpPrepare.so \
         libQnnSystem.so libQnnHtpV81Skel.so libQnnHtpNetRunExtensions.so; do
  src=$(find "$QAIRT_SDK/lib" -name "$f" -path "*aarch64-android*" | head -1)
  [ -n "$src" ] && cp "$src" "$OUT/" || echo "  WARN: $f not found"
done
cp "$QAIRT_SDK/bin/aarch64-android/genie-t2e-run" "$OUT/" 2>/dev/null || \
  echo "  NOTE: genie-t2e-run absent; the image path needs a pipeline driver (spec section 6, Stage 3)"

tar -C "$LLMDEPLOY_DATA/bundles" -czf "$OUT.tar.gz" "$BUNDLE"
ls -lh "$OUT" "$OUT.tar.gz"
echo "BUNDLE COMPLETE: $OUT.tar.gz"
```

- [ ] **Step 3: Run it**

```bash
chmod +x scripts/build/vit_bundle.sh
./scripts/build/vit_bundle.sh
```

Expected: `BUNDLE COMPLETE` and a tarball of roughly 700 MB.

- [ ] **Step 4: Commit**

```bash
git add configs/genie_image_encoder_qwen3vl.json scripts/build/vit_bundle.sh
git commit -m "feat: Genie image-encoder config + device bundle for ViT"
```

---

## Task 8: Publish to Hugging Face

The upload path has two recorded hazards — read
`~/.claude/projects/-mnt-x-code-llm-deploy/memory/hf-upload-visibility-gotcha.md`
and the `hf-artifact-repo` memory before starting. In short: the proxy at
`http://127.0.0.1:17890` is required but drops long streams, the Hub caps
repos at 128 commits/hour, and bulk upload has been observed flipping private
repos public.

The target is a **new** repo — the existing one is named `...-qwen3-w8a16-bundles`
and this is neither Qwen3-text nor W8A16.

- [ ] **Step 1: Write the model card**

Create `$LLMDEPLOY_DATA/bundles/qwen3vl_4b_vit_fp16/README.md`:

```markdown
---
license: apache-2.0
base_model: Qwen/Qwen3-VL-4B-Instruct
tags:
  - qualcomm
  - qnn
  - genie
  - sa8797p
  - vision-encoder
---

# Qwen3-VL-4B-Instruct vision tower — SA8797P (QNN FP16 ctx-bin)

Stage 1 of a Qwen3-VL-4B deployment on the Qualcomm SA8797P (Hexagon v81 HTP,
Android GVM), built against QAIRT 2.48.40.260702 / libGenie 1.19.

This is the **vision tower only**. The text tower and the multimodal pipeline
are Stages 2 and 3 and are not included here.

## Graph contract

| | Name | Shape | Precision |
|---|---|---|---|
| Input | `pixel_values` | `[1024, 1536]` | FP16 |
| Output | `image_features` | `[256, 2560]` | FP16 |
| Output | `deepstack_visual_embed_{0,1,2}` | `[256, 2560]` | FP16 |

Fixed to a single image bucket: **512×512**, `grid_thw = [1, 32, 32]`,
1024 patches → 256 visual tokens. Other resolutions require another graph.

`pixel_values` must come from the HF `Qwen3VLImageProcessor`, which emits
patches pre-shuffled into 2×2 spatial-merge order.

## Validation

Built and validated **without device access**. Four numerical gates, each
against the HF `Qwen3VLVisionModel` reference:

1. Export wrapper vs HF (torch)
2. Exported ONNX vs HF (ONNX Runtime)
3. **Converted DLC vs HF** (`qnn-net-run`, QNN CPU backend)
4. ctx-bin graph I/O contract lint

Gate 3 executes the shipped artifact's graph, but on the CPU backend in FP32 —
it proves topology and weights, not FP16 rounding or HTP scheduling. **This
bundle has never run on an SA8797P.**

## Third-party components

The tarball embeds Qualcomm QAIRT 2.48.40.260702 runtime binaries (aarch64
`.so` files) which the `apache-2.0` tag above does **not** cover. They are
redistributed under Qualcomm's SDK licence terms.
```

- [ ] **Step 2: Create the repo (private)**

Private by default — publishing was not explicitly requested for this new
artifact, and flipping to public later is one command while the reverse is not.

```bash
export https_proxy=http://127.0.0.1:17890 http_proxy=http://127.0.0.1:17890
$PY_DEPLOY -m huggingface_hub.commands.huggingface_cli repo create \
  sa8797p-qwen3vl-4b-bundles --repo-type model --private -y
```

- [ ] **Step 3: Upload**

Upload the extracted bundle directory (not the tarball) so individual files
are fetchable. Use the supervised watchdog — a bare `hf upload` hangs forever
when the proxy drops the stream.

```bash
SOCKET_CHECKS=999999 scripts/util/hf_upload_watchdog.sh \
  vinccniv/sa8797p-qwen3vl-4b-bundles \
  $LLMDEPLOY_DATA/bundles/qwen3vl_4b_vit_fp16
```

If the commit phase stalls with all blobs uploaded and `committed: N/M`
frozen, that is the 128-commits/hour 429. Diagnose with a single foreground
`HfApi().upload_file` — the 429 surfaces in seconds — then wait ~1 h and
finish with one spaced `upload_file` per file.

- [ ] **Step 4: Verify visibility and contents**

Bulk upload has been observed flipping repo visibility. Check, do not assume:

```bash
$PY_DEPLOY -c "
from huggingface_hub import HfApi
i = HfApi().model_info('vinccniv/sa8797p-qwen3vl-4b-bundles', files_metadata=True)
print('private:', i.private)
for s in i.siblings:
    print(f'  {s.rfilename:50s} {(s.size or 0)/1e6:10.1f} MB')
"
```

Expected: `private: True`, the ctx-bin at roughly 700 MB, plus
`image_encoder.json`, `htp_backend_ext_config.json`, `README.md` and the
runtime `.so` files. If `private` is `False`, set it back immediately:

```bash
$PY_DEPLOY -c "
from huggingface_hub import HfApi
HfApi().update_repo_settings('vinccniv/sa8797p-qwen3vl-4b-bundles', private=True)
print('restored to private')
"
```

- [ ] **Step 5: Commit the plan's completion state**

```bash
git add -A docs/
git commit -m "docs: Stage 1 vision tower built, validated, and published"
```

---

## Definition of done

- [ ] All four gates pass and their output is recorded
- [ ] ctx-bin exists and matches the contract in `lint_vit_contract.py`
- [ ] Bundle tarball built
- [ ] Repo `vinccniv/sa8797p-qwen3vl-4b-bundles` contains the bundle, is
      **private**, and the model card states the artifact is device-untested
- [ ] Every new script committed on branch `qwen3-vl-4b`
