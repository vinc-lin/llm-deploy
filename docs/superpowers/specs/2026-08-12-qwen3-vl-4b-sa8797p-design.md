# Qwen3-VL-4B-Instruct on SA8797P — Design

*Date: 2026-08-12 · Status: approved · Target: Qualcomm SA8797P (Hexagon v81
HTP, Android GVM), QAIRT 2.48.40.260702, libGenie 1.19*

## 1. Goal

Deploy **Qwen3-VL-4B-Instruct** as an offline Genie bundle on the SA8797P.

The success criterion is **capability, not speed**: images must work at all.
Decode throughput is explicitly not a goal for this effort — a 4B model on a
platform measured at 100% DDR-bandwidth-bound 7.4–8.2 tok/s for 0.6B will be
slow, and that is accepted.

There is **no device access**, so correctness can only be established
numerically, against the HF reference implementation, using the runtime-feed
emulation methodology that caught the 2026-08-11 prefill-logits bug.

## 2. What the SDK already knows

QAIRT 2.48.40 / Genie 1.19 has first-class Qwen3-VL support. This is not a
port to an unsupported architecture; it is a build against a supported one.

| Evidence | Location |
|---|---|
| `ROPE_QWEN3VL_MROPE` rope type | `nsp-params.hpp:24`, `nsp-params.cpp:22` |
| MRoPE computation citing `transformers v5.0.0rc2 modeling_qwen3_vl.py` | `nsp-model.cpp:3803`, `:3853` |
| `qwen3vl-mrope` accepted in dialog/engine config schema | `Engine.cpp:603,1221`, `Dialog.cpp:875,1628`, `TextGenerator.cpp:98` |
| Image-encoder engine (`pixel_values` → `image_features`) | `nsp-image-model.hpp` |
| `deepstack_visual_embed*` output recognition | `nsp-image-model.cpp:199` |
| Reference two-engine VLM config shape | `configs/glm-4v/{glm-4v,siglip}.json` |

`Qwen3VLForConditionalGeneration` is present in the local `qwen3-deploy` env
(transformers 5.14.1), giving us a reference for both export and parity.

## 3. Model facts

From `Qwen/Qwen3-VL-4B-Instruct/config.json`:

**Text tower** — 36 layers, hidden 2560, 32 heads / 8 KV heads, head_dim 128,
intermediate 9728, vocab 151936, `tie_word_embeddings: true`,
`rope_theta: 5e6`, `rope_scaling: {mrope_interleaved: true, mrope_section:
[24,20,20]}` (sums to 64 = half head_dim, matching the existing pos-id-dim 64
contract). ~3.6 B + 389 M tied embedding ≈ **4.0 B params**.

**Vision tower** — 24 layers, hidden 1024, 16 heads, intermediate 4096,
patch 16, `spatial_merge_size: 2`, `temporal_patch_size: 2`, out_hidden 2560,
`num_position_embeddings: 2304` (48×48 grid), `deepstack_visual_indexes:
[5, 11, 17]`. ≈ **350 M params**.

Two properties make a static-shape ViT export clean:

1. Qwen3-VL **dropped Qwen2.5-VL's windowed attention**. `cu_seqlens`
   degenerates to `[0, N]` for a single image, i.e. plain full attention with
   no mask input.
2. Position handling is **constant once the grid is fixed** — the bilinear
   interpolation of the 48×48 learned position embedding
   (`get_vision_bilinear_indices_and_weights`) and the rotary embeddings
   (`get_vision_position_ids`) depend only on `grid_thw`, so at a fixed
   resolution they fold into graph constants.

## 4. Fixed parameters

| Parameter | Value | Rationale |
|---|---|---|
| Variant | Qwen3-VL-4B-**Instruct** | user choice |
| Image bucket | **one**: 512×512 → 32×32 patches → 1024 patch tokens → **256 visual tokens** | Additional resolutions later become additional weight-shared graphs in the same ctx-bin — the same mechanism as the existing AR=1/AR=128 text graphs |
| Context length | CL=2048 (prefill AR=128, decode AR=1) | KV ≈ 300 MB at fp16; 256 image tokens still leaves real room for text |
| Vision precision | **FP16** (no AIMET) | 350 M params: quantizing saves ~350 MB against a 4 GB text tower while risking correctness (GELU, interpolated pos-embeds). Removes the need for any image calibration set. |
| Text precision | **W8A16** | matches the proven pipeline |
| RoPE | `qwen3vl-mrope`, section `[24,20,20]`, θ=5e6, pos-id-dim 64 | from config; Genie computes it natively |

`pixel_values` for the chosen bucket is `[1024, 1536]`, where
1536 = `in_channels(3) × temporal_patch(2) × patch(16) × patch(16)`.

## 5. Architecture

Two ctx-bins driven by a Genie pipeline, mirroring the SDK's glm-4v example:

```
image ──► preprocessor ──► ViT ctx-bin (FP16, image-encoder engine)
                             │
                             ├─ vision_embedding          [256, 2560]
                             └─ deepstack_visual_embed_{0,1,2} [256, 2560]
                                          │
text ──► tokenizer ──► embedding LUT ─────┤
                                          ▼
                            text ctx-bin (W8A16, text-generator engine)
                              inputs_embeds [1, AR, 2560]
                              + 3 optional deepstack inputs
                              + MRoPE cos/sin, mask, past-KV
                                          │
                                          ▼
                                       logits
```

### 5.1 Departures from the existing text pipeline

Everything in `docs/NOTES-genie-io.md` carries over unchanged — all-position
prefill logits `[1,AR,vocab]`, transposed past-keys `[1,n_kv,D,P]`, natural
past-values `[1,n_kv,P,D]`, new-slice-only KV outputs, rank-3 additive mask,
single encodings lineage across prefill/decode. Three things change:

1. **`inputs_embeds` replaces `input_ids`.** The runtime must splice visual
   features into the embedding sequence, so it cannot do the lookup in-graph.
   This is why the reference VLM configs carry an `embedding` LUT section that
   the current `configs/genie_dialog_qwen3_0.6b.json` does not have.
2. **Three optional deepstack inputs**, added at decoder layers 0/1/2 at
   visual-token positions. Zero-padded outside those positions, which makes
   them exactly inert when absent (see §7).
3. **MRoPE.** cos/sin remain graph *inputs* — the "nothing position-dependent
   is computed in-graph" rule survives intact — but Genie now generates them
   with interleaved-MRoPE semantics.

### 5.2 New pipeline capabilities required

- **Embedding LUT extraction** — `embedding_int8_lut.bin`, 151936×2560 int8
  with a single scale/offset, per the glm-4v config shape. Because
  `tie_word_embeddings: true`, this table is shared with the LM head.
- **Multi-part ctx-bin support** in `ctxbin.sh` / `bundle.sh`. W8A16 at 4 B is
  ~4 GB of weights; the 0.6 B model's 1.1 GB ctx-bin implies an ~1.8×
  overhead, putting this model at roughly 5–7 GB. The SDK's own glm-4v
  (`1_of_2`) and llava (`1_of_4`) examples split; the current pipeline emits a
  single binary.
- **Multimodal calibration** for the text tower — text sequences with real
  image embeddings spliced in, not text alone.

## 6. Build stages

Staged vision-first, so that the cheapest component exercises the most novel
contracts and failures stay localized.

### Stage 1 — Vision tower

New `scripts/export/modeling_vit_export.py` and
`scripts/export/export_qwen3vl_vit.py`. Static grid, constants folded, full
attention, no mask input.

- Inputs: `pixel_values [1024, 1536]`
- Outputs: `image_features [256, 2560]`, `deepstack_visual_embed_{0,1,2}
  [256, 2560]`

  `image_features` because Genie's image model initialises
  `m_layerNames[LayerType::OUTPUT] = "image_features"` by default
  (`nsp-image-model.hpp`) and only overrides it for outputs named
  `vision_embedding` or `cross_attention_states`. Using the default name means
  no override logic participates.
- FP16 conversion (`--float_bitwidth 16`), no AIMET, no calibration
- ctx-bin via the existing generator path with an image-encoder config
- **Gate**: `scripts/validate/parity_vit.py` — cosine similarity and max-abs
  error of all four outputs vs HF `Qwen3VLVisionModel` on real images

### Stage 2 — Text tower

Extends `modeling_export.py` for the Qwen3-VL text architecture per §5.1,
keeping every existing contract. AIMET W8A16 with multimodal calibration,
embedding LUT extraction, multi-part ctx-bin.

- **Gate**: extend `parity_qualla_read.py` / `parity_ladekv_read.py` to the
  embeds-input + deepstack + MRoPE feed pattern; argmax must match HF.

### Stage 3 — Integration

Genie pipeline configs (`image-encoder` in `"mode": "image"` +
`text-generator` with the embedding LUT and `qwen3vl-mrope`), plus a driver.
`genie-t2t-run` is text-only; the image path needs the pipeline API, so ship
`genie-app` if it handles image nodes, else a small custom driver.

- **Gate**: full-path parity — image → ViT → embeds → text tower → argmax vs
  HF `Qwen3VLForConditionalGeneration` on a small image+prompt set.

## 7. Known unknown: deepstack routing

The SDK proves Genie 1.19 *knows* Qwen3-VL, but the glue that routes three
deepstack tensors into the text graph's early layers is **not visible** in the
available sources. `TextGenerator::setEmbeddingInputData`
(`pipeline/TextGenerator.cpp:188`) appends to a single accumulator stream, and
the deepstack handling at `nsp-image-model.cpp:199` is a fallback that treats
a `deepstack_visual_embed*` tensor as a graph's *primary* output — hinting at
a per-graph decomposition rather than one graph with four outputs.

Three possibilities, undecidable from here:

- **(a)** Genie expects a specific graph decomposition we must match
- **(b)** The wiring is app-level and we write it
- **(c)** Deepstack is not supported end-to-end in 1.19

**Mitigation — optional by construction.** The text graph takes the deepstack
tensors as ordinary optional inputs; supplying zeros yields a model that is
exactly HF-minus-deepstack. It still sees images through the main visual
embeddings, losing only the multi-level detail injection. This keeps "images
work at all" reachable under all three outcomes.

For Stage 1 we build the single-graph/four-output form first, since it matches
the HF module structure, and keep the per-graph decomposition as plan B.

## 8. Validation strategy

No device means three tiers of numerical evidence, mirroring what caught the
prefill bug:

1. **Per-tower parity** — ONNX/DLC outputs vs HF modules
2. **Runtime-feed-pattern emulation** — Python that reproduces qualla's exact
   feed and read pattern (left-aligned input, logits row n−1, chunking). This
   is the tier that actually caught the `[1,1,vocab]` bug; a graph can pass
   tier 1 and still be garbage on device.
3. **Static contract lint** — graph I/O names, shapes and dtypes checked
   against the contract recorded in `docs/NOTES-genie-io.md`

## 9. Risks

| Risk | Severity | Handling |
|---|---|---|
| Device memory unknown; ~5–7 GB ctx-bin may not fit | High | Accepted by user decision. Multi-part ctx-bin makes layout flexible; requirement documented prominently for whoever gets device access. |
| Deepstack routing unresolved (§7) | Medium | Optional-by-construction inputs; degraded path always available |
| No device validation | High | Three-tier numerical gates (§8); inherent to the project |
| 4 B AIMET calibration on CPU (8 GB VRAM box) | Medium | `QUANT_DEVICE=cpu`, long runs; budget for it |
| Multi-part ctx-bin is new to this pipeline | Medium | Built and tested in Stage 2 |

Disk: ~80 GB needed (checkpoint ~8 GB, FP32 ONNX, DLC, ctx-bins); 623 GB free
on the ext4 data volume.

## 10. Out of scope

Decode throughput optimization, lookahead decoding, video input, multiple
resolution buckets, W4A16, and LoRA. Each is additive later; none is required
for "images work at all".
