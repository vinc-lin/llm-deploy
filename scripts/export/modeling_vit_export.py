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
