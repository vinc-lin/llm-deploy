#!/usr/bin/env python
"""Prove from_hf_vl_text actually forwards grouped_gqa.

The VL chain shipped v2 with 36 replication ops per shard because this factory
silently dropped the flag (the quantizer HAS --grouped-gqa; the VL branch
could not pass it anywhere). This test runs in seconds on a tiny random tower:
no 4B checkpoint, no AIMET.
"""
import sys
import tempfile
from pathlib import Path
from types import SimpleNamespace

import onnx
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "export"))
from modeling_export import ExportQwen3, causal_mask, rope_tables  # noqa: E402

H, LAYERS, HEADS, KV, HD, VOCAB, INTER, NDEEP, S = 64, 2, 8, 2, 8, 512, 128, 3, 16

CFG = SimpleNamespace(hidden_size=H, num_hidden_layers=LAYERS,
                      num_attention_heads=HEADS, num_key_value_heads=KV,
                      head_dim=HD, vocab_size=VOCAB, intermediate_size=INTER,
                      rms_norm_eps=1e-6)


class FakeVL:
    """Just enough of Qwen3VLForConditionalGeneration for from_hf_vl_text."""
    def __init__(self, sd):
        self.config = SimpleNamespace(text_config=CFG)
        self._sd = sd

    def state_dict(self):
        return self._sd


def make_state_dict():
    torch.manual_seed(0)
    p = "model.language_model."
    sd = {p + "embed_tokens.weight": torch.randn(VOCAB, H) * 0.02,
          p + "norm.weight": torch.ones(H)}
    for i in range(LAYERS):
        s = f"{p}layers.{i}."
        sd[s + "input_layernorm.weight"] = torch.ones(H)
        sd[s + "post_attention_layernorm.weight"] = torch.ones(H)
        sd[s + "self_attn.q_norm.weight"] = torch.ones(HD)
        sd[s + "self_attn.k_norm.weight"] = torch.ones(HD)
        sd[s + "self_attn.q_proj.weight"] = torch.randn(HEADS * HD, H) * 0.02
        sd[s + "self_attn.k_proj.weight"] = torch.randn(KV * HD, H) * 0.02
        sd[s + "self_attn.v_proj.weight"] = torch.randn(KV * HD, H) * 0.02
        sd[s + "self_attn.o_proj.weight"] = torch.randn(H, HEADS * HD) * 0.02
        sd[s + "mlp.gate_proj.weight"] = torch.randn(INTER, H) * 0.02
        sd[s + "mlp.up_proj.weight"] = torch.randn(INTER, H) * 0.02
        sd[s + "mlp.down_proj.weight"] = torch.randn(H, INTER) * 0.02
    return sd


def export_and_count_expand(model):
    embeds = torch.zeros(1, 1, S, H)
    mask = causal_mask(S, S)
    cos, sin = rope_tables(torch.arange(S), HD, 10000.0)
    deep = [torch.zeros(1, 1, S, H) for _ in range(NDEEP)]
    with tempfile.TemporaryDirectory() as d, torch.no_grad():
        path = Path(d) / "m.onnx"
        torch.onnx.export(model, (embeds, mask, cos, sin, *deep), str(path),
                          opset_version=17, dynamo=False)
        g = onnx.load(str(path)).graph
        return sum(1 for n in g.node if n.op_type == "Expand")


def main():
    sd = make_state_dict()
    rep = ExportQwen3.from_hf_vl_text(FakeVL(sd), use_past=False,
                                      n_deepstack=NDEEP)
    grp = ExportQwen3.from_hf_vl_text(FakeVL(sd), use_past=False,
                                      n_deepstack=NDEEP, grouped_gqa=True)

    # 1. numerics: the grouped form is the same math, reshaped
    torch.manual_seed(1)
    embeds = torch.randn(1, 1, S, H) * 0.02
    mask = causal_mask(S, S)
    cos, sin = rope_tables(torch.arange(S), HD, 10000.0)
    deep = [torch.randn(1, 1, S, H) * 0.02 for _ in range(NDEEP)]
    with torch.no_grad():
        lo = rep(embeds, mask, cos, sin, *deep)[0]
        lg = grp(embeds, mask, cos, sin, *deep)[0]
    d = (lo - lg).abs().max().item()
    assert d < 1e-5, f"grouped vs replicating logits diverge: max|d|={d:.3e}"

    # 2. topology: grouped export has ZERO Expand nodes; replicating has 2/layer
    n_rep, n_grp = export_and_count_expand(rep), export_and_count_expand(grp)
    assert n_rep == 2 * LAYERS, f"control: expected {2*LAYERS} Expand, got {n_rep}"
    assert n_grp == 0, f"grouped export still has {n_grp} Expand nodes"
    print(f"OK  max|dlogits|={d:.3e}  Expand: replicating={n_rep} grouped=0")


if __name__ == "__main__":
    main()
