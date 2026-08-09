"""Export-friendly Qwen3 for QNN HTP / Genie deployment.

Reconstructed from SA8797P_Deployment_Status_Summary.md (the original remote
scripts are unavailable). Encodes the validated export constraints:

- RoPE cos/sin are graph INPUTS of half-head-dim (pos-id-dim = 64), split into
  separate cos and sin tensors; nothing position-dependent is computed in-graph.
- RMSNorm decomposed into primitive ops (opset 17 has no RMSNorm; never use
  torch.nn.RMSNorm which exports a fused op).
- Q/K per-head RMSNorm (Qwen3's q_norm/k_norm) preserved exactly.
- Attention mask is an ADDITIVE float input of shape [1, 1, S, P+S]
  (width = context + AR, per the summary doc). 0 = attend, -100 = masked
  (-100 rather than -inf for FP16 safety).
- KV cache: static shapes. past_(key|value)_i are [1, n_kv, P, D] inputs,
  right-aligned (latest token last; front is masked garbage). The graph outputs
  concat(past, new) of length P+S; the runtime keeps the trailing P entries.
- Optional fusions (summary §2.3 / §4.1):
    fuse_gate_up: gate_proj+up_proj -> one gate_up_proj Linear, Split after.
    fuse_qkv:     q/k/v_proj -> one qkv_proj Linear, Split after.
- All-FP32 module; FP16 happens at qairt-converter (--float_bitwidth 16).
"""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

MASK_VALUE = -100.0


class ExportRMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x):
        var = (x * x).mean(dim=-1, keepdim=True)
        return x * torch.rsqrt(var + self.eps) * self.weight


def apply_rope(x, cos_h, sin_h):
    """x: [B, H, S, D]; cos_h/sin_h: [B, S, D/2] (half tables, duplicated here)."""
    cos = torch.cat([cos_h, cos_h], dim=-1).unsqueeze(1)  # [B, 1, S, D]
    sin = torch.cat([sin_h, sin_h], dim=-1).unsqueeze(1)
    half = x.shape[-1] // 2
    x1 = x[..., :half]
    x2 = x[..., half:]
    rotated = torch.cat([-x2, x1], dim=-1)
    return x * cos + rotated * sin


class ExportAttention(nn.Module):
    def __init__(self, cfg, fuse_qkv: bool):
        super().__init__()
        self.n_heads = cfg.num_attention_heads
        self.n_kv = cfg.num_key_value_heads
        self.head_dim = cfg.head_dim
        self.fuse_qkv = fuse_qkv
        hidden = cfg.hidden_size
        q_out = self.n_heads * self.head_dim
        kv_out = self.n_kv * self.head_dim
        if fuse_qkv:
            self.qkv_proj = nn.Linear(hidden, q_out + 2 * kv_out, bias=False)
        else:
            self.q_proj = nn.Linear(hidden, q_out, bias=False)
            self.k_proj = nn.Linear(hidden, kv_out, bias=False)
            self.v_proj = nn.Linear(hidden, kv_out, bias=False)
        self.o_proj = nn.Linear(q_out, hidden, bias=False)
        self.q_norm = ExportRMSNorm(self.head_dim, cfg.rms_norm_eps)
        self.k_norm = ExportRMSNorm(self.head_dim, cfg.rms_norm_eps)
        self.scale = 1.0 / math.sqrt(self.head_dim)

    def forward(self, x, cos_h, sin_h, mask, past_k, past_v):
        B, S, _ = x.shape
        if self.fuse_qkv:
            qkv = self.qkv_proj(x)
            q, k, v = torch.split(
                qkv,
                [self.n_heads * self.head_dim, self.n_kv * self.head_dim, self.n_kv * self.head_dim],
                dim=-1,
            )
        else:
            q, k, v = self.q_proj(x), self.k_proj(x), self.v_proj(x)

        # HF Qwen3 order: view -> per-head RMSNorm -> transpose -> rope
        q = self.q_norm(q.view(B, S, self.n_heads, self.head_dim)).transpose(1, 2)
        k = self.k_norm(k.view(B, S, self.n_kv, self.head_dim)).transpose(1, 2)
        v = v.view(B, S, self.n_kv, self.head_dim).transpose(1, 2)

        q = apply_rope(q, cos_h, sin_h)
        k = apply_rope(k, cos_h, sin_h)

        if past_k is not None:
            k = torch.cat([past_k, k], dim=2)
            v = torch.cat([past_v, v], dim=2)
        new_k, new_v = k, v
        T = k.shape[2]

        # GQA: expand kv heads to q heads (no repeat_interleave: keep ops simple)
        rep = self.n_heads // self.n_kv
        k = k.unsqueeze(2).expand(B, self.n_kv, rep, T, self.head_dim).reshape(B, self.n_heads, T, self.head_dim)
        v = v.unsqueeze(2).expand(B, self.n_kv, rep, T, self.head_dim).reshape(B, self.n_heads, T, self.head_dim)

        attn = torch.matmul(q, k.transpose(-1, -2)) * self.scale
        attn = attn + mask  # [1, 1, S, T] additive
        attn = torch.softmax(attn, dim=-1)
        out = torch.matmul(attn, v)  # [B, H, S, D]
        out = out.transpose(1, 2).reshape(B, S, self.n_heads * self.head_dim)
        return self.o_proj(out), new_k, new_v


class ExportMLP(nn.Module):
    def __init__(self, cfg, fuse_gate_up: bool):
        super().__init__()
        self.fuse_gate_up = fuse_gate_up
        self.intermediate = cfg.intermediate_size
        if fuse_gate_up:
            self.gate_up_proj = nn.Linear(cfg.hidden_size, 2 * cfg.intermediate_size, bias=False)
        else:
            self.gate_proj = nn.Linear(cfg.hidden_size, cfg.intermediate_size, bias=False)
            self.up_proj = nn.Linear(cfg.hidden_size, cfg.intermediate_size, bias=False)
        self.down_proj = nn.Linear(cfg.intermediate_size, cfg.hidden_size, bias=False)

    def forward(self, x):
        if self.fuse_gate_up:
            gu = self.gate_up_proj(x)
            g, u = torch.split(gu, [self.intermediate, self.intermediate], dim=-1)
        else:
            g, u = self.gate_proj(x), self.up_proj(x)
        return self.down_proj(F.silu(g) * u)


class ExportLayer(nn.Module):
    def __init__(self, cfg, fuse_gate_up: bool, fuse_qkv: bool):
        super().__init__()
        self.input_layernorm = ExportRMSNorm(cfg.hidden_size, cfg.rms_norm_eps)
        self.self_attn = ExportAttention(cfg, fuse_qkv)
        self.post_attention_layernorm = ExportRMSNorm(cfg.hidden_size, cfg.rms_norm_eps)
        self.mlp = ExportMLP(cfg, fuse_gate_up)

    def forward(self, x, cos_h, sin_h, mask, past_k, past_v):
        a, new_k, new_v = self.self_attn(self.input_layernorm(x), cos_h, sin_h, mask, past_k, past_v)
        h = x + a
        return h + self.mlp(self.post_attention_layernorm(h)), new_k, new_v


class ExportQwen3(nn.Module):
    """Flat-signature module suitable for torch.onnx.export.

    forward(input_ids, attention_mask, position_ids_cos, position_ids_sin,
            past_key_0, past_value_0, ..., past_key_{L-1}, past_value_{L-1})
      -> (logits, past_key_0_out, past_value_0_out, ...)

    With use_past=False the past_* arguments are omitted (prefill graph).
    """

    def __init__(self, cfg, fuse_gate_up=False, fuse_qkv=False, use_past=True, logits_last_only=False):
        super().__init__()
        self.cfg = cfg
        self.use_past = use_past
        self.logits_last_only = logits_last_only
        self.embed_tokens = nn.Embedding(cfg.vocab_size, cfg.hidden_size)
        self.layers = nn.ModuleList(
            [ExportLayer(cfg, fuse_gate_up, fuse_qkv) for _ in range(cfg.num_hidden_layers)]
        )
        self.norm = ExportRMSNorm(cfg.hidden_size, cfg.rms_norm_eps)
        self.lm_head = nn.Linear(cfg.hidden_size, cfg.vocab_size, bias=False)

    def forward(self, input_ids, attention_mask, position_ids_cos, position_ids_sin, *past):
        x = self.embed_tokens(input_ids.to(torch.long))
        new_kv = []
        for i, layer in enumerate(self.layers):
            pk = past[2 * i] if self.use_past else None
            pv = past[2 * i + 1] if self.use_past else None
            x, nk, nv = layer(x, position_ids_cos, position_ids_sin, attention_mask, pk, pv)
            new_kv.extend([nk, nv])
        x = self.norm(x)
        if self.logits_last_only:
            x = x[:, -1:, :]
        logits = self.lm_head(x)
        return (logits, *new_kv)

    @staticmethod
    def from_hf(hf_model, fuse_gate_up=False, fuse_qkv=False, use_past=True, logits_last_only=False):
        cfg = hf_model.config
        m = ExportQwen3(cfg, fuse_gate_up, fuse_qkv, use_past, logits_last_only)
        src = hf_model.state_dict()
        dst = {}
        dst["embed_tokens.weight"] = src["model.embed_tokens.weight"]
        dst["norm.weight"] = src["model.norm.weight"]
        # 0.6B ties lm_head to embeddings; state dict may or may not carry lm_head
        dst["lm_head.weight"] = src.get("lm_head.weight", src["model.embed_tokens.weight"])
        for i in range(cfg.num_hidden_layers):
            s = f"model.layers.{i}."
            d = f"layers.{i}."
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
                dst[d + "self_attn.q_proj.weight"] = src[s + "self_attn.q_proj.weight"]
                dst[d + "self_attn.k_proj.weight"] = src[s + "self_attn.k_proj.weight"]
                dst[d + "self_attn.v_proj.weight"] = src[s + "self_attn.v_proj.weight"]
            dst[d + "mlp.down_proj.weight"] = src[s + "mlp.down_proj.weight"]
            if fuse_gate_up:
                dst[d + "mlp.gate_up_proj.weight"] = torch.cat(
                    [src[s + "mlp.gate_proj.weight"], src[s + "mlp.up_proj.weight"]], dim=0)
            else:
                dst[d + "mlp.gate_proj.weight"] = src[s + "mlp.gate_proj.weight"]
                dst[d + "mlp.up_proj.weight"] = src[s + "mlp.up_proj.weight"]
        m.load_state_dict(dst, strict=True)
        return m.eval()


def rope_tables(positions, head_dim, theta):
    """Host-side cos/sin half tables: positions [S] -> ([1, S, D/2], [1, S, D/2])."""
    half = head_dim // 2
    inv_freq = 1.0 / (theta ** (torch.arange(0, half, dtype=torch.float32) / half))
    ang = positions.to(torch.float32)[:, None] * inv_freq[None, :]
    return ang.cos()[None], ang.sin()[None]


def causal_mask(seq, total):
    """Additive [1, 1, seq, total] causal mask; current tokens right-aligned."""
    m = torch.full((seq, total), MASK_VALUE, dtype=torch.float32)
    past = total - seq
    for r in range(seq):
        m[r, : past + r + 1] = 0.0
    return m[None, None]
