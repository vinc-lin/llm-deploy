"""Export-friendly Qwen3 for QNN HTP / Genie deployment.

Reconstructed from docs/archive/SA8797P_Deployment_Status_Summary.md (the original remote
scripts are unavailable). Encodes the validated export constraints:

- RoPE cos/sin are graph INPUTS of half-head-dim (pos-id-dim = 64), split into
  separate cos and sin tensors; nothing position-dependent is computed in-graph.
- RMSNorm decomposed into primitive ops (opset 17 has no RMSNorm; never use
  torch.nn.RMSNorm which exports a fused op).
- Q/K per-head RMSNorm (Qwen3's q_norm/k_norm) preserved exactly.
- Attention mask is an ADDITIVE float input of shape [1, S, P+S] (rank-3;
  Genie checkShape expects [1, AR, CTX] — nsp-model.cpp Check 3). 0 = attend,
  -100 = masked (-100 rather than -inf for FP16 safety).
- KV cache contract (verified against Genie 1.19 qualla engine source,
  examples/Genie/Genie/src/qualla/engines/qnn-htp/nsp-graph.cpp:117,156 and
  nsp-model.cpp:804ff):
    * KEYS are stored TRANSPOSED: past_key_{i}_in  [1, n_kv, D, P]
      (kv_dim = second-to-last axis, sequence = LAST axis)
    * VALUES natural:             past_value_{i}_in [1, n_kv, P, D]
    * outputs carry ONLY the new tokens' slice (Genie scatters into its cache):
      past_key_{i}_out [1, n_kv, D, S], past_value_{i}_out [1, n_kv, S, D]
    * name pattern past_<kind>_<i>_in/_out (isKVTensor: contains key|value and
      ends _in/_out; input name derived from output by replacing _out with _in)
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
    def __init__(self, cfg, fuse_qkv: bool, grouped_gqa: bool = False):
        super().__init__()
        self.n_heads = cfg.num_attention_heads
        self.n_kv = cfg.num_key_value_heads
        self.head_dim = cfg.head_dim
        self.fuse_qkv = fuse_qkv
        self.grouped_gqa = grouped_gqa
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

    def forward(self, x, cos_h, sin_h, mask, past_k_t, past_v):
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

        # Genie KV contract: keys transposed (seq last), outputs = new slice only
        new_k_t = k.transpose(-1, -2)  # [B, n_kv, D, S]
        new_v = v                      # [B, n_kv, S, D]
        if past_k_t is not None:
            k_t = torch.cat([past_k_t, new_k_t], dim=-1)  # [B, n_kv, D, T]
            v_full = torch.cat([past_v, new_v], dim=2)    # [B, n_kv, T, D]
        else:
            k_t, v_full = new_k_t, new_v
        T = k_t.shape[-1]

        rep = self.n_heads // self.n_kv
        if self.grouped_gqa:
            # Batch the attention MatMuls over the 8 KV heads instead of
            # materialising 16 replicated ones.
            #
            # WHY: the replicating form below costs 74.7% of every decode step's
            # DSP cycles (261.8M of 350.3M, device profile 2026-08-13). The
            # converter lowers each `expand` into an FP16 broadcast MULTIPLY
            # (QNN Eltwise_Binary operation 13) whose output is 4.72 MB; at 2 per
            # layer x 28 layers that is 264 MB written and 264 MB re-read by the
            # MatMuls, every step, purely to hand the MatMul a tensor whose 16
            # heads are 8 duplicated pairs.
            #
            # HOW: HF groups Q heads contiguously per KV head (kv head i serves q
            # heads [i*rep, (i+1)*rep)), so H factorises as (n_kv, rep) in that
            # order. Every reshape below is therefore a pure view -- no data
            # moves, and the mask/softmax/output paths stay structurally
            # identical to the replicating form, keeping the change to exactly
            # the two MatMuls' batch dimension.
            q_g = q.reshape(B, self.n_kv, rep * S, self.head_dim)
            attn = torch.matmul(q_g, k_t) * self.scale         # [B, n_kv, rep*S, T]
            attn = attn.reshape(B, self.n_heads, S, T)         # view: (n_kv, rep) == H
            attn = attn + mask.unsqueeze(1)
            attn = torch.softmax(attn, dim=-1)
            attn = attn.reshape(B, self.n_kv, rep * S, T)      # view, back to grouped
            out = torch.matmul(attn, v_full)                   # [B, n_kv, rep*S, D]
            out = out.reshape(B, self.n_heads, S, self.head_dim)
        else:
            # GQA: expand kv heads to q heads (no repeat_interleave: keep ops simple)
            k_t = k_t.unsqueeze(2).expand(B, self.n_kv, rep, self.head_dim, T).reshape(B, self.n_heads, self.head_dim, T)
            v_full = v_full.unsqueeze(2).expand(B, self.n_kv, rep, T, self.head_dim).reshape(B, self.n_heads, T, self.head_dim)

            attn = torch.matmul(q, k_t) * self.scale  # [B, H, S, T]
            attn = attn + mask.unsqueeze(1)  # mask [1, S, T] additive -> broadcast heads
            attn = torch.softmax(attn, dim=-1)
            out = torch.matmul(attn, v_full)  # [B, H, S, D]

        out = out.transpose(1, 2).reshape(B, S, self.n_heads * self.head_dim)
        return self.o_proj(out), new_k_t, new_v


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
    def __init__(self, cfg, fuse_gate_up: bool, fuse_qkv: bool, grouped_gqa: bool = False):
        super().__init__()
        self.input_layernorm = ExportRMSNorm(cfg.hidden_size, cfg.rms_norm_eps)
        self.self_attn = ExportAttention(cfg, fuse_qkv, grouped_gqa)
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

    Two optional modes exist for the Qwen3-VL text tower (both OFF by default,
    both only ever used together -- see from_hf_vl_text):

    input_embeds: the first argument is already [B, S, H] hidden states, not
      token ids, and the 389 M-param embedding table is NOT part of the graph.
      The runtime owns the lookup because it must splice visual features into
      the sequence before the text tower runs.

    n_deepstack: N extra full-width [B, S, H] inputs added to the residual
      stream after decoder layers 0..N-1. They precede the past-KV tensors in
      the variadic tail so that ONNX input order matches the declared
      input_names.
    """

    def __init__(self, cfg, fuse_gate_up=False, fuse_qkv=False, use_past=True, logits_last_only=False,
                 input_embeds=False, n_deepstack=0, n_layers=None, is_first=True, is_last=True,
                 grouped_gqa=False):
        super().__init__()
        self.cfg = cfg
        self.use_past = use_past
        self.logits_last_only = logits_last_only
        self.input_embeds = input_embeds
        self.n_deepstack = n_deepstack
        # Chunked (multi-ctx-bin) export: a 4B tower at W8A16 estimates 4.18 GiB
        # and qnn-context-binary-generator rejects anything over 3.5 GiB per
        # graph, so the layer stack is split across ctx-bins. A middle chunk
        # takes hidden states in and hands hidden states out; only the first
        # chunk owns the embedding path and only the last owns norm + lm_head.
        # Defaults keep the whole-model behaviour byte-for-byte.
        self.is_first = is_first
        self.is_last = is_last
        n_layers = cfg.num_hidden_layers if n_layers is None else n_layers
        # embeddings-in: the runtime owns the token LUT, keep it out of the graph
        self.embed_tokens = (
            nn.Embedding(cfg.vocab_size, cfg.hidden_size)
            if (is_first and not input_embeds) else None
        )
        self.layers = nn.ModuleList(
            [ExportLayer(cfg, fuse_gate_up, fuse_qkv, grouped_gqa) for _ in range(n_layers)]
        )
        # Weights the chunk does not own must not exist, or they land in its
        # ctx-bin as dead payload -- lm_head alone is 389 M params.
        self.norm = ExportRMSNorm(cfg.hidden_size, cfg.rms_norm_eps) if is_last else None
        self.lm_head = nn.Linear(cfg.hidden_size, cfg.vocab_size, bias=False) if is_last else None

    def forward(self, input_ids, attention_mask, position_ids_cos, position_ids_sin, *rest):
        # variadic tail: deepstack tensors first, then past-KV pairs. With the
        # defaults (n_deepstack=0) this is exactly the old `*past`.
        deep = rest[: self.n_deepstack]
        past = rest[self.n_deepstack:]

        # A non-first chunk is fed `last_hidden_states` from the previous chunk,
        # which is already [B, S, H] -- same handling as embeddings-in.
        hidden_in = self.input_embeds or not self.is_first
        x = input_ids if hidden_in else self.embed_tokens(input_ids.to(torch.long))
        if hidden_in and x.dim() == 4:
            x = x.squeeze(1)   # Genie's documented [1,1,AR,H] -> [B,S,H]
        new_kv = []
        for i, layer in enumerate(self.layers):
            pk = past[2 * i] if self.use_past else None
            pv = past[2 * i + 1] if self.use_past else None
            x, nk, nv = layer(x, position_ids_cos, position_ids_sin, attention_mask, pk, pv)
            new_kv.extend([nk, nv])
            # Deepstack (Qwen3-VL): HF applies it AFTER layer `i` runs, for
            # i < len(deepstack_visual_embeds), as a scatter-add onto the visual
            # positions only (modeling_qwen3_vl.py:835 / _deepstack_process).
            # A static graph cannot scatter, so the host zero-pads the tensor to
            # full width and we add it unconditionally -- identical arithmetic,
            # since adding 0 at non-visual positions is a no-op. All-zeros is
            # therefore exactly HF-minus-deepstack: the graceful-degradation
            # path for text-only prompts.
            if i < self.n_deepstack:
                d = deep[i]
                x = x + (d.squeeze(1) if d.dim() == 4 else d)
        if not self.is_last:
            # Boundary tensor. Genie connects splits implicitly by NAME -- an
            # input matching a previous split's output name is the same buffer
            # (nsp-model.cpp:1453) -- and it does NOT check that their shapes
            # agree ("Missing check : Shape of tensor between splits match up",
            # nsp-graph.cpp:229). The exporter names this `last_hidden_states`
            # on both sides; see docs/NOTES-genie-splits.md.
            return (x, *new_kv)
        x = self.norm(x)
        if self.logits_last_only:
            x = x[:, -1:, :]
        logits = self.lm_head(x)
        return (logits, *new_kv)

    @staticmethod
    def _map_layers(dst, src, prefix, n_layers, fuse_gate_up, fuse_qkv, src_offset=0):
        """Copy the decoder-layer weights. `prefix` is the source tower prefix:
        'model.' for plain Qwen3, 'model.language_model.' for the Qwen3-VL text
        tower (the two towers are architecturally identical).

        `src_offset` shifts the SOURCE index for chunked exports: a chunk's
        modules are always local 0..n-1, but its weights come from global
        src_offset..src_offset+n-1."""
        for i in range(n_layers):
            s = f"{prefix}layers.{src_offset + i}."
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
        return dst

    @staticmethod
    def from_hf(hf_model, fuse_gate_up=False, fuse_qkv=False, use_past=True, logits_last_only=False,
                grouped_gqa=False, input_embeds=False):
        """Plain Qwen3 (0.6B/1.7B).

        `input_embeds=True` builds the same tower fed pre-computed hidden states
        from an external LUT instead of token ids -- the Qwen3-VL feed shape, on
        a model that is known-good on device. That combination has no product
        use; it exists so the LUT/accumulator path can be tested against a
        working control instead of only inside the 4B, which is the one place it
        has ever run and the one place it fails. See
        docs/DEVICE_TEST_qwen3_06b_lut.md.
        """
        cfg = hf_model.config
        m = ExportQwen3(cfg, fuse_gate_up, fuse_qkv, use_past, logits_last_only,
                        grouped_gqa=grouped_gqa, input_embeds=input_embeds)
        src = hf_model.state_dict()
        dst = {}
        # embed_tokens is None under input_embeds (the runtime owns the lookup),
        # so loading it would fail strict load_state_dict.
        if not input_embeds:
            dst["embed_tokens.weight"] = src["model.embed_tokens.weight"]
        dst["norm.weight"] = src["model.norm.weight"]
        # 0.6B ties lm_head to embeddings; state dict may or may not carry lm_head
        dst["lm_head.weight"] = src.get("lm_head.weight", src["model.embed_tokens.weight"])
        ExportQwen3._map_layers(dst, src, "model.", cfg.num_hidden_layers, fuse_gate_up, fuse_qkv)
        m.load_state_dict(dst, strict=True)
        return m.eval()

    @staticmethod
    def from_hf_vl_text(hf_vl, fuse_gate_up=False, fuse_qkv=False, use_past=True,
                        logits_last_only=False, n_deepstack=3, layer_range=None,
                        grouped_gqa=False):
        """Qwen3-VL text tower: embeddings-in + deepstack.

        Architecturally identical to plain Qwen3 text, so it reuses ExportQwen3
        wholesale; only the checkpoint prefix and the two new modes differ. The
        embedding table is deliberately NOT loaded -- the runtime does the token
        lookup from an external LUT so it can splice visual features in first.

        `layer_range=(start, end)` builds one chunk of a split export, holding
        global layers [start, end). The chunk owns the embedding path only if it
        starts at 0, and norm + lm_head only if it ends at num_hidden_layers.
        Deepstack lands on global layers 0..n_deepstack-1, so pass n_deepstack=0
        for any chunk that does not start at 0. See docs/NOTES-genie-splits.md.
        """
        cfg = hf_vl.config.text_config
        start, end = (0, cfg.num_hidden_layers) if layer_range is None else layer_range
        if not 0 <= start < end <= cfg.num_hidden_layers:
            raise ValueError(f"layer_range {(start, end)} outside [0, {cfg.num_hidden_layers}]")
        is_first, is_last = start == 0, end == cfg.num_hidden_layers
        if not is_first and n_deepstack:
            raise ValueError(
                f"n_deepstack={n_deepstack} but this chunk starts at layer {start}; "
                "deepstack applies to global layers 0..n-1 only")

        m = ExportQwen3(cfg, fuse_gate_up, fuse_qkv, use_past, logits_last_only,
                        input_embeds=True, n_deepstack=n_deepstack,
                        n_layers=end - start, is_first=is_first, is_last=is_last,
                        grouped_gqa=grouped_gqa)
        src = hf_vl.state_dict()
        p = "model.language_model."
        dst = {}
        if is_last:
            dst["norm.weight"] = src[p + "norm.weight"]
            # tie_word_embeddings: true -> the checkpoint carries no lm_head.weight
            dst["lm_head.weight"] = src.get("lm_head.weight", src[p + "embed_tokens.weight"])
        ExportQwen3._map_layers(dst, src, p, end - start, fuse_gate_up, fuse_qkv,
                                src_offset=start)
        m.load_state_dict(dst, strict=True)
        return m.eval()


def rope_theta_of(cfg):
    """transformers 5.x moved rope_theta into cfg.rope_parameters."""
    if hasattr(cfg, "rope_theta"):
        return cfg.rope_theta
    return cfg.rope_parameters["rope_theta"]


def rope_tables(positions, head_dim, theta):
    """Host-side cos/sin half tables: positions [S] -> ([1, S, D/2], [1, S, D/2])."""
    half = head_dim // 2
    inv_freq = 1.0 / (theta ** (torch.arange(0, half, dtype=torch.float32) / half))
    ang = positions.to(torch.float32)[:, None] * inv_freq[None, :]
    return ang.cos()[None], ang.sin()[None]


def mrope_tables(pos3, head_dim, theta, mrope_section=(24, 20, 20)):
    """Qwen3-VL interleaved MRoPE half-tables: pos3 [3, S] (t,h,w) ->
    ([1, S, D/2], [1, S, D/2]).

    Layout per HF Qwen3VLTextRotaryEmbedding.apply_interleaved_mrope
    (modeling_qwen3_vl.py): half-dim k takes t/h/w as [THWTHW...TT] -- h owns
    dims 1,4,..,3*sec[1]-2, w owns 2,5,..,3*sec[2]-1, t owns the rest
    including the tail. All three rows equal -> identical to rope_tables().
    """
    half = head_dim // 2
    inv_freq = 1.0 / (theta ** (torch.arange(0, half, dtype=torch.float32) / half))
    ang = pos3.to(torch.float32)[:, :, None] * inv_freq[None, None, :]  # [3,S,half]
    out = ang[0].clone()
    for dim, off in ((1, 1), (2, 2)):
        idx = slice(off, mrope_section[dim] * 3, 3)
        out[:, idx] = ang[dim][:, idx]
    return out.cos()[None], out.sin()[None]


def causal_mask(seq, total):
    """Additive rank-3 [1, seq, total] causal mask; current tokens right-aligned."""
    m = torch.full((seq, total), MASK_VALUE, dtype=torch.float32)
    past = total - seq
    for r in range(seq):
        m[r, : past + r + 1] = 0.0
    return m[None]
