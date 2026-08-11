#!/usr/bin/env python
"""AIMET PTQ W8A16 for the export-friendly Qwen3 wrapper.

Reconstructed from SA8797P_Deployment_Status_Summary.md §2.2:
- default_param_bw=8 (per-channel symmetric via config json), default_output_bw=16
- quant_scheme = post_training_tf_enhanced
- calibration: ~10 mixed zh/en/code/math prompts
- clip_weights_to_7f7f: avoid INT8 saturation (reconstructed: clamp weights so
  no value quantizes to -128; i.e. symmetric ±127 range)  [FLAGGED for review]
- quantizers DISABLED on: embed_tokens (HTP Gather-on-INT16 err 0xc26),
  final norm, lm_head (quality), and K/V projection outputs (cross-graph FP16)
- with --fuse-gate-up: gate_up_proj output quantizer disabled (FP16 internal,
  requantized at down_proj)

Written against the AIMET v1-style API (aimet_torch.quantsim). If the installed
aimet-torch 2.x rejects these calls, see docs/LOCAL_ENV.md for the version pin.
"""
import argparse
import os
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "export"))
from modeling_export import ExportQwen3, causal_mask, rope_tables, rope_theta_of  # noqa: E402

DATA = Path(os.environ.get("LLMDEPLOY_DATA", "/home/vinc/llm-local"))

CALIB_PROMPTS = [
    "The quick brown fox jumps over the lazy dog.",
    "def quicksort(arr):\n    if len(arr) <= 1:\n        return arr",
    "解释一下什么是注意力机制,它在Transformer中起什么作用?",
    "求解方程 x^2 - 5x + 6 = 0 的两个根。",
    "The theory of relativity states that",
    "import numpy as np\nA = np.random.randn(4, 4)",
    "中国的首都是北京,美国的首都是华盛顿。",
    "What is the integral of x * exp(x) dx?",
    "SELECT name, COUNT(*) FROM users GROUP BY name;",
    "深度学习模型的量化是指将浮点权重转换为低比特整数表示。",
]


def _quantized_modules(sim):
    from aimet_torch.v2.nn import BaseQuantizationMixin
    for name, m in sim.model.named_modules():
        if isinstance(m, BaseQuantizationMixin):
            yield name, m


def clip_weights_to_7f7f(sim):
    """Clamp each quantized weight so no value maps to the asymmetric extreme
    (-128 for INT8): restrict to symmetric ±127 steps.  Reconstructed from the
    function name in the summary doc — semantics flagged for review.
    aimet-torch 2.x (v2) API: quantizers are affine QuantizeDequantize modules."""
    n = 0
    for name, m in _quantized_modules(sim):
        pq = getattr(m, "param_quantizers", None)  # torch ModuleDict — no .get()
        q = pq["weight"] if (pq is not None and "weight" in pq) else None
        if q is None or not hasattr(m, "weight"):
            continue
        if "Block" in type(q).__name__:   # LPBQ/blockwise: per-block scale shape
            continue                       # doesn't broadcast; clip is N/A
        try:
            scale = q.get_scale()
        except Exception:
            continue
        if scale is None:
            continue
        w = m.weight
        with torch.no_grad():
            s = scale.detach().to(dtype=w.dtype, device=w.device)
            while s.dim() < w.dim():
                s = s.unsqueeze(-1)
            hi = float(2 ** (q.bitwidth - 1) - 1) * s
            w.copy_(torch.minimum(torch.maximum(w, -hi), hi))
        n += 1
    print(f"clip_weights_to_7f7f: clamped {n} weight tensors")


def disable_quantizers(sim, fuse_gate_up, quant_head=False):
    """Disable per summary §2.2. Wrapper names: embed_tokens, norm, lm_head,
    layers.i.self_attn.{k,v}_proj (outputs), layers.i.mlp.gate_up_proj.
    v2 semantics: a quantizer slot is disabled by setting it to None.
    quant_head=True keeps lm_head's WEIGHT quantizer (8-bit per-channel: its
    FP16 weight is ~311MB of the ~960MB/token decode stream; W8 halves that)
    while still disabling its activation quantizers (logits stay FP16)."""
    disabled = []
    for name, m in _quantized_modules(sim):
        leaf = name.split(".")[-1]
        kill_all = leaf in ("embed_tokens", "norm", "lm_head")
        keep_params = quant_head and leaf == "lm_head"
        kill_out = kill_all or leaf in ("k_proj", "v_proj") or (fuse_gate_up and leaf == "gate_up_proj")
        if not (kill_all or kill_out):
            continue
        if getattr(m, "output_quantizers", None) is not None:
            for i in range(len(m.output_quantizers)):
                m.output_quantizers[i] = None
            disabled.append(name + ".out")
        if kill_all:
            pq = getattr(m, "param_quantizers", None)
            if pq is not None and not keep_params:
                for k in list(pq.keys()):
                    pq[k] = None
                disabled.append(name + ".params")
            iq = getattr(m, "input_quantizers", None)
            if iq is not None:
                for i in range(len(iq)):
                    iq[i] = None
    print(f"disabled quantizers on: {disabled}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=str(DATA / "models/Qwen3-0.6B"))
    ap.add_argument("--out", default=str(DATA / "work/quant/qwen3-0.6b"))
    ap.add_argument("--cl-prefill", type=int, default=128)
    ap.add_argument("--weight-bw", type=int, default=8, choices=(4, 8),
                    help="weight bitwidth (per-channel symmetric); 4 = W4A16 "
                         "experiment, k/v/o_proj stay 8-bit")
    ap.add_argument("--lpbq", action="store_true",
                    help="with --weight-bw 4: grouped blockwise (LPBQ) weight "
                         "quantizers instead of plain per-channel")
    ap.add_argument("--lpbq-block", type=int, default=64)
    ap.add_argument("--seq-mse", action="store_true",
                    help="apply Sequential MSE weight-range optimization "
                         "before compute_encodings (W4 recipes need this)")
    ap.add_argument("--quant-head", action="store_true",
                    help="quantize lm_head weight to 8-bit per-channel "
                         "(saves ~155MB/token of decode DDR stream)")
    ap.add_argument("--fuse-gate-up", action="store_true")
    ap.add_argument("--fuse-qkv", action="store_true")
    # aimet-torch 2.36 SHIPS the exact config the summary doc names — use it directly
    ap.add_argument("--config", default=str(
        Path(torch.__file__).parent.parent / "aimet_torch/common/quantsim_config/htp_quantsim_config_v81_per_channel_linear.json"))
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--eval", action="store_true",
                    help="after calibration, report quantsim-vs-FP32 last-token "
                         "argmax agreement on the reference prompts")
    ap.add_argument("--export-decode", metavar="PREFILL_OUT_DIR",
                    help="skip calibration: build the use_past=True decode wrapper, load "
                         "encodings from a previous prefill run (guarantees identical "
                         "cross-graph KV scales — the remote error-5005 constraint), export")
    ap.add_argument("--ctx", type=int, default=1024,
                    help="context size for --export-decode (past = ctx + cl_prefill - ar)")
    ap.add_argument("--decode-ar", type=int, default=1,
                    help="tokens per decode step for --export-decode. 1 = normal decode; "
                         "32 = lookahead/speculative verification graph (all-position "
                         "logits, same total window ctx + cl_prefill)")
    ap.add_argument("--adopt-encodings", metavar="PREV_OUT_DIR",
                    help="skip calibration on the PREFILL path: load encodings from a "
                         "previous prefill run of the same variant (graph-shape-only "
                         "rebuilds — keeps every scale bit-identical to the existing "
                         "decode/verify DLCs so they can be reused as-is)")
    args = ap.parse_args()

    from transformers import AutoModelForCausalLM, AutoTokenizer
    from aimet_common.defs import QuantScheme
    from aimet_torch.quantsim import QuantizationSimModel
    from aimet_torch.v2.nn import QuantizationMixin
    from modeling_export import ExportRMSNorm

    # Register quantized variant of our custom RMSNorm so AIMET v2 can wrap it
    # (summary §2.2: RMSNorm quantized, forced 16-bit via default_output_bw=16)
    if ExportRMSNorm not in QuantizationMixin.cls_to_qcls:
        @QuantizationMixin.implements(ExportRMSNorm)
        class QuantizedExportRMSNorm(QuantizationMixin, ExportRMSNorm):
            def __quant_init__(self):
                super().__quant_init__()
                self.input_quantizers = torch.nn.ModuleList([None])
                self.output_quantizers = torch.nn.ModuleList([None])

            def forward(self, x):
                if self.input_quantizers[0]:
                    x = self.input_quantizers[0](x)
                with self._patch_quantized_parameters():
                    ret = super().forward(x)
                if self.output_quantizers[0]:
                    ret = self.output_quantizers[0](ret)
                return ret

    tok = AutoTokenizer.from_pretrained(args.model)
    hf = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=torch.float32)
    cfg = hf.config
    S = args.cl_prefill
    if args.export_decode:
        ar = args.decode_ar
        past = args.ctx + S - ar  # total window (past + ar) stays ctx + cl_prefill
        model = ExportQwen3.from_hf(hf, args.fuse_gate_up, args.fuse_qkv,
                                    use_past=True, logits_last_only=False).to(args.device)
        n_kv, hd = cfg.num_key_value_heads, cfg.head_dim
        mask = torch.zeros(1, ar, past + ar, device=args.device)
        cos, sin = rope_tables(torch.arange(past, past + ar), cfg.head_dim, rope_theta_of(cfg))
        dummy = [torch.zeros(1, ar, dtype=torch.int32, device=args.device),
                 mask, cos.to(args.device), sin.to(args.device)]
        for _ in range(cfg.num_hidden_layers):
            dummy += [torch.zeros(1, n_kv, hd, past, device=args.device),
                      torch.zeros(1, n_kv, past, hd, device=args.device)]
        dummy = tuple(dummy)
    else:
        # logits_last_only MUST be False: Genie's basic dialog left-aligns input
        # and samples logits row n_process-1 (qualla nsp-model.cpp:3295) — a
        # last-only head reads out of bounds on device (2026-08-11 root cause).
        model = ExportQwen3.from_hf(hf, args.fuse_gate_up, args.fuse_qkv,
                                    use_past=False, logits_last_only=False).to(args.device)
        mask = causal_mask(S, S).to(args.device)
        cos, sin = rope_tables(torch.arange(S), cfg.head_dim, rope_theta_of(cfg))
        dummy_ids = torch.zeros(1, S, dtype=torch.int32, device=args.device)
        dummy = (dummy_ids, mask, cos.to(args.device), sin.to(args.device))
    del hf

    sim = QuantizationSimModel(
        model,
        dummy_input=dummy,
        quant_scheme=QuantScheme.post_training_tf_enhanced,
        default_param_bw=args.weight_bw,
        default_output_bw=16,
        config_file=args.config,
    )
    disable_quantizers(sim, args.fuse_gate_up, args.quant_head)
    if args.weight_bw < 8 and args.lpbq:
        # LPBQ (grouped blockwise, Qualcomm's W4A16 LLM recipe): int4 blocks of
        # 64 input channels, block scales grouped onto a per-channel int8 grid
        # (decompressed_bw). HTP supports it as BLOCKWISE_EXPANSION, arch >= v69.
        import torch.nn as nn
        from aimet_torch.quantsim.config_utils import \
            set_grouped_blockwise_quantization_for_weights

        def eligible(m):
            return (isinstance(m, nn.Linear)
                    and m.in_features % args.lpbq_block == 0)
        set_grouped_blockwise_quantization_for_weights(
            sim, eligible, bitwidth=args.weight_bw, symmetric=True,
            decompressed_bw=8, block_size=args.lpbq_block, block_grouping=-1)
        print(f"weight_bw={args.weight_bw}: LPBQ block={args.lpbq_block} "
              "applied to all divisible Linears")
    elif args.weight_bw < 8:
        # HTP W4A16: FullyConnected supports int4 per-channel symmetric weights,
        # but 4-bit grids are coarse — keep the attention projections whose
        # outputs feed the KV cache (and the output projection) at 8-bit.
        kept = 0
        for name, m in _quantized_modules(sim):
            if any(k in name for k in ("k_proj", "v_proj", "o_proj")):
                pq = getattr(m, "param_quantizers", None)
                if pq is not None and "weight" in pq and pq["weight"] is not None:
                    pq["weight"].bitwidth = 8
                    kept += 1
        print(f"weight_bw={args.weight_bw}: kept {kept} attention projections at 8-bit")

    def calibrate(m, _):
        with torch.no_grad():
            for p in CALIB_PROMPTS:
                ids = tok(p, return_tensors="pt").input_ids[:, :S]
                n = ids.shape[1]
                padded = torch.zeros(1, S, dtype=torch.int32, device=args.device)
                padded[0, -n:] = ids[0]
                cmask = causal_mask(S, S).to(args.device)  # rank-3 [1, S, S]
                cmask[:, :, : S - n] = -100.0
                pos = torch.cat([torch.zeros(S - n, dtype=torch.long), torch.arange(n)])
                c, s_ = rope_tables(pos, cfg.head_dim, rope_theta_of(cfg))
                m(padded, cmask, c.to(args.device), s_.to(args.device))

    if args.export_decode:
        # No calibration: adopt the prefill run's encodings so every shared
        # tensor (weights, activations, KV path) has IDENTICAL scales across
        # the prefill/decode graph boundary (remote error-5005 constraint).
        sim.load_encodings(str(Path(args.export_decode) / "model_torch.encodings"),
                           strict=False)
    elif args.adopt_encodings:
        # Graph-shape-only prefill rebuild: adopt the previous run's encodings
        # (already clipped) instead of recalibrating, so existing decode/verify
        # DLCs stay scale-compatible and only prefill.dlc needs regeneration.
        sim.load_encodings(str(Path(args.adopt_encodings) / "model_torch.encodings"),
                           strict=False)
    else:
        if args.seq_mse:
            # Sequential MSE picks per-layer optimal weight-encoding candidates
            # by minimizing activation MSE — Qualcomm's documented make-or-break
            # step for W4 LLM recipes. Must run BEFORE compute_encodings (it
            # freezes the chosen param encodings).
            from aimet_torch.seq_mse import apply_seq_mse
            batches = []
            with torch.no_grad():
                for p in CALIB_PROMPTS:
                    ids = tok(p, return_tensors="pt").input_ids[:, :S]
                    n = ids.shape[1]
                    padded = torch.zeros(1, S, dtype=torch.int32, device=args.device)
                    padded[0, -n:] = ids[0]
                    cmask = causal_mask(S, S).to(args.device)
                    cmask[:, :, : S - n] = -100.0
                    pos = torch.cat([torch.zeros(S - n, dtype=torch.long),
                                     torch.arange(n)])
                    c, s_ = rope_tables(pos, cfg.head_dim, rope_theta_of(cfg))
                    batches.append((padded, cmask, c.to(args.device), s_.to(args.device)))
            apply_seq_mse(sim=sim, data_loader=batches)
            print(f"seq_mse applied over {len(batches)} calibration batches")
        sim.compute_encodings(calibrate, None)
        clip_weights_to_7f7f(sim)

    if args.eval and not args.export_decode:
        eval_prompts = ["The capital of France is", "def fibonacci(n):",
                        "解释一下什么是注意力机制。", "1+2+3+...+100 ="]
        agree = 0
        for p in eval_prompts:
            ids = tok(p, return_tensors="pt").input_ids[:, :S]
            n = ids.shape[1]
            padded = torch.zeros(1, S, dtype=torch.int32, device=args.device)
            padded[0, -n:] = ids[0]
            cmask = causal_mask(S, S).to(args.device)
            cmask[:, :, : S - n] = -100.0
            pos = torch.cat([torch.zeros(S - n, dtype=torch.long), torch.arange(n)])
            c, s_ = rope_tables(pos, cfg.head_dim, rope_theta_of(cfg))
            c, s_ = c.to(args.device), s_.to(args.device)
            with torch.no_grad():
                q_logits = sim.model(padded, cmask, c, s_)[0][0, -1]
                f_logits = model(padded, cmask, c, s_)[0][0, -1]
            top1 = q_logits.argmax().item() == f_logits.argmax().item()
            agree += int(top1)
            print(f"[eval] {p!r}: quant argmax {'==' if top1 else '!='} fp32 argmax; "
                  f"max|dlogits|={float((q_logits - f_logits).abs().max()):.3f}")
        print(f"[eval] last-token argmax agreement: {agree}/{len(eval_prompts)}")

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    # aimet-torch 2.36.0 bug: quantsim.export references nn.lora.QuantizedLora,
    # which this build doesn't define. Provide an inert shim class.
    import aimet_torch.nn.lora as _lora
    if not hasattr(_lora, "QuantizedLora"):
        _lora.QuantizedLora = type("QuantizedLoraShim", (), {})
    # aimet-torch 2.36.0 bug #2: _onnx_model_size_larger_than_max_protobuf calls
    # proto.ByteSize(), which itself raises EncodeError for >2GB models (our FP32
    # 0.6B is ~2.4GB). Force the external-data path unconditionally.
    import aimet_torch.onnx_utils as _onnx_utils
    _onnx_utils._onnx_model_size_larger_than_max_protobuf = lambda _m: True
    sim.export(str(out), "model", dummy_input=tuple(t.cpu() for t in dummy))
    print(f"exported quantsim to {out} (model.onnx + model.encodings)")


if __name__ == "__main__":
    main()
