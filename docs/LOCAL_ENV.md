# Local (Device-Free) SA8797P Environment — Status

*Started 2026-08-10. Companion to `docs/superpowers/plans/2026-08-10-local-sa8797p-pipeline.md`.*

## Machine

- WSL2 (kernel 6.6.87.2), Ryzen 9 5900X (24T), 47 GB RAM, RTX 4060 Ti 8 GB (CUDA passthrough OK)
- Repo: `/mnt/x/code/llm-deploy` (drvfs/NTFS — Windows-visible; `~/code` is a symlink here)
- Heavy data: `/home/vinc/llm-local` (real ext4, 820 GB free): `envs/`, `models/`, `work/`, `sdk/`
- No sudo; `uv` manages Python toolchains. Proxy available at `http://127.0.0.1:17890` (`proxy-on`).

## Divergences from the remote (jump-host) environment

| Item | Remote | Local |
|---|---|---|
| SDK | `/mnt/code/toolchains/qairt/2.48.40/` | `$LLMDEPLOY_DATA/sdk/qairt/2.48.40.260702/` (same version, Community drop) |
| Envs | conda `qwen3-deploy` / `qairt-py312` | uv venvs, same names, `$LLMDEPLOY_DATA/envs/` |
| Pipeline scripts | `sa8797_deploy_kit` + `scripts/quant/*` (NOT available locally) | **reconstructed from the summary doc** under `scripts/` — expect drift; parity tests are the arbiter |
| Device | `ssh <JUMPHOST>` → adb REDACTED | none — build + numerics only |

## What can NEVER be validated locally

tok/s, DDR bandwidth, VTCM/unsigned-PD behavior, perf profiles, GVM effects,
Genie on-HTP execution. Local success = artifact correctness + numerical parity.

## Reconstruction uncertainties (review before trusting)

1. **`clip_weights_to_7f7f`** — reconstructed as "clamp weights to symmetric ±127
   steps so nothing maps to INT8 -128". Original semantics unknown.
2. **Genie graph I/O naming** (`position_ids_cos/sin`, `past_key_i_in/out`, KV
   layout `[1, n_kv, len, head_dim]`, right-aligned sliding window) — chosen from
   qai-hub/Genie conventions + summary values (pos-id-dim 64, kv-dim 128); MUST be
   verified against SDK Genie docs once unpacked (plan Task 2 Step 4).
3. **Config JSON schemas** (`configs/*.json`) — values are the validated ones from
   the summary; key spellings/nesting drafted from memory of QNN conventions,
   marked `_comment: DRAFT`, must be checked against SDK docs/examples.
4. **QKV surgery** realized at the ENCODINGS level (opset 17 has no INT16 QDQ);
   see header of `scripts/export/qkv_surgery.py`.
5. **Attention mask value** −100 (FP16-safe) vs remote's unknown mask constant.
6. **AIMET version**: remote version unknown; local pin recorded below after the
   API smoke test.

## Version pins (filled as installed)

- Python 3.10.19 (`qwen3-deploy`), 3.12.x (`qairt-py312`)
- onnx == 1.19.0 (REQUIRED — ≥1.20 breaks AIMET export, summary §1.3)
- numpy < 2 in qwen3-deploy
- torch / aimet-torch / transformers: TBD (see pip log)

## Version pins (final)

- torch 2.13.0+cu130 (CUDA works in WSL2), aimet-torch 2.36.0, transformers 5.14.1,
  onnx 1.19.0 (BOTH envs — converter needs `onnx.version`, gone in ≥1.20),
  onnxruntime 1.23.2, numpy 1.26.4, setuptools (py312 distutils shim)
- aimet-torch 2.36.0 bugs worked around in `quantize_aimet.py`: missing
  `nn.lora.QuantizedLora` attr; `ByteSize()` crash on >2GB proto; and
  `load_encodings` unreliable across graph variants (only 48/308 scales carried) —
  decode DLC therefore converts against the PREFILL encodings file directly
  (280/280 + 308/308 tensor-name overlap verified — bit-identical cross-graph scales)

## Validated pipeline results (2026-08-10, smoke scale CL=64)

| Stage | Result |
|---|---|
| Export wrapper vs HF forward | max abs diff 1.79e-05 |
| ONNX prefill parity (ORT) | 3.48e-05, argmax match |
| ONNX decode chain, 8 greedy steps | token-identical vs HF generate |
| AIMET W8A16 | 196 PER_CHANNEL INT8 weights, A16, embed/lm_head/kv-proj FP16 ✓ |
| Quantized DLC | 1.1 GB (matches remote §2.3: 1.1 vs 1.5 GB FP16) |
| Two-graph quantized ctx-bin | 2.0 GB, prefill(4in/57out) + decode(60in/57out), vtcm16/v81 |

AIMET-export I/O names are mangled → `scripts/quant/rename_aimet_io.py`
restores canonical Genie names (positional + consumer-pattern checks).

## Full-size build results (CL=128 / CTX=1024, weight-shared, vtcm 16, v81)

| Variant | ctx-bin | Build-time DDR read (prefill / decode) | VTCM spill |
|---|---|---|---|
| baseline W8A16 | 1.5 GB (2.1 GB without weight sharing) | 759 MB / 957 MB | 0 |
| + Gate-Up fusion | 1.5 GB | 769 MB / 961 MB | 0 |
| + QKV fusion (surgery 28/28) | 1.5 GB | 763 MB / 961 MB | 0 |

Weight sharing reproduces the remote's recorded 1.5 GB binary size exactly.

**Key observation vs summary §3.3/§4.3-Q1:** at vtcm 16, fusion gives **no
build-time DDR byte reduction** (remote's 3.4× reduction was measured at
vtcm_mb=24, which is rejected on the device's unsigned PD). All variants show
zero VTCM spill at 16 MB. Any real fusion gain must come from fewer/larger DMA
transactions (access-pattern quality), which only on-device measurement can
confirm. The QKV quantizer conflict (remote error 5005 / garbage) is resolved
at the ENCODINGS level — surgered encodings accepted by converter + generator
at vtcm 16; runtime/quality proof still requires hardware.

## Quantization quality (local proxy, CL=128, fixed clip active)

Quantsim-vs-FP32 last-token argmax agreement on the 4 reference prompts: **3/4**
(miss = "1+2+3+...+100 =", a near-tie; max|Δlogits| ≈ 1.3–1.6 across prompts).
Consistent with the remote's on-device "coherent output" finding for W8A16.

## Progress log

- 2026-08-10: full environment stood up and ENTIRE pipeline validated at smoke
  scale (see table above). One-shot pipeline: `scripts/build/full_build.sh`;
  bundling: `scripts/build/bundle.sh`. Full-size (CL=128/CTX=1024) build launched.
