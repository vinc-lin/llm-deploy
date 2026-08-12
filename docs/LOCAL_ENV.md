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

## Qwen3-VL-4B text tower: quantizes fine, does NOT export on this box (2026-08-12)

Stage 2 Phase C. The W8A16 recipe itself is **validated at 4B**: 22 multimodal
calibration windows, 396 weight tensors clipped, and `--eval` **4/4** last-token
argmax agreement (bar is 3/4) — including both all-visual windows, whose
activations span [-5.32, +4.97] against text's [-0.146, +0.124]. max|Δlogits|
1.03–1.46.

What does not fit is AIMET's **export**, killed by the OOM killer twice inside
`_create_onnx_model_with_markers`:

| attempt | env | anon-rss at kill | swap | wall |
|---|---|---|---|---|
| 1 | default (24 threads) | 37.4 GiB | 16 GiB exhausted | 23 min |
| 2 | `MALLOC_ARENA_MAX=2`, `OMP_NUM_THREADS=8`, `-u` | 45.4 GiB | 16 GiB exhausted | 20.5 min |

Machine budget is 47 GB RAM + 16 GB swap = 63 GB; attempt 2 committed ~61 GiB
and still died. The cause is structural, not fragmentation: the legacy
`sim.export` path holds **four** fp32 copies of the graph simultaneously —
`sim.model`, `model_to_export` (`get_original_model` deepcopy), the marker
deepcopy inside `_create_onnx_model_with_markers`, and the ONNX proto
`torch.onnx.export` builds in memory before writing. At 4B the wrapper is
4,022,468,096 params = 15.0 GiB, so that is 60 GiB before any runtime overhead.
Allocator tuning bought ~3% and moved the deadline, not the wall.

Things that do NOT help, measured rather than assumed:

- Making `onnx.load` skip external data — **neither globally nor scoped** to
  `aimet_torch._base.quantsim`. With the default `encoding_version = "1.0.0"`
  the re-read at `_base/quantsim.py:1084` is forwarded to
  `_derive_const_rescale_op_output_encodings`, which calls
  `numpy_helper.to_array` on Mul/Div operands to find constant scalars; both
  variants die with `ValidationError: Data of TensorProto (norm.weight) should
  be stored in <uuid>.data`. It is also memory-neutral — 0.6B peak RSS 13.64 GB
  scoped vs 13.57 GB baseline, because that read lands *after* the high-water
  mark, not at it.
- `update_all_onnx_nodes_name = False` would drop the later
  `copy.deepcopy(onnx_model)`, but both kills happened *before* it, so it does
  not move this peak either.

The real lever is the API AIMET itself deprecates the old one in favour of:
`sim.onnx.export()` / `aimet_torch/experimental/onnx/_export.py`. Switching
would need the encodings names and `rename_aimet_io.py`'s positional I/O
assumptions revalidated, and the prefill/decode encodings lineage re-proven.
Failing that: more RAM, or a bigger `.wslconfig` swap (needs an elevated
PowerShell + `wsl --shutdown`, so it cannot be done from inside the guest).

`--lean-export` (see `quantize_aimet.py`) is unrelated to the OOM but keeps the
disk honest: 66% of what quantsim writes per graph is scratch nothing reads
(`model.pth` + the all-markers temp export's per-initializer files).

## Lookahead decoding build (2026-08-10, attacks the fragmentation root cause)

The DDR collapse (49 → ~7 GB/s) comes from per-token weight streaming; the only
lever that *divides* bytes/token is multi-token decoding. Genie 1.19 ships
dialog types `lade` (lookahead, no draft model), `spd`, `eaglet`, `ssd-q1` —
contract extracted in `docs/NOTES-genie-io.md`. Built: AR=32 verification
graph (decode wrapper at S=32, past=1120, all-position logits) via
`scripts/build/lade_build.sh`, converted against the SAME prefill encodings.

| Artifact | Result |
|---|---|
| 3-graph ctx-bin (prefill128 + decode1 + verify32) | **1.5 GB** — weight sharing held across all 3 |
| verify32 dims | ids [1,32], mask [1,32,1152], past 1120, logits [1,32,151936] ✓ |
| AR=8 ONNX verify parity vs HF (CL=32/ctx=96) | all-position argmax match, max|Δ| 3.05e-05 |
| Bundle | `qwen3_06b_w8a16_lade.tar.gz` (lade + basic dialog configs, same bin) |

**Build-time DDR per verification pass (vtcm 16):** prefill 763 MB (0 spill),
decode 961 MB (0 spill), **verify32 1,906 MB with 745/750 MB VTCM spill/fill**
— AR=32 activations don't fit 16 MB VTCM. On raw bytes, lookahead breaks even
at ~2.0 accepted tokens/pass (LADE typical: 1.3–1.8 chat, higher for code).
BUT spill/fill is contiguous DMA (~49 GB/s class) while weight streaming is
the fragmented ~7 GB/s kind, so the spill may be cheap in wall-clock. Device
A/B (`genie_dialog.json` = lade vs `genie_dialog_basic.json` on the SAME
ctx-bin; Genie reports `tps.tokenAcceptance`) settles it.

Config guardrail (from qualla source): keep `(ngram-1)*(window+gcap) <= 32` —
oversized lade configs silently route batches to the prefill graph, which has
no past-KV inputs and cannot serve incremental verification.
Shipped config: window 8 / ngram 3 / gcap 8 = exactly 32.
Tuning option if spill hurts: an AR=16 graph (`lade_build.sh <name> 128 1024 16`)
with window 4 / gcap 4.

## Progress log

- 2026-08-10: full environment stood up and ENTIRE pipeline validated at smoke
  scale (see table above). One-shot pipeline: `scripts/build/full_build.sh`;
  bundling: `scripts/build/bundle.sh`. Full-size (CL=128/CTX=1024) build launched.
- 2026-08-10 (later): lookahead-decoding support built — AR=32 verification
  graph, 3-graph weight-shared ctx-bin, `lade` dialog config, ONNX parity
  passed (see section above). C: recovered to 109 GB free (VHD compacted).
- 2026-08-11: FIRST DEVICE RUN (remote tester, fuseqkvgu) → garbage from
  token 1. Root-caused device-free to OUR prefill export, not fusion: the
  last-token-only logits head `[1,1,V]` is incompatible with qualla's basic
  dialog, which left-aligns input and samples logits row `n_process-1`
  (out-of-bounds on a 1-row buffer → zero/noise logits → argmax = token 0
  = `"!"`). All bundles were affected. Full mechanism + citations:
  docs/NOTES-genie-io.md "Prefill logits contract". Encodings surgery
  verified CLEAN at JSON + DLC level (not the cause). Fix: prefill now
  exports all-position logits `[1,128,V]`; new regression guard
  `scripts/validate/parity_qualla_read.py` (left-aligned, row n−1) passed
  4/4 alongside existing parity. Rebuilds via
  `scripts/build/prefill_fix_rebuild.sh` (adopt-encodings, prefill-DLC-only).
  Device run also yielded first real KPIs: fused decode ~6.27 tok/s
  (159.6 ms/step ≈ 6 GB/s — fusion NOT beating baseline at vtcm 16);
  AR-128 prefill pass 50 ms vs AR-1 decode step 160 ms — first on-device
  evidence for multi-token (lade) amortization.
