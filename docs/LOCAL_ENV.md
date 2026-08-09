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

## Progress log

- 2026-08-10: dirs + venvs created; Qwen3-0.6B downloaded (1.5 GB, ext4);
  QAIRT 2.48.40.260702 confirmed publicly downloadable, segmented download in
  progress; export/quant/parity/surgery/build scripts written (commits in git log).
