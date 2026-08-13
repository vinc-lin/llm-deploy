# llm-deploy — SA8797P LLM build pipeline (device-free)

Builds W8A16 Qwen3 Genie ctx-bins/bundles for Qualcomm SA8797P (Hexagon v81,
QAIRT 2.48.40, libGenie 1.19) with no device access. Bundles ship via HF
`vinccniv/sa8797p-qwen3-w8a16-bundles` (Qwen3-VL artifacts:
`vinccniv/sa8797p-qwen3vl-4b-bundles`). This repo holds only
scripts/configs/docs.

**Repo visibility is switched often and deliberately by the user — this file
does not state it on purpose.** Read it live (`HfApi().repo_info(r).private`)
if it matters; change it **only** when asked in that message; if a bulk upload
flips it as a side effect, report and stop. Flipping it unasked — in either
direction — has caused four incidents.

## Setup

- `source scripts/env.sh` FIRST in every shell (QAIRT_SDK, PY envs, LD_LIBRARY_PATH).
- uv envs: `qwen3-deploy` (torch/aimet/onnx export+quant), `qairt-py312` (converter only).
- Heavy data (SDK, models, work, bundles) lives in `/home/vinc/llm-local/` — never in the repo.
- `QUANT_DEVICE=cpu` for models >0.6B (8 GB VRAM box).

## Build chain

`full_build.sh <name>` → `lade_build.sh <name>` (verify32) → `ladekv_build.sh <name>`
(past-KV prefill, 3-graph) → `bundle.sh <bundlename> <ctxbin> [dialog_json]`.
Extra flags after `full_build.sh <name> <cl> <ctx>` pass through to
`quantize_aimet.py` (e.g. `--quant-head`, `--fuse-gate-up`).

Cross-graph rule: decode/verify/prefill DLCs must convert against the SAME
encodings lineage (`--export-decode` / `--adopt-encodings`); mixed encodings
are a fatal Genie load error (KV quant params must be byte-identical).

## Hard contracts (violations = silent device garbage or SIGSEGV)

- Prefill graphs MUST emit all-position logits `[1,AR,vocab]`.
- Genie picks graphs by numeric (AR, CL) best-fit — names are cosmetic *to Genie*;
  never ship two graphs with the same (AR, CL); avoid AR==CL (bertcache) graphs with lade.
- **Graph names are NOT cosmetic to the HTP backend.** A graph's name is baked in
  at conversion time from the `--output_path` basename, dots included: converting
  to `decode.dlc.new` yields graph `decode_dlc`, and renaming the file afterwards
  does NOT change it. `htp_backend_ext_config.json`'s `graph_names` must match the
  names inside the ctx-bin exactly, or that graph silently gets backend defaults
  (4 MB VTCM, 24 MB spill) — or, for lade, a null-pointer SIGSEGV on the first
  speculation step. Always convert straight to the final filename, and verify with
  `qnn-context-binary-utility --json_file` before bundling.
- Read `docs/NOTES-genie-io.md` before touching graph topology or configs.

## Validation gates (run before shipping any bundle)

- `scripts/validate/parity_ladekv_read.py` (qualla feed pattern incl. chunking)
  and/or `parity_qualla_read.py` — argmax must match HF on all prompts.
- `quantize_aimet.py --eval` reference is 3/4 last-token argmax agreement.

## Gotchas

- HF: proxy `http://127.0.0.1:17890` required, but it drops long upload streams —
  use `scripts/util/hf_upload_watchdog.sh` (set `SOCKET_CHECKS=999999`; the
  socket detector false-positives through the proxy).
- Hub limit: 128 repo commits/hour. "Hung" commit phase with all blobs
  pre-uploaded = 429; diagnose with one foreground `HfApi().upload_file`,
  recover with spaced single-file commits after ~1h.
- Never let Windows C: run dry: `$LLMDEPLOY_DATA` sits on the ext4.vhdx, which is
  on C:. A failed vhdx grow is NOT ENOSPC — the guest still reports free space,
  the host write fails, and every mmap'd page takes SIGBUS; PID 1 dies and the VM
  hard-crashes with no OOM line anywhere (3x on 2026-08-12, during VL-4B stage 2).
  Dumps land in `%LOCALAPPDATA%\Temp\wsl-crashes`; the `-N` suffix is the signal,
  `-7` = SIGBUS.
- `disk_guard [need_gb]` lives in `scripts/env.sh` (every build script sources it) —
  call it before any multi-GB step, sized to that step: 6 GB is the converter
  floor, a 4B export writes 8.6 GB and asks 20. A flat 6 GB check passes and then
  still runs C: dry mid-step.
- The vhdx is sparse and `/` is mounted `discard`, so deleting in-guest reclaims C:
  with no compaction step. `ls` reports the ~448 GB virtual size and always will;
  `du -h <vhdx>` (no `--apparent-size`) is the real consumption.
- W4A16 is a dead end on this SDK at any size: quality 0/4 at 0.6B, AND v81's
  `htp_v2.json` ships zero INT4 matmul kernels — the converter folds s4→f16
  (`--lpbq`/`--seq-mse` stay only for a future SDK).
- `--quant-head` (W8 lm_head) is a **net LADE regression**: −14% tok/s on device
  (9.3 vs 10.8), because it costs ~10% n-gram acceptance. Quality is fine; the
  DDR win does not survive spec-decode amortization. It also needs
  `--keep-head-weight` or the filter strips the encoding and you silently get an
  FP16 head — verify with `qairt-dlc-info | grep lm_head.weight` → `sFxp_8`.

## Docs

**`docs/REFERENCE.md` first** (consolidated current truth: contracts, measured
numbers, dead ends, open questions) · `docs/BUILD_GUIDE.md` (full recipes) ·
`docs/LOCAL_ENV.md` (provenance, aimet workarounds) · `docs/NOTES-genie-io.md`
(Genie/qualla runtime contract, cited) · `docs/NOTES-genie-splits.md`
(multi-ctx-bin contract — mandatory ≳2B) ·
`docs/SA8797P_HTP_v81_Hardware_and_Deployment_Quantization_Reference_EN.md`
(device team's measured hardware truth, annotated).
