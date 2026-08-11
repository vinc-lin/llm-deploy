# llm-deploy — SA8797P LLM build pipeline (device-free)

Builds W8A16 Qwen3 Genie ctx-bins/bundles for Qualcomm SA8797P (Hexagon v81,
QAIRT 2.48.40, libGenie 1.19) with no device access. Bundles ship via HF
`vinccniv/sa8797p-qwen3-w8a16-bundles` (private); this repo holds only
scripts/configs/docs.

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
- Genie picks graphs by numeric (AR, CL) best-fit — names are cosmetic; never
  ship two graphs with the same (AR, CL); avoid AR==CL (bertcache) graphs with lade.
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
- Never let Windows C: drop below 6 GB (converter temp) — build scripts guard this.
- W4A16 is a dead end at 0.6B (per-channel, LPBQ-64, LPBQ+SeqMSE all fail the
  argmax gate); `--lpbq`/`--seq-mse` flags remain for larger models.

## Docs

`docs/BUILD_GUIDE.md` (full recipes) · `docs/LOCAL_ENV.md` (provenance, aimet
workarounds) · `docs/NOTES-genie-io.md` (Genie/qualla runtime contract, cited).
