# llm-deploy — SA8797P LLM build pipeline (device-free)

Builds W8A16 Qwen3 Genie ctx-bins/bundles for Qualcomm SA8797P (Hexagon v81,
QAIRT 2.48.40, libGenie 1.19) with no device access. Bundles ship via HF
`vinccniv/sa8797p-qwen3-w8a16-bundles` (Qwen3-VL artifacts:
`vinccniv/sa8797p-qwen3vl-4b-bundles`). This repo holds only
scripts/configs/docs.

**Building is device-free; the project is not.** A device team runs the bundles
and reports back: `reports/` holds their test reports (and dated photo drops,
transcribed to Markdown), `docs/DEVICE_*` the measurement reports and the
exchange protocol. Check `reports/` before trusting any performance or "does it
load" claim — the 2026-08-14 Qwen3-VL e2e attempt failed at load, and that is
only recorded there.

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

**Qwen3-VL (multimodal)** is a separate chain, two towers built independently:
`export_qwen3vl_vit.py` → `quantize_vit_aimet.py` → `vit_build_quant.sh` (vision)
and `vl_text_build.sh` → `vl_text_ctxbin_split.sh` (text, 2-split ctx-bin),
joined by `vl_pipeline_bundle.sh` into one `genie-app` pipeline bundle.

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
- **A stock Genie pipeline cannot drive an FP16 image encoder.** `setupInputFP16`
  is an empty stub that discards the pixel blob and returns success, and the
  requantize table has no `Float16` entries either way. The vision tower must
  ship W8A16 with `UFIXED_POINT_16` IO.
- `pos-id-dim` (backend block) alongside `positional-encoding` is a **hard
  load-time schema error**, not a warning. Declare one. Same for `rope-theta`.
- A prefill graph whose `attention_mask` is `[1,AR,AR]` registers `ctx_size ==
  AR` (bertcache) and is **never selected** for prompts longer than AR — the
  whole prompt goes through the AR=1 decode graph, silently and slowly.
  **Unsplit only.** In a split tower the last shard owns the lm_head, so shard
  0's prefill has no logits, classifies `DECODER_PREFILL`, and has its expected
  CL rewritten to the cache-group max — the mask then fails validation and the
  node **never loads** (`Failed to create the Genie Node (-1)` + SIGSEGV, one
  ShapeError per shard). Splitting is mandatory ≳2B, so **any ≳2B model must
  ship a past-KV prefill (`[1,AR,CL]`, `CL>AR`) or no prefill graph at all**;
  the 0.6B pattern does not transfer. See `docs/REFERENCE.md` §3.6.
- Image-encoder configs need `vision-param: {height, width}` in **patch units**
  (pre-merge), or MRoPE never engages and image rows fall back to plain rope.

## Validation gates (run before shipping any bundle)

- `scripts/validate/parity_ladekv_read.py` (qualla feed pattern incl. chunking)
  and/or `parity_qualla_read.py` — argmax must match HF on all prompts.
- `quantize_aimet.py --eval` reference is 3/4 last-token argmax agreement.
- Qwen3-VL: `scripts/validate/parity_e2e_vl.py` — full path (image → ViT →
  splice → text tower) vs `hf.generate`, token-for-token. `lint_pipeline_bundle.py`
  for bundle contracts. Run the gate with no `--chains` filter; a subset skips
  the mutation checks.

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
  DDR win does not survive spec-decode amortization. It pairs with
  `filter_aimet_w8a16.py --keep-head-weight` — **not** a `quantize_aimet.py`
  flag; the build scripts add it to the filter automatically when they see
  `--quant-head`. Without it the filter strips the encoding and you silently get
  an FP16 head — verify with `qairt-dlc-info | grep lm_head.weight` → `sFxp_8`.
- `genie-app` script strings **never unescape**: `"\n"` in a quoted `node set
  text` argument yields the two characters `\` and `n`. Use `node set textFile`
  (read via `rdbuf()`) whenever the prompt needs a real newline. The SDK's own
  GLM-4v example gets this wrong.
- `clip_weights_to_7f7f` assumes a symmetric grid. Guard on symmetry before
  clamping — it silently halved every asymmetric LayerNorm gain in the ViT
  (cos 0.758 → 0.997 once fixed). RMSNorm models are unaffected.

## Docs

**`docs/REFERENCE.md` first** (consolidated current truth: contracts, measured
numbers, dead ends, open questions) · `docs/BUILD_GUIDE.md` (full recipes) ·
`docs/LOCAL_ENV.md` (provenance, aimet workarounds) · `docs/NOTES-genie-io.md`
(Genie/qualla runtime contract, cited) · `docs/NOTES-genie-splits.md`
(multi-ctx-bin contract — mandatory ≳2B) · `docs/NOTES-genie-pipeline.md`
(multimodal pipeline contract: image-encoder dtype, MRoPE, deepstack-by-zeros) ·
`docs/NOTES-htp-config-keys.md` (which HTP backend-extension keys are real,
audited against the SDK's own `config.py` / `QnnHtpGraph.h`) ·
`docs/SA8797P_HTP_v81_Hardware_and_Deployment_Quantization_Reference_EN.md`
(device team's measured hardware truth, annotated).
