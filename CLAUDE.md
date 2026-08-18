# llm-deploy — SA8797P LLM build pipeline (device-free)

Builds W8A16 Qwen3 Genie ctx-bins/bundles for Qualcomm SA8797P (Hexagon v81,
QAIRT 2.48.40, libGenie 1.19) with no device access. Bundles ship via HF
`vinccniv/sa8797p-qwen3-w8a16-bundles` (Qwen3-VL artifacts:
`vinccniv/sa8797p-qwen3vl-4b-bundles`). This repo holds only
scripts/configs/docs.

**Building is device-free; the project is not.** A device team runs the bundles
and reports back: `reports/` holds their test reports, `docs/DEVICE_*` the
measurement reports and the exchange protocol. Some reports are transcribed
from screen photographs — the Markdown IS the record, and the source photos are
deleted once it is committed, so never treat a missing photo drop as lost data.
Check `reports/` before trusting any performance or "does it load" claim — the
2026-08-14 Qwen3-VL e2e attempt failed at load, and that is only recorded there.

**Current state (device, 2026-08-15):** best 0.6B decode is **44.707 tok/s** —
`gqafix_ladekv`, **basic** mode (`genie_dialog_basic.json`), TTFT 103 ms. LADE is
**parked**: post-fix it is a 30% regression. Decode was compute-bound, not
DDR-bound, so byte levers are demoted. `docs/REFERENCE.md` §0 is the live board.

**Repo visibility is switched often and deliberately by the user — this file
does not state it on purpose.** Read it live (`HfApi().repo_info(r).private`)
if it matters; change it **only** when asked in that message; if a bulk upload
flips it as a side effect, report and stop. Flipping it unasked — in either
direction — has caused four incidents.

## Setup

- `source scripts/env.sh` FIRST in every shell (QAIRT_SDK, PY envs, LD_LIBRARY_PATH).
- uv envs: `qwen3-deploy` (torch/aimet/onnx export+quant), `qairt-py312` (converter only).
- Heavy data (SDK, models, work, bundles) lives in `/home/vinc/llm-local/` — never in the repo.
- `QUANT_DEVICE=cpu` for models >0.6B locally (8 GB VRAM box); **always** on `tank` (no GPU).

### Two build hosts — know which one you are on

`scripts/env.sh` derives its paths from its own location, so the repo and every
build script run unchanged on either. What differs is what each host *can* do:

| | this WSL box | `tank` (`ssh tank`) |
|---|---|---|
| CPU / RAM | 5900X 24T / 47 GB | **44 cores / 125 GB** |
| GPU | RTX 4060 Ti 8 GB, CUDA ✓ | **none** — `torch.cuda.is_available()` is False |
| Disk | C:-backed vhdx, the SIGBUS hazard | 937 GB native, **no vhdx indirection** |
| Hugging Face | ✓ via `127.0.0.1:17890` | ✗ **unreachable** (IPv6-only DNS, no route) |
| GitHub | ✓ ssh | HTTPS only — the ssh key is not authorized |

**Consequences, not preferences:** anything needing >47 GB RAM (a 4B export peaks
at 63.5 GB) or large scratch must run on tank; anything touching HF — model
download, `bundle.sh` upload, `hf_upload_watchdog.sh` — **cannot** run there and
stays local. The proxy is bound to localhost, so this is not a config fix.

`tank:~/llm-deploy` is a real git repo (`origin` = the GitHub mirror). Re-sync it
from here with `git push ssh://tank/home/vinc/llm-deploy main` — it has
`receive.denyCurrentBranch=updateInstead`, so the working tree updates in place.
**Do this before every remote build.** A stale copy there once predated the
`ctxbin_variant.sh` readback gate, which is exactly the check that catches a knob
silently failing to bind — an unbound knob looks like a measurement, not an error.

**Scripts are not executable on tank.** This repo lives on a Windows-backed
mount where every file reads `0777`, so the exec bit never enters git; on tank
the same files land `0664` and `scripts/build/foo.sh` fails `Permission
denied` (exit 126). Always invoke build scripts there as `bash
scripts/build/foo.sh`. Verified, not theoretical — it bit this session.

### What lives where

Tank holds the canonical `work/` for **Qwen3-VL-4B** (its export does not fit
locally) and now the **Qwen3-0.6B** lineage. Retain **encodings + `.onnx` graphs
+ calib `.npz`** and let the heavy tensors be regenerable: one 0.6B quant dir is
8.6 GB of which ~135 MB is irreplaceable, so the whole 0.6B lineage is 1.1 GB
against 69 GB of source dirs. Same logic for ctx-bins — generation is
deterministic, so **store the recipe, not the variant**.
A bundle `.tar.gz` already contains its ctx-bin, so `work/ctxbin/<v>/` is a pure
duplicate for any bundled variant — verify by byte size, then drop it. And a
bin's `info.json` is ~0.01% of its size while answering most later questions:
**strip the `.bin`, keep the sidecar**.

## Build chain

`full_build.sh <name>` → `lade_build.sh <name>` (verify32) → `ladekv_build.sh <name>`
(past-KV prefill, 3-graph) → `bundle.sh <bundlename> <ctxbin> [dialog_json]`.
Extra flags after `full_build.sh <name> <cl> <ctx>` pass through to
`quantize_aimet.py` (e.g. `--quant-head`, `--fuse-gate-up`).

**`--grouped-gqa` is mandatory on every 0.6B build.** Without it you ship the
pre-fix model — 6.836 tok/s instead of 44.707. `full_build.sh` takes it as a
pass-through flag; `lade_build.sh` / `ladekv_build.sh` need
`FUSE_FLAGS="--grouped-gqa"`, because they re-export verify32 and the past-KV
prefill and will otherwise silently ship old attention in *those* graphs while
the decode graph looks correct.

Cross-graph rule: decode/verify/prefill DLCs must convert against the SAME
encodings lineage (`--export-decode` / `--adopt-encodings`); mixed encodings
are a fatal Genie load error (KV quant params must be byte-identical).

**Qwen3-VL (multimodal)** is a separate chain, two towers built independently:
`export_qwen3vl_vit.py` → `quantize_vit_aimet.py` → `vit_build_quant.sh` (vision)
and `vl_text_build.sh` → `vl_text_ctxbin_split.sh` (text, 2-split ctx-bin),
joined by `vl_pipeline_bundle.sh` into one `genie-app` pipeline bundle.
**`--grouped-gqa` is mandatory here too**: `vl_text_build.sh <name> <cl> <ctx>
--grouped-gqa` passes it through, but it must ALSO reach the separate past-KV
prefill export (`quantize_aimet.py --export-decode ... --grouped-gqa`) and
`export_qwen3vl_text.py --grouped-gqa` for the fp32 gate exports — a graph
that misses it silently ships old attention while the others look correct.
v2 shipped exactly that: 36 replication ops per shard at a 4:1 head ratio.

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
- **Weight sharing is verifiable, and silently optional.** The same `--json_file`
  dump reports `graphBlobInfoV2.sharedWeightsSize` (the context's single pool,
  printed identically on every graph — never sum it) and `constSize` (that
  graph's *private* copy; the SDK header's own map calls it "Non-Shared (Const)").
  The invariant is **pooled fraction**, not a hard zero: the pool should hold
  essentially the whole weight set. Measured — `gqafix_ladekv` 100% (const 256 B),
  `gqafix_qh_ladekv` 95%, vs **`w8a16qh` 29%** and `w8a16qh-lade` 62%, where each
  graph re-carries ~755 MB and the bin inflates 1.09 → 1.84 GB (`REFERENCE.md` §8.2).
  ⚠ **Non-zero `constSize` is not automatically a fault** — `--quant-head` moves
  ~144 MB into a private decode block by design (§8.1) and a bertcache graph
  forces a private 444 MB copy (§6.10). A gate demanding `constSize == 0` would
  reject good bins; gate on pooled fraction with the known exceptions named.
  ⚠ These key names are not guessable — `vtcmSize` (not `vtcmSizeInMB`),
  `graphBlobInfoV2` (not `graphBlobInfo`; the V1 struct has no weight fields at
  all). Grep the raw JSON before asserting on one, or the assert silently passes
  on a field that never existed.
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
- **The shipped image blob is FLOAT32 (`*_fp32.raw`), never tensor-native
  UFixed16.** Genie's image staging reinterprets the file as float32 and
  quantizes on device (`nsp-image-model.cpp:501-524`; `embedding-datatype`
  defaults to FLOAT_32 and no image-encoder config key routes to it), so a
  UFixed16 blob is read at 2× its size — a ~3 MB over-read, the `SIGSEGV
  (SEGV_ACCERR)` that blocked the device 2026-08-15..17. Padding cannot fix it
  (correction #35 — the earlier 1-byte/guard-page theory here was wrong).
  fp32 payload 1024×1536×4 = 6,291,456 B + 4096 B inert pad = 6,295,552 B;
  `preprocess_image.py` / `build_test_kit.py` emit it plus a `*_u16.raw`
  (exact tensor bytes, **qnn-net-run triage only**); `lint_pipeline_bundle.py`
  enforces sizes and fp32↔u16 agreement. `docs/NOTES-genie-pipeline.md` D1b.

## Validation gates (run before shipping any bundle)

- `scripts/validate/parity_ladekv_read.py` (qualla feed pattern incl. chunking)
  and/or `parity_qualla_read.py` — argmax must match HF on all prompts.
- `scripts/validate/lint_gqa_ops.py` — **0** replication ops in *every* graph.
  Non-zero means `--grouped-gqa` was missed, usually in verify32 or the past-KV
  prefill rather than decode.
- `quantize_aimet.py --eval` reference is 3/4 last-token argmax agreement.
- Qwen3-VL: `scripts/validate/parity_e2e_vl.py` — full path (image → ViT →
  splice → text tower) vs `hf.generate`, token-for-token. `lint_pipeline_bundle.py`
  for bundle contracts. Run the gate with no `--chains` filter; a subset skips
  the mutation checks. Also `lint_gqa_ops.py` on all four text DLCs with
  **`--layers 18`** (the per-shard count) — the flag defaults to 28 (the 0.6B
  tower), and since the lint's pass criterion is `len(matmuls) == 2 * layers`,
  the default reports FAIL on a perfectly correct shard.
  `vl_text_ctxbin_split.sh` now runs this automatically (commit `0db9643`) —
  don't add a manual duplicate.

## Gotchas — environment & infrastructure

- HF: proxy `http://127.0.0.1:17890` required (export `https_proxy`/`http_proxy`
  yourself — neither `env.sh` nor the watchdog sets them), but it drops long
  upload streams — use `scripts/util/hf_upload_watchdog.sh`.
  ⚠ **`upload-large-folder` transfers the bytes fine and then hangs at the
  COMMIT.** Measured 2026-08-17 on the VL v3 bundle: five watchdog attempts,
  each ending with every socket CLOSE-WAIT and `/proc/PID/io` flat, `Files:`
  stuck at `pre-uploaded: 8/26, committed: 0/60`. It looks like a transfer
  failure and is not: a later per-file `upload_file` of each 1.85 GB / 2.63 GB
  ctx-bin reported **`New Data Upload: 0.00B`** and committed in 9–11 s,
  i.e. the 4.5 GB was already server-side the whole time. Do not diagnose this
  from the progress bars — they reach 99–100% and keep redrawing forever, which
  also defeats the watchdog's progress-freeze detector, and `SOCKET_CHECKS=999999`
  disables the socket detector, so the watchdog waits indefinitely.
  **Working recipe when it stalls:** `HfApi().upload_folder(..., ignore_patterns=[<big files>])`
  for the bulk (58 files, one commit, 9 s) then one `upload_file` per file
  ≳1 GB. Four commits total, well inside the 128/h limit. Confirm a suspected
  stall with `/proc/PID/io` (zero delta) before killing anything.
- **Nothing is visible in the repo until the commit phase.** `upload-large-folder`
  pre-uploads every blob first and commits at the end, so `list_repo_files`
  returning 0 mid-run is normal, not a failure.
- Verify an upload against the **re-downloaded** bytes, never the local ones:
  `hf_hub_download` the `info.json`s + node config and re-run
  `scripts/validate/genie_load_check.py` on them.
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
  with no compaction step (asynchronously — free space does not jump immediately).
  `ls` reports the ~448 GB virtual size and always will; `du -h <vhdx>` (no
  `--apparent-size`) is the real consumption.
- **`df /` lies about headroom — check `df -h /mnt/c`.** In-guest `df` reports the
  ext4 filesystem inside the vhdx (recently: 537 GB "free") while the C: drive
  actually backing it had **54 GB**. That gap is the SIGBUS crash, one command
  away. Check C: before any multi-GB step, not `/`.
- **Never derive the same artifact twice — `coord_guard`, not `md5sum`.** ctx-bin
  generation is deterministic (same DLCs + config → byte-identical bin), so a
  "new" variant is often a ~20 min, multi-GB re-derivation of something already
  on disk. The old rule here said to `md5sum` the candidate against
  `work/ctxbin/*/` first; that is impossible — the candidate does not exist
  until you have paid for it — which is why concurrent sessions duplicated
  builds anyway. Determinism means you can hash the *inputs* instead:
  `coord_guard` (in `scripts/env.sh`, same shape as `disk_guard`) keys on
  DLCs + graph names + config + SDK in ~5 s cold, 0.3 ms warm, and refuses if
  `state/artifacts.tsv` already has that recipe or another session on this host
  is deriving it right now. Already wired into `ctxbin_variant.sh` and
  `ladekv_build.sh`. `coord.py scan` lists byte-identical groups **and separates
  real duplicates from hard links by inode** — equal md5 is not two copies, and
  `splitkv-flat` was hardlinked to `splitkv`, so 4.51 GB of apparent waste was
  worth 0 B. Only `ctrl` == baseline was real (1.09 GB, deleted 2026-08-17).
  Quote `actually reclaimable`, never the group sizes. `coord.py who` before
  starting work: another branch may be on it. Full convention in
  `docs/NOTES-coordination.md`.
- **`.gitignore` and `git status` are not evidence about what is public.** Photo
  drops sat in the public mirror's *history* for five days while the working
  tree looked clean (removed 2026-08-16 by delete-and-recreate). Verify with a
  throwaway unauthenticated `git clone --bare https://github.com/...` then
  `git rev-list --objects --all | grep -iE '\.(heic|jpg|png|pdf)$'`. Backups of
  both pre-rewrite histories are at `/mnt/x/llm-deploy-backup-2026-08-16/`.
  (`git filter-repo` prompts about a prior run — pipe `yes Y |`.)
- No HEIC decoder on this box (`pillow-heif` absent; ffmpeg fails the HEIF
  container), so a `reports/NNNN/*.HEIC` drop cannot be opened programmatically.
  Ask for PNG/JPEG if one needs reading.

## Gotchas — build, config & measurement

- W4A16 is a dead end on this SDK at any size: quality 0/4 at 0.6B, AND v81's
  `htp_v2.json` ships zero INT4 matmul kernels — the converter folds s4→f16
  (`--lpbq`/`--seq-mse` stay only for a future SDK).
- `--quant-head` (W8 lm_head) has bought nothing in any measurement so far:
  −14% under LADE (9.3 vs 10.8, an n-gram-acceptance cost) and ≈−2% in basic
  mode (6.70 vs 6.836, a pre-fix same-topology pair). Untested post-fix. Note
  the DLC shrinks 151 MB but the ctx-bin only 12.5 MB — the head *is* correctly
  halved in the shared pool; ~144 MB simply moves to a private const block on
  decode (`REFERENCE.md` §8.1). It pairs with
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
- **The SDK's `examples/Genie/.../src/qualla` tree is NOT the source of the
  shipped `libGenie.so`.** The device rejects unknown `QnnHtp` config keys and no
  such validation exists anywhere in that tree — a fallback derived from it
  failed on device. Use the source for runtime *contracts*; caveat any claim
  about what the binary actually does.
- **`hvx_threads` is baked in at ctx-bin build time, not read at runtime.**
  Changing it in `htp_backend_ext_config.json` does nothing (measured: −0.1%).
  Set it in `scripts/build/ctxbin_variant.sh` / `configs/htp_backend_config.json`
  and verify with `numHvxThreads` from `qnn-context-binary-utility --json_file`.
  Every shipping bin is currently **4 of the 8 available HVX units**.
- **Never quote a `read_total_bytes` without its build log and date.** The
  decode-graph figure 961,130,496 is *pre*-GQA-fix (`ctxbin-ws.log`, 2026-08-10);
  reusing it as a current number has now put two documents wrong. The converter's
  `====== DDR bandwidth summary ======` block is the source, and it is emitted
  even for builds that cannot run on device.

## Docs

**`docs/REFERENCE.md` first** (consolidated current truth: contracts, measured
numbers, dead ends, open questions) · `docs/PLAN_0.6B_max_tps.md` (the current
0.6B speed plan — unversioned and living; the old V1–V4 ladder is in
`docs/archive/`, which is never a source for a number) ·
`docs/BUILD_GUIDE.md` (full recipes) ·
`docs/LOCAL_ENV.md` (provenance, aimet workarounds) · `docs/NOTES-genie-io.md`
(Genie/qualla runtime contract, cited) · `docs/NOTES-genie-splits.md`
(multi-ctx-bin contract — mandatory ≳2B) · `docs/NOTES-genie-pipeline.md`
(multimodal pipeline contract: image-encoder dtype, MRoPE, deepstack-by-zeros) ·
`docs/NOTES-htp-config-keys.md` (which HTP backend-extension keys are real,
audited against the SDK's own `config.py` / `QnnHtpGraph.h`) ·
`docs/NOTES-coordination.md` (how concurrent sessions avoid re-deriving the same
artifact: recipe keys, the registry, claims, and the authority-document rule) ·
`docs/SA8797P_HTP_v81_Hardware_and_Deployment_Quantization_Reference_EN.md`
(device team's measured hardware truth, annotated).
