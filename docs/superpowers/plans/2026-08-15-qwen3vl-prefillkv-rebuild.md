# Qwen3-VL-4B Past-KV Prefill Rebuild + Single-Attempt Device Kit — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development
> (recommended) or superpowers:executing-plans to implement this plan task-by-task.
> Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Rebuild the Qwen3-VL-4B text tower with a past-KV prefill so the
bundle loads and runs on the SA8797P **on the first retry**, ship a weather
test-image kit with per-image expected captions, and update HF + docs.

**Architecture:** The 2026-08-14 device attempt failed at load:
`ShapeError: attention_mask Expected [1,128,2176] Found [1,128,128]` →
`Failed to create the Genie Node (-1)` + SIGSEGV. Mechanism (recorded,
`docs/REFERENCE.md` §3.6): in a **split** tower shard 0's prefill emits no
`logits` → classifies `DECODER_PREFILL` (`nsp-graph.cpp:247-249`) → its
expected CL is rewritten to the cache-group max (`nsp-model.cpp:604-605`) →
the `[1,AR,AR]` mask fails `validateModel` (`:858`). The fix is the
**root-cause** one: re-export prefill with past-KV (`mask [1,128,2176]`,
`past 2048`) on the same encodings lineage — the recipe already device-proven
by the 0.6B ladekv build — plus a static gate that replays Genie's load-time
validation so this class of failure can never ship again. The untested
`execute-select-graphs` workaround ships only as a **fallback config** in the
bundle. Also folded in: per-graph deepstack tensor names (the sub-128-token
uninitialized-read hazard becomes LIVE once prefill loads), and the weather kit.

**Tech Stack:** tank (44 cores / 125 GB RAM / 461 GB free) for AIMET export +
conversion; local WSL for ctx-bin verification, gates that need the ViT ONNX,
bundling, upload. `qwen3-deploy` env, QAIRT 2.48.40 converter, HF via proxy
watchdog.

---

## Ground truth this plan is built on (verified, not assumed)

### The target shapes (device-validated pattern)

The 0.6B ladekv ctx-bin (device-proven at 10.8 tok/s) read back from
`work/ctxbin/qwen3-0.6b-w8a16-fuseqkvgu-ladekv/info.json`, generalized from
CTX=1024 to CTX=2048 with `PAST = CTX + CL − AR = 2048`, `TOTAL = CTX + CL =
2176`:

| Tensor | prefill (NEW) | decode (unchanged) |
|---|---|---|
| `inputs_embeds` | `[1,1,128,2560]` | `[1,1,1,2560]` |
| `attention_mask` | **`[1,128,2176]`** | `[1,1,2176]` |
| `position_ids_cos/sin` | `[1,128,64]` | `[1,1,64]` |
| `past_key_N_in` | **`[1,8,128,2048]`** | `[1,8,128,2175]` |
| `past_value_N_in` | **`[1,8,2048,128]`** | `[1,8,2175,128]` |
| `past_key_N_out` | `[1,8,128,128]` (new slice) | `[1,8,128,1]` |
| `logits` (shard 1 only) | `[1,128,151936]` | `[1,1,151936]` |

`[1,128,2176]` is byte-for-byte what the device validator demanded.

### The recipe (proven by `ladekv_build.sh`)

`quantize_aimet.py --export-decode <prefill-quant-dir> --decode-ar 128
--cl-prefill 128 --ctx 2048` builds the `use_past=True` wrapper at AR=128 on
the **adopted** encodings lineage (`ladekv_build.sh:70-74`). The VL path of
`quantize_aimet.py` (embeds input + deepstack + `from_hf_vl_text`) is the same
code that produced the shipped decode. Known lineage caveat: `--export-decode`
**recomputes scales** at ~9.1e-8 relative (REFERENCE §4) — the weight-unification
pass (`unify_pair_weights.py`) is therefore mandatory, exactly as it was for
decode.

### Tank inventory (checked 2026-08-15)

`~/llm-local/work/quant/`: `qwen3vl-4b-w8a16-prefill` (lineage source,
`model_torch.encodings` needed by `--export-decode`), `-decode`,
`-decode-unified`, `-split-enc`, `qwen3vl-4b-calib-ar128.npz`.
`~/llm-local/work/dlc/`: `qwen3vl-4b-w8a16`, `-split`.
`~/llm-local/work/onnx/`: `qwen3vl-4b-aimet-split`, `-text-split`, `-vit`.
461 GB free. The local split ctx-bin work dir was reclaimed; ctx-bins are
regenerated in Phase 2 regardless.

### Facts that shape the design

1. **Deepstack hazard goes LIVE with this rebuild.** Today the mis-sized
   deepstack memset (`nsp-model.cpp:1481` keeps the LAST variant's spec → 5120
   bytes of a 128-row allocation) is harmless only because prefill never
   loads. With a loading prefill, any prompt ≤128 tokens selects it and reads
   uninitialized rpcmem beyond the first row (`NOTES-genie-pipeline.md` §A).
   Fix folded in: **rename prefill's three deepstack inputs**
   (`deepstack_visual_embed_{k}_p`) → distinct names → separate allocations →
   each memset at its own full size. Safe because those inputs are FLOAT_16
   (no encodings to orphan) and unknown names are explicitly zeroed
   (`initializeUnconnectedInputs`, probe A).
2. **The fallback keys are real but carry one unverified risk.**
   `execute-select-graphs`/`load-select-graphs` exist (`qnn-htp.cpp:80-81`,
   filter at `nsp-model.cpp:314-318`, `QNN_CONTEXT_CONFIG_ENABLE_GRAPHS` at
   `QnnApi.cpp:118-121`) — but the same name list goes to BOTH contexts, and
   whether HTP tolerates an enable-graphs name absent from a given binary is
   sealed inside `libQnnHtp`. Hence: fallback, never primary.
3. **GLM-4v precedent.** Qualcomm's own split pipeline example ships AR=32 +
   AR=128 past-KV graphs across a 2-split weight-shared ctx-bin
   (`configs/glm-4v/glm-4v.json`) — a split multi-AR past-KV tower is a
   supported, shipped configuration, not an experiment.

---

## Phase 0 — Space + prerequisites (local, ~15 min)

### Task 0.1: Reclaim C: (user-approved deletions)

C: is at ~97% / ~32 GB free. The rebuild needs ctx-bin scratch (~9 GB), a new
bundle (~6.3 GB), and upload staging (hard-linked, ~0).

- [ ] Delete stale HF staging tarballs (all already uploaded to HF long ago):
  `rm -f $LLMDEPLOY_DATA/hf-staging/qwen3_*.tar.gz` (~8.5 GB)
- [ ] Delete the HF hub download cache: `rm -rf ~/.cache/huggingface/hub` (~4.3 GB)
- [ ] Delete superseded local bundle tarballs if present:
  `ls -lh $LLMDEPLOY_DATA/bundles/*.tar.gz` — remove any regenerable by a
  `*_bundle.sh` script (keep none that aren't).
- [ ] Verify: `df -h /mnt/c` ≥ 45 GB free, and `disk_guard 20` passes.
- [ ] STOP if < 45 GB — ask the user before touching anything else.

### Task 0.2: Branch + docs delta check

Another session already recorded the incident (`CLAUDE.md` contract update,
`REFERENCE.md` §3.6, a `NOTES-genie-io.md` section). Do not duplicate.

- [ ] `git checkout main && git pull && git checkout -b qwen3vl-prefillkv-rebuild`
- [ ] Read `docs/NOTES-genie-io.md` § "Split prefill is fatal at load" and
  `docs/REFERENCE.md` §3.6; list what is still missing (expected: probe C's
  "silently skipped, no crash" paragraph in `NOTES-genie-pipeline.md` §C needs
  a correction note distinguishing unsplit vs split; DEVICE_TEST triage lacks
  the ShapeError row). Fix ONLY the gaps. Commit.

---

## Phase 1 — Probes + the load-simulation gate (local, read-only + 1 script)

Two questions must be answered from source BEFORE burning tank hours, plus the
gate that would have caught the incident pre-ship.

### Task 1.1: Probe — will the strategy loop actually SELECT the rebuilt prefill?

The whole point is a ~3-4 s TTFT. Shard-0's new prefill still has no `logits`,
so it still classifies `DECODER_PREFILL`. Probe C found an exclusion at
`kvmanager.cpp:392-394` (`output_all && choice.type == DECODER_PREFILL` →
skip). If that fires for our variant, the rebuild loads but silently runs
all-decode again — shipping a 6 GB rebuild for zero TTFT gain.

- [ ] Read `kvmanager.cpp:365-465` (`prepareInferenceStrategy`) and the variant
  registration path (`nsp-model.cpp:979-986`): which shard's `GraphType` lands
  in `m_supported_variants` for a (128, 2176) variant whose shard-0 graph is
  `DECODER_PREFILL` and shard-1 graph (has logits) is `DEFAULT`? What sets
  `output_all` for a basic pipeline TextGenerator (grep `output_all` /
  `all_logits` across qualla)?
- [ ] Cross-check against GLM-4v: its shard-0 AR=32/128 graphs also lack
  logits; conclude how the SDK's own example avoids (or hits) the exclusion.
- [ ] Record verdict + citations in `docs/NOTES-genie-io.md`. Decision matrix:
  - **Selected** → proceed as planned.
  - **Excluded** → prefill must emit *something* that reclassifies it, or the
    plan's TTFT claim dies. STOP and surface to the user with options before
    Phase 2 (candidates: export shard-1-style all-position logits pass-through
    is NOT possible in shard 0 — realistic options are accepting all-decode
    with select-graphs, or restructuring). Do not guess.

### Task 1.2: Probe — exact `validateModel` expectations for every named tensor

The load-sim gate must replicate the validator, not approximate it.

- [ ] Read `nsp-model.cpp:620-964` (`validateModel`) + `checkShape` (`:477`) +
  the CL rewrite (`:604-605`) + classification (`nsp-graph.cpp:194-262`) and
  write down, per named tensor (`inputs_embeds`, `attention_mask`,
  `position_ids_*`, `past_key/value_*_in/out`, `logits`), the exact expected
  dims as a function of (AR, CL, ctx_size-from-config, n_embd, kv-dim,
  classification). Include the cache-group-max rewrite rule for
  `DECODER_PREFILL` graphs.
- [ ] Record as a table in `docs/NOTES-genie-io.md` with line citations.

### Task 1.3: Gate — `validate_genie_load()` in `lint_pipeline_bundle.py` (check 9)

- [ ] **Write the check** (new function in `scripts/validate/lint_pipeline_bundle.py`):
  for each text-generator ctx-bin's `info.json` + the node config's
  `context.size`, replay Task 1.2's rules: classify each graph
  (has logits? has past-KV? → GraphType), derive its expected
  `attention_mask` / KV shapes including the DECODER_PREFILL CL rewrite, and
  compare against the actual tensor dims. Emit the same vocabulary as the
  device (`Expected [...] Found [...]`) so failures are directly comparable.
- [ ] **Mutation test — the incident must reproduce.** Run the check against
  the OLD shipped text info.jsons (fetch
  `qwen3vl_4b_e2e_pipeline/qwen3vl-4b-w8a16_{1,2}_of_2.info.json` from HF, or
  regenerate from the old bundle dir): it MUST fail with
  `Expected [1, 128, 2176] Found [1, 128, 128]` on both shards. A load-sim
  that cannot reproduce the real incident is vacuous — do not proceed until it
  does.
- [ ] **Control:** the check must PASS on a synthetic info.json with the
  Ground-truth table's shapes.
- [ ] Commit.

---

## Phase 2 — Rebuild the text tower (tank-primary, ~half day)

Mirror of the Stage 2 flow with prefill-kv in place of bertcache prefill.
Work on tank via ssh; artifacts land in tank's `~/llm-local/work/`; only the
final ctx-bins rsync back.

### Task 2.1: Past-KV prefill export (tank)

- [ ] On tank: `disk_guard`-equivalent check (461 GB free — fine), then:
  ```
  $PY scripts/quant/quantize_aimet.py --model <Qwen3-VL-4B-Instruct> \
      --cl-prefill 128 --ctx 2048 --decode-ar 128 \
      --export-decode ~/llm-local/work/quant/qwen3vl-4b-w8a16-prefill \
      --out ~/llm-local/work/quant/qwen3vl-4b-w8a16-prefillkv128 \
      --device cpu <VL flags exactly as the Stage 2 decode export used them>
  ```
  First recover the exact Stage 2 decode-export invocation from
  `vl_text_build.sh` / tank shell history / the decode quant dir's args dump —
  the VL path has flags (calib npz, embeds input) that MUST match.
- [ ] Verify the export: `past_key_0_in [1,8,128,2048]`,
  `attention_mask [1,128,2176]`, `logits [1,128,151936]`, deepstack inputs
  present, 36 layers.
- [ ] Expected wall: ~1-2 h (63.5 GB peak RAM fits tank's 125 GB).

### Task 2.2: Deepstack rename (prefill only)

- [ ] Small script `scripts/quant/rename_deepstack_inputs.py`: load the
  prefillkv `model.onnx`, rename graph inputs
  `deepstack_visual_embed_{0,1,2}` → `deepstack_visual_embed_{0,1,2}_p`
  (inputs + all node refs). Assert: 3 renamed, dtype float, no encodings
  entries reference the old names in the prefill encodings (they are float
  inputs — expected zero hits; if a hit exists, STOP: the input became
  quantized and the rename would orphan its encoding).
- [ ] Verify with onnx checker + input listing.

### Task 2.3: Weight unification

- [ ] `unify_pair_weights.py` — copy the LINEAGE weights into prefillkv
  exactly as was done for decode (decode-unified is the reference for what
  "unified" means here). Then the verification the first pass taught us:
  sample **differing** tensors pre-unification, confirm zero differing
  initializers post-unification vs `qwen3vl-4b-w8a16-decode-unified`.
- [ ] Gate: byte-compare a hash of every initializer between prefillkv-unified
  and decode-unified — must be identical (this is what makes ctx-bin weight
  sharing dedup work).

### Task 2.4: Split, encodings, convert (tank)

- [ ] `split_aimet_onnx.py` at layer 18 on prefillkv-unified →
  `prefill_0` / `prefill_1` chunks (per-graph external-data subdirs — the
  filename-collision lesson).
- [ ] `split_encodings.py` on the prefill lineage encodings (both `/layers.N/`
  spellings; `--renumber` OFF), same as Stage 2.
- [ ] Convert 4 DLCs — names are graph names, convert STRAIGHT to final
  filenames: `prefill_0.dlc`, `prefill_1.dlc` (new); `decode_0.dlc`,
  `decode_1.dlc` (re-convert from the surviving split ONNX/encodings on tank
  unless the existing split DLCs are verifiably the shipped lineage — check
  `~/llm-local/work/dlc/qwen3vl-4b-w8a16-split` first and prefer reuse).
  Converter dims for prefill: `-d attention_mask "1,128,2176"`,
  `-d past_key_N_in "1,8,128,2048"`, `-d past_value_N_in "1,8,2048,128"`,
  deepstack `_p` names.
- [ ] `qairt-dlc-info` on each: assert prefill logits `[1,128,151936]` in
  shard 1 only, past-KV present in all four, mask trailing dim 2176.

### Task 2.5: ctx-bins + readback gates (local or tank, then rsync)

- [ ] `vl_text_ctxbin_split.sh` (or its prefillkv variant): 2 ctx-bins,
  4 graphs, weight-sharing ON. Keep the `MIN_SHARED_GB` floors — with
  identical weights the shared sizes must come out ≥ {1.4, 2.0} GB as before;
  a collapse below floor = lineage divergence, fail the build.
- [ ] `qnn-context-binary-utility --json_file` both bins. Assertions:
  graphs `{prefill_0, decode_0}` / `{prefill_1, decode_1}`; every shape in
  the Ground-truth table; deepstack `_p` names present in `prefill_0` only;
  HTP config bound (O=3, vtcm 16, 4 HVX) on all four.
- [ ] **Run the Phase 1 load-sim gate against the new info.jsons — must PASS.**
- [ ] rsync the two ctx-bins + info.jsons to local `$LLMDEPLOY_DATA/work/ctxbin/qwen3vl-4b-w8a16-splitkv/`.
- [ ] 3.5 GiB per-graph serialization limit note: weights unchanged (~4.3 GB
  total, 1.7/2.4 per shard) — margins identical to Stage 2; if the generator
  errors, STOP (something else changed).

---

## Phase 3 — Numerical gates (tank preferred for RAM; ~2 h wall)

### Task 3.1: Extend `parity_e2e_vl.py` with the device-primary chain

- [ ] Add `chain0b-prefillkv`: the rebuilt device feed —
  prefill chunk 0 (rows 0-127, empty past, mask exposing past=0), prefill
  chunk 1 (rows 128-255, past=128), then 17 decode steps for rows 256-272,
  then free-running generation. Mask/position/KV threading pattern:
  `parity_ladekv_read.py` is the proven reference (chunked past-KV feed);
  positions from `mrope_tables` as in the existing chains; deepstack fed via
  the RENAMED `_p` inputs for prefill calls, original names for decode.
  **Bar: token-for-token identical to `hf.generate`** (GATED_EXACT), same as
  chain0/chain1.
- [ ] Keep `chain0-alldecode` gated — it is now the FALLBACK path's numerics.
- [ ] Point `--text-onnx` at the new prefillkv ONNX pair (fp32 ORT-loadable
  exports; if the AIMET model_renamed.onnx pair is ORT-friendly use it,
  else export a plain past-KV pair via `export_qwen3vl_text.py` with a
  `--prefill-past 2048` flag added — `io_spec`/`export_graph` already take
  `past_len`, this is a ~10-line arg change).
- [ ] Run the full gate (all chains, no `--chains` filter). Expect chain0b,
  chain0, chain1, chain2 exact; tierA text reported.
- [ ] Also re-run `parity_vl_text_split.py` equivalent on the new pair:
  split-vs-unsplit bit-identity (logits + all KV outs at 0.0).

### Task 3.2: Mutation checks specific to the rebuild

- [ ] In chain0b, corrupt the chunk-1 past (zero it) → generated tokens must
  diverge (proves the chunked past actually carries chunk 0).
- [ ] Feed chunk 1 with chunk-0's positions (no offset) → must diverge
  (proves position continuation).

---

## Phase 4 — Weather test kit (local, ~1.5 h)

User spec: vehicle exterior / road / surrounding environment / weather
scenarios; general-purpose prompt (`"Describe this image in one sentence."`,
unchanged — segment files already encode it).

### Task 4.1: Source candidates

- [ ] `scripts/pipeline/fetch_test_images.py`: download 10-14 candidates via
  the proxy. Primary source: COCO val2017 by direct image id (reachable during
  ViT calibration); curated id list of street/vehicle/weather scenes +
  Wikimedia Commons direct-file URLs as backup (rainy street, foggy road,
  snowy road with vehicles, wet asphalt, overcast highway, sunny road).
  Every download: verify JPEG magic + decode with PIL; discard failures.
- [ ] Time-box networking to 20 min; if fewer than 6 usable images arrive,
  STOP and report (do NOT fall back to synthetic — the user asked for real
  scenes).

### Task 4.2: Select by what the model actually sees

- [ ] Run each candidate through HF `Qwen3VLForConditionalGeneration.generate`
  (host, fp32, the same 512×512 preprocess) with the general prompt. Select
  5-6 finals whose captions (a) are correct on inspection of the caption
  text, and (b) jointly cover: rain, fog/overcast, snow, clear/sunny, road
  with vehicles. The caption IS the content check — no manual image viewing
  required.
- [ ] Record the HF caption per selected image (this is the fp32-with-deepstack
  reference).

### Task 4.3: Device-ready kit

- [ ] Per selected image `wx_{scene}.{jpg,raw,json}`: preprocess with
  `preprocess_image.py --encodings <NEW ctx-bin info.json>`; the 8-LSB clip
  gate applies (a real photo failing it = out-of-calibration finding, report
  it).
- [ ] Per-image pipeline script `wx_{scene}.script` — copy of the main script
  with only the `node set image` path changed (segments identical).
- [ ] **Expected captions**: run each image through the extended E2E gate's
  device-faithful chain (chain0b feed + ZERO deepstack = exactly what the
  device runs) via a small runner reusing the gate's machinery
  (`--kit-mode <img.raw>` flag or a sibling script importing it). Record per
  image: expected caption (device-faithful) + HF reference caption.
- [ ] `test_images/README.md`: table of image → scene → expected device
  caption → HF reference caption, with the wording-drift caveat (W8A16 on
  device; the bar is semantic agreement on weather + scene content).
- [ ] Kit cost note: ~8-10 min/image for expected captions (273-token chain on
  CPU) — run on tank in parallel if wall-clock matters.

---

## Phase 5 — Bundle v2 (local, ~30 min)

### Task 5.1: Update `vl_pipeline_bundle.sh`

- [ ] Point at the new splitkv ctx-bins; add `test_images/` (blobs + scripts +
  README, ~30 MB); add the fallback config
  `genie_text_generator_qwen3vl_4b_decodeonly.json` (copy of the main config +
  `"execute-select-graphs": ["decode_0","decode_1"], "load-select-graphs": true`
  in the QnnHtp block) and `genie_pipeline_qwen3vl_decodeonly.script`
  (identical script, textGenerator node config swapped). Regenerate info.json
  sidecars from the final binaries as before.
- [ ] Lint extensions in `lint_pipeline_bundle.py`: (a) check 9 load-sim runs
  on BOTH configs (primary must pass all four graphs; decodeonly must pass
  for the decode graphs it loads); (b) kit closure — every `wx_*.script`
  references files present in the bundle; every `wx_*.raw` is 3,145,728 bytes
  with sidecar encoding == ViT ctx-bin's; (c) fallback config parses and
  differs from primary ONLY by the two keys.
- [ ] Build; lint PASS; mutation-test the new checks against a copy (drop a
  kit blob → (b) fires; revert mask dims in a copied info.json → (a) fires
  with the device's exact vocabulary).

### Task 5.2: Docs for the single attempt

- [ ] `DEVICE_TEST.md` rewrite: primary path (expected TTFT ~3-4 s, chunked
  prefill 128+128+17) → smoke with `sample_image` → weather kit loop → only
  if load fails, flip to `_decodeonly` config (one file swap, expected TTFT
  ~30 s) and note the cross-context enable-graphs caveat + what its failure
  looks like. Triage table gains the ShapeError row (now "must not happen —
  load-sim gated") and the fallback-specific rows.
- [ ] Bundle `README.md`: update limitations — TTFT fixed, sub-128 hazard
  CLOSED (renamed tensors), fallback documented; validation table gains
  chain0b and the load-sim gate.

---

## Phase 6 — Ship (local, ~1 h)

- [ ] Stage with hard links; include: e2e bundle v2 (new ctx-bins are ~4.5 GB
  of genuinely new blobs — the upload is NOT dedup-cheap this time),
  `qwen3vl_4b_text_w8a16/genie_dialog.json` fix (already on HF — verify,
  re-push only if drifted), updated root README (TTFT + kit mention).
- [ ] `SOCKET_CHECKS=999999` watchdog upload; expect ~4.6 GB real transfer.
- [ ] Verify: file listing, `get_paths_info` sizes on the two new ctx-bins,
  re-download one info.json and re-run the load-sim gate against it
  (proves the uploaded bytes, not the local ones).
- [ ] **Read and REPORT visibility; never change it** (4 incidents on record).
- [ ] Merge branch → main, pre-push scan (no secrets / no >200 KB / no
  binaries), push. Update `NOTES-genie-io.md` select-graphs section status if
  Phase 1 probes refined it.
- [ ] Final report to user: what shipped, gate numbers, per-image expected
  captions, the exact device commands, and the fallback procedure.

---

## Risk register (single-attempt honesty)

| Risk | Mitigation | Residual |
|---|---|---|
| Strategy loop excludes DECODER_PREFILL variant → TTFT stays ~30 s | Task 1.1 probe BEFORE building; GLM-4v precedent suggests it selects | If probe is ambiguous: still ship (correctness unaffected), flag TTFT |
| `--export-decode` scale drift breaks weight sharing | Task 2.3 unification + byte-hash gate + `MIN_SHARED_GB` floors | ~0 |
| New prefill introduces a numerical bug | chain0b token-exact vs HF; split bit-identity; two rebuild-specific mutations | ~0 device-free |
| Load-time validation surprises beyond shapes | Load-sim gate replays validator rules incl. classification + CL rewrite; mutation reproduces the real incident | Unknown unknowns in libGenie |
| Fallback config itself fails (cross-context enable-graphs) | Fallback is secondary; failure mode + signature documented in DEVICE_TEST | Accepted (untested by definition) |
| HTP scheduling / memory budget (~6.3 GB + KV) | Not testable off-device | Accepted, documented |
| Weather images out of ViT calibration range | 8-LSB clip gate per image; recalibration path documented | Low |
| C: exhaustion mid-build (SIGBUS, 3 prior VM crashes) | Task 0.1 reclaim + `disk_guard` sized per step; heavy steps on tank | Low |

## STOP conditions

- Task 1.1 verdict = **Excluded** → user decision before Phase 2.
- Load-sim mutation cannot reproduce the device ShapeError → fix the gate,
  never proceed with a vacuous gate.
- chain0b not token-exact → do not bundle; diagnose (same rule as Stage 3).
- < 6 usable weather images → report, ask.
- Any upload flips repo visibility → report and stop.

## Estimates

Tank: export ~1-2 h + convert ~1 h + ctx-bins ~30 min. Gates ~2 h. Kit ~1.5 h.
Bundle + upload ~1.5 h. Wall-clock with overlap: **roughly one working day**.

## Self-review (done at planning time)

- Requirements coverage: single-attempt maximization (root-cause fix + load-sim
  gate + fallback + per-image expected outputs) ✓; e2e incl. image analysis
  (chain0b + kit captions) ✓; weather/vehicle/road/exterior images (Task 4.1
  scene list) ✓; general prompt (unchanged segments) ✓; deletions approved and
  scoped (Task 0.1) ✓; tank-primary (Phases 2-3) ✓.
- Consistency: `_p` rename appears in Task 2.2 (export), 2.4 (convert dims),
  2.5 (readback), 3.1 (gate feeds), 5.1 (load-sim on final bins).
- The plan never relies on the untested select-graphs keys for the primary
  path, and never claims device certainty anywhere.
