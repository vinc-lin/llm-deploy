# SA8797P LLM Bundle Build Guide

*Standalone guide: how to build, validate, bundle, and ship every Qwen3 model
variant for the Qualcomm SA8797P (Hexagon v81 HTP, Android GVM) — from a clean
Linux machine to a push-ready device bundle, entirely without device access.*

*Last validated end-to-end: 2026-08-11 (v2 bundles, device-confirmed correct
output). Assumes no prior knowledge of this project. ~30 min setup reading,
~1–2 h wall-clock per model build.*

---

## 1. What you are building

Each build produces a **flat device bundle** (`.tar.gz`, ~930 MB for 0.6B) that
runs offline on the SA8797P via Qualcomm Genie:

```
qwen3_06b_w8a16_local/
├── qwen3-0.6b-w8a16_ctx.bin      # the model: QNN context binary (~1.1 GB, 2-3 graphs)
├── genie_dialog.json             # Genie dialog config (points at the ctx-bin)
├── genie_dialog_basic.json       # (lade bundle only) same ctx-bin, basic decode
├── htp_backend_ext_config.json   # vtcm/PD/perf-profile config
├── tokenizer.json
├── genie-t2t-run                 # Genie CLI (aarch64-android)
└── 7 × .so                       # libGenie, libQnnHtp*, libQnnSystem (flat, no lib/)
```

The ctx-bin contains 2–3 **graphs sharing one weight set**:

| Graph | Shape | Role |
|---|---|---|
| `prefill` | AR=128, CL=128, no past-KV | prompt ingestion (+ first ~117 generated tokens, see §3.4) |
| `decode` | AR=1, CTX=1152, past-KV | steady-state generation |
| `verify32` | AR=32, CTX=1152, past-KV | (lade builds only) lookahead-decoding verification |

**Pipeline stages** (each has a script; §5 gives per-variant recipes):

```
HF checkpoint ─→ export wrapper (PyTorch) ─→ AIMET W8A16 quantsim + calibration
      ─→ ONNX export ─→ I/O rename ─→ qairt-converter (DLC per graph)
      ─→ qnn-context-binary-generator (weight-shared ctx-bin) ─→ bundle.sh
```

## 2. One-time machine setup

Reference machine: WSL2 (also fine: native Linux), 8 GB VRAM GPU (0.6B
quantizes on GPU; 1.7B needs `QUANT_DEVICE=cpu`), ~60 GB free disk on the
data volume.

### 2.1 Directory layout

Two roots, wired together by `scripts/env.sh` (edit it if your paths differ):

```
$LLMDEPLOY_ROOT   the git repo (scripts/, configs/, docs/)
$LLMDEPLOY_DATA   heavy data on fast ext4:
  ├── sdk/qairt/2.48.40.260702/   # QAIRT SDK — version MUST match device runtime
  ├── models/Qwen3-0.6B/          # HF checkpoints (config + safetensors + tokenizer)
  ├── models/Qwen3-1.7B/
  ├── envs/qwen3-deploy/          # uv venv, python 3.10
  ├── envs/qairt-py312/           # uv venv, python 3.12
  ├── work/{onnx,quant,dlc,ctxbin}/
  └── bundles/
```

### 2.2 SDK

QAIRT **2.48.40.260702** — the exact version matters: the device runs
libGenie 1.19.0 from this release, and the graph contract in §3 was extracted
from *this* SDK's shipped qualla source. Publicly downloadable from
`softwarecenter.qualcomm.com` (ranged GETs work; use `curl -L`).

### 2.3 Python environments (uv)

```bash
# build env — torch + AIMET + export
uv venv $LLMDEPLOY_DATA/envs/qwen3-deploy --python 3.10
uv pip install --python $LLMDEPLOY_DATA/envs/qwen3-deploy/bin/python \
    torch aimet-torch==2.36.0 transformers "onnx==1.19.0" \
    onnxruntime "numpy<2"

# converter env — the qairt-converter needs py3.12
uv venv $LLMDEPLOY_DATA/envs/qairt-py312 --python 3.12
uv pip install --python $LLMDEPLOY_DATA/envs/qairt-py312/bin/python \
    "onnx==1.19.0" setuptools numpy pyyaml packaging
```

Hard version pins (breaking if violated):
- `onnx == 1.19.0` in **both** envs — ≥1.20 removes `onnx.version`, breaking the converter, and breaks AIMET export.
- `numpy < 2` in the build env.
- `aimet-torch == 2.36.0` — three of its bugs are patched *inside our scripts* (LoRA attr shim, >2 GB protobuf ByteSize crash, unreliable cross-variant `load_encodings`); other versions may need different workarounds.

Validated-known-good: torch 2.13.0+cu130, transformers 5.14.1, onnxruntime 1.23.2, numpy 1.26.4.

### 2.4 Every session

```bash
source scripts/env.sh   # sets QAIRT_SDK, PY_DEPLOY, PY_QAIRT, PATH, LD_LIBRARY_PATH
```

No sudo needed anywhere: QNN tools need LLVM libc++, which env.sh points at a
locally-extracted copy (`$LLMDEPLOY_DATA/syslibs/`).

## 3. The contract you must not break

These rules come from the SDK's shipped qualla engine source (see
`docs/NOTES-genie-io.md` for line-level citations). Violating them produces
binaries that **load and run cleanly but emit garbage** — the failure is
silent.

### 3.1 Graph I/O names and shapes

| Tensor | Shape | Rule |
|---|---|---|
| `input_ids` | `[1, AR]` int32 | AR is *discovered* from numElements |
| `attention_mask` | `[1, AR, CTX]` | rank-3, additive float |
| `position_ids_cos` / `_sin` | `[1, AR, 64]` | rope-dim 64 for head_dim 128 |
| `past_key_{i}_in` | `[1, n_kv, 128, P]` | **keys transposed — sequence LAST** |
| `past_value_{i}_in` | `[1, n_kv, P, 128]` | |
| `past_key_{i}_out` | `[1, n_kv, 128, AR]` | **new-slice only**, not full cache |
| `past_value_{i}_out` | `[1, n_kv, AR, 128]` | |
| `logits` | `[1, AR, 151936]` | **ALL positions — see below** |

### 3.2 All-position logits (the 2026-08-11 lesson)

Every logit-producing graph MUST emit logits for **all AR positions**. Genie
left-aligns input tokens and samples logits **row `n_process − 1`**, with no
last-token-only mode and no load-time shape rejection. Our v1 bundles emitted
last-token-only prefill logits `[1,1,V]`: they loaded fine, ran fine, and
produced garbage from the first token (out-of-bounds logits read → zeros →
argmax = token 0 = `"!"`). The regression guard is
`scripts/validate/parity_qualla_read.py` — run it on every new export
(§6).

### 3.3 Cross-graph encodings identity

All graphs in one ctx-bin share weights, so all DLCs must convert against **the
same encodings file** (the prefill run's). Never recalibrate one graph of a
set. This is also why `--adopt-encodings` exists (§5.6): a graph-shape-only
rebuild reuses the old calibration bit-exactly and only reconverts one DLC.
(History: per-graph encodings mismatches are the class of failure behind the
remote team's "error 5005".)

### 3.4 Runtime facts that shape decisions

- `vtcm_mb: 16` and `pd_session: "unsigned"` are the device caps (24 MB VTCM
  is rejected on unsigned PD). `O: 3`, 4 HVX threads, perf profile
  `llm_decode_burst`.
- Cross-graph **weight sharing must be ON** in `configs/htp_config.json` —
  it's what makes a 3-graph bin cost ~1.1 GB instead of 3×.
- Genie drives our `(128,128)` no-past-KV prefill graph in "bertcache" mode:
  after the prompt it keeps generating through the prefill graph (~42 ms/tok,
  whole-window reprocess) until the KV passes 128 positions, then switches to
  the AR-1 decode graph (~155 ms/tok). Quote tok/s numbers per phase.
- Device-measured (v2, 2026-08-11): decode ~6.5 tok/s, prefill-phase
  ~23.8 tok/s, init ~0.8 s, RAM ~163 MB.

## 4. Quantization recipe (what the scripts implement)

W8A16 via AIMET 2.36 PTQ: per-channel symmetric INT8 weights, 16-bit
activations, `post_training_tf_enhanced`, calibration = 10 mixed
zh/en/code/math prompts (embedded in `quantize_aimet.py`). Quantizers
**disabled** (FP16) on: `embed_tokens` (HTP Gather-on-INT16 error 0xc26),
final norm, `lm_head`, and all K/V-projection outputs (cross-graph FP16
requirement). Weight encodings clipped to the ±0x7f7f-safe range (HTP
packed-pair saturation — quantsim does NOT model this, so "passes quantsim,
garbage on silicon" is its signature). Gate-Up fusion additionally keeps the
fused `gate_up` output FP16 (requantized at `down_proj`); QKV fusion grafts
the donor q_proj INT16 encoding onto the Q split (K/V splits FP16) — done by
`scripts/export/qkv_surgery.py` at the encodings level.

## 5. Build recipes — all models

All commands assume `source scripts/env.sh` first. Names below are the
canonical ones; artifacts land in `$LLMDEPLOY_DATA/work/…/<name>/`.

### 5.1 Baseline 0.6B (`qwen3-0.6b-w8a16`) — build this FIRST

```bash
./scripts/build/full_build.sh qwen3-0.6b-w8a16 128 1024
```

Runs quantize → decode export (adopting prefill encodings) → filter → rename →
2 DLC conversions → 2-graph ctx-bin. ~1 h. Every other 0.6B variant depends on
this build's artifacts (donor encodings, DLCs).

### 5.2 Gate-Up fused (`qwen3-0.6b-w8a16-fusegu`)

```bash
./scripts/build/full_build.sh qwen3-0.6b-w8a16-fusegu 128 1024 --fuse-gate-up
```

### 5.3 QKV fused (`-fuseqkv`) and QKV+GateUp (`-fuseqkvgu`)

Require the baseline (§5.1) as encodings donor:

```bash
./scripts/build/qkv_build.sh qwen3-0.6b-w8a16-fuseqkv \
    $LLMDEPLOY_DATA/work/quant/qwen3-0.6b-w8a16-prefill 128 1024
./scripts/build/qkv_build.sh qwen3-0.6b-w8a16-fuseqkvgu \
    $LLMDEPLOY_DATA/work/quant/qwen3-0.6b-w8a16-prefill 128 1024 --fuse-gate-up
```

Device verdict so far: fusion showed **no decode tok/s gain at vtcm 16**
(6.27–6.5 tok/s ≈ baseline). Keep building these only for A/B completeness or
if signed PD (vtcm 24) becomes available.

### 5.4 Lookahead decoding (`-lade`, 3-graph)

Adds an AR=32 verification graph to the baseline and packs a 3-graph ctx-bin.
Requires §5.1 complete:

```bash
./scripts/build/lade_build.sh qwen3-0.6b-w8a16 128 1024 32   # AR=16 variant: last arg 16
```

Bundle ships two dialog configs (lade + basic) for same-binary A/B. Config
guardrail: `(ngram−1)×(window+gcap)` must stay ≤ the verify graph's AR, or
Genie silently routes verification batches to the prefill graph. Shipped
config window 8 / ngram 3 / gcap 8 = exactly 32.
**Known open issue (2026-08-11): `type:"lade"` SIGSEGVs in libGenie on device
when the prompt is short; basic mode on the same ctx-bin works. Suspected
lhd-dec × bertcache-prefill interaction at `n_past < 128` — see
`reports/qwen3-0.6b-w8a16-v2-lade-vs-baseline-report.md`.**

### 5.5 Qwen3-1.7B baseline

Same pipeline, bigger model — quantize on CPU (8 GB VRAM is not enough):

```bash
MODEL=$LLMDEPLOY_DATA/models/Qwen3-1.7B QUANT_DEVICE=cpu \
    ./scripts/build/full_build.sh qwen3-1.7b-w8a16 128 1024
```

Expect ~3.9 GB ctx-bin and roughly ⅓ of 0.6B decode tok/s (DDR-bound scaling).

### 5.6 Graph-shape-only rebuilds (fast path)

If you change a graph's *shape* but not its quantization (the all-position
logits fix was exactly this), do NOT rerun the full build. Rebuild one DLC
with the variant's existing calibration:

```bash
./scripts/build/prefill_fix_rebuild.sh qwen3-0.6b-w8a16 128 1024
# fused variants add their flags; surgery variants need the donor:
SURGERY=1 DONOR=qwen3-0.6b-w8a16 ./scripts/build/prefill_fix_rebuild.sh \
    qwen3-0.6b-w8a16-fuseqkvgu 128 1024 --fuse-qkv --fuse-gate-up
```

Then regenerate the ctx-bin(s) (stage-7 command in `full_build.sh` /
stage-4 in `lade_build.sh`) and re-bundle. ~30 min per variant instead of ~1 h,
and scales stay bit-identical to the untouched DLCs.

## 6. Validation gates — run before shipping anything

Device-free gates, in order; each has a known-good reference value:

| Gate | Command | Pass looks like |
|---|---|---|
| Wrapper vs HF | `export_qwen3.py … --parity-check` | max\|Δlogits\| ~4e-05 |
| Standard parity | `scripts/validate/parity_onnx.py --onnx <dir> --cl-prefill 128 --ctx 1024` | prefill argmax match + 8-step greedy chain token-identical |
| **Device read pattern** | `scripts/validate/parity_qualla_read.py --onnx <dir> --cl-prefill 128` | 4/4 prompts OK (left-aligned, logits row n−1 — what the device actually samples) |
| Verify graph (lade) | `scripts/validate/parity_verify.py` | batched rows match HF, ~3e-05 |
| Quant quality | `quantize_aimet.py … --eval` | last-token argmax agreement ≥ 3/4 prompts |
| DLC shape | `qairt-dlc-info -i prefill.dlc \| grep logits` | `1,128,151936` — never `1,1,…` |
| Ctx-bin | `qnn-context-binary-utility --json_file info.json` | all graphs listed; logits dims per §3.1; ~1.1 GB for 0.6B |

## 7. Bundle and ship

```bash
./scripts/build/bundle.sh qwen3_06b_w8a16_local \
    $LLMDEPLOY_DATA/work/ctxbin/qwen3-0.6b-w8a16/qwen3-0.6b-w8a16_ctx.bin
# lade: pass configs/genie_dialog_qwen3_0.6b_lade.json as 3rd arg, then also
# copy a ctx-bins-patched basic config in as genie_dialog_basic.json and re-tar
```

`bundle.sh` assembles the flat layout, patches the dialog JSON's `ctx-bins` to
the real filename, and tars. Expected: ~930 MB tarball for 0.6B.

Upload to HF (`vinccniv/sa8797p-qwen3-w8a16-bundles`, staging dir
`$LLMDEPLOY_DATA/hf-staging/` with hardlinks):

```bash
scripts/util/hf_upload_watchdog.sh vinccniv/sa8797p-qwen3-w8a16-bundles \
    $LLMDEPLOY_DATA/hf-staging
```

The watchdog exists because the local proxy drops long-lived uploads (client
hangs on CLOSE-WAIT sockets looking alive). **Also beware the hub's 128
commits/hour limit**: restart storms exhaust it and the commit phase then
"hangs" with all bytes already uploaded. Diagnose with one foreground
`HfApi().upload_file` (a 429 surfaces in seconds); recover by waiting ~1 h
then committing one file at a time.

Device smoke test:

```bash
adb push qwen3_06b_w8a16_local.tar.gz /data/local/tmp/
adb shell 'cd /data/local/tmp && tar xzf qwen3_06b_w8a16_local.tar.gz'
adb shell 'cd /data/local/tmp/qwen3_06b_w8a16_local && LD_LIBRARY_PATH=. \
    ./genie-t2t-run -c genie_dialog.json -p "What is 2+2? Answer with one number."'
```

Expected: `2+2=4.` (a repetition loop after the answer is normal with raw
prompts — apply the Qwen3 chat template for clean stops). Always smoke-test
the **baseline** bundle first so failures isolate to a variant, not the
environment.

## 8. Troubleshooting

| Symptom | Cause / fix |
|---|---|
| Output is fluent garbage from token 1, no errors anywhere | Prefill logits are last-token-only, or an encodings-scale fault. Check `qairt-dlc-info` logits shape first (§6); run `parity_qualla_read.py`. |
| Converter floods `Operation Equal … version 19` warnings | Benign; ignore. |
| AIMET export crashes: `QuantizedLora` / protobuf `ByteSize` | Handled by shims inside `quantize_aimet.py`; if you see them you're calling AIMET outside our scripts. |
| Passes quantsim locally, garbage on HTP only | ±0x7f7f weight-clip class of bug (§4) — quantsim doesn't model HTP packed-pair arithmetic. |
| `vtcm_mb: 24` rejected | Unsigned PD cap; use 16 (signed PD would lift it — device-side provisioning). |
| Ctx-bin ≈ sum of DLC sizes (e.g. 2.1 GB) | Weight sharing off — check `configs/htp_config.json`. |
| 1.7B quantize OOMs GPU | `QUANT_DEVICE=cpu`. |
| lade mode SIGSEGV in libGenie, basic mode fine | Known open issue (§5.4). |
| HF upload progress frozen, sockets CLOSE-WAIT | Proxy drop — the watchdog handles it. If blobs are uploaded but commits never land: 429 rate limit (§7). |
| WSL2 C: drive filling up | Build scripts call `disk_guard` (abort < 6 GB); compact the VHD from Windows if hit. |

## 9. Map of the repo

| Path | Role |
|---|---|
| `scripts/env.sh` | environment — source in every shell (never commit secrets here) |
| `scripts/export/export_qwen3.py` | FP32 export wrapper (prefill/decode/verify graphs) |
| `scripts/export/modeling_export.py` | the wrapper model itself (Genie-shaped I/O) |
| `scripts/export/qkv_surgery.py` | QKV-fusion encodings graft |
| `scripts/quant/quantize_aimet.py` | AIMET quantsim, calibration, export (+ `--adopt-encodings`, `--export-decode`, `--decode-ar`) |
| `scripts/quant/filter_aimet_w8a16.py`, `rename_aimet_io.py` | encodings filter; canonical I/O rename |
| `scripts/build/full_build.sh` | baseline / gate-up end-to-end build |
| `scripts/build/qkv_build.sh` | QKV-fused build (+ surgery) |
| `scripts/build/lade_build.sh` | verify graph + 3-graph ctx-bin |
| `scripts/build/prefill_fix_rebuild.sh` | fast graph-shape-only rebuild |
| `scripts/build/bundle.sh` | device bundle assembly |
| `scripts/validate/parity_onnx.py`, `parity_qualla_read.py`, `parity_verify.py` | validation gates (§6) |
| `scripts/util/hf_upload_watchdog.sh` | supervised HF upload |
| `docs/NOTES-genie-io.md` | the Genie contract with SDK source citations — read before touching graph I/O |
| `docs/LOCAL_ENV.md` | environment provenance + progress log |
| `reports/` | device test reports (v1 failure analysis, v2 validation) |
| `SA8797P_Deployment_Status_Summary.md` | project-level status (authoritative) |
