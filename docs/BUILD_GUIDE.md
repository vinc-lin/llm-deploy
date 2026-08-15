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

The ctx-bin contains 2–3 **graphs sharing one weight set**. Two topologies
exist — pick by whether you need lookahead decoding:

**A. Bertcache topology** (baseline + fused variants):

| Graph | Shape | Role |
|---|---|---|
| `prefill` | AR=128, CL=128, **no** past-KV | prompt ingestion (+ first ~117 generated tokens, see §3.4) |
| `decode` | AR=1, CTX=1152, past-KV | steady-state generation |

**B. All-past-KV topology** (`-ladekv`, §5.4 — the reference build for anything
using `dialog.type: "lade"`):

| Graph | Shape | Role |
|---|---|---|
| `prefill` | AR=128, CL=1152, past=1024, past-KV | prompts 33–128 tokens, and the chunking unit for >128 |
| `decode` | AR=1, CTX=1152, past-KV | steady-state generation |
| `verify32` | AR=32, CTX=1152, past-KV | lade verification batches; also serves prompts ≤32 tokens |

Topology B is what makes lade work at all (§3.5) and additionally enables
prompts >128 tokens. It gives up the bertcache early-token burst (§3.4).

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
(History: per-graph encodings mismatches were blamed for the remote team's
"error 5005"; their HTP doc §9 instead ties 5005 to NOT_SUPPORTED triggers —
`vtcm_mb > 16`, multi-core — so treat that attribution as unconfirmed. The
mismatch itself is still a hard load failure.)

### 3.4 Runtime facts that shape decisions

- `vtcm_mb: 16` and `pd_session: "unsigned"` are the device caps (24 MB VTCM
  is rejected on unsigned PD). `O: 3`, 4 HVX threads, perf profile
  `llm_decode_burst`. Two unrun A/Bs on these numbers: the runtime always
  reports **8** HVX threads in use regardless of `hvx_threads: 4`
  (REFERENCE §8.9), and the device team's verified build sets
  `soc_id`/`soc_model: 72` explicitly where ours leaves them unset
  (REFERENCE §8.4).
- Cross-graph **weight sharing must be ON** in `configs/htp_config.json` —
  it's what makes a 3-graph bin cost ~1.1 GB instead of 3×.
- Genie drives our `(128,128)` no-past-KV prefill graph in "bertcache" mode:
  after the prompt it keeps generating through the prefill graph (~42 ms/tok,
  whole-window reprocess) until the KV passes 128 positions, then switches to
  the AR-1 decode graph (~155 ms/tok). Quote tok/s numbers per phase.
- Device-measured (v2, 2026-08-11): decode ~6.5 tok/s, prefill-phase
  ~23.8 tok/s, init ~0.8 s, RAM ~163 MB. (A later run measured
  `qwen3_06b_w8a16_local` at **11.72 tok/s** AR-1 — *faster* than the device
  team's 7.79, inverting the old "their builds are ~20% faster" premise. See
  REFERENCE §8.8, now marked resolved-backwards.)
- **Graph selection is numeric best-fit on (AR, CL) — names are cosmetic *to
  Genie*.** Genie picks the smallest CL ≥ current KV, then the smallest AR ≥ the
  batch size (largest smaller AR = chunking fallback). Consequences: two graphs
  may never share an (AR, CL) pair (hard load error), and in topology B a
  12-token prompt runs on `verify32`, not on the graph named `prefill`.
- **Graph names are NOT cosmetic to the HTP backend.** `htp_backend_ext_config.json`
  scopes its tuning block by `graph_names`; a graph whose name is absent silently
  gets backend defaults (4 MB VTCM, 24 MB spill) — no warning, exit 0. The name is
  baked in at conversion time from the `--output_path` **basename, dots included**
  (converting to `decode.dlc.new` yields graph `decode_dlc`), and renaming the file
  afterwards does not change it. Always convert straight to the final filename and
  verify against the ctx-bin before bundling (§6). See
  `docs/NOTES-vit-htp-config.md` for the measured cost: 345× the DDR spill traffic
  from a build whose log is clean.

### 3.5 Past-KV prefill contract (topology B)

Everything here is pinned to SDK source in `docs/NOTES-genie-io.md`; it is what
`parity_ladekv_read.py` reproduces. If you build a past-KV prefill for another
model, these are the rules:

- Tokens are **left-aligned** in the AR window; the remainder is filled with the
  pad token, which defaults to the **first `eos-token`** entry (151645 for us).
- Attention mask is FP16 additive: allow `+0.0`, masked **`-1000.0`** (not
  `-inf`). Our encodings calibrate the mask at −100, so the device clips
  −1000 → −100 — still e^−100 ≈ 0, harmless.
- Concat KV layout: mask columns `[0, past)` are the past region,
  `[past, CL)` the new tokens. Chunk row *i* allows past `[0, n_valid_kv)`
  plus new `past … past+i`.
- RoPE positions `iota(n_past + i)` on valid rows, 0 on pad rows.
- Logits sampled at row `n_process − 1`; all-position logits still mandatory.
- Prompt chunking works AR tokens at a time with growing `n_past`, up to
  accumulated KV = `past_dim = CL − AR` (1024 for us). Beyond that Genie
  throws `ContextLimitException`.

**Why topology B fixes the lade SIGSEGV:** only an AR==CL graph inflates
`n_process` past the lade attention-map size, which drives an out-of-bounds
position-id read whose garbage becomes a RoPE-table byte offset. Removing the
AR==CL graph removes the code path. Independently, **lade prompts must
tokenize to ≥ 2 tokens** — a 1-token prompt hits `rand() % 0` in `lhd-dec.cpp`
regardless of topology.

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

Device verdict — **revised 2026-08-13**: our "no gain at vtcm 16" A/B
(6.27–6.5 tok/s) compared against a v1-era fused build that was emitting
garbage output, so it never actually measured fusion. The device team's
*working* fused build measures **8.98 vs 7.79 unfused (+15%)**, decode DDR
read 880 vs ~960 MB (REFERENCE §6.6, correction #15). Fusion is back on the
table: re-A/B on topology B before the next device cycle, and treat it as a
first-class candidate for the 4B build (each fused-away op also refunds its
~220 µs dispatch, which compounds at 36 layers).

### 5.4 Lookahead decoding (`-lade`, 3-graph)

Adds an AR=32 verification graph to the baseline and packs a 3-graph ctx-bin.
Requires §5.1 complete:

```bash
./scripts/build/lade_build.sh qwen3-0.6b-w8a16 128 1024 32   # AR=16 variant: last arg 16
```

Config guardrail: `(ngram−1)×(window+gcap)` must stay ≤ the verify graph's AR,
or Genie routes verification batches to a graph that cannot serve them. Shipped
config window 8 / ngram 3 / gcap 8 = exactly 32.

⚠️ **This build alone is not shippable for lade** — its bertcache prefill makes
`type:"lade"` SIGSEGV on device (§3.5). It is now only an intermediate step:
`ladekv_build.sh` reuses its `verify32.dlc`. Continue to §5.4b.

### 5.4b Lookahead decoding, fixed (`-ladekv`) — the reference lade build

**This is the recipe to copy for any new lade-capable model.** It replaces the
bertcache prefill with a past-KV prefill (§3.5) and packs the 3-graph ctx-bin.

Prerequisites: §5.1 (`full_build.sh`) and §5.4 (`lade_build.sh`) complete for
the same `<name>` — this build reuses their `decode.dlc`, `verify32.dlc`, and
the prefill quant dir's encodings.

```bash
./scripts/build/ladekv_build.sh qwen3-0.6b-w8a16 128 1024 128
#                               <name>            CL  CTX AR_prefill
# VERIFY_AR=16 ./scripts/build/ladekv_build.sh …   if you built a 16-wide verify graph
```

What the five stages do, and what to check after each:

| Stage | Action | Gate |
|---|---|---|
| 1 | AIMET export of a past-KV graph at AR=128 (`--decode-ar 128 --export-decode <prefill quant dir>`) — the decode wrapper *is* the past-KV prefill, just wider. Encodings are adopted, never recalibrated, so scales stay bit-identical to `decode.dlc`/`verify32.dlc`. | skipped automatically if `model_renamed.onnx` exists; `FORCE_EXPORT=1` to redo |
| 2 | Canonical I/O rename (`--with-past`) | tensor names match §3.1 |
| 3 | `parity_ladekv_read.py` — replays qualla's exact feed pattern incl. two-chunk prompts | all prompts OK (we see 6/6, `max\|Δ\|` ≈ 1.6–6.4 vs FP32 HF) |
| 4 | `qairt-converter` → `prefill.dlc` in its own dir (graph names come from DLC filenames) | asserts `logits 1,128,151936` **and** that `past_key_0_in` exists |
| 5 | 3-graph weight-shared ctx-bin | prints all 3 graphs, 60 inputs / 57 outputs each; ~1.1 GB |

Then bundle with **both** dialog configs (§7) and A/B lade vs basic on one binary.

Two things this build also fixes, worth carrying to any new variant:

- `verify32` must be listed in `graph_names` in **both** `configs/htp_config.json`
  (compile-time) and `configs/htp_backend_ext_config.json` (device). Omitted, it
  silently compiles with defaults — device logs showed 4 MB VTCM + 24 MB spill
  instead of 16 MB + 0.
- Expect basic mode to lose the ~23 tok/s bertcache early phase; generation runs
  at true decode speed from token 1. That is the intended trade.

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

### 5.7 Weight-bitwidth variants (`--quant-head`, and the W4 dead end)

Flags go to `quantize_aimet.py`, so they pass through `full_build.sh <name> <cl>
<ctx> …`. A bitwidth change alters encodings, so it needs a **full** rebuild
chain (not the §5.6 fast path):

```bash
./scripts/build/full_build.sh   qwen3-0.6b-w8a16qh 128 1024 --quant-head
./scripts/build/lade_build.sh   qwen3-0.6b-w8a16qh 128 1024 32
./scripts/build/ladekv_build.sh qwen3-0.6b-w8a16qh 128 1024 128
```

- `--quant-head`: quantizes the `lm_head` **weight** to INT8 per-channel while
  leaving its activations FP16. Must be passed together with
  `--keep-head-weight` (the build scripts forward it automatically) — without it
  `filter_aimet_w8a16.py` strips the encoding and the converter silently emits
  `Float_16`, producing a build byte-identical to a non-qh one. Verify, don't
  assume — but **use this exact command** (2026-08-16):

  ```bash
  qairt-dlc-info -i prefill.dlc \
      | grep -oP 'lm_head\.weight \(data type: \K[A-Za-z_0-9]+'   # -> sFxp_8
  ```

  ⚠️ **The obvious `grep lm_head.weight` gives the wrong answer.** It reports
  `Float_16` on a build whose head is correctly `sFxp_8`, for two compounding
  reasons: `.` is a regex wildcard, and the op-table row for the `lm_head`
  `FullyConnected` lists its **activation** input (`Float_16`) before the weight
  tensor, so `grep -m1` matches the activation. This produced a false FAIL on a
  correct build, which is the same class of error as the trap the check exists
  to catch — just inverted. The size check is a useful cross-reference: a real
  W8 head shrinks each 0.6B DLC from ~1,074 MB to ~923 MB.

  **Note the head can also arrive without the flag.** `lade_build.sh` and
  `ladekv_build.sh` do not re-run the filter — they pass an already-filtered
  encodings file to `--quantization_overrides`. So exporting the verify/past-KV
  graphs against a **qh prefill quant dir** yields `sFxp_8` in those graphs even
  though the export log lists `lm_head.params` among its disabled quantizers.
  Confirmed on all three graphs of `gqafix_qh_ladekv`, 2026-08-16.

  What it actually buys, measured (2026-08-12):

  | | value |
  |---|---|
  | `lm_head` size | 311.2 MB FP16 → 155.6 MB INT8 (151936 × 1024) |
  | DLC | 1,074,293,920 → 922,965,680 B (**−151.3 MB**) ✅ |
  | ctx-bin | 1,106,276,352 → 1,093,767,168 B (**−12.5 MB only**) ⚠️ |
  | device, LADE mode | **9.3 vs 10.8 tok/s (−14%)** ❌ |
  | quality | unchanged (3/4 argmax local; device parity confirmed) ✅ |

  So ~139 MB of the saving reappears when the generator prepares the context
  blob, and on device the variant is a **net regression** under LADE — the head
  quantization costs ~10% n-gram acceptance, which dominates any per-call gain
  (`reports/qwen3-0.6b-w8a16qh-ladekv-test-report.md`). Do not ship `qh` for
  speculative decoding. It remains untested in AR-1 basic mode, where acceptance
  is irrelevant.

  ⚠️ Earlier revisions of this guide claimed "961 → 763 MB/token (−20.6%)" here.
  That was wrong: both numbers come from `ctxbin-ws.log` of 2026-08-10 — the
  prefill (763,410,432) and decode (961,130,496) `read_total_bytes` of one
  *non-qh* weight-shared build, two days before `--quant-head` existed. No
  converter DDR summary for a real qh build has been recorded.
- `--weight-bw 4` (+ optional `--lpbq`, `--seq-mse`): **W4A16 is a dead end,
  full stop** (2026-08-13). Two independent kills: (a) accuracy — all three
  recipes (per-channel int4, LPBQ block-64, LPBQ+SeqMSE) scored 0/4 on the
  `--eval` argmax gate that W8A16 passes 3/4 (`max|Δlogits|` 16–25 vs 1.3–1.7);
  (b) kernels — `htp_v2.json` contains **zero** INT4 MatMul/FC entries (SDK
  2.43 and 2.48 alike) and qairt-converter folds s4 weights back to f16
  (`Constant folded static tensor ... from s4 to f16` — HTP doc §5.3/§10.1).
  Even a model that quantized well would not run 4-bit on this SDK. The flags
  stay in the script only for a future SDK that ships the kernels.

## 6. Validation gates — run before shipping anything

Device-free gates, in order; each has a known-good reference value:

| Gate | Command | Pass looks like |
|---|---|---|
| Wrapper vs HF | `export_qwen3.py … --parity-check` | max\|Δlogits\| ~4e-05 |
| Standard parity | `scripts/validate/parity_onnx.py --onnx <dir> --cl-prefill 128 --ctx 1024` | prefill argmax match + 8-step greedy chain token-identical |
| **Device read pattern** (topology A) | `scripts/validate/parity_qualla_read.py --onnx <dir> --cl-prefill 128` | 4/4 prompts OK (left-aligned, logits row n−1 — what the device actually samples) |
| **Device read pattern** (topology B) | `scripts/validate/parity_ladekv_read.py --onnx <model_renamed.onnx> --ar 128 --ctx 1024` | 6/6 — 4 single-chunk + 2 chunked (129, 200 tokens); run automatically as stage 3 of `ladekv_build.sh` |
| Verify graph (lade) | `scripts/validate/parity_verify.py` | batched rows match HF, ~3e-05 |
| Quant quality | `quantize_aimet.py … --eval` | last-token argmax agreement ≥ 3/4 prompts |
| DLC shape | `qairt-dlc-info -i prefill.dlc \| grep logits` | `1,128,151936` — never `1,1,…` |
| **Graph names** | `qnn-context-binary-utility --json_file info.json` → `info.graphs[].info.graphName` | exactly the names in **both** `htp_config.json` and `htp_backend_ext_config.json` — a mismatch silently reverts that graph to backend defaults (§3.4) |
| Ctx-bin | `qnn-context-binary-utility --json_file info.json` | all graphs listed; logits dims per §3.1; ~1.09 GB for 0.6B (2- or 3-graph, weight-shared) |
| Quantized head | `qairt-dlc-info -i prefill.dlc \| grep lm_head.weight` | `sFxp_8` if `--quant-head` was used, else `Float_16` (§5.7) |

## 7. Bundle and ship

```bash
# baseline / fused (topology A)
./scripts/build/bundle.sh qwen3_06b_w8a16_local \
    $LLMDEPLOY_DATA/work/ctxbin/qwen3-0.6b-w8a16/qwen3-0.6b-w8a16_ctx.bin

# ladekv (topology B): lade config as 3rd arg, then add the basic config too
./scripts/build/bundle.sh qwen3_06b_w8a16_ladekv \
    $LLMDEPLOY_DATA/work/ctxbin/qwen3-0.6b-w8a16-ladekv/qwen3-0.6b-w8a16-ladekv_ctx.bin \
    $LLMDEPLOY_ROOT/configs/genie_dialog_qwen3_0.6b_lade.json
cd $LLMDEPLOY_DATA/bundles/qwen3_06b_w8a16_ladekv && python3 - <<'EOF'
import json
d = json.load(open("/mnt/x/code/llm-deploy/configs/genie_dialog_qwen3_0.6b.json"))
d["dialog"]["engine"]["model"]["binary"]["ctx-bins"] = ["qwen3-0.6b-w8a16-ladekv_ctx.bin"]
json.dump(d, open("genie_dialog_basic.json", "w"), indent=2)
EOF

# MANDATORY after hand-adding dialogs -- bundle.sh's own gate ran before they
# existed. A "lade" dialog carrying "max-num-tokens" SIGSEGVs on device (139);
# that pair reached three shipped bundles through exactly this ungated step.
python3 $LLMDEPLOY_ROOT/scripts/validate/lint_bundle_dialogs.py \
    $LLMDEPLOY_DATA/bundles/qwen3_06b_w8a16_ladekv

cd $LLMDEPLOY_DATA/bundles && tar -czf qwen3_06b_w8a16_ladekv.tar.gz qwen3_06b_w8a16_ladekv
```

Shipping both configs is what lets the tester A/B lade vs basic on one binary.

`bundle.sh` assembles the flat layout, patches the dialog JSON's `ctx-bins` to
the real filename, and tars. Expected: ~930 MB tarball for 0.6B.

Upload to HF (`vinccniv/sa8797p-qwen3-w8a16-bundles`, staging dir
`$LLMDEPLOY_DATA/hf-staging/` with hardlinks):

```bash
scripts/util/hf_upload_watchdog.sh vinccniv/sa8797p-qwen3-w8a16-bundles \
    $LLMDEPLOY_DATA/hf-staging
```

The watchdog exists because the local proxy drops long-lived uploads (client
hangs on CLOSE-WAIT sockets looking alive). Three hard-won rules:

1. **Set `SOCKET_CHECKS=999999`.** The watchdog's CLOSE-WAIT detector
   false-positives through this proxy and kills healthy transfers (a partial
   blob restarts from byte 0, so it can never finish). The progress-freeze
   detector alone (`STALL_SECS=240`) is the reliable signal.
2. **The hub allows 128 commits/hour.** Restart storms exhaust it; the commit
   phase then "hangs" with every byte already uploaded. Diagnose with one
   foreground `HfApi().upload_file` (a 429 surfaces in seconds). Recovery:
   stop all uploaders, wait ~1 h, then commit **one file at a time** with
   `upload_file` — the blobs dedup, so each commit is data-free and instant.
3. **`hf upload-large-folder` silently resets repo visibility and overwrites
   the hub README.** After any bulk upload, check
   `HfApi().repo_info(repo).private` and diff the README — then **report what
   changed; do not "restore" a setting from assumption.**
   Repo visibility is switched often and deliberately by the user, so no doc
   here records it. Read `HfApi().repo_info(repo).private` live when it
   matters, change it only when asked in that message, and if a bulk upload
   flipped it as a side effect, report it and stop. Acting on a remembered
   value has caused four incidents, in both directions. Single `upload_file`
   commits do not touch repo settings — prefer them once blobs are staged.
4. **Long single-file uploads need `setsid`.** Backgrounded from a shell that
   then exits, the uploader is killed and its log simply stops mid-progress-bar
   with no error — indistinguishable from a proxy drop at a glance. Use
   `setsid nohup … & disown`; a 1.08 GB blob then lands in under a minute.

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
| lade mode SIGSEGV in libGenie, basic mode fine | Bertcache prefill in the ctx-bin — rebuild with `ladekv_build.sh` (§3.5, §5.4b). If it still crashes, check the prompt is ≥ 2 tokens. |
| Genie load error about duplicate graphs / KV quant params | Two graphs share an (AR, CL) pair, or DLCs came from different encodings runs (§3.3). |
| HF upload progress frozen, sockets CLOSE-WAIT | Proxy drop — the watchdog handles it, but set `SOCKET_CHECKS=999999` (§7). If blobs are uploaded but commits never land: 429 rate limit (§7). |
| HF repo unexpectedly public | `hf upload-large-folder` reset it — restore with `update_repo_settings(private=True)` (§7). |
| WSL2 C: drive filling up | Call `disk_guard <need_gb>` (in `env.sh`) before every multi-GB step, **sized to that step**: 6 GB is the converter floor, a 4B export writes 8.6 GB and should ask 20. A flat 6 GB check passes and then still runs C: dry mid-step. No compaction step is needed to recover: the vhdx is sparse and `/` is mounted `discard`, so deleting in-guest returns the space to C:. (`ls` always reports the ~448 GB virtual size; `du -h <vhdx>` without `--apparent-size` is the real consumption.) |
| WSL2 VM hard-crashes, no OOM line anywhere | C: ran dry and the vhdx grow failed. This is **not** ENOSPC — the guest still reports free space, the host write fails, and every mmap'd page takes SIGBUS; PID 1 dies and the VM dies with it (3× on 2026-08-12 during VL-4B stage 2). Dumps land in `%LOCALAPPDATA%\Temp\wsl-crashes`; the `-N` filename suffix is the signal, `-7` = SIGBUS. Prevention is `disk_guard`, above. |
| `--quant-head` build looks identical to a normal one | `--keep-head-weight` missing → the encodings filter stripped the head encoding and the converter emitted `Float_16`. Check with `qairt-dlc-info \| grep lm_head.weight` (§5.7). |
| Device output loops until `Context Size was exceeded` | The shipped `genie_dialog*.json` is the **greedy parity config** (temp 0, no `max-num-tokens`) — right for validation, a footgun for demos; the device team hit exactly this (HTP doc §8.2). For interactive runs add `max-num-tokens` and sampling (temp 0.85 / top-k 50 / top-p 0.9 verified on device), and apply the Qwen3 chat template with the empty `<think>\n\n</think>` block so thinking mode stays off. |
| Ctx-bin much larger than one DLC in a multi-graph build | Weight sharing not effective. A 3-graph 0.6B bin should be ~1.09 GB, not 1.8–2.2 GB — check `context.weight_sharing_enabled` in `configs/htp_config.json`. |

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
| `scripts/build/lade_build.sh` | verify graph + 3-graph ctx-bin (bertcache prefill — intermediate step) |
| `scripts/build/ladekv_build.sh` | **past-KV prefill + 3-graph ctx-bin — the lade-capable build (§5.4b)** |
| `scripts/build/prefill_fix_rebuild.sh` | fast graph-shape-only rebuild |
| `scripts/build/bundle.sh` | device bundle assembly |
| `scripts/validate/parity_onnx.py`, `parity_qualla_read.py`, `parity_ladekv_read.py`, `parity_verify.py` | validation gates (§6) |
| `scripts/util/hf_upload_watchdog.sh` | supervised HF upload |
| `docs/REFERENCE.md` | **consolidated, corrected reference — start here** |
| `docs/MAX_TPS_QWEN3_0.6B.md` | the max-throughput 0.6B build path (proven 10.8 tok/s + fused candidate + A/Bs) |
| `docs/NOTES-genie-io.md` | the Genie contract with SDK source citations — read before touching graph I/O |
| `docs/NOTES-vit-htp-config.md` | why graph names must appear in the backend config (§3.4) |
| `docs/NOTES-genie-splits.md` | multi-ctx-bin (split) contract — required for any graph over the 3.5 GiB serialization limit (every text tower ≳2B) |
| `docs/SA8797P_HTP_v81_Hardware_and_Deployment_Quantization_Reference_EN.md` | device team's measured hardware/runtime reference (2026-08-12), annotated — the hardware ground truth |
| `docs/LOCAL_ENV.md` | environment provenance + progress log |
| `reports/` | device test reports (v1 failure analysis, v2 validation, ladekv, qh) |
| `SA8797P_Deployment_Status_Summary.md` | inherited remote-team status, 2026-08-09 — **partly superseded**, see its banner and `docs/REFERENCE.md` |
