# Storage inventory — where this project's files actually live (2026-08-16)

Read-only survey. Nothing was deleted, moved or modified to produce it.

> ## ⚠ Two same-day addenda — read these before quoting any number below
>
> ### A. The survey omitted a whole host
>
> It counts only the WSL box. `tank` — the second build host
> (`LOCAL_ENV.md` §Machines) — holds another **355 GB**: `work/quant` 167,
> `work/onnx` 122, `dlc` 25, `ctxbin` 8.4, `bundles` 11, `models` 8.3. True
> project footprint at survey time was therefore **~982 GB**, not 627 — every
> total below understated by ~36%. The per-root analysis is unaffected; the
> headline is not.
>
> Tank's disk is **not** the constraint the local one is — 937 GB native LVM,
> ~325 GB free, no vhdx indirection, so C: exhaustion cannot happen there.
>
> ### B. The local roots were cut by ~133 GB the same day
>
> This survey's central claim — *"~211 GB of regenerable fp32 tensors are being
> retained as if they were outputs"* — was acted on within hours. **The tables
> below are a pre-cleanup snapshot.** Current state:
>
> | | survey | now | note |
> |---|---:|---:|---|
> | `C:` free | 54 GB (95%) | **181 GB (82%)** | still rising; vhdx `discard` reclaims asynchronously |
> | `llm-local` | 297 GB | **165 GB** | |
> | `work/quant` | 70.5 GB | **3.0 GB** | regenerable tensors stripped, all `.encodings`/`.onnx` kept |
> | `work/ctxbin` | 47 GB | **32 GB** | 11 pre-fix bins removed, each matched to a backing bundle first |
> | `work/dlc` | 33 GB | **14 GB** | 8 pre-GQA-fix families |
> | `bundles` | ~80 GB | **46 GB** | 20 unpacked dirs that duplicated their own `.tar.gz` |
>
> **What was kept, and why it is the whole policy in one line:** per quant dir,
> `*.encodings` + `*.onnx` are lineage and everything else regenerates. Measured:
> 8 dirs × 8.6 GB → 122–152 MB each, i.e. **1.6% is irreplaceable** — an
> independent confirmation of this survey's own 97%-regenerable finding. The
> retained 0.6B lineage is checksum-identical to tank's copy, so it now exists
> twice.
>
> **Deletions were verified, not assumed.** A bundle tarball was confirmed to
> hold the ctx-bin at identical byte size (1,106,276,352) before any unpacked
> copy went; all 29 tarballs were indexed once and pre-fix ctx-bins matched by
> exact byte size rather than by name.
>
> **Two artifacts were held back, then mined and released.**
> `work/ctxbin/qwen3-0.6b-w8a16qh` (1.8 GB) and `…qh-lade` (2.1 GB) existed in no
> bundle and on no remote, and `REFERENCE.md` §8.2 cited them as the only known
> specimens of a silently-unshared build. Rather than delete or keep them blind,
> the question was **settled from their `info.json`** — `graphBlobInfoV2.constSize`
> names an unshared graph directly, so file size was never the only symptom
> (§8.2, resolved). The 4.0 GB of `.bin` then went; **the two `info.json` files
> are retained at 541 KB as the evidence.**
>
> Generalisable: a ctx-bin's `info.json` is ~0.01% of its size and carries most
> of what anyone later asks of it. Strip the `.bin`, keep the sidecar.

## 0. Headline

| | |
|---|---|
| **Total project footprint** | **~585 GB** across three roots on two drives, plus ~42 GB of tool caches on the same C: vhdx (**~627 GB** all in) — *local only; add ~355 GB on tank, see the addendum* |
| **The binding constraint** | **C: was 95% full — 54 GB free.** Everything under `/` (the WSL vhdx) is backed by C:. *Resolved same day: now 181 GB free — addendum B* |
| **Quantization intermediates** | **346 GB** — 59% of everything, spread across **three** locations with no shared convention |
| **Of that, irreplaceable** | **~9.6 GB (2.8%).** The other **97%** is fp32 weights that `quantize_aimet.py` regenerates |
| Bundles | 80 GB local, ~61 GB of it name-matched on HF |
| Biggest single non-project item on C: | `~/.cache/uv` at **34 GB** |

The disk is 95% full because ~211 GB of regenerable fp32 tensors are being retained
as if they were outputs. That is the whole problem in one sentence.

> **This diagnosis was acted on the same day** — ~133 GB reclaimed, C: to 82%.
> See addendum B. The sentence stayed true; only the numbers moved.

---

## 1. The map — three roots, two drives

| Root | Physical drive | Size | What it is |
|---|---|---|---|
| `/home/vinc/llm-local` (`$LLMDEPLOY_DATA`) | **C:** (via the WSL ext4.vhdx) | **297 GB** → **165 GB** | live working set: `work/` 192→49, `bundles/` 80→46, `models/` 14, `sdk/` 5.7, `envs/` 5.6 |
| `/home/vinc/.cache` | **C:** (same vhdx) | **~42 GB** | `uv` 34, `huggingface` 4.7, `codebase-memory-mcp` 1.7, playwright/puppeteer 1.7 |
| `/mnt/x/llm-archive` | X: | **134 GB** | `quant-consumed-2026-08-14` (77), `quant-2026-08-13` (57) |
| `/mnt/x/llm-local-archive` | X: | **152 GB** | a mirror of the old `work/` layout: `quant` 142, `onnx` 11 |
| `/mnt/x/code/llm-deploy` | X: | 2.3 GB | the repo — but 2.3 GB of that is **one file** (see §5) |
| `tank:~/llm-local` | tank's own LVM | **355 GB** | the second build host, omitted from this survey's totals: `work/` 324 (`quant` 167, `onnx` 122, `dlc` 25, `ctxbin` 8.4), `bundles/` 11, `models/` 8.3, `sdk/` 5.7, `envs/` 5.6 |

Free space at survey time: **C: 54 GB (95% used)** · X: 334 GB (83% used).
After the same-day cleanup: **C: 181 GB (82% used)**.

The C: number is the one that matters. Per `CLAUDE.md`, a failed vhdx grow is **not**
an ENOSPC — the guest still reports free space, the host write fails, every mmap'd page
takes SIGBUS, and the VM hard-crashes. That has already happened three times. At 54 GB
free, a 4B export (writes 8.6 GB, `disk_guard` asks 20) is not yet at risk, but the
margin is thinner than it looks because the vhdx only ever grows.

---

## 2. The quantization intermediates — the actual concern

### 2.1 Anatomy of one quant directory

Every `work/quant/<name>/` is ~8.6 GB **for a 0.6B model**. Measured contents of
`qwen3-0.6b-w8a16-prefill`:

| File | Size | Regenerable? |
|---|---|---|
| `model.pth` (AIMET quantsim checkpoint) | 3.0 GB | yes — re-running quantize rebuilds it |
| `<uuid>.data` (ONNX external data) | 3.0 GB | yes — same fp32 weights, second copy |
| `embed_tokens.marked_module.weight` | 622 MB | yes |
| ~200 loose `onnx__MatMul_*` blobs | ~1.8 GB | yes |
| **`model_torch.encodings`** | **98.5 MB** | **NO — calibration output** |
| **`model.encodings`** | **20.1 MB** | **NO** |
| **`model_filtered.encodings`** | **14.9 MB** | **NO** |
| **`model_filtered_renamed.encodings`** | **14.9 MB** | **NO** |
| **`model_renamed.onnx` / `model.onnx`** | 0.9 MB each | **NO — the graph** |

**8.4 GB regenerable, ~150 MB precious. The precious part is 1.6%.**

The encodings are the part that cannot be recreated identically, and per `CLAUDE.md`'s
cross-graph rule they are load-bearing: decode/verify/prefill must share one encodings
lineage or Genie fails to load with byte-mismatched KV quant params. **The encodings are
the asset. The weights are scratch.**

### 2.2 Measured across all three locations

| Location | Total | encodings + graph | `model.pth` | ONNX `.data` |
|---|---|---|---|---|
| C: live `work/quant` | 70.1 GB | 1.14 GB (1.6%) | 22.4 GB | 23.9 GB |
| X: `llm-archive` | 133.8 GB | 2.19 GB (1.6%) | 42.0 GB | 47.6 GB |
| X: `llm-local-archive` | 141.9 GB | 6.27 GB (4.4%) | 36.4 GB | 39.2 GB |
| **Total** | **345.8 GB** | **9.6 GB (2.8%)** | **100.8 GB** | **110.7 GB** |

`model.pth` + `.data` alone = **211.5 GB of duplicated fp32 weights**. They are two
serializations of the same tensors, kept side by side, ~35 times over.

### 2.3 Why it feels unmanaged

1. **Three locations, no stated convention.** `llm-archive` is grouped by *event*
   (`quant-consumed-2026-08-14`, `quant-2026-08-13`); `llm-local-archive` is grouped by
   *original layout* (`work/quant/...`); C: holds whatever the last build touched. Nothing
   in `docs/` defines which is authoritative or when something moves.
2. **Retention is all-or-nothing.** Archiving copies the full 8.6 GB, including the 8.4 GB
   of scratch. There is no "keep the encodings, drop the weights" step anywhere in the
   build scripts.
3. **Naming does not encode lineage.** `-prefill`, `-decode`, `-verify32`, `-prefillkv128`
   are variants of one lineage, but nothing records *which* prefill an
   `--adopt-encodings` decode was built against. That has to be reconstructed from build
   logs, which is exactly what the cross-graph rule punishes you for getting wrong.
4. **Only 3 name collisions across 37 named quant dirs** (9 live + 9 in `llm-archive` + 19
   in `llm-local-archive`, plus 4 `bisect-*`/`determinism` dirs in `quant-2026-08-13`) — so
   this is not mass duplication by name; it is ~41 *distinct* experiments each retained at
   full weight.

The three collisions, all with **different sizes** (so not identical copies — the archive
side is a superset in two cases):

| Name | C: live | X: llm-archive | X: llm-local-archive |
|---|---|---|---|
| `qwen3-0.6b-w8a16-prefill` | 8.6 GB / 207 files | — | 11 GB / 403 files |
| `qwen3-0.6b-w8a16-fuseqkvgu-prefill` | — | 8.6 GB / 125 files | 11 GB / 237 files |
| `qwen3-0.6b-w8a16-fuseqkvgu-decode` | — | 8.6 GB / 121 files | 1.8 GB / 117 files |

---

## 3. Other `work/` stages

| Stage | Size | Notes |
|---|---|---|
| `work/onnx` | 37 GB | `qwen3vl-4b-text` alone is **30 GB** — the 4B export, the single largest directory in the project |
| `work/ctxbin` | 49 GB | 30+ dirs, ~1.1–4.3 GB each. These are build *outputs*, and each shipped one is also inside a bundle |
| `work/dlc` | 33 GB | 17 dirs. Pure intermediates — converter input to the ctx-bin step |
| `work/lut`, `scratch`, `profiling`, `kit`, `reference`, `calib` | 4.1 GB | small |

`ctxbin` + `dlc` = 82 GB of a stage that exists only to feed the next stage. A DLC is
fully determined by (ONNX graph + encodings + converter flags).

---

## 4. Bundles — 80 GB, doubled by construction

**20 of 24 bundles exist as both an extracted directory and a `.tar.gz`** — roughly 2×
storage for the same content (e.g. `qwen3_06b_w8a16_ladekv` = 1.2 GB dir + 892 MB tar).
A further 9 tarballs have no directory.

Name-matched against the two HF repos:

- **21 bundles (61.1 GB local) have a matching name on HF** — `2026-08-14-gqafix`,
  `2026-08-16-regime`, and both `qwen3vl_4b_e2e_pipeline*` folders.
- **12 bundles (18.2 GB local) do not**, including `qwen3vl_4b_text_w8a16` (5.8 GB),
  `qwen3vl_4b_vit_fp16` (926 MB), and ten `*_ladekv` variants.

⚠️ **This match is by filename only.** I did not compare hashes or sizes against the
remote. Treat "on HF" as *a remote copy of that name exists*, not as *verified identical*.
Verify before relying on it to delete anything.

---

## 5. Everything else

| Item | Size | Note |
|---|---|---|
| `models/` | 14 GB | Qwen3-VL-4B 8.3, Qwen3-1.7B 3.8, Qwen3-0.6B 1.5 — all re-downloadable from HF |
| `~/.cache/uv` | **34 GB** | largest single reclaim on C:, and not project data |
| `~/.cache/huggingface` | 4.7 GB | hub 4.3 + xet 0.38; already listed as a reclaim in the active plan's Task 0.1 |
| `sdk/` 5.7 GB · `envs/` 5.6 GB | 11.3 GB | keep — `env.sh` depends on both |
| `llm-local/*.log` | 54.6 MB / 50 files | flat in the root, unrotated, back to the first build |
| **`repo/downloads/qairt-2.48.40.260702.zip`** | **2.3 GB** | the SDK installer **inside the git repo dir** — already extracted to `sdk/`. It is 99.9% of the repo's size |
| WSL crash dumps | 0 B | `wsl-crashes/` is empty — nothing to clean |

---

## 6. Reproducible vs precious — the decision table

| Class | Where | Size | Cost to recreate |
|---|---|---|---|
| **Precious** — encodings + renamed graph | inside every quant dir | **9.6 GB** | impossible to reproduce byte-identically; breaks the cross-graph rule |
| **Precious** — device-validated ctx-bins/bundles not on HF | `bundles/` | 18.2 GB | a full rebuild per bundle |
| Regenerable — `model.pth` + ONNX `.data` + loose blobs | all 3 quant roots | **~336 GB** | one `quantize_aimet.py` run each (~1–2 h for 4B, minutes for 0.6B) |
| Regenerable — DLCs | `work/dlc` | 33 GB | one converter pass from ONNX + encodings |
| Regenerable — ctx-bins | `work/ctxbin` | 49 GB | one `qnn-context-binary-generator` pass from DLCs |
| Re-downloadable | `models/`, HF cache, uv cache | 53 GB | bandwidth only |

---

## 7. What I'd suggest (not done — your call)

**A convention worth adopting: strip, don't archive.**
A "consumed" quant dir should keep only `*.encodings` + `model_renamed.onnx` + a
provenance note (source model, flags, which lineage it was adopted from). That is ~150 MB
instead of 8.6 GB — a **98% reduction** with zero loss of anything that cannot be
regenerated. Applied to all 37 dirs that is roughly **336 GB back**, ~70 of it on C:.

Ordered by value-per-risk:

1. **`~/.cache/uv` (34 GB, C:, zero project risk)** — the single biggest C: win, and it is
   not project data. `uv cache clean`.
2. **Bundle dir-or-tarball, pick one (~35 GB, C:)** — keep the tarball (it is what ships),
   drop the extracted dir, for the 20 that have both.
3. **Strip the archived quant dirs on X: (~270 GB)** — they are already declared
   "consumed"; nothing reads them.
4. **HF cache (4.3 GB, C:)** — already sanctioned in the active plan's Task 0.1.
5. **Move `downloads/qairt-*.zip` (2.3 GB) out of the repo dir** — it is already extracted
   to `sdk/`, and it does not belong on X: next to tracked scripts.
6. **`work/dlc` + superseded `work/ctxbin` (up to 82 GB, C:)** — regenerable, but check
   each against a shipped bundle first.

**Two guardrails worth adding to the build scripts,** since this will otherwise recur:
- Have `full_build.sh` strip the previous lineage's `model.pth`/`.data` once the DLCs are
  converted and verified — the point at which the weights are provably no longer needed.
- Write a `LINEAGE` file into each quant dir recording source model, flags, and the
  `--adopt-encodings` parent, so retention decisions stop depending on build logs.

**Before deleting anything**, the three name-collision pairs in §2.3 differ in size and
file count. Confirm which side is the superset rather than assuming the archive wins:

```
diff <(ls -la <live>) <(ls -la <archive>)
```

---

*Method: `du`/`find` over the three roots; per-file classification by name within quant
dirs; HF contents via `HfApi().list_repo_files` on both repos. Sizes are `du` apparent
usage at 2026-08-16. Every figure here was measured, not estimated — except the HF
name-match in §4, which is explicitly a name match only.*
