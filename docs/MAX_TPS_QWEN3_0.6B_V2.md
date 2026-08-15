# MAX TPS Qwen3-0.6B on SA8797P — Plan V2

**Date:** 2026-08-14 · **Supersedes:** `MAX_TPS_QWEN3_0.6B.md` (Phases A/B/C all built; B/C still unmeasured)
**Inputs:** 2026-08-13 device measurement report (5 tests), local DLC/ctx-bin forensics (this doc's §1–§2), supplier "Hengju" package analysis, QAIRT 2.48 SDK docs.

---

## 0. Measured baselines (device, 2026-08-13, warm, greedy, 56-token prompt)

| Config | Bundle | tok/s | Note |
|---|---|---:|---|
| Basic AR-1 | `qwen3_06b_w8a16_local` (2-graph) | **11.72 ± 0.01** | new best AR-1 |
| LADE | `qwen3_06b_w8a16_ladekv` (3-graph) | 9.18 ± 0.00 | technical prompt, low acceptance |
| LADE (earlier, simple prompt) | `qwen3_06b_w8a16_ladekv` | 10.8 | 1.94 accepted tok/call |
| Basic AR-1 | `qwen3_06b_w8a16qh_ladekv` (3-graph, W8 head) | 6.70 ± 0.00 | **confounded** — see §2.2 |
| Basic AR-1 on plain `ladekv` | — | **never measured** | the missing arm (A1) |

Decode step on `local`: ~85 ms. Decode-only profile: **350.3M aggregate DSP cycles/step, ±0.3% reproducible**.

---

## 1. Revised performance model (the central correction)

### 1.1 The 74.7% is GQA head replication, not the attention mask

The report attributes 261.8M cycles/step (74.7%) to "attention-mask Expand". DLC inspection
(`qairt-dlc-info` on `decode.dlc`) shows the mask is never expanded — it enters as `[1,1,1152]`,
gets an Unsqueeze to `[1,1,1,1152]`, and broadcasts implicitly in the Add. The 56 expensive ops
(2/layer × 28) are **`repeat_kv` — GQA KV head replication**:

```
Expand    [1,8,1,128,1152] → [1,8,2,128,1152] → Reshape → [1,16,128,1152]   (K, stored transposed)
Expand_1  [1,8,1,1152,128] → [1,8,2,1152,128] → Reshape → [1,16,1152,128]   (V)
```

Qwen3-0.6B: 16 Q heads / 8 KV heads → each KV head duplicated ×2 so a 16-head MatMul can consume it.

**Arithmetic proof:** report assumed output `[1,8,1,1152]` = 18,432 B → "260 cycles/byte,
impossibly inefficient". Real output is 4,718,592 B — exactly 256× larger — giving
**1.01 cycles/byte**, an ordinary unvectorized copy. The shape error and the impossibility cancel;
the cycle counts themselves are trustworthy.

### 1.2 Traffic and cycle budget

Per decode step, the replication costs:
- **264 MB written** (56 × 4.72 MB) + **264 MB re-read** by the attention MatMuls + 132 MB read as input ⇒ up to **~530 MB/step removable DDR traffic** (less whatever transiently stays in VTCM).
- **261.8M of 350.3M cycles.** At 4 HVX threads, 350M aggregate ≈ 87.5M/thread ≈ 85 ms at ~1 GHz — matching the measured step time. **The DSP is compute-busy essentially the whole step.**

This revises `REFERENCE.md`'s "decode is 100% DDR-bound": the 2-process linear split is equally
consistent with contention on the single HTP core, and the cycle budget says the current
bottleneck is the replication copy loop. (Do not edit REFERENCE.md until B7 confirms on device.)

### 1.3 Renormalized compute (the 88.5M cycles that remain after removing replication)

| Category | % of full step (report) | % of remaining compute |
|---|---:|---:|
| Attention GEMV (Q@K + attn@V) | 11.2% | **44.5%** |
| Weight GEMMs | 8.9% | **35.3%** |
| `lm_head` (FP16) | 1.7% | **6.9%** |
| Everything else | 4.2% | 13.3% |

Consequences: QKV+Gate-Up fusion is ~10% of real compute, not "~4% marginal" (consistent with the
device team's +15% fused result). `lm_head` is 6.9%, not negligible. "verify32 amortizes the
broadcast" is backwards — replication cost is AR-independent, which *strengthens* spec-decode.

### 1.4 Two bounding models (what decides between them)

- **Compute-bound today (likely):** removing replication drops compute to ~22 ms/thread-time;
  step becomes weight-stream-limited at 751 MB (440 MB INT8 decoder + 311 MB FP16 head) / effective BW.
  Today's step moves ~1.4 GB in 85 ms ⇒ ≥16 GB/s aggregate is achievable ⇒ post-fix ~50–55 ms ⇒ **~18–19 tok/s**.
- **Hidden weight floor (pessimistic):** if weights can't stream faster than ~85 ms regardless, gain ≈ 0.
  Evidence against: the 3-graph bundle's decode takes 149 ms with *identical* weights — there is no fixed floor at 85 ms.

Honest projection for B alone: **12–19 tok/s AR-1**, central ~15–18. One rebuild + one device run decides it.

---

## 2. New structural facts (ctx-bin forensics, 2026-08-14)

### 2.1 The `local` bundle's prefill is bertcache (AR=128, **CL=128**)

`qnn-context-binary-utility` on both shipped bins:

| Bundle | Graph | mask shape | spillFill | sharedWeights |
|---|---|---|---:|---:|
| `local` | prefill | `[1,128,128]` | 0 | 1,063,366,656 |
| `local` | decode | `[1,1,1152]` | 0 | ″ |
| `ladekv` | prefill | `[1,128,1152]` | 0 | 1,067,503,616 |
| `ladekv` | decode | `[1,1,1152]` | 0 | ″ |
| `ladekv` | verify32 | `[1,32,1152]` | 0 | ″ |

This **fully explains** the report's prompt-rate gap (1397 vs 301 tok/s) and TTFT gap (40 vs
186 ms): `local`'s prefill attends over 128 positions, `ladekv`'s over 1152 — and the GQA
replication cost scales with CL, so the small prefill graph does ~9× less of it. Not a mystery,
not "graph-switch overhead".

**Constraint:** an AR==CL bertcache prefill is *incompatible with lade* (hard rule). The fast
`local` topology cannot simply gain LADE; conversely `ladekv`'s prefill is the price of verify32.
A hybrid (both a CL=128 bertcache prefill *and* the CL=1152 past-KV prefill in one bin) is legal
by Genie's (AR,CL) best-fit rule for basic mode, but must never ship in a lade bundle.

### 2.2 The decode graphs are structurally near-identical → the "75% build gap" is suspect

Both decode graphs: CL=1152, spillFill 0, same vtcm/O/hvx, sharedWeights within 4 MB. The
report's 6.70 tok/s "3-graph" number was measured **on the qh bundle only** (W8 lm_head). So the
gap labeled "build gap" is really `qh-vs-local` and conflates three variables: W8 head, graph
count, and build lineage. **A1 (Basic on plain `ladekv`) splits this with one run:**
- ≈11.7 → the gap was the qh head all along (and prior 6.3–6.5 numbers need re-examination);
- ≈6.7 → real 3-graph/graph-switching penalty → pursue C2/C3.

### 2.3 Shipped demo config crashes on device

Report §6.1: `type:"lade"` + `max-num-tokens` ⇒ SIGSEGV. Our
`genie_dialog_qwen3_0.6b_lade_demo.json` — shipped as `genie_dialog_demo.json` in **all three
new bundles** (fuseqkvgu, socmodel72, hvx8) — is `type: lade` with `max-num-tokens: 256`.
Confirmed locally. Every demo run on device will exit 139. Fix in P0.

---

## 3. Workstreams

### P0 — Hygiene / ship-blockers (device-free, ~half a day)

| # | Task | Detail |
|---|---|---|
| P0.1 | **Fix demo dialog SIGSEGV** | Remove `max-num-tokens` from the lade demo config (context.size already bounds generation) or make demo `type:"basic"`. Update `configs/genie_dialog_qwen3_0.6b_lade_demo.json`, regenerate the three bundle tarballs' config, re-upload via single-file `HfApi().upload_file` commits. Add a bundle-gate check: no lade dialog may contain `max-num-tokens`. |
| P0.2 | **Redact the measurement report** | It reintroduces the device serial and the internal test-host name — both scrubbed from git history twice. Redact before any commit of the report or `docs/test_artifacts/measurement_2026-08-13/`. |
| P0.3 | Commit report + artifacts (post-redaction) | Artifacts are not in the repo yet. |
| P0.4 | Device-team exchange | Send our 11.72 AR-1 result; request their 7.79-run artifacts (binary, converter cmds, HTP config, info.json, dialog JSON). Unblocks Test-3 A/B. |
| P0.5 | Doc corrections queue | After B7 confirms: fix REFERENCE.md "100% DDR-bound", the report's mask misattribution, and its fusion/-lm_head projections. Not before. |
| P0.6 | **Supplier-analysis publication gate** | `reports/sharegpt-onestep-1.7b-intent-ext-build-analysis.md` + `reports/0813/IMG_3033–3039.HEIC` (7 photos incl. full terminal chrome) are **untracked** — nothing public yet. Before any commit to this public repo: (a) decide whether supplier-IP analysis belongs in-repo at all (the ASR-package analysis deliberately lives outside, in `../central-intelligence-board/`); (b) annotate the report's composer claim — refuted by direct SDK test (composer emits ONE GGUF for the CPU-only `QnnGenAiTransformer` backend, never per-graph `prefill_*/decode_*` bins; the FP16-LUT evidence item is non-discriminating — composer's FP16 path also dumps an FP16 LUT, verified 2026-08-14); (c) review the photos for chrome/identifiers before they join a repo whose history was scrubbed twice for exactly this class of leak. |

### A — De-confounding measurements (Device Session A, ~2 h, everything already built)

All arms: warm, 3 reps, greedy, both the 56-token technical prompt *and* one simple prompt.

| # | Arm | Decides | Cost |
|---|---|---|---|
| A1 ⭐ | **Basic on plain `qwen3_06b_w8a16_ladekv`** | splits qh regression from 3-graph penalty (§2.2); de-confounds report Tests 1 & 4 | 1 run |
| A2 | `spill-fill-bufsize: 0 → 640000000` on ladekv LADE | runtime-only JSON edit; verify32 moves 745 MB spill/fill | 1 edit + runs |
| A3 | Phase B fused `qwen3_06b_w8a16_fuseqkvgu_ladekv` — LADE and Basic | fusion worth ~10% of compute (§1.3); device team saw +15% | built, shipped |
| A4 | Phase C1 `_socmodel72` | supplier ships soc_model 72 — corroborated; run before C2 | built, shipped |
| A5 | Phase C2 `_hvx8` (compile-time) | exactly the report's Rec #4; expectations low (supplier uses 4) | built, shipped |
| A6 | LADE vs Basic on ONE bin × 3 prompt classes (technical / conversational / repetitive-structured) | acceptance-rate map for the real workload question | reuses A1–A3 |
| A7 | `enable-graph-switching: false` for Basic on 3-graph bin | isolates switching overhead if A1 shows the slow path | runtime edit |

### B — GQA replication elimination ⭐ (device-free until the final run; the big lever)

Target: remove 56 Expand+Reshape pairs from all graphs. **KV I/O shapes are untouched**
(`past_key_i [1,8,128,1151]`, `past_value_i [1,8,1151,128]`) — the change is graph-internal, so
the Genie feed contract and cross-graph encodings lineage survive.

| # | Step | Detail |
|---|---|---|
| B1 | Validate the hypothesis against raw profiler artifacts | when P0.3 lands, grep the `qnn-profile-viewer` output for the top ops' output dims — expect `[1,8,2,*,*]`, not `[1,8,1,1152]` |
| B2 | Converter reconnaissance | does 2.48 `qairt-converter` have a GQA/broadcast-MatMul pattern (it ships masked-softmax/topk/gathernd matchers)? Check `--apply_masked_softmax`-style flags and IR optimizer passes. If yes: flag-only fix. |
| B3 | Export-side rewrite (fallback, expected path) | In the export attention wrapper, replace `repeat_kv` + 16-head MatMul with grouped batched MatMul: Q reshaped `[8, 2·AR, 128]`, scores = Q @ K `[8,128,CL]` → `[8, 2·AR, CL]`; mask add broadcasts; attn @ V `[8,CL,128]` → `[8, 2·AR, 128]` → reshape to `[1, AR, 2048]` for o_proj. HF groups Q heads contiguously per KV head — preserve that ordering. Rank-5 broadcast MatMul (`[1,8,2,AR,128] @ [1,8,1,128,CL]`) is the alternative if the converter/HTP handles it (rank-5 tensors already exist in today's graph). |
| B4 | Encodings | weight tensor names unchanged; activations are FP16 in W8A16 → reuse existing filtered encodings via the rename/adopt path. Verify no orphaned activation encodings for deleted Expand outputs. |
| B5 | Local gates (all existing) | ONNX parity vs HF (argmax, all prompts) → `quantize_aimet.py --eval` ≥3/4 → `parity_ladekv_read.py` → graph-name check via `qnn-context-binary-utility`. Read `docs/NOTES-genie-io.md` first (topology change ⇒ mandatory). |
| B6 | Build all variants | 3-graph ladekv (same lineage rule: `--export-decode`/`--adopt-encodings`) + a 2-graph basic-topology variant. `disk_guard` before each multi-GB step. |
| B7 | Device validation (Session B) | (a) decode-only bin under `qnn-net-run --profiling_level detailed`: expect **~350M → ~90M cycles**; (b) Genie tok/s on full bundles. This is the experiment that decides §1.4. |

Expected: −75% DSP cycles, −~530 MB/step DDR. Projection **12–19 tok/s AR-1** (central 15–18).
Also speeds prefill and verify32 (their replication scales with AR·CL).

### C — Topology & the 3-graph question (contingent on A1)

| # | Task | Trigger |
|---|---|---|
| C1 | If A1 ≈ 6.7: build 2-graph past-KV bin (prefill CL=1152 + decode) to isolate graph-count from switching; A7 covers the runtime side | A1 slow |
| C2 | If A1 ≈ 11.7: close the "build gap" question as qh-artifact; re-test W8 head *cleanly* (2-graph qh bin) only if B lands — head is 6.9% of post-B compute, and the LADE acceptance regression (−14%) still stands | A1 fast |
| C3 | Hybrid prefill for basic-mode products: bertcache CL=128 prefill + past-KV prefill + decode in one bin → 40 ms TTFT + full context. **Never with lade** (AR==CL rule). | after B |

### D — Speculative decoding maximization (after B picks the winner)

| # | Task | Detail |
|---|---|---|
| D1 | Re-baseline LADE on the fastest post-B build | if the 1.68× effective LADE multiplier holds on a ~16 tok/s base → **~20+ tok/s** on favorable prompts |
| D2 | LADE parameter sweep on the A6 prompt map | guardrail `(ngram−1)×(window+gcap) ≤ 32`; current 2×16 is at the limit; try ngram=2 (deeper window/gcap), window↔gcap trades. Optimize *acceptance*, not call latency. |
| D3 | Verify-graph AR study | verify16 (cheaper calls, tighter guardrail) vs verify64 (deeper speculation, more spill — verify32 already moves 745 MB) |
| D4 | Learned draft (eaglet/spd); CPU-draft hypothesis | desk task first: read `Genie/tutorials/dialog/*/kvshare/` + spd docs to test whether a `QnnGenAiTransformer` (CPU, GGUF — we already have a validated 15-s composer path) secondary engine can draft for an HTP primary. **Unverified hypothesis** — 1 h reading before any build. |

### E — Config & compression micro-levers (cheap, batched into any session)

| # | Task | Detail |
|---|---|---|
| E1 | DLBC build A/B: `"dlbc": 1` | `htpDlbc=0` in all shipped graphs today. Plain DLBC (activations) is weight-sharing-compatible; `dlbc_weights` is NOT (SDK ≥2.36). Value shrinks post-B; ctx-bin rebuild only. |
| E2 | Config-key audit | verify `graph_configs_extra.sparse_weights_compression` and `memory.extended_udma` are actually consumed (unknown keys are silently ignored — the same failure class as the graph-names trap). Check info.json / SDK docs. |
| E3 | Adopt `soc_model: 72` as default if A4 ≥ neutral | supplier ships it; genericized `0` is our current default |
| E4 | (speculative) per-graph split bins without weight sharing to unlock `dlbc_weights` on the decode graph's 751 MB | 2× disk, INT8 may compress poorly; only if B lands and we're bandwidth-limited |

### F — Context-length products

KV traffic, attention GEMV (44.5% of post-B compute), and replication (pre-B) all scale with CL.
If product allows: CL=512 or 768 decode/verify variants alongside CL=1152. Post-B this is the
next-largest structural lever on the attention side. Requires new conversions (same lineage rule).
Corroboration: the supplier ships task-sized contexts in production (CL=128 NLU, CL=256 ASR).

---

## 4. Device session batching

| Session | Contents | Prereq |
|---|---|---|
| **A** (~2 h) | A1–A7 (all bundles already on HF) + E1 bin if ready | P0.1 fix uploaded first |
| **B** (~1 h) | B7 decode-only cycle profile + full-bundle tok/s; C1/C2 follow-up arm | B5 gates green |
| **C** (~2 h) | D1–D3 sweep on the winner; F variants if built | Session B result |

## 5. Projection ladder (explicit uncertainty)

| Milestone | AR-1 | LADE (favorable prompts) |
|---|---|---|
| Today | 11.72 | 10.8 |
| After A (best existing config) | 11.7–12.5 | ~11–13 (if A2/A3 help) |
| After B (GQA fix) | **12–19** (central 15–18) | — |
| After D (LADE retuned on B) | — | **16–26** |
| Hard ceiling (weights only, 751 MB @ 16 GB/s) | ~21 AR-1 | ~35 @ 1.68× |

The single decisive experiment is **B7**. Everything else is ±10–20% trimming.

## 6. Risk register

| Risk | Mitigation |
|---|---|
| B changes tensor names → encodings drift | weight names unchanged; adopt-encodings path; parity gates are the backstop |
| Rank-5 / broadcast MatMul unsupported on HTP | rank-3 batched fallback (B3); converter recon first (B2) |
| Genie graph contract violated by topology edit | KV I/O shapes frozen by design; read `NOTES-genie-io.md` before touching export (mandatory) |
| Prefill all-position-logits contract | unchanged by B — verify in gates anyway |
| Graph-name trap on any rebuild | convert straight to final filename; `qnn-context-binary-utility` check is in every build script |
| Disk (vhdx → C:) | `disk_guard` sized per step; 20 GB for exports |
| qh conclusions overturned by A1 | qh stays parked either way until a clean 2-graph A/B (C2) |
| Report's numbers reused before correction | P0.5 gates doc edits on B7 confirmation |

## 7. Open questions ledger

1. Does plain `ladekv` Basic run at ~11.7 or ~6.7? (A1 — everything in §2.2 hangs on this)
2. Is there a converter-level GQA matcher in 2.48? (B2)
3. Device team's 7.79 vs our 11.72 — protocol or build? (P0.4)
4. Is `sparse_weights_compression` actually applied? (E2)
5. Can Genie pair a CPU draft engine with an HTP target for spec-decode? (D4)
6. Why did the 2026-08-10 two-graph bins dedupe despite mixed export paths? (inherited, still open)
7. Are the supplier's per-graph `.bin`s HTP ctx-bins? `qnn-context-binary-utility` on the actual files
   when a package lands — 10-second definitive test (validated: accepts real ctx-bins, rejects GGUF/DLC/truncated).
   Every supplier analysis so far asserts composer; our SDK experiments refute the load-bearing evidence.
   Guard against the ASR doc's §10 recommendation to adopt composer — it would move LLM inference to CPU.
