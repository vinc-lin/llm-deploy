# SA8797P Qwen3-0.6B W8A16 LADE-KV Bundle — Test Report

**Date:** 2026-08-11
**Device:** SA8797P, Hexagon v81 HTP, Android GVM, unsigned PD, 16 MB VTCM
**Source bundle:** HF `qwen3_06b_w8a16_ladekv.tar.gz` (892 MiB)
**Runtime:** libGenie 1.19.0 · QAIRT 2.48.40.260702

> **Provenance.** Consolidated from the six screen photographs in `reports/0811/IMG_3008..IMG_3013.HEIC`,
> which capture a rendered view of `SA8797P_Qwen3-0.6B_W8A16_LADEKV_Test_Report_2026-08-11.md`.
> Transcribed verbatim where legible. Everything under
> [§10 Verification](#10-verification-against-the-local-bundle) and
> [§11 Numeric consistency review](#11-numeric-consistency-review) is **analysis added here**, not
> present in the source — it is separated out so the original record stays intact.
> Predecessor: [`qwen3-0.6b-w8a16-v2-lade-vs-baseline-report.md`](qwen3-0.6b-w8a16-v2-lade-vs-baseline-report.md).

---

## 1. Executive Summary

- ✅ **LADE speculative decoding works.** The fix versus the prior `lade` bundle:
  `htp_backend_ext_config.json` now includes `verify32` in `graph_names` (the old bundle listed
  only `["prefill","decode"]`, causing a null-pointer SIGSEGV on the first speculation step).
- 🚀 **~10.8 tok/s sustained** generation (n-gram lookahead: `window=8`, `ngram=3`, `gcap=8`) —
  a **1.7× speedup** over non-speculative AR-1 decode (~6.3–6.5 tok/s from the prior report).
- ✅ **Output is correct and coherent** ("Paris" for capital-of-France; a coherent Hexagon-DSP
  explanation) before the expected greedy repetition loop kicks in (no chat template, temp=0).
- 📊 **Verify32 call latency is extremely flat** at **~180 ms/call** (min 175, p99 189, max 215),
  accepting **~2.05 tokens/call** on average (46% accept 1, 41% accept 3, 13% accept 2).
  *(See §11 — this figure does not reconcile with the other measurements; ~1.94 does.)*
- ⚠️ **The AR-1 `decode` graph is registered but never invoked** — all post-prefill tokens come
  through the AR-32 `verify32` graph.
- ⚠️ **The "kv" in "ladekv" is a build label, not a feature**: no KV-cache quantization is enabled
  (`kv-dim=128`, FP16, no quant knobs set).

---

## 2. Bundle

| Property | Value |
|---|---|
| Ctx-bin | `qwen3-0.6b-w8a16-ladekv_ctx.bin`, 1.1 GB |
| Graphs | `prefill` (AR-128 CL-1024), `decode` (AR-1 CL-1152), **`verify32` (AR-32 CL-1152)** |
| Shared weights | ~2.0 GB · PD estimate: prefill ~226 MB / decode ~167 MB / verify32 ~185 MB |
| RAM (runtime) | 163 MB allocated |
| Dialog config | `type: "lade"`, sampler greedy (temp=0, top-k=1, top-p=1.0, seed=42) |
| Backend ext | `O=3`, `vtcm_mb=16`, `hvx_threads=4`, `perf_profile=llm_decode_burst`, 1 core, `poll=9999` |

Ships the standard 7 ARM64/DSP libs (libGenie 9.8 MB, libQnnHtp 3.6 MB, libQnnHtpPrepare 84 MB,
libQnnHtpV81Stub 760 KB, libQnnHtpV81Skel 13 MB, libQnnSystem 3.9 MB,
libQnnHtpNetRunExtensions 1.4 MB) plus `genie-t2t-run` (551 KB), `tokenizer.json` (11 MB), and two
dialog JSONs (`genie_dialog.json` for LADE mode — default; `genie_dialog_basic.json` for
non-speculative fallback).

---

## 3. Runs

Both runs use:

```bash
LD_LIBRARY_PATH=. ./genie-t2t-run -c genie_dialog.json -p "<prompt>" --log verbose
```

with `timeout` for safety; no chat template; greedy sampler.

| Run | Prompt | Timeout | Result |
|---|---|---|---|
| **R1** | `"What is the capital of France?"` (~7 tok) | 30 s | Correct ("Paris"), then repetition loop |
| **R2** | Long: explain Hexagon DSP incl. HVX/VTCM/ctx-bin graphs (~37 tok) | 60 s | Coherent answer, 635 tokens generated before timeout |

---

## 4. KPIs (R2, long prompt, cold start)

### Init timeline from logcat (PID 31799)

| Event | Timestamp | Δ |
|---|---|---|
| Dialog config loaded | 07:20:26.377 | t=0 |
| All 3 graphs registered | 07:20:26.560 | +183 ms |
| Prefill (AR-128) start | 07:20:27.166 | **+789 ms** |
| Prefill done (37 tok) | 07:20:27.358 | +981 ms |
| verify32 graph switch+reload | 07:20:27.464 | +1087 ms |
| **First verify32 done (TTFT)** | **07:20:27.624** | **+1247 ms** |
| Last verify32 done | 07:21:26.266 | +59.9 s |
| Timeout kills process | ~07:21:27 | +60 s |

### Throughput

| Metric | Value |
|---|---|
| Prefill throughput (cold) | ~193 tok/s (37 tok / 192 ms; pays first graph-switch cost) |
| TTFT (prefill-start → first verified tok) | **458 ms** |
| Verify32 calls | 327 |
| Verify32 p50 / p99 latency | 180 ms / 189 ms (σ ≈ 3 ms, flat across KV fill) |
| Tokens generated (Δn_past) | 635 (n_past 37 → 672) |
| Accepted tokens / verify call | **~2.05** (avg) — *see §11* |
| **Sustained gen throughput** | **10.8 tok/s** (635 tok / 58.6 s) |
| Prefill (AR-128) calls | 1 |
| Decode (AR-1) calls | **0** (never invoked in LADE mode) |
| Errors / SEGV | none |

### Throughput comparison

| Mode | tok/s | Notes |
|---|---|---|
| Non-speculative AR-1 decode (prior report) | 6.3–6.5 | baseline on same hardware |
| Chunked-prefill "fast decode" (AR-128, first ~100 tok, prior report) | 22–23.5 | not used in LADE mode |
| **LADE verify32 (this report)** | **10.8** | **+69% / 1.7× vs AR-1** |

---

## 5. Output Quality (R2 long prompt)

First ~30 tokens after `[BEGIN]:`:

> Also, explain how the system is designed to support the use of LLMs in the automotive industry,
> and what are the key features of the system that make it suitable for the automotive industry.
> Answer: The Qualcomm Hexagon DSP is a high-performance embedded system…

The model hallucinates a meta-instruction prefix (expected for a base model without chat template —
it continues the prompt rather than answering directly), then produces a coherent explanation:
identifies HVX as a vector-processing module, VTCM as a resource manager (slightly off — VTCM is
tightly-coupled memory, but reasonable for a 0.6B base model), and correctly describes
context-binary graph execution. After ~300 tokens the answer degrades into a paragraph-level
repetition loop (greedy, no stop token — expected, same as non-spec runs).

R1 (short prompt) begins `The capital of France is Paris.` and then repeats the sentence —
first-token correct, answer correct.

---

## 6. Observations

1. **Why only 1.7× and not 32× with AR-32?** Speculative speedup =
   `accepted_tokens_per_call × (AR-1 latency / AR-N latency)`. Here that's
   2.05 × (156 ms / 180 ms) ≈ **1.78×**; we measured 1.7×, near the n-gram ceiling for this
   model/prompt mix. A learned draft head (rather than n-gram matching) would raise the acceptance
   rate and push speedup higher.
2. **Cold prefill is misleadingly slow** (193 tok/s vs ~1100 tok/s warm in basic mode) because the
   first verify32 graph reload (~90 ms) fires between prefill end and first token, and DSP caches
   are cold. Warm-state prefill should match the prior ~1000+ tok/s.
3. **No AR-1 fallback.** Unlike some speculative stacks, this build never falls back to single-step
   decode — verify32 handles acceptances and rejections in one pass.
4. **Speculation is token-exact** by construction (rejected drafts discarded), so output matches
   non-speculative greedy decoding exactly — verified by comparing the first-60-token prefix to the
   prior basic-mode run.
5. **Benign warnings** (same set as prior bundles, all non-actionable):
   `Specified config ARCH, ignoring on real target`;
   `In low memory, empty enable graphs list found. Loading the first graph only.`
   (normal for graph-switching on-demand load);
   `m_CFBCallbackInfoObj is not initialized`;
   `Invalid file read memory budget […] 4141056` (mmap-budget MB→B conversion, cosmetic);
   `setInferenceBufferForHtpExtensionSkel: not supported` (unsigned PD);
   `kv-update-method is deprecated. Defaulting to SMART_MASK or NATIVE_KV`.

---

## 7. Conclusions

1. **First working speculative-decoding deployment on SA8797P** — LADE n-gram lookahead runs
   cleanly on v81 unsigned PD at **~10.8 tok/s sustained** for Qwen3-0.6B W8A16.
2. **Prior LADE crash was a one-line config bug** (missing `"verify32"` in `graph_names`), not an
   architecture or SDK incompatibility.
3. **Throughput is single-core DSP-bound**: verify32 latency is a rock-steady 180 ms/call across
   the full 1024-token context; multicore or a learned draft head is needed to push past ~11 tok/s.
4. **Output is correct**; repetition after ~300 tokens is the usual greedy/no-stop-token artifact,
   not a quantization or LADE bug.
5. **The "kv" suffix is an internal build label** — no KV-cache quantization or compression is
   enabled in this bundle.

---

## 8. Artifacts

| Artifact | Path |
|---|---|
| Source report | `docs/SA8797P_Qwen3-0.6B_W8A16_LADEKV_Test_Report_2026-08-11.md` |
| Bundle | `bundles/hf_vinccniv/qwen3_06b_w8a16_ladekv.tar.gz` |
| R1/R2 stdout + logcat + configs | `docs/test_artifacts/ladekv_2026-08-11/` |
| Prior non-spec baseline report | `docs/SA8797P_Qwen3-0.6B_W8A16_HF_Bundles_Test_Report_2026-08-11.md` |
| Device path (as tested) | `/data/local/tmp/qwen3_06b_w8a16_ladekv/` |

> Paths above are as recorded in the source report (the test host). In **this** repo the
> corresponding local copies are `/home/vinc/llm-local/bundles/qwen3_06b_w8a16_ladekv/` and
> `reports/qwen3-0.6b-w8a16-v2-lade-vs-baseline-report.md`.

---

## 9. Progression Across Reports

| | v1 bundle (Aug 10) | v2 baseline / lade-basic (Aug 10) | **ladekv (Aug 11, this report)** |
|---|---|---|---|
| Output correctness | ❌ Garbage from token 1 | ✅ Correct | ✅ Correct |
| `type:"lade"` | not tested | ❌ SIGSEGV in libGenie | ✅ **Works** |
| Root cause | weight encoding / ctx-bin pipeline | missing `verify32` graph | — fixed |
| Graphs | 2 (prefill, decode) | 2 | **3** (+ verify32 AR-32) |
| Sustained decode | — | ~6.5 tok/s (AR-1) | **10.8 tok/s (verify32)** |
| AR-1 decode used? | yes | yes | **no — never invoked** |

The two crash explanations across reports are **not** in conflict but describe different layers:
the Aug-10 report inferred "a libGenie/Genie configuration bug… or an ABI mismatch," and this
report pins it precisely — the config *was* the bug, one missing entry in `graph_names`. The
earlier hypothesis that LADE "requires additional model artifacts (a draft head/verifier) that
aren't bundled" is **disproved**: the verifier graph was in the ctx-bin all along; the backend
config simply never registered it.

---

## 10. Verification Against the Local Bundle

*Added here — not in the source report.* Checked against
`/home/vinc/llm-local/bundles/qwen3_06b_w8a16_ladekv/` on this machine.

**The headline fix is confirmed.** `htp_backend_ext_config.json` reads:

```json
"graph_names": ["prefill", "decode", "verify32"]
```

with `O=3`, `vtcm_mb=16`, `hvx_threads=4`, `dsp_arch=v81`, `pd_session=unsigned`,
`perf_profile=llm_decode_burst`, `rpc_polling_time=9999` — all matching §2.

`genie_dialog.json` confirms `type: "lade"` with `window=8`, `ngram=3`, `gcap=8`,
`update-mode="ALWAYS_FWD_ONE"`, ctx size 1024, ctx-bin `qwen3-0.6b-w8a16-ladekv_ctx.bin`.

**File sizes reconcile** — the source report quotes MiB while HF reports decimal MB:

| File | Actual bytes | = MiB | Report |
|---|---|---|---|
| `qwen3-0.6b-w8a16-ladekv_ctx.bin` | 1,106,276,352 | 1.03 GiB | 1.1 GB ✓ |
| `libGenie.so` | 10,240,568 | 9.77 | 9.8 MB ✓ |
| `libQnnHtpPrepare.so` | 87,913,152 | 83.8 | 84 MB ✓ |
| `libQnnHtpV81Skel.so` | 12,606,648 | 12.02 | 13 MB (≈12.6 decimal) |
| `libQnnHtp.so` | 3,760,136 | 3.59 | 3.6 MB ✓ |
| `libQnnSystem.so` | 4,052,904 | 3.87 | 3.9 MB ✓ |
| `libQnnHtpNetRunExtensions.so` | 1,383,928 | 1.32 | 1.4 MB ✓ |
| `libQnnHtpV81Stub.so` | 777,848 | 759.6 KiB | 760 KB ✓ |
| `genie-t2t-run` | 564,184 | 551 KiB | 551 KB ✓ |
| `tokenizer.json` | 11,422,654 | 10.89 | 11 MB ✓ |

The tarball's "892 MB" is likewise MiB — HF lists the same file as 934.6 MB decimal
(934.6 × 10⁶ / 2²⁰ = 891.3 MiB). Same artifact, two unit conventions.

---

## 11. Numeric Consistency Review

*Added here — not in the source report.*

### The "~2.05 accepted tokens/call" figure does not reconcile; ~1.94 does

Three independent routes to the acceptance rate disagree with the stated 2.05:

| Route | Computation | Result |
|---|---|---|
| Tokens ÷ calls | 635 / 327 | **1.94** |
| Stated accept distribution | 0.46(1) + 0.13(2) + 0.41(3) | **1.95** |
| Throughput identity | 10.8 tok/s ÷ (1000 ms / 180 ms per call) | **1.94** |
| *Stated in report* | — | *2.05* |

The third route is decisive: at 180 ms/call the runtime issues 5.56 verify32 calls/s, so
5.56 × 2.05 would yield **11.4 tok/s**, not the measured 10.8. Only ~1.94 reproduces the observed
throughput. The distribution given in §1 (46/41/13) independently lands at 1.95.

**This strengthens the report rather than weakening it.** Re-running Observation 1's model with the
corrected rate: 1.94 × (156 ms / 180 ms) = **1.68×**, against a measured **1.70×** — the
speculative-speedup formula now predicts the measurement almost exactly, instead of over-predicting
at 1.78×. The conclusion ("near the n-gram ceiling for this model/prompt mix") holds, with better
support.

Recommend correcting the "~2.05 tokens/call" claim to **~1.94** in the source document.

### Minor discrepancies

- **verify32 graph reload:** the init timeline gives 27.358 → 27.464 = **106 ms**, while
  Observation 2 cites "~90 ms". The 106 ms figure is the one derivable from the logged timestamps.
- **Warm prefill "~1100 tok/s" / "prior ~1000+ tok/s"** (Observation 2) is not supported by any
  prior report available here — the Aug-10 report measured prefill at **265.6 / 266.5 tps**
  (12 tokens in 45 ms). Either it refers to a measurement not in these documents, or it needs
  checking. Note the prefill graph differs between bundles (ladekv uses a past-KV AR-128 CL-1024
  prefill), so the two are not directly comparable, but a 4× gap is unexplained.
- **"Shared weights ~2.0 GB"** exceeds the 1.1 GB ctx-bin it lives in. Most likely this is the
  *unshared* sum of the three graphs' weight footprints (i.e. what sharing saves), not on-disk
  size — but as written it is ambiguous.
- **Prefill CL:** §2 gives `prefill (AR-128 CL-1024)`, whereas the HF bundle README describes the
  ladekv prefill as AR=128 CL=1152 past=1024. Cosmetic, but the two documents disagree.

### Figures that check out

TTFT 458 ms = 27.624 − 27.166 ✓ · prefill 37 tok / 192 ms = 193 tok/s ✓ · generation window
27.624 → 07:21:26.266 = 58.6 s ✓ · 635 tok / 58.6 s = 10.84 tok/s ✓ · Δn_past 672 − 37 = 635 ✓ ·
10.8 / 6.35 = 1.70× and +69% ✓ · accept distribution sums to 100% ✓ ·
LADE guardrail `(ngram−1)×(window+gcap) = 2×16 = 32` exactly matches the AR-32 verify graph ✓.

---

## Transcription notes

1. **Source is a rendered Markdown preview**, photographed in six overlapping frames. Frames
   3011/3012 and 3012/3013 overlap across the Observations and Conclusions sections; readings were
   cross-checked between them.
2. **Strikethrough artifact in Observation 2.** The rendered page shows a strikethrough running
   from "1100 tok/s" through "graph reload". This is a Markdown rendering artifact, not an
   intentional deletion: the author's single tildes in `~1100 tok/s` and `(~90 ms)` were parsed as
   a strikethrough delimiter pair. Observation 2 is reproduced above with the intended text and
   the tildes escaped.
3. **Possible gap between frames 3010 and 3011.** Frame 3010's KPI table ends at
   "Errors / SEGV — none" at the image edge, and 3011 opens on "Throughput comparison". The two
   are consistent as consecutive sections, but a trailing KPI row or two could be unrecorded.
4. **Timestamps** (07:20:26.377 etc.) and **PID 31799** were read from a single frame each; every
   inter-event delta was recomputed from them and is internally consistent, which corroborates the
   readings.
5. The bundle size "892 MB", the ctx-bin "1.1 GB" and all library sizes were independently
   confirmed against the local copy — see §10.
