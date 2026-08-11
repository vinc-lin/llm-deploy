# Qwen3-0.6B W8A16 — v2 Bundle Refresh: LADE vs Baseline Test Report

> **Test date:** 2026-08-10 (bundles re-uploaded ~17:00 UTC that day)
> **Source:** HF `vinccniv` W8A16 device bundles
> **Reconstructed from** the three screen photographs in `reports/IMG_3004..IMG_3006.HEIC`.
> Content is transcribed verbatim where legible; derived figures are marked as such.
> Supersedes the failure analysis in
> [`qwen3-0.6b-w8a16-fuseqkvgu-test-report.md`](qwen3-0.6b-w8a16-fuseqkvgu-test-report.md) (v1 bundles).

---

## 1. Headline Finding

**The v2 rebuild works.** ✅

The author uploaded rebuilt bundles around 17:00 UTC on 2026-08-10. Both the updated baseline
and the LADE bundle **in basic mode** produce correct, coherent output. The garbage-output issue
that made the v1 bundles unusable is fixed.

Two qualifiers on that headline:

1. **LADE's own mode is still broken.** With `dialog.type="lade"` the run crashes with SIGSEGV
   inside `libGenie.so` on the first inference step. Only `type:"basic"` works, so the LADE
   bundle currently delivers **zero speculative-decoding speedup** — it is byte-for-byte a
   baseline model in practice.
2. **The two ctx-bins are effectively the same model.** Performance and output are identical
   within noise; the 12 MB size delta is metadata/encoding, not a model difference.

---

## 2. KPI Comparison — LADE bundle (basic mode) vs New Baseline

| Metric | LADE bundle (`genie_dialog_basic.json`) | New Baseline (`genie_dialog.json`) |
|---|---|---|
| File tested | `qwen3_06b_w8a16_lade.tar.gz` (889 MB) | `qwen3_06b_w8a16_local.tar.gz` (885 MB) |
| Ctx-bin | `qwen3-0.6b-w8a16-lade_ctx.bin` (1.10 GB, 1,099,091,968 B) | `qwen3-0.6b-w8a16_ctx.bin` (1.09 GB, 1,087,074,304 B) |
| Model init | 422 ms | 401 ms |
| Total init (dialog + backend) | 772 ms | 796 ms |
| Allocated RAM | 171,123,200 B (163 MB) | 171,082,240 B (163 MB) |
| Prefill throughput (prompt processing) | 265.6 tps (12 tokens in 45 ms) | 266.5 tps (12 tokens in 45 ms) |
| Chunked-prefill decode (first ~117 tokens, AR-128 graph, 1 tok/step) | ~42 ms/step = **~23.8 tok/s** | ~42 ms/step = **~23.8 tok/s** |
| True decode (AR-1 graph, post-KV > 128) | ~155 ms/step = **~6.5 tok/s** | ~155 ms/step = **~6.5 tok/s** |
| Total tokens in 30 s run | ~272 (117 prefill-mode + 155 decode) | ~271 (117 prefill-mode + 154 decode) |
| Sampler | Greedy: seed=42, temp=0.0, top-k=1, top-p=1.0 | Same |
| Chat template applied | No (raw prompt) | No (raw prompt) |
| Coherence — "2+2" prompt | ✅ `"2+2=4."` (then rep loop) | ✅ `"2+2=4."` (then rep loop) |
| Coherence — "capital of France" | ✅ `"The capital of France is Paris."` | ✅ `"The capital of France is Paris."` |
| Errors / warnings | None (in basic mode) | None |

**Conclusion (as recorded):** the two ctx-bins produce **essentially identical performance and
output** — they are effectively the same model. The 12 MB size difference is likely minor
metadata/encoding, not model changes.

### Derived figures (computed from the table above, not printed on screen)

| Derived metric | Value |
|---|---|
| Ctx-bin size delta (LADE − baseline) | 12,017,664 B ≈ 11.5 MiB / 12.0 MB |
| Allocated-RAM delta | 40,960 B (40 KiB) — noise |
| Tarball delta | 4 MB |
| Blended rate over the 30 s run | ~9.1 tok/s (272 tokens / 30 s) |
| Two-phase split | ~43 % of the window in fast prefill-mode, ~57 % in slow true decode |

> **Reading the two decode rates:** Genie stays on the AR-128 prefill graph, emitting one token
> per step at ~42 ms, until the KV cache passes 128 positions (~117 tokens here). It then switches
> to the AR-1 decode graph, and per-step cost jumps ~3.7× to ~155 ms. Any tok/s number for this
> bundle is meaningless without stating which phase it refers to.

---

## 3. LADE-Specific Mode (`genie_dialog.json`, `dialog.type="lade"`)

❌ **Crashes with SIGSEGV inside `libGenie.so` on the first inference step** (signal 11, `SEGV_MAPERR`).

| Field | Value |
|---|---|
| Fault address | `0xb4000072e7eec6a0` |
| Faulting module | `libGenie.so` @ pc `0x4c2d58` |
| Dialog config | `"type":"lade"`, `window:8`, `ngram:3`, `gcap:8`, `update-mode:"ALWAYS_FWD_ONE"` |
| Same ctx-bin with `"type":"basic"` | Works fine (`genie_dialog_basic.json`) |

**Assessment recorded on screen:** this is a **libGenie/Genie configuration bug, not a model
problem**. LADE (speculative n-gram lookahead) appears to require additional model artifacts —
a draft head / verifier — that aren't bundled, or there is an ABI mismatch between this
`libGenie.so` and the LADE config path.

**Cross-check against this repo:** the LADE parameters transcribed above match
`configs/genie_dialog_qwen3_0.6b_lade.json` exactly (`window: 8`, `ngram: 3`, `gcap: 8`,
`update-mode: "ALWAYS_FWD_ONE"`), so the on-screen config is the same shape we generate locally.
One difference worth noting: the local config points at `qwen3_06b_w8a16_lade_ctx.bin`
(underscores) whereas the HF bundle ships `qwen3-0.6b-w8a16-lade_ctx.bin` (hyphens) — the two
configs are not interchangeable without editing the `ctx-bins` entry.

---

## 4. Old (v1, broken) Baseline vs New (v2, working) Baseline

| Aspect | Old (v1, 1.3 GB tar, 1.52 GB ctx-bin) | New (v2, 885 MB tar, 1.09 GB ctx-bin) |
|---|---|---|
| Output coherence | ❌ Garbage (Hebrew/CJK/English noise, rep loop) | ✅ Correct answers (`"4"`, `"Paris"`) |
| Allocated RAM | 132 MB | 171 MB (+39 MB, likely extra buffers/caches) |
| Ctx-bin size | 1.52 GB | 1.09 GB (~400 MB smaller) |
| Genie version | Same (1.19.0) | Same (1.19.0) |
| Config JSON | Identical structure | Identical structure |

### Root-cause reading

> The fix is in the **quantization / ctx-bin generation pipeline, not the config**. The smaller
> ctx-bin and larger RAM allocation suggest the author changed how weights are laid out/encoded
> — possibly corrected the per-channel axis alignment issue hypothesized earlier, or fixed a
> weight scaling/encoding mismatch that caused garbage output while appearing to load successfully.

This is consistent with the v1 diagnosis in the earlier report: the v1 bundle loaded cleanly,
switched graphs correctly and tore down without crashing, yet emitted garbage from the very first
token — a symptom of bad weight encoding rather than a runtime or configuration fault. Genie
version and config JSON structure are unchanged across v1 → v2, which rules both out as the fix.

---

## 5. What Was Correct About the Config All Along

Every one of these settings was validated as correct and needed no change across the rebuild:

- ✅ `kv-dim=128`, `rope-dim=64`, `rope-theta=1000000`
- ✅ 2-graph ctx-bin (prefill AR-128 CL-128 + decode AR-1 CL-1152) with graph switching
- ✅ `vtcm_mb=16`, unsigned PD, `perf_profile=llm_decode_burst`, `O=3`, `hvx_threads=4`
- ✅ `n-threads=3`, `cpu-mask=0xe0`, `mmap=true`, `mmap-budget=25`, `poll=true`
- ✅ Flat layout, `LD_LIBRARY_PATH=.`

---

## 6. Status Summary

| Item | Status |
|---|---|
| v2 baseline bundle loads and runs | ✅ |
| v2 baseline output correctness | ✅ Correct (`"4"`, `"Paris"`) |
| v2 LADE bundle in **basic** mode | ✅ Works, identical to baseline |
| v2 LADE bundle in **lade** mode | ❌ SIGSEGV in `libGenie.so` on first inference step |
| Speculative-decoding speedup realized | ❌ None — LADE path unusable |
| Prefill-phase throughput | ✅ ~265–267 tps prompt processing; ~23.8 tok/s emitting |
| True decode throughput | ⚠️ ~6.5 tok/s (~155 ms/step) — unchanged from v1's ~6.3 tok/s |
| Init time | ✅ ~0.8 s total (dialog + backend) |
| Memory footprint | ✅ 163 MB allocated, 1.09 GB ctx-bin |
| Repetition loop after correct answer | ⚠️ Present in both bundles — expected for the base (non-Instruct) model with no chat template |

---

## 7. Open Questions / Next Steps

1. **LADE SIGSEGV** — determine whether `libGenie.so` 1.19.0 requires a draft-head/verifier
   artifact that the bundle omits, or whether the LADE config path is ABI-incompatible with this
   build. Until resolved, the LADE bundle has no advantage over the baseline and the 4 MB / 12 MB
   size premium buys nothing.
2. **Decode throughput is the frontier, and it did not move.** The correctness fix left true
   decode at ~6.5 tok/s. The ~23.8 tok/s seen in the first ~117 tokens is the AR-128 prefill graph
   emitting one token per step — it is not a sustainable rate and it ends when KV passes 128.
3. **Repetition loop** — both bundles loop after answering. Confirm whether applying the Qwen3
   chat template and/or moving to the Instruct checkpoint cleans this up, since the bundles use the
   base model with raw prompts.
4. **Reproduce v2 locally** — the fix is in the quantization/ctx-bin generation pipeline. The
   ~400 MB ctx-bin shrink plus +39 MB runtime RAM is the signature to match when re-running our own
   export/quantize path.

---

## Transcription notes

The source images are photographs of a laptop screen with glare and reflection. Readings that
warrant caution:

1. **Image overlap.** `IMG_3004` → `IMG_3005` → `IMG_3006` are consecutive scroll captures of one
   continuous session, with substantial overlap. The Sampler row, the LADE crash bullets and the
   v1-vs-v2 table each appear in two images; readings were cross-checked between them.
2. **Duplicated bullets in `IMG_3005`.** The LADE crash bullet list appears twice within the same
   screenshot (once wrapped narrow, once wide). This is a terminal re-wrap/redraw artifact during
   scrolling, not two separate crash reports.
3. **Fault address `0xb4000072e7eec6a0` and pc `0x4c2d58`** — read from a single unmagnified line;
   digit-exact reading was verified at full resolution but no second capture exists to confirm.
4. **`~400 MB smaller`** is quoted as printed. The stated sizes (1.52 GB → 1.09 GB) imply ~430 MB.
5. **Byte counts** (`1,099,091,968`, `1,087,074,304`, `171,123,200`, `171,082,240`) were verified by
   cropping the originals at full resolution.

### Session footer from the capture

```
* Brewed for 1h 37m 52s
/m/c/s/bundles/hf_vinccniv master  ctx:1%/200k  cost:$0.028  [complexity-router|default|low]
```
