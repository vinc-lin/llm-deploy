# SA8797P Qwen3-0.6B W8A16QH LADEKV — On-Device Test Report

**Date:** 2026-08-12
**Device:** SA8797P (nordy / Gen5, Hexagon v81 unsigned PD, Android GVM)
**Runtime:** QAIRT 2.48.40.260702, libGenie 1.19.0
**Bundle:** `vinccniv/sa8797p-qwen3-w8a16-bundles/qwen3_06b_w8a16qh_ladekv.tar.gz`

> **Provenance.** Consolidated from the six screen photographs formerly at
> `reports/0812/IMG_3024..IMG_3029.HEIC` (deleted 2026-08-12 once transcribed —
> this document is now the record), which captured a rendered view of
> `SA8797P_Qwen3-0.6B_W8A16QH_LADEKV_Test_Report_*.md`. Transcribed verbatim where legible.
> The only material **not** in the source is [§10 Verification](#10-verification-lm_head-is-int8-in-the-dlcs-but-the-ctx-bin-did-not-shrink)
> and the [transcription notes](#transcription-notes), both clearly marked, so the original
> record stays intact.
> Baseline for every comparison below: [`qwen3-0.6b-w8a16-ladekv-test-report.md`](qwen3-0.6b-w8a16-ladekv-test-report.md).

---

## 1. Executive Summary

`qh` = **W8 `lm_head`** (INT8-quantized language-model head, previously FP16). Author projection:
**~+19%** decode tok/s vs baseline ladekv. **Actual result: −14%.**

| Metric | qh-ladekv (measured) | ladekv baseline | Δ |
|---|---|---|---|
| Sustained tok/s (LADE) | **9.3** | 10.8 | **−14%** |
| Tokens accepted / verify call | 1.74 | 2.05 | −15% |
| Steady verify latency | 187 ms | 180 ms | +4% |
| Cold init → first logits | 1070 ms | ~458 ms | +134% |
| VTCM spill | 0 B | 0 B | — |
| Errors | none | none | — |

**Why it's slower:** the `lm_head` INT8 quant reduces per-call DDR by ~155 MB/tok (as predicted),
but speculative acceptance **dropped** from 2.05 → 1.74 tok/call (−15%), which dominates the small
per-call latency gain. The quantized `lm_head` changes logit distributions enough to nudge the
n-gram match rate down, and the spec-decode ceiling depends more on acceptance rate than on
per-call latency.

Output quality is identical to ladekv (greedy "Paris" for the short prompt, coherent structured
Hexagon explanation for the long prompt, with repetition patterns).

---

## 2. Test Environment

| | |
|---|---|
| SoC | SA8797P (Snapdragon Ride Gen5, nordy) |
| DSP | Hexagon v81 HTP, 1 core, unsigned PD |
| VTCM | 16 MB (unsigned PD ceiling) |
| Runtime | libGenie 1.19.0 / QNN 2.37.0 / QAIRT 2.48.40 |
| Mode | LADE speculative decoding (`type:"lade"`, lhd-dec) |
| Configs | `window=8`, `ngram=3`, `gcap=8`, `ALWAYS_FWD_ONE`, greedy (temp=0, top-k=1) |
| Graphs | prefill AR-128 + decode AR-1 + verify32 AR-32, CL-1152 |
| Graph switching | enabled, `use-mmap=true`, `mmap-budget=25` |
| Perf | `O=3`, `hvx_threads=4`, `llm_decode_burst`, `rpc_polling_time=9999` |
| CPU | `n-threads=3`, `cpumask=0xe0` |
| Device path | `/data/local/tmp/qwen3_06b_w8a16qh_ladekv/` |

---

## 3. Bundle Description

Ctx-bin: **`qwen3-0.6b-w8a16qh-ladekv_ctx.bin`** (1.09 GB, 3 graphs).

Configs identical to `qwen3_06b_w8a16_ladekv` except the ctx-bin filename. The "qh" difference is
**inside the weights**: `lm_head` is per-channel INT8 (was FP16 in baseline ladekv). Expected to
save ~155 MB/token DDR on the output projection → ~+19% tok/s in AR-1 decode.

Flat layout, 7 required `.so` files present, `genie-t2t-run` 551 KB.

---

## 4. KPI Summary

### Init / Cold Start

| Phase | Duration |
|---|---|
| `init_start` → Graphs loaded | 2 ms |
| `init_start` → first verify execute | 812 ms |
| Cold first-verify (incl. graph switch) | 258 ms |
| **Total cold start (init → first logits)** | **1070 ms** |

Note: cold start is much higher than ladekv's ~458 ms — this is because the very first inference
step (prompt processing) runs verify32 with graph-switch overhead from prefill→verify32, and the
initial `n_process=22` prompt processing happens on verify32 (not prefill) in LHD mode.

### Throughput

*(long Hexagon prompt, 22-token prompt, 409 generated tokens in 235 verify calls over 43.8 s)*

| Metric | Value |
|---|---|
| Total verify calls | 235 |
| Total tokens generated | ~409 |
| Total duration | 43.8 s |
| **Sustained tok/s** | **9.3** |
| **Tokens per verify call** | **1.74** |
| Avg verify latency (steady) | 187 ms |
| Min / median / max latency | 181 / 187 / 217 ms |
| VTCM spill | 0 B |

---

## 5. Acceptance Distribution (`n_process`)

| `n_process` | calls |
|---|---|
| 16 | 64 |
| 18 | 23 |
| 20 | 25 |
| 22 | 20 |
| 24 | 14 |
| 26 | 15 |
| 28 | 6 |
| 30 | 3 |
| 32 | 66 |

Skewed toward lower acceptance (mode=16 at 27% of calls) vs ladekv, which had more 24–32 token
accepts. Full 32-token acceptance rate: 28% of calls.

---

## 6. Comparison Table

| | qh-ladekv | ladekv | Δ |
|---|---|---|---|
| Sustained tok/s | 9.3 | 10.8 | **−14%** |
| Tokens/verify | 1.74 | 2.05 | −15% |
| Verify latency | 187 ms | 180 ms | +4% |
| VTCM spill | 0 | 0 | — |
| `lm_head` quant | INT8 | FP16 | — |

---

## 7. Output Quality

- **Short prompt** ("capital of France?"): correct answer "Paris", followed by the expected greedy
  repetition pattern (identical to ladekv).
- **Long prompt** (Hexagon DSP explanation): structured, coherent answer with key features,
  benefits, and numbered lists. Content is reasonable and factually plausible. Same repetition
  patterns as ladekv at ~200+ tokens.

No quality degradation observed (but no improvement either — the README's "quality gate 3/4 =
baseline" is consistent with on-device observation).

---

## 8. Observations

1. **Spec-decode ceiling is acceptance-rate-bound, not latency-bound.** The W8 `lm_head` saves
   ~20 ms/call in compute (DDR-bound), but n-gram acceptance drops 15% because logit quantization
   shifts the greedy next-token distribution enough to break n-gram matches. Net effect: slower.
2. **Graph switching works correctly.** Prefill → verify32 switch took 91 ms (cold), then all
   subsequent calls run verify32 resident. No further switches observed.
3. **No VTCM spill.** Zero spill-fill across all 3 graphs. Verify32 PD footprint: 186.79 MB
   (vs decode 304 MB, prefill 223 MB).
4. **First-prompt chunk runs on verify32 (not prefill) in LHD mode.** `n_process=22` for step 0 —
   the 22-token prompt gets processed in a single verify32 call (fits within AR-32). This means the
   prefill graph may be unused in LHD mode for short prompts.
5. **No errors, no crashes.** Benign warnings only (deprecated kv-update-method, FastRPC
   non-domain, PrepareLib not loaded, audit selinux permissive).

---

## 9. Conclusions

1. **W8 `lm_head` (qh variant) is a net regression on LADE**: 9.3 vs 10.8 tok/s (−14%). The DDR
   savings are real but overwhelmed by the reduced spec-decode acceptance rate.
2. **`qh` might still help in AR-1 (basic) mode.** If acceptance rate is irrelevant, the
   ~20 ms/call (11%) latency reduction should translate directly to a tok/s gain. Worth testing
   basic mode to confirm.
3. **ladekv remains the best LADE bundle** for throughput. The "qh" tradeoff only makes sense if:
   (a) you're running in basic (non-spec) mode, or (b) you're memory-bandwidth starved at higher
   context lengths where per-token DDR dominates.
4. **Quality parity holds.** INT8 `lm_head` does not visibly degrade output quality with greedy
   sampling at 0.6B scale.
5. **The author's +19% projection was for AR-1 decode, not spec-decode.** The README correctly
   frames it as "~155 MB/token less DDR, est. +19% decode tok/s" — the "decode tok/s" refers to
   per-step decode cost, not end-to-end spec-decode throughput. The DDR savings are real;
   spec-decode just amortizes them differently.

---

## 10. Verification: `lm_head` is INT8 in the DLCs, but the ctx-bin did not shrink

> **Added here — not in the source report.** Everything above is transcribed as written. This
> section records checks run against the local build artifacts.

### The quantized head is real

All three DLCs the tested ctx-bin was built from carry a **per-channel INT8** `lm_head`
(`qairt-dlc-info`):

```
lm_head  FullyConnected
  lm_head.weight (data type: sFxp_8; tensor dimension: [151936,1024]; tensor type: STATIC)
  lm_head.weight encoding for channel_0: bitwidth 8, min -0.117441192269,
                                         max 0.116523683071, scale 0.000917509315, offset 0.0
```

| DLC | bytes | mtime | `lm_head.weight` |
|---|---|---|---|
| `qwen3-0.6b-w8a16qh-ladekv/prefill.dlc` | 922,965,680 | 08-12 18:45:26 | `sFxp_8` |
| `qwen3-0.6b-w8a16qh/decode.dlc` | 922,799,320 | 08-12 18:30:32 | `sFxp_8` |
| `qwen3-0.6b-w8a16qh/verify32.dlc` | 922,965,696 | 08-12 18:38:00 | `sFxp_8` |
| ctx-bin built from them | 1,093,767,168 | 08-12 **18:49:20** | — |

All three precede the ctx-bin, which precedes the bundle tarball (18:51). So the
`--keep-head-weight` fix was already in the build when this bundle was made; commit `f486eec`
(19:49) is the write-up an hour later, not the fix arriving after the fact. The pre-fix state is
preserved alongside as `*.dlc.fp16old` (1,074,293,920 B, 08-11 16:28), and the 08-11
`prefill_info.txt` shows what the bug looked like: `lm_head.weight (data type: Float_16)`. The
baseline ladekv `prefill.dlc` is byte-for-byte the same size as those `.fp16old` files
(1,074,293,920), confirming its head is FP16.

**§1's premise holds.** The −14% is a real measurement of a real INT8-head build.

### But ~139 MB of the saving reappears at prepare time

The DLC shrank by 151.3 MB; the ctx-bin shrank by only 12.5 MB:

| | DLC (prefill) | ctx-bin | ctx − DLC |
|---|---|---|---|
| baseline ladekv | 1,074,293,920 | 1,106,276,352 | +32.0 MB |
| qh ladekv | 922,965,680 | 1,093,767,168 | **+170.8 MB** |
| Δ | **−151.3 MB** | **−12.5 MB** | +138.8 MB |

(151.3 MB against a 155.6 MB ideal — 151936 × 1024 bytes — the rest being per-channel scales and
container overhead.)

So ~139 MB comes back when `qnn-context-binary-utility`'s prepare step turns the DLCs into a
context blob. A plausible reading is that HTP materializes the INT8 head back to 16-bit at prepare
time, since the `FullyConnected`'s input and output are both `Float_16` — in which case the
on-device weight traffic never changed, and §1's "reduces per-call DDR by ~155MB/tok (as
predicted)" is the unverified claim, not the dtype. That would also fit the measurement the report
itself records: steady verify latency went **up** 180 → 187 ms, where the projection called for
−20 ms/call.

**This is a hypothesis, not a finding** — ctx-bin dumps expose no weight tensors
(`numContextTensors: 0`), so it cannot be settled from the blob. On-device DDR counters, or a
prepare-time weight dump, would settle it.

*Also confirmed:* the ctx-bin holds exactly 3 graphs named `prefill` / `decode` / `verify32`, with
`input_ids` `[1,128]` / `[1,1]` / `[1,32]` and all-position FP16 `logits` `[1,AR,151936]` — graph
names and the all-position-logits contract both hold.

---

## 11. Artifacts

| Location | Description |
|---|---|
| `bundles/hf_vinccniv/qwen3_06b_w8a16qh_ladekv.tar.gz` | Downloaded bundle (923 MB) |
| `/data/local/tmp/qwen3_06b_w8a16qh_ladekv/` | Extracted bundle on device |
| `docs/test_artifacts/qh_ladekv_2026-08-12/genie_dialog.json` | LADE config |
| `docs/test_artifacts/qh_ladekv_2026-08-12/htp_backend_ext_config.json` | HTP backend config |
| `docs/test_artifacts/qh_ladekv_2026-08-12/short_stdout.txt` | Short-prompt output |
| `docs/test_artifacts/qh_ladekv_2026-08-12/long_stdout.txt` | Long-prompt output |
| `docs/test_artifacts/qh_ladekv_2026-08-12/long_logcat.txt.gz` | Verbose logcat (61 MB → 3.2 MB gz) |

> Paths are as recorded in the source report (the test host). In **this** repo the corresponding
> local bundle is `/home/vinc/llm-local/bundles/qwen3_06b_w8a16qh_ladekv/`; the
> `docs/test_artifacts/` tree is not present here.

---

## Transcription notes

1. **Six overlapping frames**, read in order IMG_3024 → IMG_3029. Frames overlap across
   *Test Environment*, *KPI Summary*, *Output Quality* and *Conclusions*; those sections were
   cross-checked between frames. The document ends at *Artifacts* — no content follows it in
   frame 3029.
2. **The `n_process` rows sum to 236, but total verify calls is 235.** Transcribed as shown
   (digits re-read at 2× zoom to be sure). Both stated percentages are computed against 236:
   64/236 = 27.1% and 66/236 = 28.0%.
3. **"mode=16" is not what the table shows.** `n_process=32` has 66 calls against 16's 64, so 32 is
   the most frequent bin. Transcribed as written in §5.
4. **Source filename vs date.** The editor status bar in every frame reads
   `SA8797P_Qwen3-0.6B_W8A16QH_LADEKV_Test_Report_2026-08-11.md`, while the document's own **Date**
   field and the artifact directory (`qh_ladekv_2026-08-12/`) both say 2026-08-12. The 08-11 in the
   filename appears to be a carry-over from the baseline report.
5. **§8.1 "saves ~20 ms/call" and §9.2's "~20 ms/call (11%)" are projections, not measurements** —
   the measured steady verify latency went *up*, 180 → 187 ms (§1, §6). Transcribed as written;
   see §10 for why the DDR saving may not have reached the device at all.
6. **§1's "+134% cold start" compares two different quantities.** 1070 ms is this run's
   init→first-logits; the ~458 ms it is measured against is ladekv's **TTFT**, timed from
   prefill start and excluding ~789 ms of init. Like-for-like, the ladekv init timeline gives
   1247 ms to first logits, so this build is ~177 ms *faster* to first logits. Transcribed as
   written; see `docs/REFERENCE.md` §6.1.
