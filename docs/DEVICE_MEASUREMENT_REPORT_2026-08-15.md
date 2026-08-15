# GQA-Fix Device Measurement Report — SA8797P

**Date:** 2026-08-15
**Device:** `REDACTED` (SA8797P / nordy / Gen5, Hexagon v81 HTP, unsigned PD, 16 MB VTCM)
**SDK:** QAIRT 2.48.40.260702, QNN API v2.37.0, libGenie 1.19.0
**Build source:** HF `vinccniv/sa8797p-qwen3-w8a16-bundles`, `2026-08-14-gqafix/`

> **Provenance.** Transcribed 2026-08-16 from five screen photographs of the
> device team's `SA8797P_GQAFIX_Measurement_Report_2026-08-15.md`
> (`reports/0815/IMG_3053`–`IMG_3057.HEIC`). The photographs overlap and cover
> the document end to end with no gaps. **This Markdown is the record** — per
> repo convention the source photos are deleted once it is committed.

---

## 0. Editorial annotations (added 2026-08-16 during transcription, not part of the original report)

### 0.1 The device serial is redacted

The original names the device serial in its header. It is replaced with
`REDACTED` here, exactly as was done for the 2026-08-13 report — this repo's
history has been scrubbed twice for that class of leak, and both the HF repo and
the GitHub mirror are currently **public**. Nothing else in the body has been
altered; §1–§7 below are a faithful transcription.

### 0.2 The result overshoots the pre-agreed bandwidth ceiling by 2.5× — because the ceiling was built on a misused number

> **Revised 2026-08-16.** The first version of this annotation argued the fix
> "destroyed and then restored the effective bandwidth of the whole step",
> reasoning from a pre-fix byte count and a derived pre-fix total. Both inputs
> were wrong. The original text is preserved in git history; what follows
> replaces it. See `REFERENCE.md` §6.9 and corrections #22–#25.

The kit's decision table derived its ceiling from `read_total_bytes` =
961,130,496 per decode step and projected **≈18 tok/s** as the top of the range.
The measured 44.707 tok/s is 2.5× beyond it. The projection failed because the
byte model it rests on does not describe this workload:

1. **961,130,496 is a pre-fix figure.** It comes from `ctxbin-ws.log` dated
   2026-08-10, four days before `--grouped-gqa` existed. It was used here as if
   it were the post-fix byte count. **No post-fix `read_total_bytes` has ever
   been recorded** — the gqafix build directories retain the ctx-bins and
   `info.json` but no build logs. Capturing it is a ~20-minute device-free job
   and it is the single measurement that would settle this section.
2. **The replication ops never wrote to DDR.** The same converter summary reports
   the decode graph writing **419,840 bytes** per step — 420 KB, not the 264 MB
   the "destroyed bandwidth" story assumed. That traffic was VTCM-resident, and
   `spill_bytes`/`fill_bytes` are both 0.

So the fix removed ~262M of 350.3M DSP cycles and **approximately zero DDR
bytes**, and throughput rose 6.5×. A change that moves no DDR bytes cannot
produce a 6.5× speedup in a DDR-bound regime — which is why the byte-bound
projection undershot by 2.5× rather than bounding the result.

Note also that "effective bandwidth rose from ~17.5 to ~43 GB/s" is **circular**
if the byte count did not change: with bytes constant it is a restatement of
"the step got 6.5× faster", not an explanation of why.

The competing reading — that decode was **compute-bound** — predicted this
measurement out-of-sample: `88.5M remaining cycles ÷ 4 HVX threads @ ~1 GHz ≈
22.1 ms` against **22.37 ms measured**, written before the fix shipped. Its
divisor of 4 is now confirmed by reading `numHvxThreads` out of the shipped
ctx-bin.

This remains the strongest argument for completing the P1 cycle profile (§6 rec
6), which would separate "compute freed" from "streaming improved" directly.
`REFERENCE.md` §8.11 tracks the adjudication and names the two device-free
experiments that decide it.

### 0.3 The P1 blocker is a build-side packaging defect, not a device-side failure

§5 records that P1 could not run because the shipped profiling inputs were
pre-fix format. Those inputs (`decode_profile_inputs.tar.gz`) were generated on
2026-08-13 against the **pre-fix** decode graph and were re-shipped unchanged
inside the 2026-08-14 gqafix drop — the drop's own README pointed at them as
"priority 1, the decisive measurement". The generator
(`gen_decode_profile_inputs.py`) ships in the same package, so regenerating
against the gqafix graph and re-running is cheap. **Recorded here so the next
drop's checklist includes regenerating profiling inputs whenever graph I/O
changes.**

---

## 1. Background

An op-level profile on 2026-08-13 found **74.7% of each decode step's DSP
cycles** (261.8M of 350.3M) consumed by 56 ONNX `Expand` → `Eltwise_Binary`
MULTIPLY-by-ones ops (`op:13`). These were initially mislabeled as "attention
mask broadcast" but direct DLC inspection revealed them as **GQA KV-head
replication**: materializing 8 KV heads into 16 to match the 16-head Q
projection, writing/reading 264 MB per step.

The fix batches attention MatMuls over 8 KV heads directly (`1×8×2×1152` instead
of `1×16×1×1152`), removing all 56 replication ops. The KV I/O contract is
unchanged. Numerical equivalence is verified (max |Δ| 6e-16, bit-identical for
decode).

**The open question** was whether the freed cycles translate to throughput. This
report answers that.

## 2. Protocol

All measurements follow the 2026-08-13 protocol exactly:

| Rule | Value |
|---|---|
| Warm-up | 1 discarded run per bundle (cold init ~1.8–2.0 s → warm ~800 ms) |
| Reps | 3 per arm, greedy (`temp=0`, `top-k=1`) |
| Prompt | 56-token technical prompt (identical to 2026-08-13) |
| `perf_profile` | `llm_decode_burst`, `rpc_polling_time: 9999` |
| TTFT | Reported separately from init time |

## 3. Results

### 3.1 Primary A/B: GQA fix vs pre-fix (same topology, same 3-graph, same 1.09 GB)

Both bundles are 3-graph past-KV prefill, W16 head, ~1.09 GB ctx-bin,
weight-shared. The only variable is attention — KV-head replication vs grouped
GQA MatMul.

| Arm | Bundle | Mode | TGR (tok/s) | Init (ms) | TTFT (ms) | Tokens |
|---|---|---|---:|---:|---:|---:|
| P3 (A1) | pre-fix ladekv | basic | **6.836 ± 0.000** | 771 | 186 | 563 |
| P2 (B7b) | **gqafix ladekv** | basic | **44.707 ± 0.030** | 796 | 103 | 554 |

**Delta: +6.5× throughput, −45% TTFT.**

### 3.2 P2 LADE arm

| Arm | Mode | TGR (tok/s) | TTFT (ms) | Init (ms) | Acceptance | Tokens |
|---|---|---:|---:|---:|---|---:|
| P2 (B7b) | LADE | 31.342 ± 0.090 | 102 | 806 | 1.61 tok/iter | 493 |

**LADE is a regression:** 31.3 vs 44.7 tok/s in basic mode on the same binary.
Acceptance is only 1.61 tokens per verification call, and per-call latency
dominates.

### 3.3 P5: 2-graph past-KV

The `gqafix_pastkv2g` bundle isolates "graph count" from "prefill type" —
2 graphs with past-KV prefill, ~1.08 GB, weight-shared.

| Rep | TGR (tok/s) | Init (ms) | TTFT (ms) |
|---:|---:|---:|---:|
| 1 | 23.43 | 873 | 117 |
| 2 | **44.54** | 811 | 103 |
| 3 | 29.34 | 854 | 103 |

**Variance is high.** Rep 2 matches the 3-graph 44.7 tok/s, suggesting 2-graph
does not add measurable overhead. Reps 1 and 3 are slower for reasons not yet
isolated (thermal throttle, DSP scheduling, or kernel launch variance). The best
rep confirms 2-graph and 3-graph past-KV bins perform at the same ceiling.

### 3.4 P7: Hybrid prefill

The `gqafix_hybrid` dual-prefill bundle (bertcache CL=128 + past-KV CL=1152)
produced **degenerate output** — infinite loop on `"and parallel, and parallel,
..."` after the first few tokens. This is a **quality failure**, not a
performance result. The hybrid layout likely has a prefill graph wiring issue;
TTFT numbers are invalid because the output is not a valid continuation.

## 4. Baselines (from 2026-08-13)

For reference, all pre-fix measurements:

| Bundle | Mode | TGR (tok/s) | Notes |
|---|---|---:|---|
| `qwen3_06b_w8a16_local` | basic | 11.72 | 2-graph bertcache, 1.09 GB |
| `qwen3_06b_w8a16_ladekv` | LADE | 9.18 | 3-graph past-KV, technical prompt |
| `qwen3_06b_w8a16_ladekv` | LADE | 10.8 | 3-graph past-KV, simple prompt |
| `qwen3_06b_w8a16_qh` | basic | 6.70 | 3-graph, W8 `lm_head` |

## 5. Classification

Per the README §5.4 decision matrix:

| | tok/s rises (≥ 14) | tok/s flat |
|---|---|---|
| **cycles ↓** | **A. Compute-bound, fix works ✅** | B. Byte floor |
| **cycles flat** | C. Impossible | D. Build defect |

**Result: Quadrant A.** 44.7 tok/s is 3.8× the pre-fix 2-graph ceiling (11.72)
and 6.5× the pre-fix 3-graph floor (6.84). Decode was compute-bound, the GQA fix
removed the bottleneck, and the freed cycles translated to throughput.

The P1 decode-only cycle profile was **not completed** (profiling input files
were pre-fix format with 128-dim KV and 128-byte `position_ids`, incompatible
with the gqafix graph's 64-dim KV). The magnitude of the throughput delta (6.5×)
is sufficient to confirm quadrant A without the cycle count, but the cycle
profile should still be done for completeness.

## 6. Recommendations

1. **Ship `gqafix_ladekv` in basic mode.** 44.7 tok/s is the new baseline. The
   3-graph past-KV prefill is production-ready — no size confound, weight sharing
   perfect, zero spill/fill.
2. **Drop LADE.** Post-fix, basic mode is faster than LADE (44.7 vs 31.3). The
   `verify32` graph's AR-scaling and the n-gram acceptance penalty (~1.6
   tok/iter) make LADE a net negative. This confirms the README's pre-committed
   decision: *"LADE loses ... post-fix — basic is the ship configuration."*
3. **2-graph vs 3-graph is a wash** for past-KV prefill. The 3-graph bundle
   (`ladekv`) is the safe choice: it supports LADE if needed later, and the
   `verify32` graph costs ~0 extra memory (weight-shared). No reason to switch to
   2-graph.
4. **Hybrid prefill needs debugging.** The quality degradation (infinite loop)
   suggests a prefill graph wiring issue in the bertcache+past-KV hybrid layout.
   Do not ship until the output is verified.
5. **P6 (LADE acceptance map) is not needed** since LADE is a confirmed
   regression. P4 (byte-bound branch) is also unnecessary since the decode is
   clearly compute-bound.
6. **Complete the P1 cycle profile** for documentation. Re-generate profiling
   inputs from the gqafix graph (64-dim KV, 64-byte `position_ids`) and run the
   `qnn-net-run` detailed profile to confirm the 56 GQA replication ops are gone
   and total cycles are ~90M (vs 350M).

## 7. Data

All per-arm `--profile` JSON and stdout files are on the device at:

```
/data/local/tmp/results/p3_a1_ladekv_basic/      (3 reps, pre-fix basic)
/data/local/tmp/results/p2_gqafix_ladekv_basic/  (3 reps, gqafix basic)
/data/local/tmp/results/p2_gqafix_ladekv_lade/   (3 reps, gqafix LADE)
/data/local/tmp/results/p5_gqafix_pastkv2g/      (3 reps, 2-graph basic)
/data/local/tmp/results/p7_hybrid_ttft/          (warmup only, quality failure)
```

---

## Follow-ups this report creates

Not part of the original document — derived from it, for the build side.

| # | Action | Owner | Blocking? |
|---|---|---|---|
| 1 | Regenerate `decode_profile_inputs` against the gqafix decode graph; re-ship and re-run P1 | build | no — quadrant A already confirmed |
| 2 | Debug the `gqafix_hybrid` prefill wiring; do not ship until output is verified | build | yes, for the TTFT workstream |
| 3 | Update the HF repo README + bundle README: 44.7 tok/s is the new baseline, LADE parked, basic is the ship config | build | no |
| 4 | Isolate the `pastkv2g` rep variance (23.4 / 44.5 / 29.3) — thermal vs scheduling | device | no |
| 5 | Retire the "~20% build gap" open question — the pre-fix 3-graph floor is now measured at 6.84 and superseded | build | no |
| 6 | Pull `results/` off the device before it is wiped; `/data` runs 98–99% full | device | yes, for the per-op record |
