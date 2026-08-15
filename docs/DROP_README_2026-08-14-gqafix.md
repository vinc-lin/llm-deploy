# 2026-08-14 — GQA replication fix

Everything from this drop lives in this folder. Nothing here depends on the
files at the repo root except the two pre-fix bundles named below, which you
almost certainly already have on the device.

## What changed

An op-level profile on 2026-08-13 found **74.7% of every decode step's DSP
cycles** (261.8M of 350.3M) in 56 ops that the report labelled "attention-mask
broadcast". Inspecting the shipped `decode.dlc` showed that label was wrong:
the mask is never expanded — it enters as `[1,1,1152]`, gets one `Unsqueeze`,
and broadcasts implicitly inside the `Add`. The 56 ops are **GQA KV-head
replication**, materialising 8 KV heads into 16 so a 16-head MatMul can consume
them, writing 264 MB and re-reading it every step. Their QNN type is
`Eltwise_Binary` with `operation: 13` (MULTIPLY) against a static
`[1,1,2,1,1]` coefficient — the converter lowers ONNX `Expand` into a broadcast
multiply-by-ones.

These bundles batch the attention MatMuls over the 8 KV heads instead
(`1x8x2x1152` in place of `1x16x1x1152`). The KV I/O contract is untouched, so
the Genie feed pattern is unchanged.

Verified device-free before shipping: numerically equivalent to the replicating
form (max |Δ| 6e-16 in float64, bit-identical for decode); ONNX-vs-HF prefill
argmax identical and an 8-step greedy chain token-identical; qualla-read parity
6/6 including chunked prompts; and all four exported graphs — bertcache
prefill, decode, verify32, past-KV prefill — assert **zero** replication ops.

## Layout

```
2026-08-14-gqafix/
├── README.md          ← you are here
├── kit/               the device session: START AT kit/runsheet.md
│   ├── runsheet.md            priority-ordered arms, protocol, what to send back
│   ├── decision_table.md      what each result means, agreed in advance
│   ├── run_all.sh             runs priorities 2–7 unattended; skips absent bundles
│   ├── prompts/               technical (56 tok), simple, structured
│   └── expected/              greedy HF references — check quality before speed
├── bundles/           8 tarballs
└── profiling/         priority-1 material (qnn-net-run, not genie-t2t-run)
    ├── qwen3-0.6b-w8a16-gqafix-decodeonly_ctx.bin
    ├── decode_profile_inputs.tar.gz
    ├── htp_config_decodeonly.json
    └── htp_backend_config_decodeonly.json
```

## Start here

1. **`kit/runsheet.md`** — the arms in priority order. If you stop anywhere, the
   questions already answered are the ones that mattered most.
2. **`kit/decision_table.md`** — every outcome already has an agreed meaning, so
   the session should need no round-trip with the build side.
3. Extract and run:

```sh
cd /data/local/tmp
for f in *.tar.gz; do tar xzf "$f" && rm "$f"; done   # /data runs 98–99% full
sh kit/run_all.sh results
```

Priority 1 is **not** in `run_all.sh` — it uses `qnn-net-run`. Commands are in
the runsheet.

> **This drop has been run — results in `DEVICE_MEASUREMENT_REPORT_2026-08-15.md`.**
> The fix worked: **44.707 tok/s** basic, +6.5× (`REFERENCE.md` §6.8). The two
> "if you only have time for one thing" sections below are kept as the record of
> what was asked; both are now answered or blocked. Current priorities are in
> `docs/PLAN_0.6B_max_tps.md`.

### If you only have time for one thing — ⚠ BLOCKED, could not run

Profile `profiling/qwen3-0.6b-w8a16-gqafix-decodeonly_ctx.bin` against the known
**350,302,972**-cycle baseline. **Expect ~90M.**

**This did not run on 2026-08-15.** The `decode_profile_inputs.tar.gz` shipped
alongside it was generated 2026-08-13 against the **pre-fix** decode graph
(128-dim KV, 128-byte `position_ids`) and is incompatible with the gqafix
graph's 64-dim layout. The ctx-bin is fine; the inputs are not. Build-side
packaging defect — regenerate with `scripts/util/gen_decode_profile_inputs.py`
before re-shipping, and add "regenerate profiling inputs whenever graph I/O
changes" to the drop checklist. The ~90M expectation stands, untested.

### If you only have time for one *Genie* run — ✅ answered

`p3_a1_ladekv_basic` — basic mode on the plain `qwen3_06b_w8a16_ladekv` bundle.
**Measured 6.836 ± 0.000 tok/s.** It did what it was for: it de-confounded the
3-graph numbers (the 6.70 tok/s in Test 4 came from the **qh** bundle and
conflated W8 head, graph count and build lineage), and it is the same-topology
predecessor against which the fix measures **+6.5×**.

## The bundles

Baselines this drop was measured against (2026-08-13, warm, greedy, 56-token
technical prompt): basic AR-1 `local` = 11.72 tok/s, LADE `ladekv` = 9.18 tok/s.
**Both are pre-fix. The current baseline is 44.707 tok/s** (`REFERENCE.md` §6.8).

| `bundles/…` | ctx-bin | Prefill | Outcome |
|---|---:|---|---|
| `qwen3_06b_w8a16_gqafix_ladekv.tar.gz` | 1.087 GB | past-KV ×3 | ✅ **the artifact that shipped — 44.707 tok/s basic, the current baseline** |
| `qwen3_06b_w8a16_gqafix_pastkv2g.tar.gz` | 1.080 GB | past-KV ×2 | ✅ **a wash.** Best rep 44.54 matches 3-graph; reps 1/3 were 23.43 / 29.34 — the rep-variance finding that forced the 5-rep protocol. 3-graph stays the safe choice |
| `qwen3_06b_w8a16_gqafix_local.tar.gz` | 1.523 GB | bertcache | ⚠ **not size-matched** to the 11.72 baseline — carries the 444 MB bertcache weight duplication (§6.10) |
| `qwen3_06b_w8a16_gqafix_qh.tar.gz` | 1.525 GB | bertcache | ⏳ **not run** — skipped as "unnecessary" on a byte-bound inference that was wrong. Still open (`REFERENCE.md` §8.3). Note it is a bertcache 2-graph bin, so it answers the science but cannot ship |
| `qwen3_06b_w8a16_gqafix_cl512.tar.gz` | 1.523 GB | bertcache | ⏳ not run. Context 512: `read_total_bytes` 961 → 873 MB — both figures **pre-fix-derived**; the −9.2% ratio survives, the absolute base does not (§6.9) |
| `qwen3_06b_w8a16_gqafix_dlbc.tar.gz` | 1.523 GB | bertcache | ⏳ not run. `dlbc: 1` — note DLBC is defined as *inter-layer* (activation) compression, so it likely does not touch the weight stream |
| `qwen3_06b_w8a16_gqafix_udma.tar.gz` | 1.524 GB | bertcache | ⏳ not run. `context.extended_udma` |
| `qwen3_06b_w8a16_gqafix_hybrid.tar.gz` | 1.532 GB | bertcache + past-KV | ⛔ **DEGENERATE OUTPUT — do not ship, do not recommend.** Infinite loop on `"and parallel, and parallel, …"` after the first few tokens. A quality failure, so its TTFT numbers are invalid. Suspected prefill-graph wiring bug; reproducible device-free (`PLAN_0.6B_max_tps.md` A4) |

## Two things to know before interpreting anything

**1. The primary comparison is not the obvious one.** Run
`gqafix_ladekv` with `genie_dialog_basic.json`, and compare it against the
pre-fix `qwen3_06b_w8a16_ladekv` with `genie_dialog_basic.json`. Those two are
the same topology, same graph count, same 1.087 GB ctx-bin, same encodings
recipe — **only the attention differs**, so any delta is attributable.
`gqafix_local` vs the 11.72 tok/s baseline is corroboration, not primary
evidence, because of the next point.

**2. Bins containing the CL=128 bertcache prefill are ~1.52 GB, not ~1.09 GB.**
Under the new attention that graph requires one private ~444 MB copy of the
INT8 decoder weights; the hybrid bin shows it cleanly, with a full 1,067 MB
shared pool *plus* 444 MB of constants on the bertcache graph alone. Weight
*bytes* per step are unchanged, so this should not move decode traffic, but it
costs disk on a device whose `/data` runs 98–99% full and may affect init time.
Root cause is not yet established — it is in the ctx-bin generator's layout
decision, not the graph topology, which gates clean on all four graphs.

A consequence worth stating outright: **`gqafix_qh` is 1.525 GB, no smaller than
the FP16-head version**, because the head lands in that duplicated pool. Do not
read that as head quantisation failing. Per *step* the decode graph still reads
155 MB instead of 311 MB, so the streaming saving is intact; only storage
doubles. Judge that arm on tok/s alone.

## Also fixed

`type: "lade"` + `max-num-tokens` SIGSEGVs (exit 139) — your report §6.1. That
pair had shipped in `genie_dialog_demo.json` in the `fuseqkvgu`, `socmodel72`
and `hvx8` bundles, so **every demo run of those three died**. Fixed and
re-uploaded 2026-08-14; a linter now refuses the combination at build time. If
you pulled any of those three before 2026-08-14, re-pull them from the repo
root.

## What to send back

The whole `results/` directory (per-rep `--profile` JSON, stdout, and the dialog
+ backend configs each arm actually used), plus the priority-1
`qnn-profile-viewer` output. `results/MANIFEST.txt` records what ran and what
was skipped.

If you still have them, the raw
`test2_decode_profile/qnn_profile_r{1,2,3}_viewer.txt` from 2026-08-13 would
also help — only the narrative report reached us, so there is no per-op baseline
to diff the new profile against.
