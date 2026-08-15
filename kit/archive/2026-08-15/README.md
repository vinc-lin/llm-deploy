> # ⛔ ARCHIVED — the 2026-08-15 session has run (archived 2026-08-16)
>
> Kept for provenance: this is the pre-commitment record that makes the results
> credible, and `DEVICE_MEASUREMENT_REPORT_2026-08-15.md` §5 cites its decision
> matrix by name. **It is not live guidance.**
>
> - Baselines here are **pre-GQA-fix**. Current: **44.707 tok/s** basic
>   (`REFERENCE.md` §6.8).
> - The **quadrant A / quadrant B** matrix resolved to **A**.
> - **LADE is parked** — the pre-committed "LADE loses on all three post-fix"
>   row fired.
> - Byte-side reasoning here is **withdrawn**: `read_total_bytes = 961,130,496`
>   is a pre-fix figure and the "264 MB written / 264 MB re-read / ~530 MB
>   removable" derivation is refuted by `write_total_bytes = 419,840`
>   (`REFERENCE.md` §6.9, corrections #22-#25). The "~18 tok/s ceiling" computed
>   from it was exceeded 2.5x.
> - Priority 1's commands **cannot execute** — the shipped profiling inputs are
>   pre-fix format.
>
> The bertcache 444 MB weight-duplication finding (decision_table 1b) was
> migrated to `REFERENCE.md` §6.10 before archiving. Next session: build kit v2
> from `docs/PLAN_0.6B_max_tps.md` §4, per `kit/README.md`.

# SA8797P decode-throughput test kit

A self-contained device session. Everything needed to run it is here, and every
outcome already has an agreed meaning — **you should not need to contact the
build side to interpret anything.**

## Why this session exists

An op-level profile on 2026-08-13 found that **74.7% of every decode step's DSP
cycles** (261.8M of 350.3M) went to 56 ops that the report labelled
"attention-mask broadcast". Inspecting the shipped `decode.dlc` showed the label
was wrong: the mask is never expanded, and those 56 ops are **GQA KV-head
replication** — materialising 8 KV heads into 16 so a 16-head MatMul can consume
them, writing 264 MB and re-reading it, every step.

The bundles here remove that. The attention MatMuls now batch over the 8 KV
heads directly (`1x8x2x1152` instead of `1x16x1x1152`), which is numerically
equivalent — verified to 6e-16 in float64 — and leaves the KV I/O contract
untouched.

**The open question is what the device does with the freed cycles.** Either
decode was compute-bound and throughput rises to ~15–18 tok/s, or there is a
weight/KV streaming floor and it barely moves. Priorities 1 and 2 answer that
together, and the answer redirects everything downstream.

## Files

| File | What |
|---|---|
| `runsheet.md` | **Start here.** Priority-ordered arms, protocol, what to send back |
| `decision_table.md` | What each possible result means, agreed in advance |
| `run_all.sh` | Runs priorities 2–7 unattended; skips absent bundles |
| `prompts/` | `technical`, `simple`, `structured` — full Qwen3 chat template, empty `<think>` block |
| `expected/` | Greedy reference continuations, for checking quality before trusting speed |

## Quick start

```sh
cd /data/local/tmp
for f in *.tar.gz; do tar xzf "$f" && rm "$f"; done   # /data runs 98-99% full
sh kit/run_all.sh results
```

Then do priority 1 (B7a) by hand — it uses `qnn-net-run`, not `genie-t2t-run`;
the commands are in `runsheet.md`.

## If you have time for exactly one run

`p3_a1_ladekv_basic` — basic mode on the plain `qwen3_06b_w8a16_ladekv` bundle
you already have. It costs one run and retroactively de-confounds every 3-graph
measurement taken so far, because the 6.70 tok/s in Test 4 was measured on the
`qh` bundle and therefore conflates three variables at once.

## Notes

- The `type: "lade"` + `max-num-tokens` SIGSEGV you found (report §6.1) is fixed
  in every bundle here, and a linter now refuses the combination at build time.
  If you pulled fuseqkvgu / socmodel72 / hvx8 before 2026-08-14, re-pull.
- `hvx_threads` is **build-time only** — your Test 5 established that, so the
  variants here that change it are separate ctx-bins, not config edits.
- Backend-config keys are silently ignored when misspelled or misplaced. If you
  see `Unknown Key` warnings, that arm is not testing what it claims.
