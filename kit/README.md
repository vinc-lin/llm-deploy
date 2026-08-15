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
