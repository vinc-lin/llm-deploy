# Runsheet — SA8797P decode-throughput session

**Priority-ordered.** If you run out of time and stop anywhere, the questions
already answered are the ones that mattered most. Nothing here needs a
round-trip with the build side: every outcome has a pre-agreed meaning in
`decision_table.md`.

Estimated total: ~2 h including B7a. Priorities 1–3 alone (~45 min) decide the
plan.

## Setup

```sh
# on the device
cd /data/local/tmp
# /data runs 98-99% full: clean old bundles first, and delete each tarball
# after extracting it
for f in *.tar.gz; do tar xzf "$f" && rm "$f"; done
sh kit/run_all.sh results
```

`run_all.sh` skips any arm whose bundle is absent, so pulling a subset is fine.

## Protocol (identical to 2026-08-13, so results compare)

- **Warm.** One discarded warm-up per bundle; cold init is ~1.8–2.0 s vs
  ~800 ms warm and pollutes averages. `run_all.sh` does this for you.
- **3 reps per arm**, greedy, same prompt across arms being compared.
- **Report init time and TTFT separately** — never compare init→first-logits
  against TTFT; they differ by ~800 ms and this has already produced one phantom
  regression.
- Leave `perf_profile: llm_decode_burst` everywhere. Change nothing except what
  an arm explicitly changes.

---

## Priority 1 — B7a: decode-only cycle profile ⭐ the decisive measurement

**Not in `run_all.sh`** — this uses `qnn-net-run`, not `genie-t2t-run`.

This is the single measurement the whole plan turns on. Expectation:
**350.3M → ~90M aggregate DSP cycles (−75%)**.

```sh
cd /data/local/tmp/profiling
tar xzf decode_profile_inputs.tar.gz          # -> ar1_decode/
cd ar1_decode

export ADSP_LIBRARY_PATH=.:/vendor/lib/rfsa/adsp
LD_LIBRARY_PATH=. qnn-net-run \
    --backend libQnnHtp.so \
    --retrieve_context ../qwen3-0.6b-w8a16-gqafix-decodeonly_ctx.bin \
    --input_list input_list.txt \
    --output_dir out_gqafix \
    --profiling_level detailed \
    --config_file ../htp_config_decodeonly.json

qnn-profile-viewer --input_log out_gqafix/qnn-profiling-data_0.log \
    > profile_gqafix_viewer.txt
```

Run 3 reps. Send back `profile_gqafix_viewer.txt` for each, plus the raw
`.log` files.

**Also send the 2026-08-13 originals if you still have them** — we never
received `test2_decode_profile/`, so we have no per-op baseline to diff against,
only the narrative table.

What to look for in the viewer output:

- **Aggregate cycles.** ~90M = the fix landed. ~350M = it did not (see decision
  table box D — that is a build defect, not a disproved hypothesis).
- **No `Expand` ops.** The 56 ops that were 74.7% of cycles should be entirely
  absent. Their replacement is a batched MatMul with output `1x8x2x1152`.
- **The new top-10.** With replication gone we expect `lm_head` first, then the
  attention GEMVs and weight GEMMs. Send the top 20 — the new shape of the
  profile decides where the *next* round of work goes.

## Priority 2 — B7b: post-fix throughput (the headline)

| Arm | Bundle | Dialog |
|---|---|---|
| `p2_gqafix_local_basic` | `qwen3_06b_w8a16_gqafix_local` | `genie_dialog.json` |
| `p2_gqafix_ladekv_basic` | `qwen3_06b_w8a16_gqafix_ladekv` | `genie_dialog_basic.json` |
| `p2_gqafix_ladekv_lade` | `qwen3_06b_w8a16_gqafix_ladekv` | `genie_dialog.json` |

Compare against **11.72 tok/s** (basic) and **9.18 tok/s** (lade, same prompt).

## Priority 3 — A1: the one run that de-confounds everything

| Arm | Bundle | Dialog |
|---|---|---|
| `p3_a1_ladekv_basic` | `qwen3_06b_w8a16_ladekv` (**pre-fix**, you already have it) | `genie_dialog_basic.json` |

Basic mode on the plain `ladekv` bin has never been measured. The 6.70 tok/s in
report Test 4 came from the **qh** bundle, so "3-graph penalty", "W8 head" and
"build lineage" are still conflated. One run separates them. **If you run
nothing else from this kit, run this one.**

## Priority 4 — where do the bytes bind?

Only informative once priority 2 is known; see decision-table box B.

| Arm | Bundle | Tests |
|---|---|---|
| `p4_gqafix_qh_basic` | `qwen3_06b_w8a16_gqafix_qh` | W8 `lm_head`, clean 2-graph comparison at last |
| `p4_gqafix_cl512_basic` | `qwen3_06b_w8a16_gqafix_cl512` | KV read 132 → 59 MB |

## Priority 5 — config trims

`p5_gqafix_dlbc_basic` (activation DLBC), `p5_fuseqkvgu_lade` (fused QKV+Gate-Up
on the pre-fix base, for continuity with earlier numbers).

Also worth one runtime edit if time allows: in the `ladekv` bundle's
`genie_dialog.json`, set `"spill-fill-bufsize": 640000000` (from 0) and re-run
the LADE arm — verify32 moves 745 MB of spill/fill.

## Priority 6 — LADE acceptance map

Three prompt classes (`technical`, `simple`, `structured`) × {lade, basic} on
**one** binary. LADE lost to basic on the technical prompt (9.18 vs 11.72) and
won on an earlier simple prompt (10.8 vs ~6.5), so its value is entirely a
question of prompt distribution. This decides whether LADE ships at all.

## Priority 7 — TTFT product metric

`p7_hybrid_ttft`: a bin carrying **both** a CL=128 bertcache prefill and the
CL=1152 past-KV prefill. Genie picks by (AR, CL) best fit, so short prompts
should take the fast path. Target: **~40 ms TTFT with full context available**,
versus 186 ms today. Report TTFT and prompt rate, not just tok/s.

Basic mode only — never with `type: "lade"` (an AR==CL bertcache graph breaks
speculation; that is what SIGSEGV'd on 2026-08-10/11).

---

## Quality check — before trusting any speed number

`expected/` holds the greedy continuation each bundle should produce for each
prompt, generated from the local ONNX parity harness. Speed on a broken graph is
meaningless, and W8A16 quantisation bugs have historically produced *fluent but
wrong* output rather than obvious garbage. Diff the first ~30 tokens of each
arm's `stdout_r1.txt` against `expected/`; flag any arm that diverges early.

## What to send back

The whole `results/` directory (per-rep `--profile` JSON, stdout, and the dialog
+ backend configs each arm actually used), plus the B7a viewer outputs. The
`MANIFEST.txt` records what ran and what was skipped.
