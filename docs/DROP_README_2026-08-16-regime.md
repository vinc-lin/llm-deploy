# 2026-08-16 — the regime drop

Everything from this drop lives in this folder.

> ## 👉 Start at [`DEPLOYMENT_AND_TEST_GUIDE.md`](DEPLOYMENT_AND_TEST_GUIDE.md)
>
> That one document is self-contained: deployment, the full test procedure, the
> exact metrics to record and report, what every outcome means, and the gotcha
> table. `kit-v2/` is the scripted form of the same procedure — the guide is the
> authority if they ever disagree.
>
> This README is just the drop's inventory and rationale.

## What this drop is for

The GQA fix landed: **44.707 tok/s** basic mode against a like-for-like pre-fix
control of **6.836** — a measured **6.54×**. Basic beats LADE, so basic ships and
LADE is parked.

What we do not know is *why* 44.707 is the number. Three models remain
admissible (`REFERENCE.md` §8.11) and they imply completely different next years
of work:

| Model | Standing |
|---|---|
| **Compute-bound** | Predicted the post-fix step **out-of-sample** (88.2M ÷ 4 HVX @ ~1 GHz ≈ 22.1 ms vs 22.37 measured). Weakness: the clock is invisible, so threads × clock is one product with two unknowns |
| **Byte-bound** | Plausible at the operating point (~43 GB/s, 88% of ceiling). **But the fix removed ~0 DDR bytes and still gave 6.5×** — pre-fix and post-fix decode both read 961,130,496 B — so it cannot explain how we got here |
| **Access-pattern fragmentation** | The ~220 µs RPC / 30–60 µs inter-op costs are real, but they survive the fix unchanged, so they cannot explain a 6.5× change either |

This drop exists to settle which binds *now*, and its priority-1 arm does so
without assuming anything about clock or thread count.

## ⚠️ Read this before running anything

**The 2026-08-14 drop's priorities 4 and 5 must not be run.** Six of its eight
bundles carry a CL=128 bertcache prefill graph, which makes Genie generate
*through* that graph until the KV cache passes 128. Their basic-mode tok/s is
therefore a two-phase blend, not a decode rate — and the blend runs *fast*, so
the failure mode is a confident wrong number.

That is also what the old **11.72 tok/s** figure was. Three previously-reported
findings were artifacts of quoting it as a decode rate, and all three are
withdrawn: the "~75% build gap", "LADE is −22%", and "our builds are +51% faster
than the device team's". Details in `DEVICE_TEAM_EXCHANGE_2026-08-16.md` §2.

Every bundle in *this* drop is topologically **pure** and was gated with
`lint_bundle_topology.py` before shipping.

## The bundles

All are 3-graph past-KV (`prefill` AR=128 / `decode` AR=1 / `verify32` AR=32),
weight-shared, spill 0, and directly comparable to the 44.707 baseline.

| Bundle | What it changes | Byte model | Compute model |
|---|---|---:|---:|
| **`..._gqafix_hvx8_ladekv`** ⭐ | `hvx_threads: 8` at build time — **priority 1** | **0.0%, by construction** | up to large |
| **`..._gqafix_cl512_ladekv`** | `context.size` 512 (ctx-bin CL 640) — priority 2 | +10.1% | **+26.0%** |
| `..._gqafix_qh_ladekv` | W8 `lm_head` — priority 3, **confounded** | +17.9% | +3.6%, likely ~0% |
| ~~`..._gqafix_ctrl_ladekv`~~ | **nothing** — byte-identical to the baseline; **skip** | — | — |
| `..._gqafix_udma_ladekv` | `extended_udma` | 0.0% | unknown |
| `..._gqafix_socmodel72_ladekv` | `soc_model`/`soc_id` 72 | 0.0% | unknown |
| `..._gqafix_dlbc_ladekv` | activation DLBC | 0.0% | unknown |
| `..._gqafix_wpack_ladekv` | `weights_packing` | 0.0% | unknown |

**`hvx8` is the experiment.** It is the only arm that varies compute while
holding DDR bytes *exactly* constant, so it settles the regime with no assumption
about clock or thread count — the byte model predicts exactly 0.0% for it.
`cl512` is second and discriminates by magnitude. `qh` is **third and optional**:
it changes bytes *and* is independently suspected of never reaching the device,
so it confounds the question it was meant to answer (`REFERENCE.md` §8.11).

### ⚠️ `htp_backend_ext_config.json` says `hvx_threads: 4` in *every* bundle, including `hvx8`

That is deliberate and correct. `hvx_threads` is a **build-time** parameter —
your own 2026-08-13 Test 5 established that changing it at runtime does nothing
(9.17 vs 9.18 tok/s), and the SDK documents the runtime value as ignored. The 8
lives inside `..._hvx8_ladekv`'s ctx-bin, baked in at generation time, which is
why that bin is 2,277,376 bytes larger than the baseline, and why `numHvxThreads`
reads back as **8** out of it.

Leaving the runtime config at 4 across all bundles is what keeps the arm a
single-variable test. **Do not "fix" it to 8** — that would change nothing
functionally and would make the arm look like a two-variable change.

### The `ctrl` bundle is redundant — skip it

ctx-bin generation is **deterministic**: rebuilding the baseline config from the
same DLCs reproduces the baseline bin exactly. `ctrl`'s ctx-bin is byte-identical
to the shipped `gqafix_ladekv` (md5 `9c6024ad5b141137fbe22f3a4972eb96` on both),
so running it as a separate arm would measure the same binary twice.

**Compare every arm against `p0_rebaseline`.** `ctrl` ships only so the folder is
self-contained.

### Which knobs actually reached the binary

Offline evidence, before anyone spends device time:

| Knob | ctx-bin Δ vs baseline | Verdict |
|---|---:|---|
| `hvx_threads: 8` | **+2,277,376 B** | consumed |
| `soc_model: 72` | **+249,856 B** | consumed |
| `extended_udma` | **+212,992 B** | consumed — **its first real build ever** (the key sat in the wrong config section previously) |
| `dlbc` | 0 | identical — unproven, rank last |
| `weights_packing` | 0 | identical — unproven, rank last |

## Protocol change — not optional

The 2026-08-15 session measured `pastkv2g` at **23.43 / 44.54 / 29.34 tok/s on
one binary**. That spread is larger than every effect this drop chases.

**5 reps per arm, median reported, every raw value kept**, 30 s cool-down between
arms, temperature logged either side. One arm spends 8 reps isolating the
variance itself, because it currently bounds the resolution of every measurement
either side can take.

## Also in this drop

| Path | What |
|---|---|
| `kit-v2/` | runsheet, decision table, `run_all.sh`, prompts, expected outputs |
| `verify_profile_inputs.py` | **run before the cycle profile.** Last session's stated reason for skipping it was wrong — the shipped inputs score 60/60 against the gqafix bin. This names the real mismatch, or confirms there is none |
| `DEVICE_TEAM_EXCHANGE_2026-08-16.md` | the full correction note, and the three artifacts still outstanding from 2026-08-14 |

The pre-existing `gqafix_ladekv` and `gqafix_pastkv2g` bundles from the
2026-08-14 drop are still needed — they are the baseline and the variance arm,
and they are the only two bundles from that drop that were ever pure.
