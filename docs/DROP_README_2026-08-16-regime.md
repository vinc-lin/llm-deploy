# 2026-08-16 — the regime drop

Everything from this drop lives in this folder. **Start at
[`kit-v2/README.md`](kit-v2/README.md), then `kit-v2/runsheet.md`.**

## What this drop is for

The GQA fix landed: **44.707 tok/s** basic mode against a like-for-like pre-fix
control of **6.836** — a measured **6.54×**. Basic beats LADE, so basic ships and
LADE is parked.

What we do not know is *why* 44.707 is the number, and the two candidate answers
imply completely different next years of work:

| | fits the measured 22.37 ms/step? |
|---|---|
| **Byte-bound** — 961 MB ÷ 22.37 ms = 43.0 GB/s | yes: 88% of the device team's own 49 GB/s excl-wait ceiling |
| **Compute-bound** — 88.2M residual DSP cycles ÷ 4 HVX @ ~1 GHz = 22.06 ms | yes: to 1.4% |

They are **degenerate at this operating point**. This drop exists to separate
them, using arms whose predicted *orderings* are opposite.

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
| `..._gqafix_ctrl_ladekv` | **nothing** — the control | — | — |
| **`..._gqafix_qh_ladekv`** | W8 `lm_head` | **+17.9%** | +3.6% |
| **`..._gqafix_cl512_ladekv`** | `context.size` 512 (ctx-bin CL 640) | +10.1% | **+26.0%** |
| **`..._gqafix_hvx8_ladekv`** | `hvx_threads: 8` at build time | **0.0%** | up to large |
| `..._gqafix_udma_ladekv` | `extended_udma` | 0.0% | unknown |
| `..._gqafix_socmodel72_ladekv` | `soc_model`/`soc_id` 72 | 0.0% | unknown |
| `..._gqafix_dlbc_ladekv` | activation DLBC | 0.0% | unknown |
| `..._gqafix_wpack_ladekv` | `weights_packing` | 0.0% | unknown |

**The first three rows are the experiment.** `qh` and `cl512` predict opposite
orderings, which settles the regime with no assumption about clock or thread
count. `hvx8` changes zero DDR bytes *by construction*, so any result above the
rep spread falsifies the byte model outright.

### ⚠️ `htp_backend_ext_config.json` says `hvx_threads: 4` in *every* bundle, including `hvx8`

That is deliberate and correct. `hvx_threads` is a **build-time** parameter —
your own 2026-08-13 Test 5 established that changing it at runtime does nothing
(9.17 vs 9.18 tok/s), and the SDK documents the runtime value as ignored. The 8
lives inside `..._hvx8_ladekv`'s ctx-bin, baked in at generation time, which is
why that bin is 2,277,376 bytes larger than the control.

Leaving the runtime config at 4 across all bundles is what keeps the arm a
single-variable test. **Do not "fix" it to 8** — that would change nothing
functionally and would make the arm look like a two-variable change.

### Why the control matters

`ctrl` is built from the same DLCs through the same config path as every other
variant, with default knobs. Compare the knob arms against **`ctrl`**, not
against the shipped `gqafix_ladekv` — otherwise a delta could be the config path
rather than the knob.

### Which knobs actually reached the binary

Offline evidence, before anyone spends device time:

| Knob | ctx-bin Δ vs control | Verdict |
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
arms, temperature logged either side. Priority 2 spends 8 reps isolating the
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
