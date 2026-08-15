# Qwen3-0.6B max tok/s on SA8797P — the plan

**Living document, no version number.** It replaces the `MAX_TPS_QWEN3_0.6B`
V1–V4 ladder, which is in `docs/archive/` (see `archive/README.md` for what each
one got wrong). Analytical basis and all measured numbers live in
`docs/REFERENCE.md`; this file is only *what to do next and why*. Where the two
disagree, REFERENCE wins.

**Last revised:** 2026-08-16, after the 2026-08-15 GQA-fix device results.

---

## 1. Where we are

**44.707 ± 0.030 tok/s**, `gqafix_ladekv`, basic mode, TTFT 103 ms
(`REFERENCE.md` §6.8). That is +6.5× over the same topology pre-fix, and it is
the ship configuration. LADE is parked as a regression.

The GQA fix worked by removing **compute**, not bytes: 74.7% of decode DSP
cycles, against ~0 DDR bytes (§6.9). That refuted the DDR-bound model every
earlier version of this plan was built on (corrections #22–#25).

## 2. The one question that sets the direction

**What binds decode now?** Three models are live and they imply different work
(`REFERENCE.md` §8.11). Until one is picked, any byte-reduction campaign is a
bet, not a plan — and the byte model has already mispredicted once (V3 §7 said
18.1 tok/s; reality was 44.7).

So the next two actions are *discriminating experiments*, not optimisations.

## 3. Part A — device-free, do now

Ranked by value per hour. A1 and A2 are the discriminators; everything below
them is worth doing regardless of how they land.

| # | Action | Why it is first | Cost |
|---|---|---|---|
| **A1** | **Build a `gqafix_ladekv` ctx-bin with `hvx_threads: 8`** and read `numHvxThreads` back out of the binary | We ship on **4 of 8 HVX units** (§8.9). Changes compute capacity while holding bytes *exactly* constant, so it is orthogonal by construction. Compute-bound predicts a large gain; byte-bound predicts ~0 | ctx-bin regen only — same DLCs, same encodings, **zero numerical risk**, ~20 min |
| **A2** | **Capture post-fix `read_total_bytes`** for the gqafix decode graph | No post-fix byte count has *ever* been recorded (§6.9). It is the input the byte model claims to stand on | ~20 min, same build |
| **A3** | **`soc_model: 72`** variant | Shipping bins carry `socModel = 0`; the device team's own "verified working" build sets 72. Never A/B'd (§8.4) | ctx-bin regen |
| **A4** | **Debug `gqafix_hybrid`'s degenerate output** | Blocks the TTFT product win (103 → ~40 ms). Reproducible device-free: drive the bertcache graph's exact I/O with a `parity_ladekv_read.py`-style feed and expect an argmax divergence. Suspects, in order: per-graph input naming, the position-id path under grouped attention, graph selection handing the wrong prefill its mask | half a day |
| **A5** | **Regenerate `decode_profile_inputs`** against the gqafix decode graph | The 08-15 P1 profile died on pre-fix input format (128-dim KV vs 64-dim). Build-side packaging defect. `gen_decode_profile_inputs.py` ships in the same package | ~1 h |
| **A6** | **Fusion on the gqafix base** (`gqafix_fuseqkvgu_ladekv`) | The two winning changes have never been combined. **Genuinely uncertain**: fusion's +15% was measured pre-fix, and if it was dispatch-overhead recovery the GQA fix may already have collected it — the same reasoning that killed LADE | one build |
| **A7** | **Byte-accounting gate** — record `read_total_bytes` / `write_total_bytes` per graph for every variant and check against prediction | A variant whose bytes did not move did not do what it claims. Device-free analogue of the `Unknown Key` rule | free, add to build scripts |

**Build discipline** (non-negotiable, see `BUILD_GUIDE.md`): `FUSE_FLAGS="--grouped-gqa"`
on every `lade_build.sh` / `ladekv_build.sh` call — they re-export verify32 and the
past-KV prefill, and omitting it silently ships old attention there (`lint_gqa_ops.py`
gates it). `disk_guard` sized per step. Convert straight to final filenames.
`qnn-context-binary-utility` per bin. No duplicate (AR, CL) pairs.

**Per-build gates, none skippable:** numerical equivalence → ONNX parity vs HF →
`--eval` ≥3/4 → `parity_ladekv_read.py` 6/6 incl. chunked → `lint_gqa_ops.py` 0
replication ops → graph names + weight sharing → byte gate (A7).

### Deliberately *not* leading with the W8 head

Earlier versions made `--quant-head` priority 1. It is now demoted: it changes
bytes *and* is independently suspected of never reaching the device (the DLC
shrinks 151 MB, the ctx-bin only 12.5 MB — §6.4, §8.1). It therefore confounds
the exact question it was meant to settle. Build it, but behind A1/A2. Its
measured −14% was a LADE acceptance effect and does not apply in basic mode.

## 4. Part B — the device session, when hardware returns

**B0, first and mandatory.** The `pastkv2g` rep spread (23.4 / 44.5 / 29.3 on one
binary) is larger than most effects worth chasing. So: **5 reps per arm, report
the median and every raw value**; record thermal state before and after; fixed
cool-down between arms; never compare arms from different thermal regimes; and
**an A/B whose delta falls inside the rep spread decides nothing** — re-run it,
do not interpret it. Re-measure the 44.707 baseline under this protocol first;
every delta is computed against it.

| Pri | Arm | Decides |
|---|---|---|
| — | **Pull `/data/local/tmp/results/` off the device** | `/data` runs 98–99% full; the per-op record is one cleanup away from gone. Do this **first** |
| **1** | **hvx8 vs hvx4**, same lineage | The regime question (§2). Everything else re-ranks on the answer |
| 2 | P1 cycle profile with A5's regenerated inputs | Confirms ~90M cycles / zero replication ops, and decomposes 22.37 ms into compute vs streaming — the input every future plan needs |
| 3 | `soc_model: 72` | Free knob, never tested |
| 4 | Fusion on gqafix (A6) | Does fusion survive the fix? |
| 5 | W8 head, basic mode | The byte lever, now that basic mode removes the acceptance confound |
| 6 | CL ladder (`cl512`, `cl768`) | Product variants |
| 7 | Hybrid prefill TTFT, **if A4 fixed it** | 103 → ~40 ms, basic only |
| 8 | Basic-mode rates on `simple` / `structured` prompts | 44.707 is one prompt; the record needs a distribution |

**If the device is available for only 15 minutes:** run priority 1, 5 reps, both
arms.

### Pre-committed decisions

| Observation | Action |
|---|---|
| hvx8 ≥ +20% | Compute-bound confirmed. It becomes the ship base; byte levers demoted; pursue further compute/scheduling work |
| hvx8 +5…20% | Partially compute-bound. Ship it; re-derive from priority 2's cycle profile |
| hvx8 < +5% | Compute model is wrong despite its out-of-sample hit. Priority 2 becomes the only admissible next input; revisit byte levers with A2's real number |
| fusion flat | Retire fusion as a lever — it was pre-fix dispatch recovery, already collected |
| W8 head < +5% | The 139 MB never reaches the device (§8.1 confirmed). Park it next to W4A16 |
| any arm's spread > its delta | Undecided. Re-run under B0; do not interpret |

## 5. What we explicitly do not do

- **LADE tuning** — parked by measurement (§6.8). `verify32` stays in the bins
  (weight-shared, free); the workstream does not. This also demotes learned draft
  heads (`eaglet`, `spd`): they raise acceptance, which only pays if speculation
  is worthwhile, and post-fix it is not at 0.6B.
- **W4A16 / INT4 anything** — v81 ships zero INT4 matmul kernels (§4.1).
- **Sparse weights compression** — measured, 0 bytes saved, model isn't sparse.
- **Multi-core / multi-process** — Genie returns 5005; a 2-core bin measured
  slower. ⚠ Note the 2-core comparison was against a *failed* v17 quant build, so
  it is weaker evidence than it reads; the 5005 is the real blocker.
- **DLBC for the weight stream** — it is defined as *inter-layer* DDR
  compression, i.e. activations. Confirm scope before spending a build.
- **KV INT8 via Genie config** — the flag exists only in the CPU backend.
  An ONNX-level path needs an unconfirmed kernel; FAE question, not a build.
- **x86 HTP simulation** — closed 2026-08-14.
- **More compute-removal in attention** — the 74.7% mine is exhausted; softmax,
  RMSNorm and elementwise sum to <5% of post-fix compute.

## 6. Open questions

Tracked in `REFERENCE.md` §8. The ones this plan turns on: **§8.11** (what binds
decode now), **§8.9** (hvx_threads), **§8.1** (does the W8 head reach the
device), **§8.4** (soc_model), plus rep-variance cause and the hybrid wiring bug.

One is outside our reach and possibly larger than everything here: whether
`llm_decode_burst` is actually the maximum HTP/DDR frequency. Only `virtio_clk`
is visible under the GVM, so it is a platform/hypervisor question. Route it to
the device/platform team, not to a build.
