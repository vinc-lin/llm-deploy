# Test kit v2 — SA8797P decode-regime session

**Start at [`runsheet.md`](runsheet.md).** Every outcome already has a
pre-agreed meaning in [`decision_table.md`](decision_table.md), so the session
should need no round-trip with the build side.

**This supersedes `kit/` (2026-08-14). Do not run that one** — its priorities 4
and 5 point at bundles that cannot answer the questions they were written for.

## The 60-second version

The GQA fix landed: **44.707 tok/s** basic mode, 6.54× over the like-for-like
pre-fix control. Basic beats LADE, so basic ships and LADE is parked.

What we do not know is **why** 44.707 is the number. Two models fit it exactly —
one says the step is bound by DDR bytes (43.0 GB/s, 88% of the streaming
ceiling), the other says it is bound by DSP cycles (88.2M ÷ 4 HVX ≈ 22.06 ms vs
22.37 measured). They are indistinguishable at this operating point, and they
imply completely different next years of work.

They stop being indistinguishable the moment you perturb the model.

**Priority 1 is `hvx_threads: 8`** — ~10 minutes, and the only arm that varies
compute while holding DDR bytes *exactly* constant. The byte model predicts
exactly **0.0%** for it; anything above the rep spread falsifies that model
outright. It needs no assumption about clock speed or thread count.

**Priority 2 is `cl512`**, which discriminates by magnitude (+10.1% byte vs
+26.0% compute). **Priority 3 is the W8 head, and it is optional** — it changes
bytes *and* is suspected of never reaching the device, so it confounds the
question it was meant to answer.

## The one rule

**Never compare a decode rate from a BLENDED bundle against one from a PURE
bundle.** A bundle carrying a graph whose AR equals its CL generates *through*
that graph until the KV cache passes AR, so its tok/s is a blend of two rates —
and the blend runs fast, so the failure mode is a confident wrong number, not an
obviously broken one. That is what the old 11.72 tok/s figure was, and it
produced three phantom findings before it was caught.

Every arm in this kit is labelled, and every bundle shipped here is pure.

## Contents

| Path | What |
|---|---|
| `runsheet.md` | priority-ordered arms, the protocol, what to send back |
| `decision_table.md` | what each result means, agreed in advance |
| `run_all.sh` | runs the genie-t2t-run arms unattended; skips absent bundles |
| `prompts/` | three prompt classes (technical / simple / structured) |
| `expected/` | greedy reference continuations — diff before trusting any speed number |

Two things are deliberately **not** in `run_all.sh`:

- **P0** — pull `/data/local/tmp/results` off the device first. `/data` runs
  98–99% full and the 2026-08-15 per-op record is one cleanup away from gone.
- **P1** — the decode-only cycle profile uses `qnn-net-run`, not
  `genie-t2t-run`. Run `verify_profile_inputs.py` before it; last session's
  stated reason for skipping it turned out to be wrong, so the package now
  self-checks.

## Protocol change you will notice

**5 reps per arm, median reported, every raw value kept**, with a 30 s cool-down
between arms and temperature logged either side. The last session measured one
arm at 23.43 / 44.54 / 29.34 tok/s on a single binary — a spread larger than
every effect this session is chasing. Until that is isolated (priority 2, 8
reps), an A/B whose delta falls inside the rep spread decides nothing.
