# Device session kit

**The 2026-08-15 session has run. Its kit is in `archive/2026-08-15/` and is no
longer live guidance** — every priority in it is resolved, blocked, or
re-ranked, and its decision table is the origin of two numbers that later had to
be retracted (`REFERENCE.md` §6.9).

Reusable assets that carry forward:

| Path | What it is |
|---|---|
| `prompts/` | the three prompt classes — `simple`, `structured`, `technical`. Every measurement in this project uses one of these; keep them byte-stable or results stop comparing |
| `expected/` | reference outputs for the same three, for quality-checking a bundle before trusting its speed |
| `archive/2026-08-15/run_all.sh` | the `run_arm()` harness — bundle/dialog presence checks, warm-up discard, per-rep `--profile` capture, non-fatal arm failures. Fork kit v2's runner from this rather than rewriting it |

## Building kit v2

Do it when the device is next available, from `docs/PLAN_0.6B_max_tps.md` §4,
not from the archived runsheet. Three things must change:

1. **5 reps per arm, median reported, every raw value kept**, with thermal state
   recorded before and after and a fixed cool-down between arms. The archived
   script uses 3, and the `pastkv2g` arm it produced spread 23.43 / 44.54 /
   29.34 tok/s on **one binary** — wider than most effects worth measuring. An
   A/B whose delta falls inside the rep spread decides nothing.
2. **Priority 1 is the `hvx_threads: 8` A/B**, not the byte arms. See
   `REFERENCE.md` §8.9 and §8.11.
3. **Regenerate `decode_profile_inputs` before shipping any profiling package.**
   The 2026-08-15 cycle profile was lost because pre-fix inputs (128-dim KV)
   were re-shipped against a 64-dim graph. Add "regenerate profiling inputs
   whenever graph I/O changes" to the drop checklist.

Two rules from the archived kit are still true and worth carrying verbatim:

- **`hvx_threads` is build-time only.** Variants that change it are separate
  ctx-bins, not config edits.
- **Backend-config keys are silently ignored when misspelled or misplaced.** If
  you see `Unknown Key` warnings, that arm is not testing what it claims.
