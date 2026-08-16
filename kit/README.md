# Device session kit — superseded by `../kit-v2/`

**Use [`../kit-v2/`](../kit-v2). This directory is history.**

The 2026-08-15 session has run. Its kit is in `archive/2026-08-15/` and is no
longer live guidance: every priority in it is resolved, blocked, or re-ranked,
and its decision table is where two later-retracted numbers originated — the
pre-fix `read_total_bytes` used as a post-fix figure, and the ≈18 tok/s ceiling
derived from it (`REFERENCE.md` §6.10, corrections #27–#32).

`kit-v2/` was rebuilt around the discriminating pair rather than a byte-descent
ladder, and it carries the protocol change this session forced: **5 reps per
arm, median reported, thermal state recorded, and an A/B whose delta falls
inside the rep spread decides nothing.**

## What is still here, and why

| Path | Status |
|---|---|
| `prompts/` | **Live.** The three prompt classes — `simple`, `structured`, `technical`. Every measurement in this project uses one of these; keep them byte-stable or results stop comparing. `kit-v2/prompts/` carries the same files |
| `expected/` | **Live.** Reference outputs for quality-checking a bundle before trusting its speed |
| `archive/2026-08-15/` | History. Its `run_all.sh` is the `run_arm()` harness that `kit-v2/run_all.sh` was forked from |

⚠ One caveat that outlived the kit: `simple.txt` is short enough that a
bertcache bundle may never leave its prefill phase on it, so a decode rate
measured on that prompt alone is a blend, not a rate (`REFERENCE.md` §6.9).
`kit-v2` flags this inline.
