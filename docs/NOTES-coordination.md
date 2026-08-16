# Coordination — how concurrent sessions avoid re-deriving the same artifact

Status: live convention as of 2026-08-16. Mechanised in `scripts/util/coord.py`
and `coord_guard` / `coord_done` in `scripts/env.sh`.

---

## 0. Why the previous rule could not work

CLAUDE.md carried this, and sessions duplicated builds anyway:

> **Before building a variant, check whether it already exists.** … `md5sum` the
> candidate against `work/ctxbin/*/` first. Concurrent sessions have duplicated
> builds this way.

The rule is unfollowable. You cannot `md5sum` a candidate that does not exist,
and producing it means running the ~20 min, multi-GB build the rule exists to
avoid. The check can only ever fire **after** the cost is sunk. Nobody ignored
this rule; it was impossible to obey.

That is the general lesson, and it outranks the specific mechanism below:

> **A convention that cannot be executed cheaply, at the moment the decision is
> made, is not a convention — it is a wish.** If following a rule costs more
> than breaking it, it will be broken, and writing it more emphatically does
> not change that.

## 1. The property that makes a real check possible

ctx-bin generation is **deterministic** (`REFERENCE.md` §8.5: two independent
rebuilds hours apart produced byte-identical bins). So the output is a pure
function of its inputs:

```
bytes = f(DLC set + order, graph names + order, backend config, SDK version)
```

Determinism means you can hash the **inputs** and get a key that identifies the
output *before it exists*. Measured on the real 0.6B DLC set (3 × 1.07 GB):

| | cost |
|---|---:|
| recipe key, cold | **5.5 s** |
| recipe key, warm (sidecar cache) | **0.3 ms** |
| the build it prevents | **~20 min + 1.09 GB** |

Cheap enough that it can run unconditionally, which is the whole point of §0.

## 2. Four different problems — do not conflate them

They fail differently and need different mechanisms.

| # | Problem | Mechanism | Where it lives |
|---|---|---|---|
| 1 | **Already built** (an earlier session, maybe on the other host) | registry lookup on the recipe key | `state/artifacts.tsv`, in git |
| 2 | **Being built right now** (two sessions start within minutes) | atomic advisory claim | `$LLMDEPLOY_DATA/work/claims/`, per host |
| 3 | **Same recipe, second name** (`ctrl` vs the baseline) | recipe key collides by construction; md5 index confirms | registry |
| 4 | **Overlapping non-artifact work** (docs, analysis, reports) | scope declaration + authority rule | §6 |

**Neither 1 nor 2 subsumes the other.** The registry cannot see an in-flight
build — the entry is written when the build *finishes*. A claim cannot see a
finished one — it is released when the build ends. You need both.

## 3. The registry — `state/artifacts.tsv`

One line per **binary** (not per directory), tab-separated, sorted by
`(recipe, name)`:

```
recipe  md5  bytes  kind  name  state  location  host  utc  commit  note
```

- **In git**, because git is the only channel that already synchronises the WSL
  box, `tank`, and every worktree. It is text and tiny; the artifacts stay in
  `$LLMDEPLOY_DATA`.
- **Sorted TSV, not JSON.** A single JSON object conflicts on every concurrent
  write. Sorted lines usually land at different offsets and git auto-merges
  them; a true conflict is resolved by keeping both lines.
- `state` distinguishes `present` / `stripped` / `bundled` / `remote`.
  **`stripped` is load-bearing.** The retention policy deletes the `.bin` and
  keeps the sidecar, so `work/ctxbin/qwen3-0.6b-w8a16qh/` contains only
  `info.json` — indistinguishable, from a directory listing, from "never
  built". Without an explicit `stripped` row, deliberately reclaimed artifacts
  read as gaps and invite exactly the rebuild the reclaim was worth.

**Known latency:** a registry entry crosses worktrees only at commit + merge.
Within a host that gap is covered by claims (§4), which are visible the instant
they are taken. Across hosts it is covered by the fact that the two hosts have
disjoint capabilities — tank cannot reach HF, this box cannot fit a 4B export —
so the same derivation rarely starts in both places. Push before a remote build
anyway; that was already the rule.

## 4. Claims — advisory, self-healing

`mkdir` is atomic on ext4. That is the entire mutual-exclusion primitive.

A claim records host, pid, branch, worktree, ISO start time, label and ETA.
Three deliberate properties:

- **Advisory, never enforcing.** A held claim prints who/where/how-long and
  exits non-zero; `COORD_FORCE=1` proceeds. A dead session's leftover must never
  be able to block real work.
- **Liveness beats timeouts.** A claim whose owning pid is gone *on this host*
  is reaped automatically, so build scripts need **no `EXIT` trap** to release.
  They already own one — `ctxbin_variant.sh`'s `rm -rf "$CFGDIR"` — and
  composing traps in bash is fragile enough that a botched compose would leak
  the temp config dir, a worse bug than the one being fixed.
  ⚠ `kill(pid, 0)` raises **EPERM** for a process owned by another user, which
  means it is *alive*. A blanket `except OSError` inverts that and reaps a live
  claim. Liveness for a claim taken on *another* host is undecidable, so those
  fall back to the `STALE_HOURS` age report.
- **Age is reported, not acted on.** Six hours (a 4B export peaks around three)
  marks a claim stale in the output; a human decides.

## 5. Using it

```bash
source scripts/env.sh                 # brings coord_guard / coord_done

# in a build script, next to disk_guard:
coord_guard "<label>" "<dlc_csv>" "<graph_csv>" '<overrides_json>' <eta_min>
...build...
coord_done  "<name>" "<path/to.bin>" ctxbin "<note>"
```

`coord_guard` exits the script with **3** (already built) or **4** (claimed by
someone else) and builds nothing. It **fails open**: if `coord.py` is missing or
cannot hash, it warns and lets the build proceed, because a coordination bug
must never be able to block real work.

Already wired into `scripts/build/ctxbin_variant.sh` and
`scripts/build/ladekv_build.sh`. `ladekv_build.sh` reads the *shared*
`configs/htp_backend_config.json` rather than taking a config argument, so its
recipe folds in that file's md5 — otherwise editing `configs/` would leave the
key unchanged and the registry would vouch for a stale bin.

By hand:

```bash
python3 scripts/util/coord.py who        # live claims, worktrees, recent branches
python3 scripts/util/coord.py lookup --dlc a,b,c --graphs x,y,z --overrides '{}'
python3 scripts/util/coord.py scan       # backfill/reconcile from disk, list aliases
```

**`scan` is the alias detector.** Run it after any batch of builds; it prints
byte-identical groups, which is how §2's problem 3 gets caught.

### What is in the recipe, and what is deliberately not

Included: each DLC's md5, **in order**; graph names, **in order**; the
canonicalised overrides dict (`sort_keys`, so `{"O":3,"hvx_threads":8}` and
`{"hvx_threads":8,"O":3}` agree); the SDK version string.

Order is part of the key for DLCs and graphs because the generator lays graphs
into the bin in the order given — the same set in a different order is not
guaranteed to produce the same bytes, so it must not collide.

Not included: output path, bin name, timestamps, host. Two names for one recipe
**should** collide — that is the point.

## 6. Non-artifact work: scope and authority

Hashing cannot help with two sessions editing the same documents, which happened
on 2026-08-15/16 (a session on `main` and this worktree revising the same drop).
Two rules:

**One session, one branch/worktree, and check before you start.**
`coord.py who` prints worktrees and every branch touched in the last 24 h. If
another branch has recent commits in your area, read them before writing —
`main` had already made four of the corrections this worktree was about to make.

**Every artifact set names exactly one authority document, and a change to the
authority must reach every landing page in the same commit.**
This is not bookkeeping. On 2026-08-16 the authority
(`DEPLOYMENT_AND_TEST_GUIDE.md`) was correctly reordered while the drop's
`README.md` — the file the guide itself tells readers to open first — still
sent the device team to the confounded arm. The authority was right and everyone
following instructions would still have done the wrong thing. A document nobody
opens first cannot carry a correction by itself.

## 7. What this found on its first run

Backfilling 27 artifacts surfaced two byte-identical groups that had never been
noticed:

| md5 | names | wasted |
|---|---|---:|
| `9c6024ad…` | `gqafix-ctrl-ladekv` == `gqafix-ladekv` | 1.09 GB, ~20 min, nearly a device arm |
| `235e71af…` / `fd552818…` | `splitkv-flat/*` == `splitkv/{1,2}_of_2` | **4.3 GB**, a 4B tower that only builds on tank |

Neither is deleted here — reclaiming is a separate, explicit decision. But both
were invisible before, and the second is on the model whose export does not fit
on this box at all.
