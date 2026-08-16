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
they are taken.

Across hosts there is no live channel at all, and the disjoint-capability
argument — tank cannot reach HF, this box cannot fit a 4B export — is weaker
than it looks: the 4B splitkv tower turned out to exist on **both** hosts
(§7), so artifacts do cross even when the derivation cannot. Treat the registry
as the cross-host mechanism and keep it current: push before a remote build
(already the rule), and run `coord.py scan` on the far side afterwards, then
bring the file back. That round trip is what makes tank's builds visible here.

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

## 7. What this found on its first runs

Backfilling 32 artifacts across both hosts surfaced three byte-identical groups
that had never been noticed. **Only one of them was real:**

| md5 | names on the WSL box | size each | reclaimable |
|---|---|---:|---:|
| `9c6024ad…` | `gqafix-ctrl-ladekv` == `gqafix-ladekv` | 1.09 GB | **1.09 GB** |
| `235e71af…` | `splitkv-flat/…_1_of_2` == `splitkv/1_of_2` | 1.86 GB | 0 — hard link |
| `fd552818…` | `splitkv-flat/…_2_of_2` == `splitkv/2_of_2` | 2.65 GB | 0 — hard link |

### Equal md5 is not two copies

The first version of this report claimed 5.60 GB. It was worth **1.09 GB**. The
`splitkv-flat` bins share inodes 109473 / 169230 with `splitkv/{1,2}_of_2` —
one object under two names — so deleting a name returns nothing.

An md5 index answers "same bytes", which is the right question for *don't ship
these as two arms* but the wrong one for *reclaim space*. Those need different
tests, and conflating them produces a report that promises 5× what it can
deliver. `scan` now stats each local row and groups by `(st_dev, st_ino)`,
prints hard links in a separate section that says deleting frees nothing, and
ends with a single `actually reclaimable` figure. **Quote that line, never the
group sizes.** Inode identity is only decidable for present files on this host;
anything else is reported as unknown rather than guessed.

**Resolved 2026-08-17.** `gqafix-ctrl-ladekv` deleted (1.09 GB; its bytes
survive in `gqafix-ladekv`, and the bundle tarball is on disk and on the hub —
it is now registered `bundled` against that tarball, so the alias warning still
fires if anyone tries to rebuild it). The `splitkv-flat` names were removed too
as namespace hygiene — `splitkv/` is the canonical layout, matches tank, and
carries the per-shard `info.json` and configs that flat lacked — but that freed
0 bytes and was never the point. Verified after: both `splitkv` bins retain
inodes 109473 / 169230 at full size with link count 1, and `gqafix-ladekv` still
hashes `9c6024ad…`.

`scan` also prunes: a row for a file that is gone would answer "already built"
for something absent, sending a session hunting for a bin instead of building
it — a worse failure than the duplicate build this file exists to prevent.

### Two bugs the cross-host run exposed, worth keeping in mind

Both were in the *reporting*, which is where a coordination tool fails quietly.

1. **A legacy key was `legacy:<name>`, so tank skipped its own copies** of
   `qwen3vl-4b-w8a16-splitkv` as already-known from this box's scan. A name on
   two hosts is no evidence of the same bytes. Now `legacy:<host>:<name>`.
   Real recipe keys stay host-agnostic on purpose — those are content-derived,
   and a build on tank *should* spare this box from repeating it.
2. **The alias report classified on "is more than one host involved"**, so the
   4.51 GB group — duplicated locally *and* legitimately present on tank — was
   filed entirely as expected-cross-host. The test has to be "does any one host
   hold this twice". A report that hides waste inside a line saying everything
   is fine is worse than no report.
