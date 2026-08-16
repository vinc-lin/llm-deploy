#!/usr/bin/env python3
"""Artifact coordination -- never derive the same ctx-bin twice.

WHY THIS EXISTS
---------------
The rule this replaces was "md5sum the candidate against work/ctxbin/*/ first".
It cannot be followed: the candidate does not exist until you build it, so the
check can only ever fire *after* paying the ~20 min and multiple GB it was meant
to save. It sat in CLAUDE.md for days and concurrent sessions duplicated builds
anyway -- `gqafix_ctrl_ladekv` and `gqafix_ladekv` are two 1.09 GB files with
one md5 (9c6024ad5b141137fbe22f3a4972eb96), built separately, shipped as two
device arms.

The fix uses the same property that makes the waste possible. ctx-bin
generation is deterministic (REFERENCE.md 8.5: two independent rebuilds hours
apart produced byte-identical bins), so the output is a pure function of its
inputs -- which means you can hash the INPUTS. That key costs milliseconds and
exists before the artifact does.

TWO MECHANISMS, BECAUSE THERE ARE TWO DIFFERENT RACES
-----------------------------------------------------
  registry  state/artifacts.tsv, in git  -- "someone already built this"
  claims    $LLMDEPLOY_DATA/work/claims  -- "someone is building it RIGHT NOW"

Neither subsumes the other. The registry cannot see an in-flight build (the
entry is written when the build finishes), and a claim vanishes when the build
ends. The registry also only crosses worktrees at commit+merge time, whereas
claims are shared by every session on the host the moment they are taken.

Claims are ADVISORY. A dead session's leftover claim must never block real work,
so a held claim prints who/where/how-long and lets you proceed with
COORD_FORCE=1. It buys a decision, not a lock.
"""
import argparse
import hashlib
import json
import os
import socket
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
REGISTRY = REPO / "state" / "artifacts.tsv"
DATA = Path(os.environ.get("LLMDEPLOY_DATA", "/home/vinc/llm-local"))
CLAIMS = DATA / "work" / "claims"

COLUMNS = ["recipe", "md5", "bytes", "kind", "name", "state",
           "location", "host", "utc", "commit", "note"]

# A claim older than this is reported as stale rather than trusted. Sized to the
# longest single derivation we run: a 4B export peaks around three hours.
STALE_HOURS = 6


# --------------------------------------------------------------------------
# identity
# --------------------------------------------------------------------------

def file_md5(path: Path, cache: bool = True) -> str:
    """md5 with a sidecar cache keyed on (size, mtime_ns).

    Hashing three ~700 MB DLCs on every build start is ~10 s -- affordable
    against a 20 min build, but not against an interactive `lookup`. The sidecar
    makes repeat lookups instant. It is keyed on size+mtime, so a rebuilt DLC
    invalidates it automatically.
    """
    path = Path(path)
    st = path.stat()
    side = path.with_suffix(path.suffix + ".md5")
    if cache and side.exists():
        try:
            got, size, mtime = side.read_text().split()
            if int(size) == st.st_size and int(mtime) == st.st_mtime_ns:
                return got
        except ValueError:
            pass  # malformed sidecar: fall through and rewrite it
    h = hashlib.md5()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(8 << 20), b""):
            h.update(chunk)
    digest = h.hexdigest()
    if cache:
        try:
            side.write_text(f"{digest} {st.st_size} {st.st_mtime_ns}\n")
        except OSError:
            pass  # read-only tree is fine; the cache is an optimisation
    return digest


def tool_version() -> str:
    sdk = os.environ.get("QAIRT_SDK", "")
    return Path(sdk).name or "unknown-sdk"


def recipe_hash(dlcs, graphs, overrides, tool=None) -> str:
    """The key: sha256 over everything that determines the output bytes.

    ORDER IS PART OF THE RECIPE for both dlcs and graphs -- the generator lays
    graphs into the bin in the order given, so the same set in a different order
    is not guaranteed to produce the same bytes and must not collide. Only the
    overrides dict is order-insensitive, and it is canonicalised with sort_keys
    so {"O":3,"hvx_threads":8} and {"hvx_threads":8,"O":3} agree.
    """
    if isinstance(overrides, str):
        overrides = json.loads(overrides or "{}")
    payload = {
        "tool": tool or tool_version(),
        "dlcs": [file_md5(Path(d)) for d in dlcs],
        "graphs": list(graphs),
        "overrides": overrides,
    }
    blob = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(blob.encode()).hexdigest()[:16]


# --------------------------------------------------------------------------
# registry
# --------------------------------------------------------------------------

def load_registry():
    if not REGISTRY.exists():
        return []
    rows = []
    for line in REGISTRY.read_text().splitlines():
        if not line.strip() or line.startswith("#"):
            continue
        parts = line.split("\t")
        parts += [""] * (len(COLUMNS) - len(parts))
        rows.append(dict(zip(COLUMNS, parts)))
    return rows


def save_registry(rows):
    """Sorted by (recipe, name) so concurrent appends land at different offsets.

    A single JSON object would conflict on every concurrent write; sorted TSV
    lets git auto-merge two sessions' entries, and the rare true conflict is
    resolved by keeping both lines.
    """
    REGISTRY.parent.mkdir(parents=True, exist_ok=True)
    rows = sorted(rows, key=lambda r: (r["recipe"], r["name"]))
    out = ["# artifact registry -- see docs/NOTES-coordination.md",
           "# " + "\t".join(COLUMNS)]
    out += ["\t".join(r.get(c, "") for c in COLUMNS) for r in rows]
    REGISTRY.write_text("\n".join(out) + "\n")


def git_commit() -> str:
    try:
        return subprocess.run(["git", "-C", str(REPO), "rev-parse", "--short", "HEAD"],
                              capture_output=True, text=True, timeout=10).stdout.strip()
    except Exception:
        return ""


def find(rows, recipe=None, md5=None):
    return [r for r in rows
            if (recipe and r["recipe"] == recipe) or (md5 and r["md5"] == md5)]


# --------------------------------------------------------------------------
# claims
# --------------------------------------------------------------------------

def claim_path(recipe: str) -> Path:
    return CLAIMS / recipe


def pid_alive(pid: int) -> bool:
    """Signal 0 liveness, distinguishing ESRCH from EPERM.

    A blanket `except OSError` is WRONG here and silently inverts the answer:
    kill(pid, 0) against a process owned by another user raises EPERM, which
    means the process *exists* and merely is not ours to signal. Treating that
    as "gone" reaps a live session's claim -- the precise failure this file is
    supposed to prevent, reintroduced one except-clause deeper.
    """
    try:
        os.kill(int(pid), 0)
        return True
    except ProcessLookupError:      # ESRCH -- really gone
        return False
    except PermissionError:         # EPERM -- alive, owned by someone else
        return True
    except (OSError, ValueError, TypeError):
        return False


def read_claim(recipe: str):
    who = claim_path(recipe) / "who"
    if not who.exists():
        return None
    try:
        return json.loads(who.read_text())
    except (OSError, ValueError):
        return {"label": "(unreadable claim)", "epoch": 0}


def claim_is_dead(info) -> bool:
    """A claim whose owning process is gone on THIS host is garbage, not a lock.

    This is why build scripts need no EXIT trap to release. They already own one
    (ctxbin_variant.sh's `rm -rf "$CFGDIR"`), and composing traps in bash is
    fragile enough that a botched compose would leak the temp config dir -- a
    worse bug than the one being fixed. Liveness moves the check to the reader,
    so a killed build, a crashed VM, or a closed session self-heals.

    Only decidable for claims taken on this host; a remote claim falls back to
    the STALE_HOURS age report.
    """
    return (info.get("host") == socket.gethostname()
            and not pid_alive(info.get("pid", -1)))


def take_claim(recipe: str, label: str, eta_min: int = 0):
    """mkdir is atomic on ext4 -- that is the entire mutual-exclusion primitive.

    Returns (True, None) on acquire, (False, holder) if already held.
    """
    CLAIMS.mkdir(parents=True, exist_ok=True)
    try:
        claim_path(recipe).mkdir()
    except FileExistsError:
        held = read_claim(recipe)
        if held is not None and claim_is_dead(held):
            drop_claim(recipe)            # reap and retry once
            try:
                claim_path(recipe).mkdir()
            except FileExistsError:
                return False, read_claim(recipe)
        else:
            return False, held
    info = {
        "label": label,
        "host": socket.gethostname(),
        "pid": os.getpid(),
        "branch": subprocess.run(["git", "-C", str(REPO), "rev-parse", "--abbrev-ref", "HEAD"],
                                 capture_output=True, text=True).stdout.strip(),
        "worktree": str(REPO),
        "epoch": int(time.time()),
        "utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "eta_min": eta_min,
    }
    (claim_path(recipe) / "who").write_text(json.dumps(info, indent=2))
    return True, info


def drop_claim(recipe: str):
    p = claim_path(recipe)
    if (p / "who").exists():
        (p / "who").unlink()
    if p.exists():
        p.rmdir()


def claim_age_h(info) -> float:
    return (time.time() - info.get("epoch", 0)) / 3600.0


# --------------------------------------------------------------------------
# subcommands
# --------------------------------------------------------------------------

def cmd_hash(a):
    print(recipe_hash(a.dlc.split(","), a.graphs.split(","), a.overrides))
    return 0


def cmd_lookup(a):
    recipe = a.recipe or recipe_hash(a.dlc.split(","), a.graphs.split(","), a.overrides)
    print(f"recipe {recipe}")
    hits = find(load_registry(), recipe=recipe)
    if not hits:
        print("  not in the registry -- this derivation is new")
        return 3
    for r in hits:
        print(f"  ALREADY BUILT  {r['name']}  [{r['state']}]  {r['location']}")
        print(f"                 md5 {r['md5']}  {r['bytes']} B  by {r['host']} on {r['utc']}")
        if r["note"]:
            print(f"                 note: {r['note']}")
    return 0


def cmd_guard(a):
    """lookup + claim, for build scripts. Non-zero means 'do not build'."""
    recipe = a.recipe or recipe_hash(a.dlc.split(","), a.graphs.split(","), a.overrides)
    forced = os.environ.get("COORD_FORCE") == "1"

    hits = find(load_registry(), recipe=recipe)
    if hits and not forced:
        r = hits[0]
        sys.stderr.write(
            f"\nCOORD: recipe {recipe} is already built as '{r['name']}' [{r['state']}]\n"
            f"       {r['location']}\n"
            f"       md5 {r['md5']}  built by {r['host']} on {r['utc']}\n"
            f"       Building it again reproduces the same bytes. If you need it\n"
            f"       under a second name, copy it -- do not re-derive.\n"
            f"       Override with COORD_FORCE=1 if you are testing determinism.\n\n")
        return 3

    ok, holder = take_claim(recipe, a.label or "(unlabelled)", a.eta_min)
    if not ok:
        age = claim_age_h(holder)
        stale = age > STALE_HOURS
        sys.stderr.write(
            f"\nCOORD: recipe {recipe} is being built RIGHT NOW.\n"
            f"       {holder.get('label')}\n"
            f"       host {holder.get('host')} pid {holder.get('pid')} "
            f"branch {holder.get('branch')}\n"
            f"       started {holder.get('utc')} ({age:.1f} h ago)"
            f"{'  <-- STALE, the session probably died' if stale else ''}\n"
            f"       Release with: coord.py release --recipe {recipe}\n"
            f"       Override with COORD_FORCE=1.\n\n")
        if not forced:
            return 4
    print(f"COORD: claimed {recipe} ({a.label})")
    return 0


def cmd_claim(a):
    ok, holder = take_claim(a.recipe, a.label or "(unlabelled)", a.eta_min)
    if not ok:
        print(f"held by {holder.get('label')} on {holder.get('host')} "
              f"({claim_age_h(holder):.1f} h ago)")
        return 4
    print(f"claimed {a.recipe}")
    return 0


def cmd_release(a):
    drop_claim(a.recipe)
    print(f"released {a.recipe}")
    return 0


def cmd_record(a):
    path = Path(a.path)
    rows = [r for r in load_registry() if not (r["recipe"] == a.recipe and r["name"] == a.name)]
    if path.exists() and path.is_file():
        digest, size, state = file_md5(path, cache=False), path.stat().st_size, a.state or "present"
    else:
        digest, size, state = a.md5 or "", a.bytes or "", a.state or "stripped"
    rows.append({
        "recipe": a.recipe, "md5": digest, "bytes": str(size), "kind": a.kind,
        "name": a.name, "state": state, "location": a.location or str(path),
        "host": socket.gethostname(),
        "utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "commit": git_commit(), "note": a.note or "",
    })
    save_registry(rows)
    print(f"recorded {a.name} recipe={a.recipe} md5={digest}")

    dupes = [r for r in rows if r["md5"] == digest and r["name"] != a.name and digest]
    if dupes:
        print("  ALIAS: byte-identical to " + ", ".join(sorted(r["name"] for r in dupes)))
        print("  These are the same binary. Do not ship them as separate arms.")
    return 0


def cmd_who(a):
    print("== live claims ==")
    live = sorted(CLAIMS.glob("*/who")) if CLAIMS.exists() else []
    if not live:
        print("  (none)")
    for who in live:
        info = json.loads(who.read_text())
        age = claim_age_h(info)
        if claim_is_dead(info):
            flag = "  <-- DEAD (owning pid is gone; will be reaped on next guard)"
        else:
            flag = "  <-- STALE" if age > STALE_HOURS else ""
        print(f"  {who.parent.name}  {info.get('label')}")
        print(f"    host {info.get('host')} pid {info.get('pid')} "
              f"branch {info.get('branch')}  {age:.1f} h ago{flag}")

    print("\n== worktrees ==")
    wt = subprocess.run(["git", "-C", str(REPO), "worktree", "list"],
                        capture_output=True, text=True).stdout.strip()
    print("  " + wt.replace("\n", "\n  "))

    print("\n== branches touched in the last 24 h ==")
    log = subprocess.run(
        ["git", "-C", str(REPO), "for-each-ref", "--sort=-committerdate",
         "--format=%(refname:short)\t%(committerdate:relative)\t%(subject)",
         "refs/heads"], capture_output=True, text=True).stdout.strip()
    for line in log.splitlines()[:8]:
        print("  " + line)
    return 0


def cmd_scan(a):
    """Backfill the registry from what is on disk.

    Pre-convention artifacts have no recoverable recipe -- the DLC set that
    produced them was never recorded -- so they enter as recipe
    'legacy:<host>:<name>'. That is deliberately not a real key: it will never
    match a future lookup, so it cannot cause a false 'already built'. What it
    DOES give immediately is the md5 index, which is what surfaces aliases like
    ctrl == baseline.

    The host belongs in a LEGACY key and must NOT go in a real one. A legacy key
    is a name-shaped placeholder, and the same name on two hosts is no evidence
    of the same bytes -- unscoped, tank's scan skipped its own copies of
    qwen3vl-4b-w8a16-splitkv as "already known" from this box's scan, hiding
    exactly the cross-host divergence worth knowing about. A real recipe key is
    derived from content and must stay host-agnostic, so that a build on tank
    does spare this box from repeating it.
    """
    rows = load_registry()
    known = {(r["recipe"], r["name"]) for r in rows}
    root = DATA / "work" / "ctxbin"

    # A split tower keeps one bin per shard in a subdir (1_of_2/, 2_of_2/), so a
    # top-level-only walk silently misses exactly the artifacts whose rebuild is
    # most expensive. Descend one level when the dir itself holds no bin.
    dirs = []
    for d in sorted(p for p in root.iterdir() if p.is_dir()):
        shards = sorted(p for p in d.iterdir() if p.is_dir() and any(p.glob("*.bin")))
        dirs += shards if (shards and not any(d.glob("*.bin"))) else [d]

    def stem(d):
        return d.name if d.parent == root else f"{d.parent.name}/{d.name}"

    # ONE ENTRY PER BIN, not per directory. A flattened split tower keeps both
    # shards side by side in one dir, so indexing bins[0] silently hid the second
    # 2.5 GB shard -- and hiding an artifact is the exact failure this registry
    # exists to prevent.
    units = []
    for d in dirs:
        bins = sorted(d.glob("*.bin"))
        if bins:
            multi = len(bins) > 1
            units += [(f"{stem(d)}/{b.stem}" if multi else stem(d), b, d) for b in bins]
        elif any(d.glob("*.json")):
            units.append((stem(d), None, d))

    host = socket.gethostname()
    seen = 0
    for name, b, d in units:
        recipe = f"legacy:{host}:{name}"
        if (recipe, name) in known and not a.rehash:
            continue
        if b is not None:
            digest, size, state, loc = file_md5(b, cache=False), b.stat().st_size, "present", str(b)
            print(f"  {name}: {digest}")
        else:
            digest, size, state, loc = "", "", "stripped", str(d)
            print(f"  {name}: stripped (sidecar only)")
        rows = [r for r in rows if not (r["recipe"] == recipe and r["name"] == name)]
        rows.append({
            "recipe": recipe, "md5": digest, "bytes": str(size), "kind": "ctxbin",
            "name": name, "state": state, "location": loc,
            "host": socket.gethostname(),
            "utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "commit": git_commit(), "note": "backfilled by scan",
        })
        seen += 1
    save_registry(rows)
    print(f"\nscanned {seen} artifact dirs into {REGISTRY.relative_to(REPO)}")

    # Same-host and cross-host duplicates are NOT the same finding. Two copies on
    # one host are waste. A copy on each host is often correct -- tank is the
    # canonical home for the 4B lineage precisely because that export does not
    # fit here -- so reporting them identically would train people to ignore the
    # report, which costs more than the duplicates do.
    by_md5 = {}
    for r in rows:
        if r["md5"]:
            by_md5.setdefault(r["md5"], []).append((r["host"], r["name"], r["bytes"]))
    same, cross = [], []
    for m, entries in sorted(by_md5.items()):
        if len(entries) < 2:
            continue
        (same if len({h for h, _, _ in entries}) == 1 else cross).append((m, entries))

    def show(title, groups):
        if not groups:
            return
        print(f"\n== {title} ==")
        for m, entries in groups:
            gb = int(entries[0][2] or 0) / 1e9
            print(f"  {m}  ({gb:.2f} GB each)")
            for h, n, _ in sorted(entries):
                print(f"    {h}:{n}")

    show("byte-identical ON ONE HOST -- each group is ONE binary stored twice", same)
    show("same bytes on BOTH hosts -- expected for tank-canonical lineages", cross)
    return 0


def main():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = p.add_subparsers(dest="cmd", required=True)

    def recipe_args(sp, required_recipe=False):
        sp.add_argument("--recipe", required=required_recipe)
        sp.add_argument("--dlc", default="", help="comma-separated DLC paths (order matters)")
        sp.add_argument("--graphs", default="", help="comma-separated graph names (order matters)")
        sp.add_argument("--overrides", default="{}")

    s = sub.add_parser("hash", help="print the recipe hash for a proposed build")
    recipe_args(s); s.set_defaults(fn=cmd_hash)

    s = sub.add_parser("lookup", help="has this exact derivation been built before?")
    recipe_args(s); s.set_defaults(fn=cmd_lookup)

    s = sub.add_parser("guard", help="lookup + claim; non-zero means do not build")
    recipe_args(s)
    s.add_argument("--label", default="")
    s.add_argument("--eta-min", type=int, default=0)
    s.set_defaults(fn=cmd_guard)

    s = sub.add_parser("claim")
    s.add_argument("--recipe", required=True)
    s.add_argument("--label", default="")
    s.add_argument("--eta-min", type=int, default=0)
    s.set_defaults(fn=cmd_claim)

    s = sub.add_parser("release")
    s.add_argument("--recipe", required=True)
    s.set_defaults(fn=cmd_release)

    s = sub.add_parser("record", help="add a finished artifact to the registry")
    s.add_argument("--recipe", required=True)
    s.add_argument("--name", required=True)
    s.add_argument("--kind", default="ctxbin")
    s.add_argument("--path", default="")
    s.add_argument("--location", default="")
    s.add_argument("--state", default="")
    s.add_argument("--md5", default="")
    s.add_argument("--bytes", default="")
    s.add_argument("--note", default="")
    s.set_defaults(fn=cmd_record)

    s = sub.add_parser("who", help="what else is running: claims, worktrees, recent branches")
    s.set_defaults(fn=cmd_who)

    s = sub.add_parser("scan", help="backfill the registry from disk")
    s.add_argument("--rehash", action="store_true")
    s.set_defaults(fn=cmd_scan)

    a = p.parse_args()
    sys.exit(a.fn(a))


if __name__ == "__main__":
    main()
