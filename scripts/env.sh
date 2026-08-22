#!/usr/bin/env bash
# Local SA8797P pipeline environment. Source me.
# Repo (scripts/configs/docs) lives on /mnt/x (drvfs, Windows-visible).
# Heavy data (envs/sdk/models/work) lives on real ext4 for speed.
# Derived from this file's own location so a checkout elsewhere (tank, a
# worktree) works unchanged; still overridable from the environment.
export LLMDEPLOY_ROOT=${LLMDEPLOY_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}
export LLMDEPLOY_DATA=${LLMDEPLOY_DATA:-/home/vinc/llm-local}

export QAIRT_SDK=$LLMDEPLOY_DATA/sdk/qairt/2.48.40.260702
export PY_DEPLOY=$LLMDEPLOY_DATA/envs/qwen3-deploy/bin/python   # py3.10: torch+AIMET+onnx1.19
export PY_QAIRT=$LLMDEPLOY_DATA/envs/qairt-py312/bin/python     # py3.12: qairt-converter only

if [ -d "$QAIRT_SDK" ]; then
    export PATH=$QAIRT_SDK/bin/x86_64-linux-clang:$PATH
    # syslibs: locally-extracted libc++/libc++abi (QNN tools need LLVM libc++; no sudo)
    export LD_LIBRARY_PATH=$QAIRT_SDK/lib/x86_64-linux-clang:$LLMDEPLOY_DATA/syslibs/extracted/usr/lib/x86_64-linux-gnu:/home/vinc/.local/share/uv/python/cpython-3.12.12-linux-x86_64-gnu/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}
    export PYTHONPATH=$QAIRT_SDK/lib/python${PYTHONPATH:+:$PYTHONPATH}
fi

# Standing constraint: never let Windows C: run dry. $LLMDEPLOY_DATA lives on the
# WSL ext4.vhdx, which sits on C:, grows on demand and never shrinks by itself.
# A failed grow does NOT surface as ENOSPC -- the guest still believes it has
# hundreds of free GB, so the host write fails and the kernel delivers SIGBUS to
# every process touching an mmap'd page. That killed PID 1 and hard-crashed the
# VM three times on 2026-08-12. Lives here, not per-script, so a new build script
# cannot forget it. Pass the GB the NEXT step writes; 6 is the converter floor.
# Guard the volume the writes actually land on: on WSL that is Windows C:, since
# the guest's own df reports the vhdx's virtual free space and not the host's;
# off WSL (e.g. tank) there is no vhdx indirection, so check the data volume.
disk_guard() {
    local need_gb=${1:-6} free_gb target
    if [ -d /mnt/c ]; then target=/mnt/c; else target=$LLMDEPLOY_DATA; fi
    free_gb=$(df --output=avail -BG "$target" 2>/dev/null | tail -1 | tr -dc 0-9)
    if [ -z "$free_gb" ]; then
        echo "ABORT: disk_guard cannot read free space on $target" >&2; exit 1
    fi
    if (( free_gb < need_gb )); then
        echo "ABORT: $target free space ${free_gb}GB < ${need_gb}GB" >&2; exit 1
    fi
}

# Standing constraint: never derive the same artifact twice. Same shape as
# disk_guard and here for the same reason -- a per-script rule gets forgotten,
# and this one already was. The rule it replaces ("md5sum the candidate against
# work/ctxbin/*/ first") could not be followed: the candidate does not exist
# until you have paid the ~20 min and multiple GB it was meant to save.
#
# ctx-bin generation is deterministic, so the bytes are a pure function of
# (DLCs, graph names, config, SDK). coord_guard hashes THOSE -- in milliseconds,
# before anything is built -- and refuses if the registry already has that
# recipe, or if another session on this host is deriving it right now.
#
#   coord_guard <label> <dlc_csv> <graphs_csv> <overrides_json> [eta_min]
#
# Advisory by design: COORD_FORCE=1 proceeds anyway (use it when deliberately
# testing determinism). Fails OPEN -- if coord.py itself is broken or missing,
# it warns and lets the build run, because a coordination bug must never be
# able to block real work.
coord_guard() {
    local label=${1:?label} dlcs=${2:?dlc csv} graphs=${3:?graph csv}
    local overrides=${4:-'{}'} eta=${5:-20}
    local coord=$LLMDEPLOY_ROOT/scripts/util/coord.py
    [ -f "$coord" ] || { echo "WARN: coord.py missing; skipping dedup check" >&2; return 0; }

    COORD_RECIPE=$(python3 "$coord" hash --dlc "$dlcs" --graphs "$graphs" \
                          --overrides "$overrides" 2>/dev/null) || {
        echo "WARN: coord_guard could not hash the recipe; proceeding unguarded" >&2
        COORD_RECIPE=""; return 0; }
    export COORD_RECIPE

    python3 "$coord" guard --recipe "$COORD_RECIPE" --label "$label" --eta-min "$eta"
    local rc=$?
    if (( rc != 0 )); then
        echo "ABORT: coord_guard -- see above. Nothing was built." >&2
        exit $rc
    fi
}

# Wait for a long build to finish. Builds run far past any agent tool timeout,
# so the wait lives in a background shell -- and the obvious form of that wait
# is broken:
#
#   until ! pgrep -f "full_build.sh foo"; do sleep 20; done      # NEVER EXITS
#
# pgrep -f matches whole COMMAND LINES, and the pattern is sitting in the
# waiter's own command line, so the waiter matches itself and loops forever. It
# omits its own pid but not the shell that invoked it, and never the *other*
# waiter watching the same build. Measured 2026-08-19: three such shells were
# still spinning 1h36m-2h26m after their builds had completed. The bracket trick
# ([f]ull_build) does not fix it -- that only works when the BRACKETED text is
# what sits in the command line, which stops being true the moment the pattern
# is passed as an argument. So exclude by pid instead: self, every ancestor of
# self, and any other build_wait.
#
#   build_wait <pid>                          exact -- prefer this
#   build_wait <pattern> [poll_s] [max_min]
#
# 0 = the process is gone, 1 = timed out (default 4h -- an unbounded wait is how
# the zombies above happened), 2 = nothing matched at the START. 2 is loud on
# purpose: a typo'd pattern otherwise returns instantly and reads as success.
_build_wait_pids() {
    local pat=$1 pid p cl self_cl skip=" "
    p=$$
    while [ -n "$p" ] && [ "$p" != 0 ] && [ "$p" != 1 ]; do
        skip="$skip$p "
        p=$(ps -o ppid= -p "$p" 2>/dev/null | tr -dc 0-9)
    done
    self_cl=$(tr '\0' ' ' < "/proc/$$/cmdline" 2>/dev/null)
    for pid in $(pgrep -f -- "$pat" 2>/dev/null); do
        case "$skip" in *" $pid "*) continue ;; esac          # self + ancestors
        # Read the cmdline ONCE and treat "cannot read" as "already exited".
        # Checking /proc per-candidate instead fails OPEN on that race: every
        # $( ) below forks a subshell that inherits this shell's command line --
        # pattern included -- and dies within microseconds, so pgrep lists a pid
        # whose /proc entry is gone by the time the filter looks. Measured: that
        # phantom alone kept the loop alive indefinitely.
        # Braces + one 2>/dev/null on the group: a failed REDIRECTION is
        # reported by this shell, not by tr, so tr's own 2>/dev/null is silent
        # about exactly the case that happens most.
        cl=$( { tr '\0' ' ' < "/proc/$pid/cmdline"; } 2>/dev/null ) || continue
        [ -n "$cl" ] || continue                              # exited, or a kernel thread
        case "$cl" in *build_wait*) continue ;; esac          # another waiter, incl. siblings
        [ "$cl" = "$self_cl" ] && continue                    # a fork of this very shell
        echo "$pid"
    done
}

build_wait() {
    local target=${1:?pid or pattern} poll=${2:-20} max_min=${3:-240}
    local t0=$SECONDS deadline=$(( max_min * 60 )) pids el

    if [ -z "${target//[0-9]/}" ]; then                        # all digits -> pid mode
        kill -0 "$target" 2>/dev/null || {
            echo "build_wait: pid $target is not running -- NOT a success signal" >&2; return 2; }
        echo "build_wait: waiting on pid $target"
        while kill -0 "$target" 2>/dev/null; do
            (( SECONDS - t0 > deadline )) && {
                echo "build_wait: TIMEOUT after ${max_min}min; pid $target still alive" >&2; return 1; }
            sleep "$poll"
        done
    else
        pids=$(_build_wait_pids "$target")
        [ -n "$pids" ] || {
            echo "build_wait: nothing matches '$target' -- already finished, or the" >&2
            echo "            pattern is wrong. NOT a success signal." >&2; return 2; }
        echo "build_wait: waiting on [$(echo $pids)] matching '$target'"
        while pids=$(_build_wait_pids "$target"); [ -n "$pids" ]; do
            (( SECONDS - t0 > deadline )) && {
                echo "build_wait: TIMEOUT after ${max_min}min; still running: $(echo $pids)" >&2; return 1; }
            sleep "$poll"
        done
    fi
    el=$(( SECONDS - t0 ))
    echo "build_wait: done after $(( el / 60 ))m$(( el % 60 ))s"
}

# Record the finished artifact and drop the claim. Call at the end of a build.
coord_done() {
    local name=${1:?name} path=${2:-} kind=${3:-ctxbin} note=${4:-}
    local coord=$LLMDEPLOY_ROOT/scripts/util/coord.py
    [ -n "${COORD_RECIPE:-}" ] && [ -f "$coord" ] || return 0
    python3 "$coord" record --recipe "$COORD_RECIPE" --name "$name" \
            --kind "$kind" --path "$path" --note "$note" || true
    python3 "$coord" release --recipe "$COORD_RECIPE" >/dev/null 2>&1 || true
}
