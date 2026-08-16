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

# Record the finished artifact and drop the claim. Call at the end of a build.
coord_done() {
    local name=${1:?name} path=${2:-} kind=${3:-ctxbin} note=${4:-}
    local coord=$LLMDEPLOY_ROOT/scripts/util/coord.py
    [ -n "${COORD_RECIPE:-}" ] && [ -f "$coord" ] || return 0
    python3 "$coord" record --recipe "$COORD_RECIPE" --name "$name" \
            --kind "$kind" --path "$path" --note "$note" || true
    python3 "$coord" release --recipe "$COORD_RECIPE" >/dev/null 2>&1 || true
}
