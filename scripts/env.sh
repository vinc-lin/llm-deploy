#!/usr/bin/env bash
# Local SA8797P pipeline environment. Source me.
# Repo (scripts/configs/docs) lives on /mnt/x (drvfs, Windows-visible).
# Heavy data (envs/sdk/models/work) lives on real ext4 for speed.
export LLMDEPLOY_ROOT=/mnt/x/code/llm-deploy
export LLMDEPLOY_DATA=/home/vinc/llm-local

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
