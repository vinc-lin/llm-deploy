#!/usr/bin/env bash
# Qwen3-0.6B SA8797P standalone build kit -- environment. Source me first.
#
# Self-deriving: KIT_ROOT comes from this file's own location, so a checkout
# anywhere works unchanged. Everything else is overridable from the environment
# so the kit runs on a machine that looks nothing like the one it was written on.
#
# Verified hosts (2026-08-23):
#   tank         44 cores / 125 GB / no GPU / 937 GB native disk   <- builds run here
#   a WSL box    CUDA-capable, but its data volume sits on an ext4.vhdx on C:
#
# Quantization is bit-identical on both despite the CPU/CUDA difference
# (measured: model.encodings f4ff518b..., model_filtered_renamed.encodings
# 3e54eded..., model_renamed.onnx f588152... produced independently on each).
# That is what lets an md5 acceptance gate survive a host move.

export KIT_ROOT=${KIT_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)}
export KIT_DATA=${KIT_DATA:-$HOME/llm-local}

export QAIRT_SDK=${QAIRT_SDK:-$KIT_DATA/sdk/qairt/2.48.40.260702}
export PY_DEPLOY=${PY_DEPLOY:-$KIT_DATA/envs/qwen3-deploy/bin/python}
export PY_QAIRT=${PY_QAIRT:-$KIT_DATA/envs/qairt-py312/bin/python}
export MODEL=${MODEL:-$KIT_DATA/models/Qwen3-0.6B}

# Tests and gates run on the SYSTEM interpreter: the deploy env has no pytest,
# and adding one there would perturb an environment whose numerics are load-bearing.
export PY_TEST=${PY_TEST:-python3}

# Where the kit writes. Every heavy byte lands under here, never in the repo.
export KIT_OUT=${KIT_OUT:-$KIT_DATA/kit-out}

# Locally-extracted libc++/libc++abi, if the host needs them (QNN tools want
# LLVM libc++ and there may be no sudo to install it system-wide).
export KIT_SYSLIBS=${KIT_SYSLIBS:-$KIT_DATA/syslibs/extracted/usr/lib/x86_64-linux-gnu}

# cuda where genuinely present, cpu otherwise. The nvidia-smi probe short-circuits
# on a GPU-less host (tank) so sourcing this file stays instant there; only a host
# that looks GPU-capable pays for the multi-second torch import that settles it.
if [ -z "${QUANT_DEVICE:-}" ]; then
    QUANT_DEVICE=cpu
    if command -v nvidia-smi >/dev/null 2>&1 && [ -x "$PY_DEPLOY" ]; then
        "$PY_DEPLOY" -c 'import torch,sys; sys.exit(0 if torch.cuda.is_available() else 1)' \
            >/dev/null 2>&1 && QUANT_DEVICE=cuda
    fi
    export QUANT_DEVICE
fi

if [ -d "$QAIRT_SDK" ]; then
    export PATH=$QAIRT_SDK/bin/x86_64-linux-clang:$PATH
    export LD_LIBRARY_PATH=$QAIRT_SDK/lib/x86_64-linux-clang${KIT_SYSLIBS:+:$KIT_SYSLIBS}${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}
    export PYTHONPATH=$QAIRT_SDK/lib/python${PYTHONPATH:+:$PYTHONPATH}
fi

# Which volume the writes actually land on.
#
# On WSL, KIT_DATA sits inside an ext4.vhdx backed by Windows C:, and the guest's
# own df reports the vhdx's VIRTUAL free space, not the host's. A failed vhdx grow
# does not surface as ENOSPC: the host write fails and the kernel delivers SIGBUS
# to every process touching an mmap'd page. That killed PID 1 and hard-crashed the
# VM three times on 2026-08-12. Off WSL (tank) there is no indirection, so the data
# volume is the honest target.
disk_guard_target() {
    if [ -d /mnt/c ]; then echo /mnt/c; else echo "$KIT_DATA"; fi
}

# Call before any multi-GB step, sized to THAT step -- 6 GB is the converter
# floor, an export writes ~8.6 GB and should ask 20. A flat 6 GB check passes and
# then still runs the volume dry mid-step.
#
# Returns non-zero rather than exiting, so sourcing this file in a test or an
# interactive shell cannot kill the shell. Callers use `disk_guard N || exit 1`.
disk_guard() {
    local need_gb=${1:-6} free_gb target
    target=$(disk_guard_target)
    free_gb=$(df --output=avail -BG "$target" 2>/dev/null | tail -1 | tr -dc 0-9)
    if [ -z "$free_gb" ]; then
        echo "ABORT: disk_guard cannot read free space on $target" >&2
        return 1
    fi
    if (( free_gb < need_gb )); then
        echo "ABORT: $target free space ${free_gb}GB < ${need_gb}GB" >&2
        return 1
    fi
}

# Wait for a detached build. NEVER wait with `pgrep -f <pattern>`: the pattern
# sits in the waiter's own command line, so it matches itself and loops forever.
# Three such shells were measured still spinning 1h36m-2h26m after their builds
# had finished. A pid cannot self-match.
kit_wait() {
    local pid=${1:?pid} poll=${2:-30}
    kill -0 "$pid" 2>/dev/null || {
        echo "kit_wait: pid $pid is not running -- NOT a success signal" >&2
        return 2
    }
    while kill -0 "$pid" 2>/dev/null; do sleep "$poll"; done
}
