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
    export LD_LIBRARY_PATH=$QAIRT_SDK/lib/x86_64-linux-clang:$LD_LIBRARY_PATH
    export PYTHONPATH=$QAIRT_SDK/lib/python:$PYTHONPATH
fi
