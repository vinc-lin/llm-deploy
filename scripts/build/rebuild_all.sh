#!/usr/bin/env bash
# Sequential rebuild of ALL deliverables with the fixed clip_weights_to_7f7f,
# plus the combined QKV+GateUp variant and the 1.7B baseline. ~3h unattended.
set -euo pipefail
cd "$(dirname "$0")/../.."
source scripts/env.sh
D=$LLMDEPLOY_DATA

echo "######## [A] baseline 0.6B W8A16 ########"
bash scripts/build/full_build.sh qwen3-0.6b-w8a16 128 1024
bash scripts/build/bundle.sh qwen3_06b_w8a16_local \
    "$D/work/ctxbin/qwen3-0.6b-w8a16/qwen3-0.6b-w8a16_ctx.bin"

echo "######## [B] Gate-Up fused ########"
bash scripts/build/full_build.sh qwen3-0.6b-w8a16-fusegu 128 1024 --fuse-gate-up
bash scripts/build/bundle.sh qwen3_06b_w8a16_fusegu_local \
    "$D/work/ctxbin/qwen3-0.6b-w8a16-fusegu/qwen3-0.6b-w8a16-fusegu_ctx.bin"

DONOR=$D/work/quant/qwen3-0.6b-w8a16-prefill

echo "######## [C] QKV fused (surgery) ########"
bash scripts/build/qkv_build.sh qwen3-0.6b-w8a16-fuseqkv "$DONOR" 128 1024
bash scripts/build/bundle.sh qwen3_06b_w8a16_fuseqkv_local \
    "$D/work/ctxbin/qwen3-0.6b-w8a16-fuseqkv/qwen3-0.6b-w8a16-fuseqkv_ctx.bin"

echo "######## [D] QKV + GateUp fused (surgery) — §4.1 target ########"
bash scripts/build/qkv_build.sh qwen3-0.6b-w8a16-fuseqkvgu "$DONOR" 128 1024 --fuse-gate-up
bash scripts/build/bundle.sh qwen3_06b_w8a16_fuseqkvgu_local \
    "$D/work/ctxbin/qwen3-0.6b-w8a16-fuseqkvgu/qwen3-0.6b-w8a16-fuseqkvgu_ctx.bin"

echo "######## [E] 1.7B baseline (CPU quant — 8GB VRAM too small) ########"
MODEL=$D/models/Qwen3-1.7B QUANT_DEVICE=cpu \
    bash scripts/build/full_build.sh qwen3-1.7b-w8a16 128 1024
bash scripts/build/bundle.sh qwen3_17b_w8a16_local \
    "$D/work/ctxbin/qwen3-1.7b-w8a16/qwen3-1.7b-w8a16_ctx.bin" \
    configs/genie_dialog_qwen3_0.6b.json \
    "$D/models/Qwen3-1.7B/tokenizer.json"

echo "######## DDR summary comparison ########"
grep -h "read_total_bytes" /home/vinc/llm-local/rebuild-all.log | tail -10 || true
echo "REBUILD ALL COMPLETE"
