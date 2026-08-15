# ⛔ v1 e2e pipeline — SUPERSEDED, binaries removed

**This bundle never loaded on an SA8797P, and its binaries have been deleted.**
Use [`../qwen3vl_4b_e2e_pipeline_v2/`](../qwen3vl_4b_e2e_pipeline_v2) instead.

## What happened

On the 2026-08-14 device attempt, node creation died before a single token:

```
ShapeError : attention_mask — Expected [ 1, 128, 2176] bitwidth=*. Found [ 1, 128, 128] bitwidth=2   (x2, one per shard)
Failed to create the Genie Node (-1)
SIGSEGV
```

Cause: in a **split** tower, shard 0's prefill graph has no `logits`, so
libGenie classifies it `DECODER_PREFILL` and rewrites its expected context
length to the cache-group maximum. An `AR == CL` "bertcache" prefill mask
(`[1,128,128]`) can never satisfy that.

v2 rebuilds the text tower with a past-KV prefill — `attention_mask
[1,128,2176]`, `past_key_N_in [1,8,128,2048]` — which is what the validator
demanded.

## Why this folder still exists

The failure is worth being able to reproduce, and reproducing it does **not**
need the 6.1 GB of binaries — it needs the tensor metadata. What remains here is
exactly that:

| File | Why it is kept |
|---|---|
| `qwen3vl-4b-w8a16_{1,2}_of_2.info.json` | The graph/tensor dumps that make the failure reproducible. Feed them to `scripts/validate/genie_load_check.py` and it regenerates the device's exact error, both shards |
| `qwen3vl-4b-vit-w8a16_ctx.info.json` | Vision tower metadata from the same drop |
| `genie_*.json`, `*.script`, `htp_backend_ext_config_*.json` | The exact configuration that was run |
| `prompt_seg{1,2}.txt`, `sample_image.{png,json}` | The exact prompt and image |
| `DEVICE_TEST.md` | The instructions the device team followed |

Deleted: both text ctx-bins, the ViT ctx-bin, the embedding LUT, the tokenizer,
the runtime `.so` set, `genie-app`, and `sample_image.raw` — 6.1 GB of bytes
that are either superseded by v2 or byte-identical to files in it.

> **Note on storage.** Removing files in a new commit hides them from the file
> listing, but the blobs remain referenced by the repository's git history.
> Reclaiming the underlying storage would require rewriting history, which would
> also discard the provenance of this drop. That trade-off has not been taken.

## Reproducing the failure

```bash
python scripts/validate/genie_load_check.py \
    --info qwen3vl-4b-w8a16_1_of_2.info.json qwen3vl-4b-w8a16_2_of_2.info.json \
    --config genie_text_generator_qwen3vl_4b.json
```

Expected output:

```
prefill_0 : attention_mask - Expected [ 1, 128, 2176] Found [ 1, 128, 128]
prefill_1 : attention_mask - Expected [ 1, 128, 2176] Found [ 1, 128, 128]
GENIE LOAD WOULD FAIL
```

The same check passes on v2. It is the gate that would have caught this before
shipping, and it now runs as part of every bundle build.
