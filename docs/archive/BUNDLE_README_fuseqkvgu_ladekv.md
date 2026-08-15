> # ⛔ ARCHIVED — do not build, ship, or follow this (2026-08-16)
>
> Kept for provenance only. Four independent reasons:
>
> 1. **It predates the GQA fix.** Its projected 11.5–12.4 tok/s is roughly a
>    quarter of the measured 44.707 tok/s baseline (`REFERENCE.md` §6.8).
> 2. **It is built for LADE, which is parked** — post-fix LADE is a 30%
>    regression (31.3 vs 44.7 basic).
> 3. **⚠ Its recommended interactive config crashes.** This file presents
>    `genie_dialog_demo.json` — `type: "lade"` **+** `max-num-tokens: 256` — as
>    the config to use. That exact pair SIGSEGVs (exit 139) on the first
>    speculation step; every demo run of the three bundles shipping it died.
>    `scripts/validate/lint_bundle_dialogs.py` now refuses the combination at
>    build time. **Do not copy any config from this document.**
> 4. **It names a pre-fix LADE bundle as the fallback ship config.** The ship
>    config is `gqafix_ladekv` in **basic** mode.
>
> Its two facts that existed nowhere else — the DLC-mtime staleness gate and the
> no-weight-sharing counterfactual — were migrated to `REFERENCE.md` §5 and §8.2
> before archiving. Current build recipes: `docs/BUILD_GUIDE.md` §5.

# qwen3_06b_w8a16_fuseqkvgu_ladekv — Qwen3-0.6B W8A16, QKV+Gate-Up fused, LADE

Built 2026-08-13 from `docs/MAX_TPS_QWEN3_0.6B.md` Phase B (that plan doc is now
superseded by V3 — see its header before reusing the sequencing), device-free, on
QAIRT 2.48.40.260702 for Hexagon v81 / SA8797P.

**This is the max-TPS candidate, and its speed is NOT yet measured.** It stacks
QKV+Gate-Up fusion onto the LADE build that measured **10.8 tok/s** on the GVM
guest (2026-08-11). Fusion measured **+15% in basic mode** on the device team's
build (8.98 vs 7.79 tok/s); nobody has yet combined it with lookahead decoding.
The projection is ~11.5–12.4 tok/s — treat it as a hypothesis to test, not a
spec. A prior projection on this device (`--quant-head`) came out **−14%**
on-device despite sound reasoning, so please measure rather than assume.

The unfused 10.8 tok/s build (`qwen3_06b_w8a16_ladekv.tar.gz`) remains the
fallback ship and the A/B baseline. Same topology, same encodings recipe, same
runtime configs — the only difference is fusion.

## Run it

```bash
adb push qwen3_06b_w8a16_fuseqkvgu_ladekv.tar.gz /data/local/tmp/
adb shell 'cd /data/local/tmp && tar xzf qwen3_06b_w8a16_fuseqkvgu_ladekv.tar.gz'
cd /data/local/tmp/qwen3_06b_w8a16_fuseqkvgu_ladekv
LD_LIBRARY_PATH=. ./genie-t2t-run -c genie_dialog.json -p '<templated prompt>'
```

Flat layout, `LD_LIBRARY_PATH=.`, no `lib/` subdir and no `ADSP_LIBRARY_PATH`.

## Three dialog configs on one binary

| File | Mode | Sampling | Use for |
|---|---|---|---|
| `genie_dialog.json` | `lade` (window 8 / ngram 3 / gcap 8) | greedy, unbounded | **the throughput number** and greedy parity |
| `genie_dialog_basic.json` | `basic` (AR-1) | greedy, unbounded | the AR-1 A/B baseline on identical weights |
| `genie_dialog_demo.json` | `lade` | temp 0.85 / top-k 50 / top-p 0.9, `max-num-tokens: 256` | free-form / interactive use |

**Use the demo config for anything interactive.** The other two are greedy
parity configs (temp 0, no `max-num-tokens`) — correct for validation, but in
free-form use they produce repetitive output until `Context Size was exceeded`.
That footgun is the shipped default's fault, not the model's.

## Rules that have each cost a debugging session

- **Apply the Qwen3 chat template, with an empty `<think>\n\n</think>` block**
  in the assistant prefix. Without it thinking mode triggers and latency
  balloons. `bos-token` is `-1` in all three configs because the template
  supplies `<|im_start|>` itself — do not set both.
- **Prompts must tokenize to ≥ 2 tokens.** A 1-token prompt hits `rand() % 0`
  in qualla's warmup and reads ~7 GB out of bounds.
- **Do not retune the lade block.** The guardrail is
  `(ngram−1) × (window + gcap) ≤ 32`, the verify graph's AR. Shipped 8/3/8 is
  exactly 32. Oversizing silently routes verification batches to a graph that
  cannot serve them.
- **Do not edit `graph_names` in `htp_backend_ext_config.json`.** The three
  names must stay exactly `prefill`, `decode`, `verify32` — they are baked into
  the ctx-bin at conversion time. A name that does not match means that graph
  silently gets backend defaults (4 MB VTCM, 24 MB spill); for the verify
  graph that is a null-deref SIGSEGV on the first speculation step.
- Keep `perf_profile: "llm_decode_burst"` and `rpc_polling_time: 9999`. The
  4-tier profile ladder spans 1.95×.
- Dialog `context.size` ≤ 1024. The ctx-bin's max CL is 1152; prompts longer
  than 128 tokens are chunked automatically by the past-KV prefill.

## What to report

Warm runs only, 3 reps, fixed prompt set — first run after a reconnect pays
1.8–2.0 s init vs ~790 ms warm.

1. **Sustained tok/s** from the profile's TGR, `genie_dialog.json`.
2. **Accepted tokens per verify call** (total tokens ÷ verify calls) and the
   per-call latency. `tok/s ≈ acceptance ÷ latency`, and acceptance is the
   variable that moves — the unfused build sits at ~1.94 accepted per ~180 ms
   call. **If fusion buys per-call latency but costs acceptance, it is a
   regression**, which is exactly how `--quant-head` failed. Please report both
   numbers, not just tok/s.
3. The same two numbers for `genie_dialog_basic.json`, so fusion's basic-mode
   +15% can be confirmed independently of LADE.

Compare cold-start numbers only like-for-like (init→first-logits vs the same,
never vs TTFT — that unit mismatch once produced a phantom "+134% regression").

## Build provenance

- **W8A16**: INT8 per-channel symmetric weights, FP16 activations, AIMET 2.36
  PTQ. `lm_head` is deliberately **FP16** — `--quant-head` is a measured −14%
  under LADE because it costs ~10% n-gram acceptance.
- **Fusion**: QKV and Gate-Up fused, with encodings surgery on all 28 layers
  (the shared qkv output tensor carries no encoding; the Q split gets a
  non-fused donor calibration's INT16 `q_proj` encoding; K/V stay FP16).
- **Topology B**, 3 graphs, all sharing one encodings lineage
  (`model_surgery.encodings`) so KV quant params are byte-identical across
  graphs: `prefill` AR=128 CL=1152 past-KV all-position logits, `decode` AR=1
  CL=1152, `verify32` AR=32 CL=1152.
- **Build config**: `O:3`, `vtcm_mb:16`, `hvx_threads:4`, `soc_model:0`,
  unsigned PD, weight sharing on.

Gates passed before shipping (all device-free):

| Gate | Result |
|---|---|
| AIMET `--eval` last-token argmax vs FP32, fused | **3/4** — identical to the non-fused donor, same prompt failing |
| Fused export wrapper vs HF logits | max abs diff **4.67e-05** |
| Fused AR=32 batched verify vs HF, all positions | max abs diff **3.05e-05**, 8/8 argmax match |
| QKV encodings surgery | **28/28 layers** |
| `parity_ladekv_read.py` (qualla's exact feed pattern, incl. chunking) | **6/6** — 4 single-chunk + 2 chunked (129 and 200 tokens), all argmax-identical to HF |
| ctx-bin graph names / weight sharing | see table below |

Final ctx-bin, verified with `qnn-context-binary-utility` after generation:

```
qwen3-0.6b-w8a16-fuseqkvgu-ladekv_ctx.bin   1,102,467,072 B (1.102 GB)

prefill    60 in / 57 out   input_ids [1,128]  mask [1,128,1152]  logits [1,128,151936]
decode     60 in / 57 out   input_ids [1,1]    mask [1,1,1152]    logits [1,1,151936]
verify32   60 in / 57 out   input_ids [1,32]   mask [1,32,1152]   logits [1,32,151936]
```

- Graph names are exactly `prefill`, `decode`, `verify32`, matching `graph_names`
  in the build-time and runtime HTP configs.
- **1.102 GB for three 1.074 GB graphs** — weight sharing is working (Phase A's
  unfused equivalent is 1,106,276,352 B; without sharing this would be ~3.2 GB).
- `prefill` has 60 inputs including `past_key_0_in [1,8,128,1024]` — it is the
  past-KV graph, not a bertcache graph, and it emits **all-position** logits.
  `verify32` likewise emits all 32 positions.
- `lm_head.weight` is `Float_16 [151936,1024]` — confirmed **not** `sFxp_8`.
- All three DLCs are newer than `model_surgery.encodings`, i.e. every graph was
  converted against this build's encodings; no stale DLC from an earlier build
  leaked into the ctx-bin.
