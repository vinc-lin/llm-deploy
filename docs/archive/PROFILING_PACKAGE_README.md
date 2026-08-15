> # ⛔ ARCHIVED — superseded pre-GQA-fix package (2026-08-16)
>
> Kept for provenance. Three cautions:
>
> 1. **Its inputs are the known-defective ones.** `decode_profile_inputs.tar.gz`
>    was generated against the **pre-fix** decode graph (128-dim KV, 128-byte
>    `position_ids`) and was re-shipped unchanged inside the 2026-08-14 gqafix
>    drop, where it broke the P1 cycle profile — the graph is 64-dim. Regenerate
>    with `scripts/util/gen_decode_profile_inputs.py` against the target graph.
> 2. **Its ctx-bin is not the gqafix one.** `qwen3-0.6b-w8a16-decodeonly_ctx.bin`
>    and `qwen3-0.6b-w8a16-**gqafix**-decodeonly_ctx.bin` differ by one token in
>    the filename. Conflating them is how P1 was lost.
> 3. **`read_total_bytes = 961,130,496` here is pre-fix**, and its gloss "the
>    ~961 MB/token that makes decode DDR-bound" is a hypothesis that predicted
>    ~18 tok/s where 44.707 was measured (`REFERENCE.md` §6.9, correction #22).
>
> Its unique caveat about the `−100` vs `−1000` mask constant was migrated to
> `REFERENCE.md` §3.5 before archiving. Its live content is otherwise duplicated
> by `DEVICE_MEASUREMENT_REQUEST_2026-08-13.md`.
>
> **If a post-fix profiling package is ever shipped, write a fresh README rather
> than editing this one.**

# Decode profiling package — Qwen3-0.6B W8A16 (Test 2)

*Shipped 2026-08-13 for `docs/DEVICE_MEASUREMENT_REQUEST_2026-08-13.md` Test 2
(op-level profile of one decode step). Everything here is prebuilt — no build
step on your side.*

## What is on HF

Repo `vinccniv/sa8797p-qwen3-w8a16-bundles`, under `profiling/`. Access to this
repo is toggled as needed — if a download 404s while `hf auth whoami` succeeds,
it is private at that moment; ask and it will be opened or your account added.

| File | Size | What |
|---|---|---|
| `qwen3-0.6b-w8a16-decodeonly_ctx.bin` | 1.075 GB | **single-graph** ctx-bin holding only the AR-1 `decode` graph, so `qnn-net-run` needs no graph selection |
| `decode_profile_inputs.tar.gz` | ~1 MB | input tensors + `input_list.txt` for AR-1 and AR-32, plus the generator script |

The ctx-bin is built from the **same** `decode.dlc` as the shipped
`qwen3_06b_w8a16_local` / `ladekv` bundles — same encodings, same weights,
`O:3`, `vtcm_mb:16`, `hvx_threads:4`, `soc_model:0`, weight sharing on. Verified
after generation: 1 graph named `decode`, 60 inputs / 57 outputs,
`logits [1,1,151936]`, **spill_bytes = 0, fill_bytes = 0**, and the converter
reported `read_total_bytes = 961,130,496` (the ~961 MB/token that makes decode
DDR-bound). It is the production decode graph, isolated — not a rebuild with
different settings.

## Run it

```bash
tar xzf decode_profile_inputs.tar.gz     # -> ar1_decode/, ar32_verify/
cd ar1_decode

qnn-net-run --backend libQnnHtp.so \
    --retrieve_context /path/to/qwen3-0.6b-w8a16-decodeonly_ctx.bin \
    --input_list input_list.txt \
    --profiling_level detailed \
    --config_file htp_backend_ext_config.json \
    --output_dir out

qnn-profile-viewer --input_log out/qnn-profiling-data-0.log
```

Use the `htp_backend_ext_config.json` from any shipped bundle, but **narrow
`graph_names` to `["decode"]`** — this bin contains only that graph, and a name
listed for a graph that is not present is untested here. Keep
`perf_profile: "llm_decode_burst"` and `rpc_polling_time: 9999`.

Run 3 reps warm; discard the first (cold init is 1.8–2.0 s vs ~790 ms warm).

## What we are looking for in the output

One AR-1 decode step takes ~155 ms on device and streams ~961 MB of weights.
At the 49–67 GB/s your microbenchmarks measured for contiguous reads, that
traffic should cost ~15–20 ms; per-op dispatch overhead (~250 ops ×
30–60 µs) adds only ~10–15 ms. **~100+ ms per step is unaccounted for.** The
per-op table should show whether it is:

- MatMul/FullyConnected ops running far below streaming speed (→ fragmented
  weight access is the whole story, and fusion is the fix), or
- attention / softmax / RoPE / Convert ops costing more than expected (→ a
  specific op class is the target), or
- gaps *between* ops that no op accounts for (→ scheduling/sync, and the
  `hvx_threads` and `soc_model` A/Bs matter more than we thought).

Please send the raw `qnn-profile-viewer` output (or the profiling log) rather
than a summary — the per-op rows are the point.

## The AR-32 set (optional, "if time permits")

`ar32_verify/` holds the same tensors shaped for the AR=32 verify graph
(`input_ids [1,32]`, mask `[1,32,1152]`, past dim 1120). Comparing per-op
tables for AR-1 vs AR-32 shows exactly what amortizes across 32 positions —
the mechanism behind LADE's 1.7×.

**Note:** `verify32` lives in the multi-graph `ladekv` ctx-bin. If `qnn-net-run`
can select it there, these inputs work as-is. If graph selection is awkward,
say so and we will ship a `verify32`-only ctx-bin the same way (~20 min build,
~1.07 GB) — the inputs are already here either way.

## Regenerating the inputs

`gen_decode_profile_inputs.py` is included. KV tensors are **zero-filled** so
the package compresses to ~1 MB instead of ~132 MB; shapes and byte counts are
exact. Dense matmul on HVX/HMX should not be data-dependent, but if you want to
rule that out:

```bash
python3 gen_decode_profile_inputs.py --out ./ar1_random --random
```

Non-KV inputs are already realistic: `input_ids` is a real token id, the mask
is additive fp16 (`0.0` allow / `-1000.0` deny) for a near-full context, and the
RoPE tables are true cos/sin at the matching position with `rope_theta = 1e6`
(Qwen3, not Qwen2's 1e5).

> The `-1000.0` here is **this harness's synthetic input**, not a runtime
> constant. Our exported graphs are traced with `MASK_VALUE = -100.0`
> (`scripts/export/modeling_export.py:35`); qualla's own deny constant is not a
> literal in `attention-mask.cpp` and remains unconfirmed (`LOCAL_ENV.md` §5
> records it as unknown). Any deny value large enough to saturate softmax is
> equivalent for profiling, so the discrepancy does not affect these numbers —
> but do not cite this line as the runtime's value.
