# Qwen3-0.6B W8A16 + QKV + GU Fusion — Full Test Data Sheet

> Reconstructed from the ten screen photographs in `reports/IMG_2992..IMG_3003.jpeg`.
> Content is transcribed verbatim where legible; a few glyphs obscured by screen glare are
> marked with `[?]` and listed under [Transcription notes](#transcription-notes).

---

## 1. Bundle Info

| Field | Value |
|---|---|
| Bundle file | `qwen3_06b_w8a16_fuseqkvgu_local.tar.gz` (1.3 GB, 1,319,664,435 bytes) |
| Ctx-bin filename | `qwen3-0.6b-w8a16-fuseqkvgu_ctx.bin` (1.4 GB on-device) |
| Source | HuggingFace `vinccniv/qwen3-06b-w8a16-sa8797p` |
| Build date | 2026-08-10 |
| SDK / runtime | QAIRT 2.48.40.260702, libGenie 1.19.0, QNN v2.37.0 |
| Model | `Qwen/Qwen3-0.6B` base (**not** Instruct) |
| Quantization | AIMET 2.36 PTQ, per-channel symmetric INT8 weights + FP16 activations (W8A16) |
| Fusions | QKV fusion (encodings-level surgery) + Gate-Up MLP fusion |
| Graphs | 2-graph ctx-bin: prefill (AR-128 CL-128) + decode (AR-1 CL-1152), weight-shared |
| VTCM | 16 MB |
| Opt level | `O=3`, `pd_session=unsigned`, `dsp_arch=v81`, 4 HVX threads |
| Perf profile | `llm_decode_burst` |
| KV dim | 128, RoPE dim = 64, theta = 1,000,000 |
| Vocab size | 151,936 |
| Context size | 1024 |
| Layout on device | Flat (no `lib/` subdir), `LD_LIBRARY_PATH=.`, all 7 required `.so` present |

## 2. Device

| Field | Value |
|---|---|
| Board | SA8797P (nordy / Gen5), Hexagon v81 HTP, Android GVM |
| Serial | `REDACTED` (via ssh <JUMPHOST> → adb) |
| Device path | `/data/local/tmp/qwen3_06b_w8a16_fuseqkvgu_local/` |

---

## 3. Exact Genie Command Line (primary smoke test)

```bash
cd /data/local/tmp/qwen3_06b_w8a16_fuseqkvgu_local
LD_LIBRARY_PATH=. ./genie-t2t-run \
  -c genie_dialog.json \
  -p "What is 2+2? Answer with one number." \
  --log error
```

Verbose log version:

```bash
LD_LIBRARY_PATH=. ./genie-t2t-run \
  -c genie_dialog.json \
  -p "What is the capital of France?" \
  --log verbose
```

## 4. Exact `genie_dialog.json`

```json
{
  "dialog": {
    "version": 1,
    "type": "basic",
    "context": {
      "version": 1,
      "size": 1024,
      "n-vocab": 151936,
      "bos-token": -1,
      "eos-token": [151645, 151643]
    },
    "sampler": {
      "version": 1,
      "seed": 42,
      "temp": 0.0,
      "top-k": 1,
      "top-p": 1.0
    },
    "tokenizer": {
      "version": 1,
      "path": "tokenizer.json"
    },
    "engine": {
      "version": 1,
      "n-threads": 3,
      "backend": {
        "version": 1,
        "type": "QnnHtp",
        "QnnHtp": {
          "version": 1,
          "use-mmap": true,
          "spill-fill-bufsize": 0,
          "mmap-budget": 25,
          "poll": true,
          "cpu-mask": "0xe0",
          "kv-dim": 128,
          "allow-async-init": true,
          "enable-graph-switching": true
        },
        "extensions": "htp_backend_ext_config.json"
      },
      "model": {
        "version": 1,
        "type": "binary",
        "binary": {
          "version": 1,
          "ctx-bins": ["qwen3-0.6b-w8a16-fuseqkvgu_ctx.bin"]
        },
        "positional-encoding": {
          "type": "rope",
          "rope-dim": 64,
          "rope-theta": 1000000
        }
      }
    }
  }
}
```

## 5. `htp_backend_ext_config.json`

```json
{
  "graphs": [
    {
      "graph_names": ["prefill", "decode"],
      "O": 3,
      "vtcm_mb": 16,
      "hvx_threads": 4
    }
  ],
  "devices": [
    {
      "dsp_arch": "v81",
      "pd_session": "unsigned",
      "cores": [
        {
          "core_id": 0,
          "perf_profile": "llm_decode_burst",
          "rpc_polling_time": 9999
        }
      ]
    }
  ]
}
```

## 6. Sampler Configuration

| Field | Value |
|---|---|
| Mode | Greedy (argmax) |
| Temperature | 0.0 |
| Top-k | 1 |
| Top-p | 1.0 |
| Seed | 42 |

---

## 7. Input Prompt & Chat Template

- **Prompt (raw):** `"What is 2+2? Answer with one number."`
- **Chat template:** NOT applied. Genie T2T with `-p` passes plain text directly to the
  tokenizer. The bundle uses the base model (`Qwen/Qwen3-0.6B`, not Instruct), so there is
  no chat template to apply.
- **Tokenized prompt** (from logcat, `dialog-tokens`): `[3838, 374, 279, 6722, 315, 9625, 30]`
  — 7 tokens (for `"What is the capital of France?"`).

## 8. Raw Model Output (verbatim, 1024-ctx run)

```
Using libGenie.so version 1.19.0

[INFO]  "Using create From Binary List Async"
[INFO]  "Allocated total size = 132490496 across 1 buffers"
[PROMPT]: What is 2+2? Answer with one number.

[BEGIN]: !!!!!ロ[?]Half!!!!!!ledger storm!!!!!!earned!!!!!!ῶ}">
!!!!!!légi temple!!!!!! BitteHUD!!!!!! sprzedaż!!!!!!懸bases!!!!!!ﾞﾞ_IS!!!!!!\Helpers Company!!!!!!.upper
cook!!!!!ζcccc!!!!!!椤(score!!!!!! jakieś Mega!!!!!! nearer_IM!
2+2+2+2+2+2+2+2+2+2+2+2+2+2+2+2+2+2+2+2+2+2+2+2+2+2+2+2+2+2+2+2+2+2+2+2+2+2+2+2+2!!!!!!!!!!!
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!![END]
Context Size was exceeded.
Failed to query.
```

## 9. First ~20 Generated Tokens (approximated from raw output)

The model outputs in a `"!!!!!!<token_chunk>!!!!!!"` pattern, not cleanly token-separated.
Rough tokenization of the first 20 tokens (from `[BEGIN]`):

| # | Token (approx) |
|---|---|
| 1 | `!` |
| 2 | `!` |
| 3 | `!` |
| 4 | `!` |
| 5 | `!` |
| 6 | `!` |
| 7 | `ロ[?]` (non-Latin glyph, unreadable in photo) |
| 8 | `Half` |
| 9 | `!` |
| 10 | `!` |
| 11 | `!` |
| 12 | `!` |
| 13 | `!` |
| 14 | `!` |
| 15 | `ledger` |
| 16 | ` storm` |
| 17 | `!` |
| 18 | `!` |
| 19 | `!` |
| 20 | `!` |

### Was the first generated token already incorrect?

**Yes.** The very first output character is `!` (exclamation mark), which is completely wrong
for the question "What is 2+2?". The model never produces a digit or any coherent response.
The failure is immediate — the first sampled token is garbage.

This is confirmed by the logits analysis: the first `input-logits` entry after prompt
processing shows zeroed-out logits (`[0, 0, 0, 0, 0, 0, 0, 0, 0, 0]`) for the first 10
positions (before the full logit array), meaning the KV cache state is already corrupt after
just one prefill step.

---

## 10. Genie KPI Block

From logcat verbose, `"What is the capital of France?"` — 7-token prompt:

```
init:      [last:762.20  total:762.20  min:0.00        max:762.20  avg:762.20 (msec) count:1]
prompt:    [last: 53.41  total: 53.41  min:53.41       max: 53.41  avg: 53.41 (msec) count:1]
generate:  [last:  0.00  total:  0.00  min:UINT64_MAX  max:  0.00  avg:  0.00 (msec) count:0]
tps-prompt:    131.06
tps-generate:    0.00
```

> **Note:** the `generate` counter = 0 because the run was cut off by the context-exceeded
> error before the final summary was written to logcat. The `tps-generate: 0.00` is from the
> prefill-phase summary, not the full run.

### Decode Throughput (measured)

| Metric | Value |
|---|---|
| Decode steps measured | 144 steps (from logcat `run-inference complete` timestamps, decode graph) |
| First decode step latency (graph switch) | 323.3 ms |
| Average decode step latency (steps 2–144) | ~159.6 ms |
| Steady-state decode throughput | **~6.27 tok/s** |

### Prefill Throughput (measured)

| Metric | Value |
|---|---|
| Prompt length | 7 tokens (single prefill batch) |
| Prefill latency | 53.4 ms (first batch) |
| Steady per-token prefill | ~50.0 ms/token (chunked AR-128 prefill batches, 1 token added per step) |
| Reported `tps-prompt` | 131.06 tok/s (Genie's calculation, likely batch-based) |

> **Note on prefill strategy:** Genie uses a chunked prefill approach with the AR-128 prefill
> graph. With a 7-token prompt, the first step processes all 7 in one prefill call (53 ms),
> then subsequent 1-token-at-a-time generation steps continue using the prefill graph until
> the KV fills past 128 positions, at which point it switches to the AR-1 decode graph.

---

## 11. Full Genie Runtime Logs — Key Events (init → prefill → decode → error)

### Initialization phase (04:34:04.97 → 04:34:05.74, ~770 ms)

- Genie Logger created with level: 4 (verbose)
- `dialog-new`: config parsed, 1024 ctx, 151936 vocab, greedy sampler, 3 threads, QnnHtp
- `qnn-api` initialized with 2 graph(s): prefill + decode
- Graphs loaded: `{(1, 1152): 1, (128, 128): 1}`
- Initializing HtpProvider → HTP initialization completed successfully
- FastRPC: user PD created on domain 25600, unsigned, skel loaded
- `QnnBackend_create` → success
- `QnnDevice_create` → success (deviceId=0, v81 arch, unsigned PD)
- Perf infrastructure: `llm_decode_burst` (profile 6), power config applied
- `QnnContext_createFromBinaryListAsync` → 1 context, 2 graphs
- Graph prefill loaded: graph prefill is loaded 1
- Graph decode initially: graph decode is loaded 0 (loaded lazily via graph-switching)
- Allocated total size = 132,490,496 across 1 buffers (132 MB)

### Warnings during init (non-fatal, expected on this platform)

- `kv-update-method` is deprecated. Defaulting to `SMART_MASK` or `NATIVE_KV`
- Specified config ARCH, ignoring on real target
- In low memory, empty enable graphs list found. Loading the first graph only.
  (graph-switching: prefill first, decode on-demand)
- `setInferenceBufferForHtpExtensionSkel`: not supported for DeviceId 0 coreId 0 pdId 0
- Graph settings specified by user cannot be applied to an already composed graph.
- Various FastRPC sysfs file missing errors (benign — unsigned PD)

### Prompt phase (04:34:05.74 → 04:34:11.92)

- 7 prompt tokens: `[3838, 374, 279, 6722, 315, 9625, 30]`
- 121 prefill graph steps (AR-128 CL-128, `n_process` increments from 7 → 128)
- First prefill step: 53 ms for 7 tokens
- Each subsequent +1 token: ~50 ms
- `KV$ Update` dispatched after each step

### Decode phase (04:34:11.92 → ~04:34:35, when test was stopped)

- First decode step: 323 ms (graph-switching overhead + decode graph load)
- Subsequent decode steps: ~160 ms each
- 144 decode steps total before timeout
- Graph switching confirmed: `Executing graph 0 – decode`
- Logits are garbage: oscillate between all-zeros and random non-zero values
- Output is incoherent from token 1

### Termination

- `Context Size exceeded` / `Failed to query`
- Clean teardown: `QnnContext_free`, `QnnDevice_free`, `QnnBackend_free` — all successful,
  no crashes

### Profile JSON

Not generated. The `--profile` output file was not written. This is likely because the run
terminated with `"Context Size was exceeded"` / exit code 1 before Genie could finalize the
profile JSON.

---

## 12. Key Diagnostic Finding: KV Cache Corruption

The `input-logits` trace shows the pattern of corruption clearly:

- **Step 0** (7 tokens prefill): first 10 logits = all zeros → KV state empty/corrupt
- **Step 1** (8 tokens): all zeros
- **Step 2** (9 tokens): partial non-zero values (random)
- **Step 3** (10 tokens): mixed values
- **Step 4+**: oscillates between zero and random values

This indicates the **fused QKV attention block produces incorrect K and/or V values on HTP v81
hardware**, despite passing AIMET quantsim validation locally. The KV cache accumulates
garbage, and the model's output logits are correspondingly random.

---

## 13. Summary

| Aspect | Result |
|---|---|
| Bundle deploys and loads | ✅ Yes — 2-graph weight-shared ctx-bin loads cleanly on SA8797P |
| Runtime init | ✅ Success (~770 ms), unsigned PD, v81, `llm_decode_burst` applied |
| Graph switching (prefill → decode) | ✅ Confirmed working |
| Prefill performance | ✅ 53.4 ms for 7 tokens; `tps-prompt` 131.06 tok/s |
| Decode performance | ⚠️ ~6.27 tok/s steady-state (~159.6 ms/step) |
| Teardown | ✅ Clean, no crashes |
| **Output correctness** | ❌ **Broken — garbage from the very first token** |
| Root cause | Fused QKV block emits wrong K/V on HTP v81 → KV cache corruption |
| Profile JSON | ❌ Not written (run aborted with context-exceeded) |

Session footer from the capture: *Crunched for 1h 3m 50s* —
`/m/code/sa8797-deploy-kit master ctx:29%/200k cost:$0.233 [complexity-router|default|low]`

---

## Transcription notes

The source images are photographs of a laptop screen with significant glare and reflection.
The following readings are uncertain:

1. **Token 7 / `[BEGIN]` glyph** — a non-Latin character (appears CJK/Hebrew-like under glare)
   immediately before `Half`. Rendered here as `ロ[?]`.
2. **Raw output mojibake** — the multilingual garbage string in §8 contains characters
   (`ῶ`, `懸`, `椤`, `ζ`, `ﾞ`, `ż`) that are legible but may not be byte-exact. The repeated
   `!` runs and the `2+2+2+...` sequence are exact in structure; exact repetition counts were
   not countable from the photos.
3. **Serial `REDACTED`** — read from a low-contrast region.
4. **`min:UINT64_MAX`** in the `generate` KPI row — literal string as printed by Genie.
5. Image `IMG_2994` is absent from the set; the `genie_dialog.json` block is nonetheless
   complete because `IMG_2993`, `IMG_2995` and `IMG_2996` overlap across the whole file.
