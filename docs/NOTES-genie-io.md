# Genie 1.19 LLM Graph I/O Contract — extracted from SDK 2.48.40 sources

*Source of truth: `$QAIRT_SDK/examples/Genie/Genie/src/qualla/engines/qnn-htp/`
(the actual qualla engine code shipped with the SDK) and `examples/Genie/configs/`.
This replaces guesswork — plan Task 2 Step 4. Extracted 2026-08-10.*

## Tensor names (nsp-model.hpp:47-63, LayerType map)

| Role | Name |
|---|---|
| token input | `input_ids` |
| attention mask | `attention_mask` |
| rope tables | `position_ids_cos`, `position_ids_sin` |
| logits output | (LayerType::OUTPUT) `logits` |
| KV cache group prefix | `past_` (default, nsp-model.hpp:174) |

## Shapes (nsp-model.cpp "Check 3", ~line 804+; nsp-graph.cpp:70-190)

| Tensor | Shape | Note |
|---|---|---|
| `input_ids` | `[1, AR]` (size deduced as numElements/batch) | int32 |
| `attention_mask` | `[1, AR, CTX]` rank-3, checkShape(1, arn, ctx) | additive float |
| `position_ids_cos/sin` | `[1, AR, rope-dim]` rank-3, checkShape(1, n_tokens, dim) | rope-dim = 64 for head_dim 128 |
| `past_key_{i}_in` | `[1, n_kv, kv_dim, P]` — **keys TRANSPOSED, seq LAST** | kv_dim read from key tensor width (2nd-to-last) |
| `past_value_{i}_in` | `[1, n_kv, P, kv_dim]` | |
| `past_key_{i}_out` | `[1, n_kv, kv_dim, AR]` — **new slice ONLY** | "last dim of past_key_out will always be the input size" (nsp-graph.cpp:117) |
| `past_value_{i}_out` | `[1, n_kv, AR, kv_dim]` | |
| `logits` | `[1, AR, vocab]` — **all positions MANDATORY on every logit-producing graph** | see "Prefill logits contract" below |

- KV naming rule (`qnn-utils.hpp:234 isKVTensor`): name contains `key`/`value` AND
  ends `_in`/`_out`; input name derived from output name by `_out` → `_in`
  (nsp-graph.cpp:160). → `past_key_0_in` / `past_key_0_out` etc. are correct.
- ctx_size = past_key_out.lastdim + past_key_in.lastdim = AR + P (nsp-graph.cpp:156).
- The graph outputs ONLY new-token KV; Genie's KVCache manager
  (`KVCache/{native-kv,smart-mask,context-manager}`) scatters into its buffers.
- Fallback: AR/CL can also be parsed from graph names matching `ar<N>` / `cl<N>`
  (nsp-graph.cpp:139,168) — name graphs e.g. `qwen3_ar128_cl1152`, `qwen3_ar1_cl1152`.

## Prefill logits contract — root cause of the 2026-08-11 device garbage

*Found after the first on-device run (fuseqkvgu) produced garbage from token 1.
Sources verified line-by-line 2026-08-11.*

The basic dialog **requires all-position logits** `[1, AR, vocab]` on the
prefill graph. A last-token-only head `[1, 1, vocab]` loads and runs without
any error but produces garbage:

- Sampling row: `getLogits` offsets by `(n_process - count) * vocab * bw`
  with `count == 1` from the basic dialog — i.e. **row `n_process − 1`**
  ("Note this assumes right-padded input", nsp-model.cpp:3294-3295, call at
  :2918-2920). There is NO last-only mode; the only shape guard requires
  `count > 1` (:3289) and never fires for basic decode.
- Input alignment: decoders are **left-aligned, right-padded** — tokens at
  window slots `0..n-1`, `m_pad_token` after (nsp-model.cpp:1994-2016;
  right-alignment exists only for ModelArchitectureType::ENCODER). Pad rows
  are fully masked and get RoPE position 0.
- Load-time validation explicitly ACCEPTS `numElements == vocab`
  (nsp-model.cpp:792-798) — the mismatch is silent.
- Consequence of a last-only head: qualla reads ~`(n_process-1)*vocab*2` bytes
  past the end of a 1-row buffer (fused allocations → neighboring tensors /
  unwritten pages) → logits oscillate between zeros and noise → greedy argmax
  of zeros = token 0 = `"!"` (the observed `!!!!!` floods).
- Our `(128,128)` no-past-KV prefill is treated as an AR-ctx "bertcache"
  variant: the whole window is re-processed per generated token
  (kvmanager.cpp:421-429), so the broken read repeats until the window fills,
  then the AR-1 decode graph takes over (`n_process-1 == 0` — correct row,
  but the KV window is already poisoned with garbage tokens).
- Do NOT "fix" this by removing logits from prefill: with `input_ids` present
  and no past-KV, such a graph classifies as `GraphType::LUT`, not
  DECODER_PREFILL (nsp-graph.cpp:245-251).

Fixed 2026-08-11: `export_qwen3.py` / `quantize_aimet.py` now export prefill
with `logits_last_only=False`. Regression guard:
`scripts/validate/parity_qualla_read.py` (left-aligned feed, row n−1 read).

## Dialog JSON schema (examples/Genie/configs/llama3-3b/llama3-3b-htp-long-context.json)

- `dialog.engine.backend.QnnHtp`: `use-mmap`, `spill-fill-bufsize`, `mmap-budget`,
  `poll`, `cpu-mask`, `kv-dim`, `allow-async-init`, `enable-graph-switching` ✓
- `dialog.engine.backend.extensions` → htp_backend_ext_config.json path
- `dialog.engine.model.positional-encoding`: `{type: rope, rope-dim, rope-theta}`
  → summary's "pos-id-dim 64" = `rope-dim: 64`
- backend ext example (configs/htp_backend_ext_config.json): `devices[].cores[]`
  with `perf_profile`, `rpc_control_latency`.

## Lookahead decoding (dialog type `lade` → qualla `lhd-dec`) — extracted 2026-08-10

*Sources: `Genie/src/Dialog.cpp:2227` (validateDialogLadeConfig), `:2598`
(translateDialogConfig), `qualla/dialogs/lhd-dec.cpp`,
`qualla/engines/qnn-htp/attention-mask.cpp`.*

- Genie dialog schema: `dialog.type: "lade"` + section
  `dialog.lade: {version: 1 (MANDATORY), window, ngram, gcap, update-mode}`.
  Only these keys allowed; update-mode ∈ ALWAYS_FWD_ONE | FWD_MAX_HIT | FWD_LEVEL.
  qualla defaults: window 8, ngram 3, gcap 8, ALWAYS_FWD_ONE.
- Per-step batch: warmup `window*(ngram-1)`, main loop `(window+g_cur)*(ngram-1)`
  with g_cur ≤ gcap → max `(ngram-1)*(window+gcap)` (defaults: 2×16 = **32**).
- Constructor check is only `≤ getMaxSingleTurnInputLength()` = **max AR over ALL
  graphs including prefill** (nsp-model.cpp:1434). A config bigger than the
  verify graph would silently route batches onto the prefill graph — which has
  no past-KV inputs and cannot serve incremental verification. **Keep
  (ngram-1)*(window+gcap) ≤ verify-graph AR.**
- The verify graph MUST output logits for ALL AR positions
  (`logits.getIndexedTensor(sample_idx, n_vocab)` samples at arbitrary rows).
- Host-side responsibilities (NOT graph properties): tree attention masks are
  built by `AttentionMask` from a parent-pointer attention_map (1D: parent
  index per row, -1 = root; row positions = n_past + tree depth via
  `m_cached_attention_counts`); accepted-token KV promotion via
  `engine.updateKV(n_past, selected)`. The graph contract is unchanged from
  basic decode — just AR>1 and full logits.
- Genie reports acceptance KPI: `tps.tokenAcceptance = (n_generated-1)/iterations`.
- Speculative alternatives shipped in this SDK: `spd` (classic draft-target,
  draft on CPU via QnnGenAiTransformer or on HTP), `eaglet` (EAGLE draft head;
  Qwen3-4B-class example configs in examples/Genie/configs/qwen3/), `ssd-q1`
  (self-speculative with forecast tokens). Example target ctx-bins are named
  `base_tm_ar128_ar32_*` — i.e. prefill AR128 + verify AR32, matching our layout.

## Other findings

- x86_64 tools INCLUDE `genie-t2t-run` + `libGenie.so` → local e2e Genie smoke
  tests are possible (CPU backend).
- `lib/hexagon-v81/unsigned/` has BOTH `libQnnHtpV81Skel.so` (doc's name) and
  `libQairtHtpV81Skel.so`.
- x86 offline prepare: `libHtpPrepare.so` (x86 lib dir).
- No obvious x86 HTP *simulator* library in the Community drop (only LPAI sims);
  HTP-kernel-level numerics can't be simulated locally — CPU backend + AIMET
  quantsim outputs are the local numerics proxies.
- SDK also ships `qairt-quantizer` (x86) — potential alternative to AIMET for
  A/B experiments (summary §3.1 noted 2.43's built-in quantizer produced the
  same W8A8 garbage; untested for W8A16-style flows on 2.48).
