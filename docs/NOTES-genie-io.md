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
| `logits` | `[1, AR, vocab]` (or last-token) | numElements/channel used for AR |

- KV naming rule (`qnn-utils.hpp:234 isKVTensor`): name contains `key`/`value` AND
  ends `_in`/`_out`; input name derived from output name by `_out` → `_in`
  (nsp-graph.cpp:160). → `past_key_0_in` / `past_key_0_out` etc. are correct.
- ctx_size = past_key_out.lastdim + past_key_in.lastdim = AR + P (nsp-graph.cpp:156).
- The graph outputs ONLY new-token KV; Genie's KVCache manager
  (`KVCache/{native-kv,smart-mask,context-manager}`) scatters into its buffers.
- Fallback: AR/CL can also be parsed from graph names matching `ar<N>` / `cl<N>`
  (nsp-graph.cpp:139,168) — name graphs e.g. `qwen3_ar128_cl1152`, `qwen3_ar1_cl1152`.

## Dialog JSON schema (examples/Genie/configs/llama3-3b/llama3-3b-htp-long-context.json)

- `dialog.engine.backend.QnnHtp`: `use-mmap`, `spill-fill-bufsize`, `mmap-budget`,
  `poll`, `cpu-mask`, `kv-dim`, `allow-async-init`, `enable-graph-switching` ✓
- `dialog.engine.backend.extensions` → htp_backend_ext_config.json path
- `dialog.engine.model.positional-encoding`: `{type: rope, rope-dim, rope-theta}`
  → summary's "pos-id-dim 64" = `rope-dim: 64`
- backend ext example (configs/htp_backend_ext_config.json): `devices[].cores[]`
  with `perf_profile`, `rpc_control_latency`.

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
