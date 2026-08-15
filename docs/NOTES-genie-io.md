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

## Graph selection, lade SIGSEGV root cause, past-KV prefill contract — extracted 2026-08-11

*Sources: `qualla/engines/qnn-htp/KVCache/kvmanager.cpp`, `nsp-model.cpp`,
`nsp-graph.cpp`, `attention-mask.cpp`, `dialogs/lhd-dec.cpp`, `context.cpp`.*

**Graph selection is pure numeric best-fit on (AR, CL)** — names are cosmetic
*to Genie's graph picker*. They are **not** cosmetic to the HTP backend: names are
what `htp_backend_ext_config.json` keys its per-graph tuning on, and they are baked
into the DLC at conversion time from the `--output_path` basename (dots included),
so a name that stops matching silently costs that graph its VTCM/O/HVX settings —
see `docs/NOTES-vit-htp-config.md`.
CL: `m_supported_variants.lower_bound(n_valid_kv)` (kvmanager.cpp:408) —
smallest CL ≥ current KV. AR: smallest `arN >= n_tokens`, else largest smaller
(chunking fallback) (kvmanager.cpp:388-406). AR from `numElements(input_ids)`,
CL from mask channel dim (nsp-graph.cpp:83-87, 154-155); name regex is a
fallback only. Duplicate (AR, CL) pairs are a hard load error
(nsp-graph.cpp:297-302). With AR-1/32/128 all at CL-1152: ≤32-token prompts
and ALL lade batches run on AR-32; AR-128 serves 33–128-token prompts and
chunking; basic decode runs AR-1.

**The KV cache advances by `n_process`, NOT by the AR window** —
`n_valid_kv += n_process` (kvmanager.cpp:454). So with a left-aligned prompt of
n tokens in an AR=128 window, only columns `0..n-1` enter the cache and the
next step starts at cache offset `n`; the pad slots' KV is discarded. Any
host-side emulation of the feed pattern must scatter the new-slice KV outputs
at offset `n`, not `AR`. (Confirmed 2026-08-12 while validating the Qwen3-VL
text tower — previously inferred from `parity_ladekv_read.py` rather than read
off the source.)

**lade SIGSEGV has TWO independent mechanisms** (both consistent with the
2026-08-10/11 device crashes at libGenie pc 0x4c2d58, x0=0x6b8b4567):

1. *Bertcache (AR==CL) graph present*: only `ctx_size == variant` inflates
   `n_process` beyond the batch size (`n_remain += n_past`,
   kvmanager.cpp:421-429). lhd-dec's RELATIONAL attention path then reads
   `m_cached_attention_counts[i]` for i up to n_process with a vector sized
   n_inputs (attention-mask.cpp:236-240) — heap OOB — and the garbage position
   ids become byte offsets in a RoPE-table `memcpy` (nsp-model.cpp:2196-2204)
   → host SIGSEGV. Removing the AR==CL graph provably removes this path.
2. *1-token prompt*: `lhd_branch` warmup does
   `rand() % (tokens.size()-1)` (lhd-dec.cpp:120) — modulo ZERO for a 1-token
   prompt; aarch64 returns the dividend → index `1+0x6b8b4567` → ~7 GB OOB
   read. 0x6b8b4567 is rand()'s first output — this exactly matches the device
   crash register, and the crashing run used prompt "Hi". **Unconditional
   qualla bug, independent of graph topology: lade prompts must tokenize to
   ≥ 2 tokens.**

**Past-KV prefill (AR=128 CL=1152 past=1024) feeding contract** (basis of
`scripts/validate/parity_ladekv_read.py`):
- Tokens LEFT-aligned, remainder filled with pad token; `pad-token` defaults
  to the FIRST `eos-token` entry (context.cpp:51) — 151645 for our config.
- Mask FP16 additive: allow=+0.0, masked=fp16(**-1000.0**) — not -inf
  (nsp-model.cpp:1382-1386). Our encodings calibrate the mask input at -100,
  so device -1000 clips to -100: still e^-100 ≈ 0, harmless (and already true
  of all working device builds).
- Concat layout: mask cols [0,1024) = past region, [1024,1152) = new tokens.
  Chunk row i allows past cols [0,n_valid_kv) + new cols 1024..1024+i.
- RoPE positions `iota(n_past+i)` for valid rows, 0 for pad rows.
- Logits sampled at row `n_process-1` (all-position logits still mandatory).
- Prompts chunk AR at a time with growing n_past (kvmanager.cpp:430-465);
  ceiling: accumulated KV ≤ past_dim = CL−AR = **1024** (kvmanager.cpp:457-464),
  and `context.size` config must be ≤ max CL (nsp-model.cpp:402-406).

**Cross-graph invariants** (fatal load errors if violated):
- KV quant params byte-identical across all graphs for same-named tensors
  (`_in` normalized to `_out`) — nsp-model.cpp:922-961. Guaranteed by our
  encodings-adoption pipeline; do NOT convert graphs from different encodings.
- All graphs same KV style: concat requires `past_key_in.channel == CL − AR`
  per graph (nsp-model.cpp:552-565).
- A logits-less prefill classifies `DECODER_PREFILL` and is excluded whenever
  all-position output is requested (kvmanager.cpp:392-394) — keep logits.

**Perf note**: after every dialog-level KV update the cache is reshaped to the
smallest registered AR (kvmanager.cpp:934). In lade this means AR-32↔AR-1
reshapes every iteration; if device KPIs show this hurting, a lade-only
ctx-bin without the AR-1 graph is the lever.

## Split prefill is fatal at load — and `execute-select-graphs` is the escape hatch — extracted 2026-08-15

Source: the 2026-08-14 device attempt on the Qwen3-VL-4B e2e pipeline
(`reports/qwen3vl-4b-e2e-deployment-status-2026-08-14.md`). The node failed to
create — twice over — before any graph-selection logic ran:

```
ShapeError : attention_mask — Expected [ 1, 128, 2176] bitwidth=*. Found [ 1, 128, 128] bitwidth=2
ShapeError : attention_mask — Expected [ 1, 128, 2176] bitwidth=*. Found [ 1, 128, 128] bitwidth=2
Error validating model. Failed to create the Genie Node (-1).
Segmentation fault
```

The report attributes this to "Genie validates every graph against
`context.size + AR = 2176`". That is not the rule. The real chain is narrower,
and it is a **split-only** trap. Every step below is pinned in 2.48.40 sources
and re-confirmed against the shipped ctx-bins with
`qnn-context-binary-utility --json_file`:

1. **Split 0's prefill emits no `logits`.** In a 2-shard tower the lm_head lives
   in the last shard, so `prefill_0` outputs 36 KV tensors + `last_hidden_states`
   (verified in `qwen3vl-4b-w8a16_1_of_2.bin`); `prefill_1` is the one with
   `logits`.
2. **That classifies it `DECODER_PREFILL`, not `DEFAULT`** (`nsp-graph.cpp:247-249`).
   The branch needs `!inputIDExists && pastKVExists && !logitsExists`, and
   `matchedAllOutputTensors` still holds because `last_hidden_states` is
   explicitly inserted into the matched set (`:232`). `inputIDExists` is false
   because our input is `inputs_embeds` while `LayerType::INPUT` is `input_ids`
   (`nsp-model.hpp:47`).
3. **A `DECODER_PREFILL` in the `past_` cache group gets its expected CL
   rewritten to the group maximum** (`nsp-model.cpp:604-605`):
   ```cpp
   if (variant->variantType == GraphType::DECODER_PREFILL && (prefix == "past_")) {
     ctx = static_cast<int32_t>(m_cache_group_ctx_size[prefix]);
   ```
   `m_cache_group_ctx_size` is the running max across variants (`:566`, `:569`) —
   2176 here, from decode's `past_key_0_in [1,8,128,2175]` + AR 1.
4. **`validateModel` checks each variant's mask against that map**
   (loop `:844`, `checkShape` `:858`, error text `:493`). The map is keyed by the
   *global* `(AR, CL)` variant and is filled from the **first split only**
   (`:608`, then `break`), so both splits' AR=128 prefill variants inherit
   `{128, 2176}` — hence **two identical** ShapeErrors, one per shard.

**This is why 0.6B never hit it.** Unsplit (topology A), prefill emits logits →
`DEFAULT` → `ctx = arn = 128` (`:601-603`, no `key_in`) → its `[1,128,128]` mask
validates fine, and the graph is merely never *selected*
(`docs/NOTES-genie-pipeline.md` probe C). **The same mask shape is
silent-and-slow unsplit, and fatal at load once the tower is split.** Splitting
is mandatory ≳2B (`docs/NOTES-genie-splits.md`), so every split model carrying an
AR==CL prefill inherits this.

### The escape hatch: `execute-select-graphs` / `load-select-graphs`

Two **undocumented** config keys, read at `qnn-htp.cpp:80-81`:

```json
"execute-select-graphs": ["decode_0", "decode_1"],
"load-select-graphs": true
```

They are not equivalent, and only the first one clears the load failure:

| Key | Where it acts | Effect |
|---|---|---|
| `execute-select-graphs` (list) | `nsp-model.cpp:314-318`, *before* `m_variant_list.emplace_back` at `:320` | Unlisted graphs never become variants, so they are never shape-validated and never selectable. **This is what clears the ShapeError.** Mirrored for image nodes at `nsp-image-model.cpp:208-210`. |
| `load-select-graphs` (bool) | `QnnApi.cpp:120` → `ContextEnableGraphsConfig` (`QnnContextConfig.hpp:29-42`) → `QNN_CONTEXT_CONFIG_ENABLE_GRAPHS` on `QnnContext_createFromBinary` | Unlisted graphs are not deserialized at all — init time and memory. **No-op on its own:** the guard is `loadSelectGraphs && !execSelectGraphs.empty()`. |

So the report's framing ("with `load-select-graphs: true` … everything else is
skipped, including validation") has it backwards: the validation skip comes from
`execute-select-graphs`; `load-select-graphs` only avoids paying to deserialize
what you already excluded.

Guards: an empty match set only logs `__ERROR("No matching graphs based on conf
file")` (`:351`) and continues, so a typo'd name degrades to "no graphs" rather
than failing loudly. Names must match the ctx-bin exactly — same caveat as the
graph-names section above (they are baked in from the converter's
`--output_path` basename).

**Status: UNVERIFIED ON DEVICE.** Staged 2026-08-14; the SA8797P board dropped
off USB before it could run. The mechanism is confirmed in source and against
our own ctx-bins — the on-device result is not. If it holds, it is also a
general repair for a miscompiled multi-graph ctx-bin without a rebuild, which
matters because `QnnContext_getBinary()` returns
`QNN_COMMON_ERROR_OPERATION_NOT_PERMITTED` on contexts created from a binary.

## A past-KV prefill IS selected in a basic dialog — the `output_all` exclusion never fires — extracted 2026-08-15

Probe for the prefill-KV rebuild. The open question was whether rebuilding
shard-0's prefill with past-KV is pointless: shard 0 still has no `logits`, so
it still classifies `DECODER_PREFILL`, and `prepareInferenceStrategy`'s `pick`
lambda skips exactly that type:

```cpp
// kvmanager.cpp:388-394
auto pick = [output_all](int32_t n, std::set<GraphVariantInfo>& choices) -> int32_t {
  for (auto choice : choices) {
    if (output_all && choice.type == GraphType::DECODER_PREFILL) {
      continue;
    }
```

**The exclusion is gated on `output_all`, and a basic dialog always passes
`false`.** Every `engine.process(...)` call in `dialogs/basic.cpp` passes
`false` explicitly (`:104`, `:225`, `:237`, `:332`, `:413`, `:781`) or takes the
`= false` default (`:613`, the embeddings+per-layer overload — the Qwen3-VL
path). `output_all == true` appears **only** in `lhd-dec.cpp:161,264` (lade),
`ssd-q1.cpp`, `spec-dec.cpp:255` and `multistream.cpp:86,102`. Our text
generator is `"type": "basic"`, so the DECODER_PREFILL variant stays eligible.

Two further reasons the rebuild is safe even under `output_all == true`:

1. **Registration is per shard, not per model.** `initializeKVManager` walks
   *every* `m_nsp_graphs` entry and registers each variant with that shard's own
   type (`nsp-model.cpp:979-986`). `GraphVariantInfo::operator<` orders on
   `(arN, type)` (`kvmanager.hpp:390-395`), so AR=128 is inserted **twice** into
   `m_supported_variants[2176]` — once as shard-0 `DECODER_PREFILL`, once as
   shard-1 `DEFAULT`. `pick` only reads `choice.arN`, so the surviving DEFAULT
   entry keeps AR=128 selectable regardless.
2. **Logit variants come from the last split only.** `logit_containing_variants`
   is built by iterating `m_nsp_graphs.back().variants` (`nsp-model.cpp:1198-1204`),
   so `{128, 2176}` registers as a logit variant via `prefill_1`. The
   "last step must produce logits" post-process (`kvmanager.cpp:467-490`) is
   therefore a no-op for us and never pops the prefill step.

### Corollary: the device runs THREE prefill calls for a 273-token prompt, not 2 + 17 decodes

`variant` is picked **once** before the loop (`kvmanager.cpp:409`) and only
re-picked if the cache boundary is hit (`:439-445`). With choices `{1, 128}` and
`n_inputs = 273`, `pick` finds nothing `>= 273`, so it returns the largest
choice below — `128`. The loop then emits `n_process = min(n_remain, 128)`:

| Step | variant | ctx | n_past | n_valid_kv | n_process |
|---|---|---|---|---|---|
| 1 | 128 | 2176 | 0 | 0 | 128 |
| 2 | 128 | 2176 | 128 | 128 | 128 |
| 3 | 128 | 2176 | 256 | 256 | **17** |

The tail of the prompt runs on the **AR=128 graph padded to 17 valid tokens** —
it does not fall back to 17 AR=1 decode steps. `past_dim = ctx_size - variant =
2048` (`:448`), which is exactly the rebuilt `past_key_N_in [1,8,128,2048]`.
Any device-faithful parity chain must model three prefill calls.

## `validateModel` expectations, per named tensor — extracted 2026-08-15

`checkShape(name, tensor, height, width, channel, bitwidth, errors)`
(`nsp-model.cpp:477-497`) compares only `Dims.height/width/channel/bitwidth`;
`-1` means "ignore". Its message is the device-visible one:
`Expected [ H, W, C] bitwidth=BW. Found [ h, w, c] bitwidth=bw`.

**Axis mapping** (`qnn-utils.cpp:113-122`, `:66-76`): dims are **right-aligned**
into a 4-vector padded with 1s, then `(batch,height,width,channel) = (v0,v1,v2,v3)`.
So `[1,128,2176]` → `h=1, w=128, c=2176`, and `[1,8,128,2048]` → `h=8, w=128,
c=2048`. With `batchSize == 1` a hack applies: `height = (v0!=1 && v1==1) ? v0 :
height`, `batch = (v0>1 && v1!=1) ? v0 : 1`.

Let `AR = variant.n_tokens`, `CL = variant.ctx_size`, and per cache-group prefix `p`:

- `(group_arn, group_ctx) = m_cache_group_variant_map[p][{AR,CL}]`, defaulting to
  `{AR,CL}` (`:516`) and then recomputed at `:601-608` from the KV tensors —
  **with the rewrite that caused the incident**:
  ```cpp
  // nsp-model.cpp:604-605
  if (variant->variantType == GraphType::DECODER_PREFILL && (prefix == "past_")) {
    ctx = static_cast<int32_t>(m_cache_group_ctx_size[prefix]);   // cache-group MAX
  }
  ```
- `past_dim = use_scatter ? group_ctx : group_ctx - group_arn` (`:894-895`)
- `group_kv_dim` = config `kv-dim`, else auto-detected from `past_key*` width (`:826-842`)

| Tensor | Expected `[h, w, c]` | Source |
|---|---|---|
| `attention_mask` | `[1, group_arn, group_ctx]` | `:858` |
| `position_ids_cos` / `_sin` (ROPE) | `[1, AR, rope.dims]` | `:871-873` |
| `past_key_*_in` | `[*, group_kv_dim, past_dim]` | `:903` |
| `past_value_*_in` | `[*, past_dim, group_kv_dim]` | `:905` |
| `past_key_*_out` | `[*, group_kv_dim, group_arn]` | `:911` |
| `past_value_*_out` | `[*, group_arn, group_kv_dim]` | `:913` |
| `inputs_embeds` (first split) | bitwidth only, **plus** `numElements == AR × batch × embd_size` | `:694-716` |
| `logits` (last split) | `numElements/batch ∈ {vocab, vocab×AR}` | `:792-798` |

Two exemptions worth knowing: `logits` is **not required** when
`variantType == DECODER_PREFILL` (`:787`), and KV shapes are skipped entirely
for `KV_SHARE_NO_KV_OUTPUT` variants (`:889-890`). Check 4 (`:921-949`)
additionally requires byte-identical `(scale, offset)` for every tensor name
across all graphs, mapping `*_in` → `*_out` — this is the "same encodings
lineage" contract, enforced at load.

Worked example, the 2026-08-14 failure: shipped `prefill_0` had
`attention_mask [1,128,128]`, `group_arn = 128`, and `group_ctx` rewritten to
the cache-group max `2176` → `Expected [ 1, 128, 2176] Found [ 1, 128, 128]`,
once per shard. The rebuild ships `[1,128,2176]` and matches by construction.

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
