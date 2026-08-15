# HTP backend-extension config keys — which are real, and where they belong

**Audited 2026-08-14** against QAIRT 2.48.40.260702 (V3-1.4 / E2). Source of
truth: `$QAIRT_SDK/lib/python/qairt/api/common/backends/htp/config.py` and
`$QAIRT_SDK/include/QNN/HTP/QnnHtpGraph.h`.

## Why this file exists

**Unknown keys are silently ignored.** There is no error, no warning, and the
generated ctx-bin looks completely normal — it simply does not have the feature
you asked for. This repo has now been bitten by that failure class three times:

1. `graph_names` not matching the names baked into the ctx-bin → the graph
   silently reverts to backend defaults (4 MB VTCM, 24 MB spill) or SIGSEGVs
   under lade. Documented in `CLAUDE.md`.
2. `memory.extended_udma` → wrong section, never applied (below).
3. `graph_configs_extra.sparse_weights_compression` → the section does not
   exist and neither does the key, never applied (below).

The only reliable check is to read the schema, then verify the built ctx-bin
with `qnn-context-binary-utility --json_file`.

## Recognised top-level sections

`HtpConfigHelper._CONFIG_TYPES` is exactly:

```
{"context", "graphs", "devices", "memory", "groupContext"}
```

Anything else in the JSON — **including our `graph_configs_extra`** — is not a
config type and is dropped whole. `grep -rn graph_configs_extra $QAIRT_SDK/lib/python/`
returns nothing.

## The two keys audited

### `extended_udma` — real, but we put it in the wrong section

Defined on **`HtpContextConfig`** (i.e. the `"context"` section), not
`HtpMemoryConfig`:

```python
extended_udma: bool = False
"""Enables user direct memory access (UDMA). Only supported on HTP v81 and above."""
```

`HtpMemoryConfig` is `model_config = ConfigDict(extra="forbid")` and defines
exactly one field, `mem_type: Literal["shared_buffer"]`. So our

```json
"memory": { "extended_udma": true }
```

has never enabled UDMA in any build. It belongs under `"context"`, alongside
`weight_sharing_enabled`:

```json
"context": { "weight_sharing_enabled": true, "extended_udma": true }
```

Note this is a **v81-and-above** feature, i.e. exactly our target part — so it
is a real unexplored lever, not a no-op we can ignore.

### `sparse_weights_compression` — not a config key at all

The string does not appear anywhere in the SDK's python or headers. The
capability is real, but it is a **graph optimization type**, not a named config
field (`QnnHtpGraph.h:52`):

```c
QNN_HTP_GRAPH_OPTIMIZATION_TYPE_ENABLE_SPARSE_WEIGHTS_COMPRESSION = 6
```

`HtpGraphConfig` exposes no field for it. The escape hatch is
`graphs[].finalize_config` (`Optional[dict]`), which is how arbitrary
`QNN_HTP_GRAPH_OPTIMIZATION_TYPE_*` values are meant to be passed. Any attempt
to enable sparsity must go through there and then be **verified in the built
bin**, not assumed.

## Full `HtpGraphConfig` field list (2.48)

Everything the `graphs` section actually accepts. Serialization aliases in
parentheses are the JSON spellings.

| Field (JSON) | Default | Note |
|---|---|---|
| `name` | — | must equal the graph name inside the ctx-bin |
| `vtcm_size_in_mb` (`vtcm_mb`) | 0 | 0 = device max |
| `vtcm_size` | 0 | |
| `hvx_threads` | 0 | **build-time only** — the runtime value is ignored (report Test 5) |
| `optimization_type` (`O`) | 2 | we ship 3 |
| `finalize_config` | None | dict escape hatch for `QNN_HTP_GRAPH_OPTIMIZATION_TYPE_*` |
| `dlbc` | 0 | activations; weight-sharing-compatible (E1) |
| `dlbc_weights` | 0 | **not** weight-sharing-compatible (SDK ≥2.36) |
| `weights_packing` | False | **unexplored** — surfaced by this audit |
| `num_cores` | None | |
| `short_depth_conv_on_hmx_off` | False | |
| `fold_relu_activation_into_conv_off` | False | |
| `advanced_activation_fusion` | True | |
| `monolithic_lstm` | False | |

## `HtpContextConfig` fields

`weight_sharing_enabled`, `file_read_memory_budget_in_mb`,
`io_memory_estimation`, `max_spill_fill_buffer_for_group`, `group_id`,
`extended_udma`, `init_acceleration`, `lora_weight_sharing`,
`lora_weight_sharing_ram_preload`.

(`max_spill_fill_buffer_for_group` requires `group_id == 0`, enforced by a
validator.)

## Deliberately NOT fixed in the gqafix builds

The two corrections above are **not** folded into the GQA-fix trunk build. Doing
so would confound the one measurement the whole plan turns on: if the post-fix
bundle changed attention topology *and* gained UDMA *and* gained weight
sparsity, a tok/s delta could not be attributed. They get their own ctx-bin-only
variant, the same discipline `docs/archive/MAX_TPS_QWEN3_0.6B_V3.md` applied to `dlbc` (E1).
