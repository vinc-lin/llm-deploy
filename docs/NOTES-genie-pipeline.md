# Genie pipeline / multimodal contract — Stage 3 probe findings (2026-08-14)

*Source of truth: `$QAIRT_SDK/examples/Genie/Genie/src/` (the qualla + pipeline
sources shipped with QAIRT 2.48.40.260702) and `examples/Genie/genie-app/`.
Every claim below carries a file:line citation. Read together with
`NOTES-genie-io.md` (dialog/engine IO contract) and `NOTES-genie-splits.md`
(multi-ctx-bin contract).*

Four probes were run before writing any Stage 3 code. Two returned good news,
two found blockers. **The headline: a stock `GeniePipeline` cannot drive an
FP16 vision tower on this SDK build.**

---

## A. Unconnected graph inputs ARE explicitly zeroed — but the memset is mis-sized

**Verdict: ZERO-GUARANTEED by explicit memset (not by the allocator), and
LOAD-ACCEPTS unknown input names.** This is what makes deepstack-by-zeros a
legitimate degradation rather than a gamble.

qualla has a routine built for exactly this case —
`QnnNspModel::initializeUnconnectedInputs()`, `nsp-model.cpp:1442-1489`:

```cpp
for (auto& [tname, tspecPtr] : unconnectedInputTensors) {
  __DEBUG("Found unconnected input tensor \"{}\": Initializing to zero.", tname);
  clearBuffer(tspecPtr);
}
```

An input counts as "connected" (and is skipped) only if it is a known
`m_layerNames` value, the name of any output, cache-group-prefixed (`past_`,
`qnn-htp.cpp:144`), or `anchor_buffer`-prefixed. `deepstack_visual_embed_*`
matches none, so all three are cleared. `clearBuffer` (`nsp-model.cpp:1954-1977`)
memsets 0 for float dtypes and fills `-offset` (the quantized encoding of 0.0)
for fixed-point — either way a true additive identity.

It runs unconditionally on the stock load path (`qnn-htp.cpp:253`), after IO
allocation, guarded only by `m_lazyInitialization` (= `shared-engine`, default
**false**, `qnn-htp.cpp:55`).

Load acceptance: `validateModel()` (`nsp-model.cpp:620-964`) is structurally
positive lookups (`getInput("name")`) plus `checkShape`, which no-ops on a null
tensor (`nsp-model.cpp:477`). Nothing enumerates inputs demanding recognition.
The only `throw`s are `cross_attention_states`-specific (`:645`, `:653`).
Check 4 (quant-param consistency, `:922-949`) *does* compare our tensors across
`prefill_0`/`decode_0`, but non-quantized tensors all carry `{0.0, 0}`
(`qnn-utils.cpp:148-150`) so they match. **If deepstack is ever made quantized,
its encodings must be byte-identical across the pair** — same rule as the KV
lineage contract.

### ⚠ The sizing gap (finding 9) — a latent bug for SHORT prompts

Same-named tensors across graphs collapse into ONE allocation sized to the max
(`QnnApi.cpp:985-995`). But `initializeUnconnectedInputs` de-dupes by name and
keeps only the **last** variant's spec (`nsp-model.cpp:1481`), and `clearBuffer`
sizes the memset from *that* spec's dims, not from the allocation size.

Our export order is prefill-then-decode (`export_qwen3vl_text.py`), so
`decode_0` (AR=1) is last → only `1*2560*2 = 5120` bytes are zeroed out of a
prefill-AR-sized allocation.

This is **benign only by accident**, via probe C: prompts >128 tokens never
touch the prefill graph, so the unzeroed remainder is never read. A prompt
**shorter than 128 tokens does select prefill**, which then reads deepstack
from memory zeroed only in its first 5120 bytes — the rest is whatever
`rpcmem_alloc` returned (unprovable from source; it is behind `libcdsprpc.so`).

**Fix if we ever ship a short-prompt path: give the tensors per-graph-distinct
names** (`deepstack_visual_embed_0_prefill` / `_decode`) so each gets its own
allocation cleared at full size. Costs nothing at the graph level.

Other conditions on the verdict: the guarantee is *zero on first allocation*,
not *zero every time* — `finalizeState` (`nsp-model.cpp:4222-4283`) re-runs
`allocateAll()` without re-calling `initializeUnconnectedInputs()`, reachable
only via engine sharing, which a single stock pipeline does not do.

---

## B. qualla's `qwen3vl-mrope` matches HF's interleaved layout exactly

**Verdict: INTERLEAVED — bit-for-bit identical to HF
`apply_interleaved_mrope`.** The table construction is in the shipped source,
not hidden in `libGenie.so`.

`nsp-model.cpp:3865-3876`:

```cpp
for (size_t j = 0; j < pos_dim; ++j) {
  size_t use_dim_idx = 0;
  size_t dim_idx     = j % 3;   // cycle temporal, height, width
  size_t freq_idx    = j / 3;
  if (dim_idx > 0 && freq_idx < mrope_section[dim_idx]) use_dim_idx = dim_idx;
  freqs[i][j] = position_ids[use_dim_idx][i] * inv_freq[j];
}
```

`j%3==1 && j/3<20` → H at j = 1,4,…,58; `j%3==2 && j/3<20` → W at 2,5,…,59;
everything else T including the tail 60-63. Exactly `slice(1,60,3)` /
`slice(2,60,3)` from HF. The blocked Qwen2-VL layout is a separate branch
reached only by `rope-type: "qwen2vl-mrope"` (`:3785`, `:3793-3801`).

`inv_freq[j] = 1/pow(theta, j/pos_dim)` (`:3623-3627`) with `pos_dim = rope-dim
= 64` equals HF's `theta^(-2j/128)`. Guard at `:3856-3863`: with default
`rope-freqs-type`, `sum(mrope-section)` must equal `rope-dim` — 24+20+20 = 64. ✔

Every key our config declares is read, with correct spelling
(`nsp-params.cpp:46,85-100,268-272`). Two landmines:

- `rope-theta` is parsed as **`int32_t`** (`nsp-params.cpp:271`). 5e6 is fine;
  anything > 2^31 silently overflows.
- Declaring `positional-encoding` *together with* backend-level `pos-id-dim` or
  `rope-theta` is a hard schema error. `pos-id-dim` sets a static
  `position_dim_set` flag (`Engine.cpp:159-161`) which the positional-encoding
  validator then rejects (`Engine.cpp:677-680`):
  `"Specify one config from pos-id-dim and positional-encoding"`.
  **`genie_dialog_qwen3vl_4b.json` declared both** (`"pos-id-dim": 64` in the
  QnnHtp block *and* a `positional-encoding` block) — i.e. the Stage 2 bundle
  already on HF **fails to load with `GENIE_STATUS_ERROR_JSON_SCHEMA`**. Fixed
  here by dropping `pos-id-dim`; `positional-encoding.rope-dim: 64` supersedes
  it. See Action 3. (Backend-level `rope-theta` was never declared, so only the
  first of the two throws applied.)
- Omitting `mrope-section` defaults to `{16,24,24}` (Qwen2-VL's), which also
  sums to 64 and so passes the guard while producing a *wrong* ownership map.
  Always declare it explicitly.

### The image grid is a static config constant, not inferred

`ImageEncoder.cpp:46-47` reads `engine.model.vision-param.{height,width}`;
`:136-138` records `setVisionParam(visionPos, 1, m_height, m_width)` (temporal
hardcoded 1). `nsp-model.cpp:3813-3817` divides both by `spatial-merge-size`.
Units are **patches (pre-merge)** — corroborated by the vision tower's own
validator requiring `height*width` in the `pixel_values` shape
(`nsp-image-model.cpp:373`). For us: `height: 32, width: 32` → 1024 patches →
16×16 = 256 rows.

`temporal*height*width` (post-merge) must equal the row count the encoder
appended, computed independently at `ImageEncoder.cpp:144-146`. **If they
disagree nothing cross-checks it and every subsequent position id is silently
wrong.**

Post-image text positions correctly implement HF's `rope_deltas` continuation
(`nsp-model.cpp:3839-3850`, `max(t,h,w)+1`). Two caveats: **multi-image is
wrong** (`:3827` resets the base to the raw row index, discarding the
accumulated delta), and `visionPos` is batch-local (`Accumulator.hpp:45`,
flushed per execute at `TextGenerator.cpp:326,350`) while the rope table is
indexed by absolute KV position (`:2196-2205`) — so an image introduced in turn
≥2 lands at the wrong offset. **Single image, first turn is correct.**

### Reachable only from a Pipeline, never from a Dialog

The only callers of `Dialog::setVisionParam` are `TextGenerator.cpp:302,333`,
both gated on `m_usingMRope && useEmbedding`. There is **no C API entry point** —
`grep -i "vision|mrope"` over `include/Genie/*.h` returns nothing.
`genie-t2t-run` contains zero occurrences of `image`/`vision`.

**Consequence for bring-up:** running the text tower under `genie-t2t-run` with
a `qwen3vl-mrope` config produces *correct text-only output* and proves nothing
about the image path — `m_visionParam` stays empty and both mrope branches are
skipped (`:3736`, `:3803`), falling to plain rope (`:3900-3905`), which is
numerically identical when t==h==w.

---

## C. The AR=128 prefill graph is dead weight for prompts > 128 tokens

**Verdict: CHUNKS-VIA-DECODE — but prefill runs for ZERO of the tokens, not
128 of them.** Correctness is unaffected; throughput is not.

`determineGraphContextSize` (`nsp-graph.cpp:146-155`) derives a graph's
`ctx_size` from the **attention_mask's trailing dim**, before any past-KV
fallback:

```cpp
tensor = getInput(m_layerNames[LayerType::ATTN_MASK]);
if (tensor) return static_cast<int32_t>(tensor->dims.channel);
```

Our prefill has `attention_mask [1,128,128]` (`io_spec`: `past_len + seq` with
`past_len=0`), so its registered `ctx_size` is **128 == its own AR**. That is
the AR==CL "bertcache" shape this repo already warns about in CLAUDE.md,
arrived at via the no-past-KV design rather than via lade.

In `prepareInferenceStrategy` (`kvmanager.cpp:365-517`) with `n_inputs=290`:
the escape loop at `:411-416` sees `iter_ctx->first(128) == variant(128)` and
`290 > 128`, so it advances to the only larger bucket (2176 → AR=1) *before
processing a single token*. The main loop `:430-465` then runs 290 times at
`n_process=1`. The prefill `(128,128)` variant is never looked up
(`nsp-graph.cpp:331-346`).

The hypothesized `DECODER_PREFILL` exclusion (`kvmanager.cpp:392-394`) is a red
herring — our prefill classifies as `GraphType::DEFAULT`, not
`DECODER_PREFILL`, because it emits logits (`nsp-graph.cpp:247-251,260-261`).

Neither safety check fires: the overall context limit (`:370-373`) is fine at
290 ≪ 2176, and the bertcache "input too large" error (`:425-428`) is skipped
precisely *because* the escape loop already left that bucket. **Hence silent.**

Sampled row confirmed: `getDequantLogits` offsets by `(n_process - count)`
(`nsp-model.cpp:3242`) → row 0 for a decode step.

**Cost:** ~290 sequential single-token HTP dispatches for an image prompt
(≈30 s at 10 tok/s), versus the ~15-20 s originally budgeted. Acceptable for
"images work at all"; not acceptable long-term.

**Fix:** give prefill a runtime-discoverable `ctx_size` strictly greater than
its AR — i.e. `attention_mask [1,128,CL]` with `CL>128`, which means exporting
prefill **with past-KV inputs** (the ladekv-style past-KV prefill this repo
already builds for Qwen3-0.6B). That is a text-tower re-export, not a config
change.

---

## D. 🚫 BLOCKER — a stock pipeline cannot drive an FP16 vision tower

Two independent gaps in the shipped sources, neither fixable by config. Both
were re-verified by hand after the probe reported them.

### D1. `setupInputFP16` is an unimplemented stub

`nsp-image-model.cpp:526-530`:

```cpp
bool QnnNspImageModel::setupInputFP16(const std::vector<uint8_t>& /*inputs*/,
                                      const std::string& /*name*/) {
  // Placeholder for FP16 inputs
  return true;
}
```

Both parameters are unnamed. It copies **zero bytes** and returns `true`. It is
dispatched for exactly our case (`:547-548`, `case QNN_DATATYPE_FLOAT_16`), and
because it returns true, `runInference` proceeds (`:573-574`) and reports
success all the way up.

**So our 3,145,728-byte `pixel_values` blob is silently discarded and the vision
tower executes on an uninitialized buffer.** No error, correct-looking exit
code. Right size, wrong size and garbage all behave identically.

(The templated `setupInput<DType>` path used for UFixed8/UFixed16/INT32,
`:500-524`, does copy real data.)

### D2. The requantize dispatch table has no `Float16` entries at all

`quantization/src/Quantization.cpp:163-192` — `getConverterMap()` contains only
pairs among `SFixed8/16`, `UFixed8/16`, `UFixed4`, `Float32`. `Float16` is a
valid enum member and `toDataType()` parses it (`:27`), but it is **never a key,
as source or destination — not even `{Float16, Float16}`**.

`ImageEncoder` correctly reports its output as FP16 (read from the compiled
tensor, `nsp-image-model.cpp:670-682`), so the pipeline-connected append
(`ImageEncoder.cpp:145-146`) builds `TypePair{Float16, Float32}`, misses, and
throws `"Unsupported requantization operation"` (`Quantization.cpp:207-209`).
That surfaces as `GENIE_STATUS_ERROR_GENERAL` — **loud**, unlike D1.

The accumulator's destination dtype is float32 regardless: `lutDataType`
defaults to `QNN_DATATYPE_FLOAT_32` (`dialog.hpp:490`) and is only overridden by
a dialog's *own embedded* encoder block, which a multi-node pipeline does not
have (`qualla/dialog.cpp:444-460`, `TextGenerator.cpp:106-121`).

**No config key exists to change any of this** — `ImageEncoder`'s quant params
come only from the compiled ctx-bin, and the SDK's own `configs/glm-4v/siglip.json`
has the same minimal schema we do. Note that Qualcomm's reference vision encoder
is **ufixed8-quantized**, not FP16 — the FP16 encoder path is simply not a
supported configuration in this build.

### Escape routes (both verified to exist)

1. **`qnn-net-run` ships for aarch64-android** (`bin/aarch64-android/qnn-net-run`,
   5.3 MB) — runs the ViT ctx-bin standalone and can dump **all four** outputs,
   deepstack included.
2. **`genie-app` accepts raw embeddings**: `node set embedding <node> <IO> <file>`
   reads an opaque blob and calls `setData` (`genie-app/main.cpp:1185-1207`).
   Crucially, `TextGenerator::setEmbeddingInputData` routes
   `GENIE_NODE_TEXT_GENERATOR_EMBEDDING_INPUT` to the **raw byte-concat**
   `Accumulator::append(uint8_t*, size_t)` overload with **no requantization**
   (`TextGenerator.cpp:197-198`, `Accumulator.cpp:28-53`) — sidestepping D2
   entirely. Bytes must therefore already be in the accumulator's dtype
   (float32).

`accumulator-size` is in **bytes**, a soft `reserve()` hint that `resize()` will
silently exceed (`Accumulator.cpp:22-23,41-42`). At float32 × 2560 × ~290 tokens
≈ 2.97 MB, so 64 MB is comfortable.

---

## Actions this forces

1. **Stock single-pipeline Tier A is not buildable as planned** (D1+D2). Choose
   between quantizing the vision tower to fixed-point IO, or the two-step
   `qnn-net-run` → `genie-app` raw-embedding flow.
2. Add `engine.model.vision-param.{height: 32, width: 32}` to the image-encoder
   config if any ImageEncoder-node path is used (B) — it is mandatory for mrope
   to engage at all, and it is in patch units.
3. **`configs/genie_dialog_qwen3vl_4b.json` declared `positional-encoding`
   alongside backend `pos-id-dim`** — a hard schema error
   (`Engine.cpp:159-161`, `677-680`) that would have made the Stage 2 bundle
   fail to load on device. Fixed in the repo by dropping `pos-id-dim`;
   **the copy already uploaded to HF is still broken and must be re-uploaded.**
4. Deepstack-by-zeros is sound (A), but rename the tensors per-graph before any
   sub-128-token prompt path is shipped.
5. Long-term: re-export prefill with past-KV so it is not bertcache-shaped (C).
