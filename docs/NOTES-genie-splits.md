# Genie multi-split (multi-ctx-bin) contract

*Extracted from QAIRT 2.48.40 qualla source on 2026-08-12, while splitting the
Qwen3-VL-4B text tower. Sources: `qualla/engines/qnn-htp/nsp-model.cpp`,
`nsp-graph.cpp`.*

## Why we split

`qnn-context-binary-generator` enforces a **hard per-graph serialization limit
of 3,670,016 KB (3.5 GiB)**:

```
fa_alloc.cc:2813: graph requires estimated allocation of 4383384 KB, limit is 3670016 KB
graph_prepare.cc:905: error during serialize: memory usage too large
```

There is no flag to raise it. Measured on the Qwen3-VL-4B text tower (36 layers,
hidden 2560, W8A16, embeddings-in so no token table in the graph):

| configuration | estimate |
|---|---|
| 36 layers + FP16 lm_head | 4383384 KB (4.18 GiB) — **rejected** |
| same, `--quant-head` (W8 lm_head) | ~3.74 GiB — still rejected |
| 18 layers/chunk | ~1.9 GiB — fits |

**The estimate is weights only.** Proven by generating a ctx-bin from the
prefill graph alone, which has zero past-KV tensors: it reported the
byte-identical `4383384 KB`. So reducing context length does **not** move this
number — KV buffers are runtime, not serialized. Don't reach for that lever.

## The contract

**1. Splits are ordered by SORTED GRAPH NAME.** This is the one that will bite.
`nsp-model.cpp:365-372` groups graphs by `(AR, CL)` into a `std::set<std::string>`
and assigns split index by iteration order:

```cpp
uint32_t idx = 0;  // Graph names are sorted by default (std::set<>), so iterate by split
for (auto& graph_name : graphs)
    m_nsp_graphs[idx++].addGraph(m_graph_map.at(graph_name));
```

So the graph *name* determines which split a graph is. Names must sort into the
intended chunk order within each AR group. Since a graph's name is baked in at
conversion time from the `--output_path` basename, **the DLC filename is the
wiring**.

`n_splits` is `max(count)` over the `(AR, CL)` groups (`:354-358`), so every AR
variant must contribute the same number of graphs.

**2. Splits connect implicitly BY TENSOR NAME.** `nsp-model.cpp:1453-1457`:

> Include output tensor names from all splits. Any input with the same name as
> an output refers to the same buffer and is therefore implicitly connected.

There is no explicit wiring config. Chunk N's output tensor and chunk N+1's
input tensor are the same buffer iff they share a name.

**3. Genie does NOT validate inter-split shapes.** `nsp-graph.cpp:225-233`
lists the checks and states the gap outright:

> Missing check : Shape of tensor between splits match up

A boundary mismatch is therefore silent. Lint it in the build.

**4. Quant params must match across identically-named tensors.** Check 4 of
`validateModel`: "All tensors with identical names (incl kv_in/kv_out) have
identical quantization params." The boundary tensor appears in two chunks, so
the encodings lineage has to span them — same reason prefill and decode share
one encodings file.

**5. First and last split are special.** Check 1a: `input_ids` or
`inputs_embeds` must exist in the **first** split. Check 2: `logits` must exist
in the **last** split.

**6. `last_hidden_states` is a recognised output name** (`nsp-graph.cpp:231`),
special-cased so it is not treated as an unexpected output. It is the natural
name for the boundary tensor.

## Layout used for Qwen3-VL-4B text

Two ctx-bins, each holding both AR variants of one chunk, so the chunk's weights
are shared between its prefill and decode graphs:

| ctx-bin | graphs | layers | boundary |
|---|---|---|---|
| `..._1_of_2` | `prefill_0`, `decode_0` | 0–17 | out: `last_hidden_states` |
| `..._2_of_2` | `prefill_1`, `decode_1` | 18–35 | in: `last_hidden_states` |

Names sort correctly within each AR group (`prefill_0` < `prefill_1`,
`decode_0` < `decode_1`).

**KV tensors keep GLOBAL layer indices** — chunk 1 uses `past_key_18_in` …
`past_key_35_in`, not a renumbered 0–17. Renumbering would collide with chunk
0's names and, by rule 2, silently alias the two chunks' caches to the same
buffers.

Deepstack injects at layers 0/1/2, all inside chunk 0, so it needs no
cross-split handling.

`graph_names` in the HTP backend config must list all four graph names, or the
unlisted ones silently compile at O=0 / 4 MB VTCM (see
`docs/NOTES-vit-htp-config.md`).
