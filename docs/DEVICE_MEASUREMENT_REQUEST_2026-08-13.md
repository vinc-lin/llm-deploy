# Device measurement request — 2026-08-13

**To:** SA8797P device team (authors of the HTP v81 hardware reference)
**From:** llm-deploy build side (device-free)
**Device:** `REDACTED` via `ssh <JUMPHOST> "adb -s REDACTED …"`
**Bundles referenced:** HF `vinccniv/sa8797p-qwen3-w8a16-bundles` — each bundle
is self-contained (binary + 7 `.so` + configs). `/data` runs 98–99% full, so
clean old bundles before pulling; the tarball can be deleted after extraction.

**Why this request.** Your microbenchmarks say a 6.3 MB matmul streams its
weights at ~63 GB/s *during compute* (§3.1 of your doc), yet an AR-1 decode
step takes ~155 ms for ~0.95 GB of weights — an effective 6–7 GB/s. Per-op
sync (~250 dispatches × 30–60 µs) explains only ~10–15 ms of that. **Roughly
100+ ms of every decode step is unaccounted for**, and nobody has ever
op-profiled the decode graph on this device. Depending on where that time
goes, there may be significant free performance nobody has claimed. Tests 1–2
answer that. Tests 3–5 are independent quick wins that need no new build.

None of these tests require anything new from us; Test 2 has an optional
input package we can ship on request.

---

## 0. Protocol — applies to every test

- **Warm state**: discard the first run after any reconnect (cold init is
  1.8–2.0 s vs ~790 ms warm and pollutes averages).
- **3 reps per arm**, identical prompt across arms. Use the Qwen3 chat
  template with the empty `<think>\n\n</think>\n\n` assistant prefix (your
  §8.6) so thinking mode never triggers.
- Fix generation length for rate comparisons: set `"max-num-tokens": 128` in
  the dialog JSON. Greedy (temp 0) is fine — determinism helps here.
- Keep `perf_profile: "llm_decode_burst"` everywhere; change nothing else in
  the configs except where a test says so.
- Never compare init→first-logits against TTFT — they differ by ~800 ms of
  init and this has already produced one phantom regression. Report both,
  labeled.

**Send back per run:** the `--profile` JSON, full stdout, and the exact
dialog + backend-ext configs used (a `tar` of the bundle dir minus the
ctx-bin is perfect).

---

## 1. Test 1 — Genie profile capture, basic vs LADE ⭐ highest value, ~30 min

Runnable immediately on the deployed bundles.

```bash
# Arm A: AR-1 decode (basic bundle)
cd /data/local/tmp/qwen3_06b_w8a16_local
LD_LIBRARY_PATH=. ./genie-t2t-run -c genie_dialog.json \
    -p "<templated prompt>" --profile profile_basic_rep{1,2,3}.json

# Arm B: LADE (ladekv bundle — pull qwen3_06b_w8a16_ladekv from HF if absent)
cd /data/local/tmp/qwen3_06b_w8a16_ladekv
LD_LIBRARY_PATH=. ./genie-t2t-run -c genie_dialog.json \
    -p "<templated prompt>" --profile profile_lade_rep{1,2,3}.json
```

We want the **raw JSONs including `traceEvents`**, not just the headline
numbers. What we will extract: the per-execute timeline of a decode step vs a
verify32 call (180 vs 155 ms for 32× the work), any DVFS sag between calls,
and graph-switch events (your 79–93 ms figure) inside a LADE iteration.

---

## 2. Test 2 — op-level profile of one decode step ⭐ the key unknown, ~1–2 h

The question: inside one 155 ms AR-1 step, how much time is (a) MatMul weight
streaming, (b) attention/elementwise/softmax/RoPE ops, (c) gaps between ops?

**Everything for this test is prebuilt and already on HF** — no build step on
your side. Under `profiling/` in `vinccniv/sa8797p-qwen3-w8a16-bundles`:

| File | Size | What |
|---|---|---|
| `qwen3-0.6b-w8a16-decodeonly_ctx.bin` | 1.075 GB | **single-graph** ctx-bin, only the AR-1 `decode` graph — no graph selection needed |
| `decode_profile_inputs.tar.gz` | 276 KB | input tensors + `input_list.txt` for AR-1 **and** AR-32, plus the generator script |
| `README.md` | — | run commands, verification record, what to look for |

```bash
tar xzf decode_profile_inputs.tar.gz && cd ar1_decode
qnn-net-run --backend libQnnHtp.so \
    --retrieve_context /path/to/qwen3-0.6b-w8a16-decodeonly_ctx.bin \
    --input_list input_list.txt \
    --profiling_level detailed \
    --config_file htp_backend_ext_config.json \
    --output_dir out
qnn-profile-viewer --input_log out/qnn-profiling-data-0.log
```

Notes:

- The ctx-bin is built from the **same `decode.dlc`** as the shipped bundles —
  same encodings and weights, `O:3` / `vtcm_mb:16` / `hvx_threads:4` /
  weight-sharing on. Verified after generation: 1 graph named `decode`,
  60 in / 57 out, `logits [1,1,151936]`, **spill = fill = 0**, converter
  `read_total_bytes = 961,130,496`. It is the production decode graph in
  isolation, not a differently-configured rebuild.
- Narrow `graph_names` to `["decode"]` in the backend-ext config you pass —
  this bin holds only that graph.
- KV tensors in the package are zero-filled (so it compresses to 276 KB
  instead of ~132 MB); shapes and byte counts are exact, and the mask/RoPE/
  token inputs are realistic. If you want to rule out data-dependent timing,
  the included script regenerates with `--random`.
- **AR-32 (optional, "if time permits"):** `ar32_verify/` in the same tarball
  holds the AR=32 shapes (`input_ids [1,32]`, mask `[1,32,1152]`, past 1120).
  `verify32` lives inside the multi-graph `ladekv` ctx-bin — if `qnn-net-run`
  can select it there, these work as-is; if not, say so and we ship a
  `verify32`-only ctx-bin the same way (~20 min build). Comparing the AR-1 and
  AR-32 per-op tables shows exactly what amortizes across 32 positions, which
  is the mechanism behind LADE's 1.7×.

**Deliverable:** the `qnn-profile-viewer` output (or raw profile log). This
either explains the 155 ms — in which case we stop chasing it — or localizes
free performance to specific ops, which would redirect our next build.

---

## 3. Test 3 — the 20% build gap: your build vs our bundle, ~30 min + files

Your unfused W8A16 decodes at 7.79–7.80 tok/s; our v2 bundle measured 6.5.
Runtime configs match field-for-field, so we believe the difference is in the
build — your unfused ctx-bin is also 77 MB smaller than ours (1.01 vs
1.087 GB). If it transfers, it is worth ~20% on every future build including
4B. Please:

1. Run **your unfused build** and **our `qwen3_06b_w8a16_local` bundle**
   back-to-back, §0 protocol, same prompt — confirms the gap under one
   measurement protocol (rules out protocol as the explanation).
2. Send the **exact `qairt-converter` and quantizer invocations** (full
   command lines) used for your unfused W8A16 build, plus the build-time HTP
   config JSON.
3. Send `qnn-context-binary-utility --context_binary <your ctx.bin>
   --json_file your_bin.json` — we will diff graph shapes, tensor dtypes, and
   graph count against ours.
4. Send the dialog JSON you measured 7.79 with (to confirm context size and
   sampler match ours).

---

## 4. Test 4 — does the INT8 head help in basic mode? ~15 min

Closes our open question §8.3. The `qwen3_06b_w8a16qh_ladekv` bundle (HF,
2026-08-12) ships `genie_dialog_basic.json` against the same ctx-bin:

```bash
cd /data/local/tmp/qwen3_06b_w8a16qh_ladekv
LD_LIBRARY_PATH=. ./genie-t2t-run -c genie_dialog_basic.json \
    -p "<templated prompt>" --profile profile_qh_basic_rep{1,2,3}.json
```

Compare AR-1 tok/s against Test 1 Arm A (baseline basic). Interpretation is
pre-registered: if qh ≈ baseline, the INT8 head's DDR saving never reaches
the device (prepare-time re-materialization confirmed) and `--quant-head` is
fully dead; if qh is measurably faster, it has a niche in non-speculative
deployments only (it remains −14% under LADE — do not re-test that).

---

## 5. Test 5 — runtime `hvx_threads: 8`, ~10 min

Your open question 7, runtime half. In the **ladekv** bundle's
`htp_backend_ext_config.json`, change `"hvx_threads": 4` → `8`, run LADE
3 reps (§0), revert. The runtime already reports 8 threads in use regardless
of this value, so flat-zero is the expected outcome — but it has never been
measured, and if it moves anything we need to know before the 4B build. The
build-time half of this A/B (compiler told 8 at ctx-bin generation) will come
from our side in a later bundle.

---

## 6. Priority and effort summary

| # | Test | Time | What it decides |
|---|---|---|---|
| 1 | Genie profile capture, basic + LADE | ~30 min | where a decode step / verify call spends its time (coarse) |
| 2 | Op-level decode profile | ~1–2 h | the unaccounted ~100 ms — stop chasing it, or claim it |
| 3 | Build-gap A/B + build artifacts | ~30 min | whether ~20% transfers to all our future builds |
| 4 | qh basic mode | ~15 min | closes the last `--quant-head` question |
| 5 | Runtime `hvx_threads: 8` | ~10 min | rules the runtime knob in or out before 4B |

If time allows only one thing: **Tests 1+2 together** — every other
optimization decision downstream depends on where the decode step's time
actually goes.

---

## 7. One note on your reference doc (no action needed)

Your §7 records `type:"lade"` → SIGSEGV. Our 2026-08-11/12 runs on this same
device measured LADE at **10.8 tok/s** (9.3 for the qh variant) using the
`ladekv` bundles — the crash is config-side: the verify graph must be listed
in `htp_backend_ext_config.json`'s `graph_names` (your §8.3 example lists
only `["prefill","decode"]`), the ctx-bin must not contain an AR==CL graph,
and prompts must tokenize to ≥ 2 tokens. Details are in the annotation at the
top of your doc's copy in our repo, with the mechanism written up in our
`docs/REFERENCE.md` §3.3–3.4. The deployed `qwen3_06b_w8a16_ladekv` bundle
demonstrates it if you want to reproduce.
