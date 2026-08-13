# Device-team exchange — 2026-08-14

**To:** SA8797P device team
**From:** llm-deploy build side (device-free)
**Re:** corrections to the 2026-08-13 measurement report, three artifact
requests, and a heads-up on the test kit coming next.

Thank you for the 5-test report — the op-level decode profile in Test 2 is the
most useful measurement anyone has taken on this model, and it redirected our
entire optimization plan. Three things below: **one correction that changes what
Test 2's headline means** (please read before acting on your Rec #1), the
artifacts we still need, and what we are sending you next.

---

## 1. Correction: the 74.7% is GQA KV replication, not mask broadcasting

Test 2's headline attributes 261.8M cycles/step (74.7%) to broadcasting the
causal mask scalar to `[1,8,1,1152]`. We inspected the shipped `decode.dlc`
directly (`qairt-dlc-info`) and the graph says otherwise:

- **The attention mask is never expanded.** It enters as a `[1,1,1152]` graph
  input, gets one `Unsqueeze` to `[1,1,1,1152]`, and broadcasts *implicitly*
  inside the `Add`. There is no mask `Expand` op in the graph.
- **The 56 expensive ops are `repeat_kv`** — GQA KV-head replication, expanding
  8 KV heads to 16 Q heads so a 16-head MatMul can consume them:

  ```
  Expand    [1,8,1,128,1152] -> [1,8,2,128,1152] -> Reshape -> [1,16,128,1152]   (K, transposed)
  Expand_1  [1,8,1,1152,128] -> [1,8,2,1152,128] -> Reshape -> [1,16,1152,128]   (V)
  ```

- Their QNN type is `Eltwise_Binary` with `operation: 13` (**MULTIPLY**) against
  a `[1,1,2,1,1]` static coefficient — the converter lowers ONNX `Expand` into a
  broadcast multiply-by-ones.

**This also resolves the "260 cycles/byte, extremely inefficient" anomaly.** That
figure came from assuming an 18,432-byte output. The real output is 4,718,592
bytes — exactly 256× larger — which gives **1.03 cycles/byte**, entirely normal
throughput for a broadcast FP16 multiply. Your cycle counts were right; only the
op label was wrong, and the two errors happened to cancel into a plausible-
looking anomaly.

### What this changes

- **Rec #1 as written targets an op that does not exist.** Fusing the mask into
  the Q@K kernel would gain nothing. The correct fix is to stop materialising
  the replicated KV at all.
- **The 75% is still the right target**, and it is addressable purely in the
  export: feed the un-replicated `[1,8,...]` cache into a MatMul batched over
  the 8 KV heads, with Q grouped as `[1,8,2·AR,128]`. We have implemented this
  and verified numerically that it is equivalent (max |diff| 6e-16 in float64)
  and that the converter preserves it — the attention MatMuls come out as
  `1x8x2x1152` instead of `1x16x1x1152`, and all 56 `Expand` ops are gone.
- **Your "verify32 amortizes the broadcast" note is backwards.** Replication
  cost is AR-independent, which *strengthens* the case for speculative decoding
  rather than explaining its cost away.
- **The fusion and `lm_head` projections need renormalising.** Against the 88.5M
  cycles that remain after removing replication, attention GEMV is 44.5%, weight
  GEMMs 35.3%, and `lm_head` 6.9% — so QKV/Gate-Up fusion is worth ~10% of real
  compute rather than "~2.5% marginal", and `lm_head` is not negligible.

## 2. Test 1's TTFT/prompt-rate gap is not graph-switch overhead

`qnn-context-binary-utility` on the two shipped bins shows the prefill graphs
differ in context length, not just count:

| Bundle | prefill mask shape | decode | verify32 |
|---|---|---|---|
| `local` | `[1,128,128]` (bertcache, CL=128) | `[1,1,1152]` | — |
| `ladekv` | `[1,128,1152]` (past-KV, CL=1152) | `[1,1,1152]` | `[1,32,1152]` |

`local`'s prefill attends over 128 positions, `ladekv`'s over 1152, and
replication cost scales with CL — so the small prefill does ~9× less of it.
That fully accounts for 40 ms vs 186 ms TTFT and 1397 vs 301 tok/s, with no
graph-switching term needed.

## 3. Test 4 / §6.3: the "75% build gap" is confounded, and one run splits it

The 6.70 tok/s figure was measured on the **qh** bundle only, so the gap labelled
"build gap" conflates three variables: W8 `lm_head`, graph count, and build
lineage. **Basic mode on the plain `qwen3_06b_w8a16_ladekv` bin has never been
measured** — that single run separates them:

- ≈11.7 tok/s → the regression was the W8 head all along, and the 6.3–6.5 tok/s
  numbers from earlier sessions need re-examining;
- ≈6.7 tok/s → there is a real 3-graph/graph-switching penalty worth chasing.

It is one run on a bundle you already have. If you run nothing else from this
message, please run this.

## 4. Artifacts we need

1. **The Test 2 raw profiler output.** We received the narrative report but not
   `docs/test_artifacts/measurement_2026-08-13/`. Most valuable are
   `test2_decode_profile/qnn_profile_r{1,2,3}_viewer.txt` — with the per-op
   output dimensions we can confirm §1 from your side rather than only from the
   DLC. The Genie `--profile` JSONs from Tests 1/4/5 would also let us re-derive
   your rates independently.
2. **Your 7.79 tok/s build.** Test 3 could not be completed from one arm. We
   need the binary, the converter command lines, the build-time HTP config, the
   `qnn-context-binary-utility` JSON dump, and the dialog JSON used for that
   measurement. Our unfused `local` build measures 11.72 tok/s — **+51% faster**,
   i.e. the gap runs opposite to what we assumed, and we cannot explain it from
   our side alone.
3. Confirmation of the device-side `perf_profile` and clock state during
   `qnn-net-run`, given the 1170 ms vs 85 ms wall-clock discrepancy you flagged.

## 5. Fixed on our side: the LADE SIGSEGV

Your §6.1 finding (`type: "lade"` + `max-num-tokens` → exit 139) shipped in
`genie_dialog_demo.json` in three bundles — fuseqkvgu, socmodel72, hvx8 — so
every demo run of those died. Fixed and re-uploaded to HF as of 2026-08-14; the
combination is now refused by a linter that runs during bundling. If you pulled
those three bundles before today, re-pull them.

## 6. Everything is on HF now — `vinccniv/sa8797p-qwen3-w8a16-bundles`

Eight bundles that eliminate the GQA replication, the profiling bin for the
decisive measurement, and a **self-contained test kit**. Start at
`kit/runsheet.md`; every outcome already has a pre-agreed meaning in
`kit/decision_table.md`, so **the session should need no round-trip with us**.

| Path | What |
|---|---|
| `kit/runsheet.md` | priority-ordered arms, protocol, what to send back |
| `kit/decision_table.md` | what each result means, agreed in advance |
| `kit/run_all.sh` | runs priorities 2–7 unattended; skips absent bundles |
| `kit/prompts/`, `kit/expected/` | 3 prompt classes + greedy HF references |
| `profiling/qwen3-0.6b-w8a16-gqafix-decodeonly_ctx.bin` | priority 1 (B7a) |
| `qwen3_06b_w8a16_gqafix_*.tar.gz` | the eight bundles |

**Priority 1 is the decisive one:** the decode-only cycle profile on the fixed
graph. We expect **350.3M → ~90M aggregate cycles**. We tried to pre-answer it
without hardware using the x86 HTP emulation backend, but `libQnnHtpQemu.so`
rejects v81 ctx-bins outright (`Request feature arch with value 81
unsupported`), so it has to be measured on the device.

**Priority 3 is the cheapest high-value run** and needs nothing new from us:
basic mode on the plain `qwen3_06b_w8a16_ladekv` bundle you already have (§3).

Two things to know before you interpret anything:

- **`gqafix_local`, `gqafix_qh`, `gqafix_cl512`, `gqafix_dlbc`, `gqafix_udma`
  and `gqafix_hybrid` are ~1.32 GB, not 1.09 GB.** Any bin containing the
  CL=128 bertcache prefill graph carries one private ~444 MB copy of the
  decoder weights under the new attention. Weight *bytes* per step are
  unchanged, so this should not move decode traffic, but it costs disk on a
  device whose `/data` runs 98–99% full and may affect init time. Root cause
  not yet established; it is in the ctx-bin generator, not the graph topology.
  `gqafix_ladekv` and `gqafix_pastkv2g` are unaffected at 0.93 GB, and so is
  the single-graph profiling bin — **the decisive measurement has no confound.**
- Because of that, **the cleanest A/B is `gqafix_ladekv` basic mode versus the
  pre-fix `ladekv` basic mode** — same topology, same graph count, same size,
  only the attention differs. Treat `gqafix_local` vs the 11.72 tok/s baseline
  as corroboration rather than primary evidence.

If you pulled `fuseqkvgu` / `socmodel72` / `hvx8` before 2026-08-14, re-pull —
their demo config was the SIGSEGV one (§5).
