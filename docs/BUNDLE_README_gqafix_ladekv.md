# qwen3_06b_w8a16_gqafix_ladekv — Qwen3-0.6B W8A16, grouped GQA attention, LADE

Built 2026-08-14, device-free, on QAIRT 2.48.40.260702 for Hexagon v81 /
SA8797P. This is **the primary artifact of the 2026-08-14 GQA-fix drop** — the
bundle the whole session is designed around.

Everything here is self-contained: deploy, run, measure, interpret, and send
results back. The wider session context is in
[`../README.md`](../README.md) (drop landing page) and
[`../kit/runsheet.md`](../kit/runsheet.md) (all seven priorities).

---

## 1. What this bundle is, and why it exists

An op-level profile on 2026-08-13 found **74.7% of every decode step's DSP
cycles** — 261.8M of 350.3M — in 56 ops the report labelled "attention-mask
broadcast". Direct inspection of the shipped `decode.dlc` showed the label was
wrong: the mask is never expanded. Those 56 ops (2 per layer × 28 layers) are
**GQA KV-head replication**, materialising 8 KV heads into 16 so a 16-head
MatMul can consume them — 264 MB written and re-read every single step.

This bundle removes them. The attention MatMuls now batch over the 8 KV heads
directly (`1x8x2x1152` instead of `1x16x1x1152`). The KV I/O contract is
untouched, so the Genie feed pattern is identical to the pre-fix bundle.

**The open question is what the device does with the freed cycles.** Either
decode was compute-bound and throughput rises, or a weight/KV streaming floor
binds and it barely moves. This bundle answers that.

## 2. Contents

Flat layout — no `lib/` subdir, no `ADSP_LIBRARY_PATH` needed for Genie runs.

| File | Size | Notes |
|---|---:|---|
| `qwen3-0.6b-w8a16-gqafix-ladekv_ctx.bin` | 1,086,570,496 B | 3 graphs, weight-shared |
| `genie_dialog.json` | — | LADE, greedy — **the throughput number** |
| `genie_dialog_basic.json` | — | basic AR-1, greedy — **the primary A/B** |
| `genie_dialog_demo.json` | — | LADE + sampling — interactive use only |
| `htp_backend_ext_config.json` | — | O3, vtcm_mb 16, hvx_threads 4, v81, unsigned PD |
| `genie-t2t-run` | 564 KB | aarch64-android |
| `tokenizer.json` | 11.4 MB | |
| 7 × `lib*.so` | ~121 MB | QAIRT 2.48.40.260702 runtime |

### Verified ctx-bin contents

From `qnn-context-binary-utility --json_file` after generation — not inferred:

```
contextBlobSize   1,086,521,344 B      numGraphs  3      socModel  0

prefill    60 in / 57 out   input_ids [1,128]  mask [1,128,1152]   logits [1,128,151936]
                            past_key_0_in [1,8,128,1024]
decode     60 in / 57 out   input_ids [1,1]    mask [1,1,1152]     logits [1,1,151936]
                            past_key_0_in [1,8,128,1151]
verify32   60 in / 57 out   input_ids [1,32]   mask [1,32,1152]    logits [1,32,151936]
                            past_key_0_in [1,8,128,1120]
```

| Graph | sharedWeightsSize | constSize | spillFillBufferSize |
|---|---:|---:|---:|
| `prefill` | 1,067,499,520 | 0 | 0 |
| `decode` | 1,067,499,520 | 256 | 0 |
| `verify32` | 1,067,499,520 | 0 | 0 |

Three things that table confirms, each of which has broken a previous build:

- **Weight sharing is perfect.** One 1,067 MB pool, ~0 per-graph constants. This
  bin does **not** carry the CL=128 bertcache prefill, so it avoids the ~444 MB
  weight-duplication that inflates `gqafix_local` / `_qh` / `_cl512` / `_dlbc` /
  `_udma` / `_hybrid` to ~1.52 GB. **This bundle has no size confound.**
- **Zero spill/fill on all three graphs**, including `verify32` — the graph that
  silently got backend defaults (4 MB VTCM, 24 MB spill) in the 2026-08-10 build
  and null-deref SIGSEGV'd on the first speculation step.
- **`prefill` is a past-KV graph** (it has `past_key_0_in`, and mask `[1,128,1152]`
  with CL > AR), not a bertcache graph, and it emits **all-position** logits. That
  is what makes `type: "lade"` safe here and what enables prompts > 128 tokens.

## 3. Deploy and run

```sh
adb push qwen3_06b_w8a16_gqafix_ladekv.tar.gz /data/local/tmp/
adb shell 'cd /data/local/tmp && tar xzf qwen3_06b_w8a16_gqafix_ladekv.tar.gz \
                              && rm qwen3_06b_w8a16_gqafix_ladekv.tar.gz'
adb shell
cd /data/local/tmp/qwen3_06b_w8a16_gqafix_ladekv
LD_LIBRARY_PATH=. ./genie-t2t-run -c genie_dialog.json --prompt_file ../kit/prompts/technical.txt
```

`/data` on this device runs 98–99% full — **delete each tarball as you extract
it**, as above.

Use `--prompt_file`, not `-p`, for the kit prompts: they contain real newlines,
and `genie-app`-style script strings never unescape `\n`.

### The three dialog configs

One binary, three configs. Do not mix them up — two of the three are greedy
parity configs and are the wrong tool for a demo.

| File | Mode | Sampling | Use for |
|---|---|---|---|
| `genie_dialog.json` | `lade` — window 8 / ngram 3 / gcap 8, `ALWAYS_FWD_ONE` | greedy (temp 0, top-k 1), unbounded | the LADE throughput number |
| `genie_dialog_basic.json` | `basic` AR-1 | greedy, unbounded | **the primary A/B** vs the pre-fix bin |
| `genie_dialog_demo.json` | `lade` | temp 0.85 / top-k 50 / top-p 0.9 | anything interactive |

All three share: `context.size` 1024, `n-vocab` 151936, `bos-token` -1,
`eos-token` [151645, 151643] (demo adds 151647), `n-threads` 3,
`cpu-mask` 0xe0, `kv-dim` 128, `mmap-budget` 25, `spill-fill-bufsize` 0,
`enable-graph-switching` true, rope `rope-dim` 64 / `rope-theta` 1e6.

**Use the demo config for free-form runs.** The greedy configs are correct for
measurement but produce repetitive output until `Context Size was exceeded` —
that is the shipped default's fault, not the model's.

## 4. Rules that have each cost a debugging session

- **Apply the Qwen3 chat template with an empty `<think>\n\n</think>` block** in
  the assistant prefix. Without it, thinking mode triggers and latency balloons.
  The kit prompts already do this. `bos-token` is `-1` because the template
  supplies `<|im_start|>` itself — do not set both.
- **LADE prompts must tokenize to ≥ 2 tokens.** A 1-token prompt hits
  `rand() % (tokens.size()-1)` = modulo zero in qualla's warmup
  (`lhd-dec.cpp:120`) and reads far out of bounds. No bundle can fix this; the
  crash register `x0=0x6b8b4567` is exactly `rand()`'s first output.
- **Never add `max-num-tokens` to a `type: "lade"` config.** That pair SIGSEGVs
  (exit 139) on the first speculation step — device report §6.1. It is why every
  demo run of `fuseqkvgu` / `socmodel72` / `hvx8` died before 2026-08-14.
  Generation here is bounded by `context.size` and EOS instead, and a linter now
  refuses the combination at build time.
- **Do not retune the lade block.** The guardrail is
  `(ngram − 1) × (window + gcap) ≤ 32`, the verify graph's AR. Shipped 8/3/8 is
  exactly 32. Oversizing silently routes verification batches to a graph that
  cannot serve them.
- **Do not edit `graph_names` in `htp_backend_ext_config.json`.** They must stay
  exactly `prefill`, `decode`, `verify32` — those names are baked into the
  ctx-bin at conversion time and renaming files afterwards does not change them.
  A mismatch means that graph silently gets backend defaults (4 MB VTCM, 24 MB
  spill); for `verify32` that is a null-deref SIGSEGV.
- **Keep `perf_profile: "llm_decode_burst"` and `rpc_polling_time: 9999`.** The
  4-tier profile ladder spans 1.95×.
- **`Unknown Key` warnings mean the arm is not testing what it claims.** The key
  was silently ignored. This has bitten this project three times; see
  `docs/NOTES-htp-config-keys.md`.
- `hvx_threads` is **build-time only** (established by the device team's Test 5).
  Editing it at runtime does nothing.

## 5. How to test

### 5.1 Protocol — identical to 2026-08-13, so results compare

- **Warm.** One discarded warm-up per bundle. Cold init is 1.8–2.0 s vs ~800 ms
  warm and pollutes averages.
- **3 reps per arm**, greedy, same prompt across arms being compared.
- **Report init time and TTFT separately.** Never compare init→first-logits
  against TTFT — they differ by ~800 ms, and that unit mismatch has already
  produced one phantom regression.
- Change nothing except what the arm explicitly changes.

### 5.2 The arms this bundle carries

This one bundle serves **6 of the kit's 15 Genie arms** — all of priority 2 and
all of priority 6. `kit/run_all.sh` runs them unattended:

| Arm | Dialog | Prompt | Priority |
|---|---|---|---|
| `p2_gqafix_ladekv_basic` | `genie_dialog_basic.json` | technical | 2 — **the headline** |
| `p2_gqafix_ladekv_lade` | `genie_dialog.json` | technical | 2 |
| `p6_lade_simple` | `genie_dialog.json` | simple | 6 |
| `p6_lade_structured` | `genie_dialog.json` | structured | 6 |
| `p6_basic_simple` | `genie_dialog_basic.json` | simple | 6 |
| `p6_basic_structured` | `genie_dialog_basic.json` | structured | 6 |

Prompt token counts: `simple` 36, `structured` 71, `technical` 56. The technical
prompt matches the 2026-08-13 protocol exactly, so its rates are directly
comparable to the baselines below.

### 5.3 The comparison that matters

**Run `p2_gqafix_ladekv_basic`, and compare it against the pre-fix
`qwen3_06b_w8a16_ladekv` in basic mode** (`p3_a1_ladekv_basic`, on a bundle the
device already has).

Those two are the same topology, same graph count, same 1.087 GB ctx-bin, same
encodings recipe — **only the attention differs**, so any delta is attributable.

`gqafix_local` vs the 11.72 tok/s baseline is *corroboration, not primary
evidence*, because that bin is 1.52 GB rather than 1.09 GB and therefore not
size-matched.

Baselines to beat — all 2026-08-13, warm, greedy, 56-token technical prompt:

| Baseline | Value |
|---|---|
| Basic AR-1, `local` (2-graph) | **11.72 ± 0.01 tok/s** |
| LADE, `ladekv` (3-graph), technical prompt | 9.18 ± 0.00 tok/s |
| LADE, earlier simple prompt | 10.8 tok/s |
| Decode-only aggregate DSP cycles | 350,302,972 |
| Decode step time (Genie) | ~85 ms |

### 5.4 What the result means — pre-agreed

Read this together with the priority-1 cycle profile (the decode-only bin in
`../profiling/`, expectation 350.3M → ~90M cycles). Neither measurement alone is
decisive; together they separate the two competing models of the machine.

| | **tok/s rises** (≥ 14) | **tok/s flat** (11.5–13) |
|---|---|---|
| **cycles fall** (≈ 90M) | **A. Compute-bound, fix works.** `gqafix` becomes the ship base; proceed to priority 4, then re-tune LADE on the new base. | **B. Byte floor is real.** The DSP got 4× cheaper and the step did not — decode is bound by streaming 883 MB (751 weights + 132 KV). Priority 4 becomes the lead workstream; stop optimising compute. |
| **cycles flat** (≈ 350M) | **C. Impossible — investigate.** Check you ran the gqafix bundle, and check graph names against the ctx-bin. Report before concluding. | **D. The fix did not reach the device.** Verify the decode graph's attention MatMuls are `1x8x2x1152`. Report as a **build defect**, not as "the GQA fix does not help". |

**Converter-side bound on box B:** `read_total_bytes` is **961,130,496 for the
gqafix decode graph — unchanged from pre-fix**. That is expected: replication
cost was intermediate-tensor traffic, not weight reads. Pre-fix the step moved
≈961 MB + ≈530 MB ≈ 1.49 GB; post-fix ≈961 MB. Purely bandwidth-bound at an
unchanged rate, 85 ms would become ≈55 ms, i.e. **≈18 tok/s**. A result far
below that, with cycles confirmed down, is box B.

**Sanity check before trusting any of the above:** the fixed decode graph must
contain **zero** `Eltwise_Binary` ops with `operation: 13` whose output is
`[1,8,2,...]`. If any remain, the export flag did not propagate and the arm is
invalid.

For the priority-6 LADE map:

| Observation | Pre-committed conclusion |
|---|---|
| LADE beats basic on ≥ 2 of 3 prompt classes | LADE is the default for those workloads; proceed to the parameter sweep, optimising **acceptance** |
| LADE wins only on `structured` | Ship LADE only for structured/repetitive workloads |
| LADE loses on all three post-fix | **Basic is the ship configuration.** Park LADE — post-fix, verify32 regains AR-scaling, so the 1.68× measured pre-fix does not survive |

### 5.5 What to report

1. **Sustained tok/s** from the profile's TGR.
2. **Accepted tokens per verify call** (total tokens ÷ verify calls) and per-call
   latency, for the LADE arms. `tok/s ≈ acceptance ÷ latency`. The pre-fix
   unfused build sits at ~1.94 accepted per ~180 ms call. **If a change buys
   per-call latency but costs acceptance it is a net regression** — that is
   exactly how `--quant-head` failed at −14%. Report both numbers, never tok/s
   alone.
3. **Init time and TTFT, separately labelled.**

### 5.6 Quality check — before trusting any speed number

`../kit/expected/` holds the greedy continuation this bundle should produce for
each prompt, generated from the local ONNX parity harness. Diff the first ~30
tokens of each arm's `stdout_r1.txt` against it.

An INT8 device run will not match forever — **early divergence is the signal,
not eventual divergence.** Flag any arm that diverges in the first few tokens or
degenerates into repetition. W8A16 quantisation bugs have historically produced
*fluent but wrong* output rather than obvious garbage, so speed on a broken
graph looks perfectly plausible.

### 5.7 Things that invalidate a result — report, don't work around

- Any arm exits **139** (SIGSEGV). Should not happen now; if it does, send the
  dialog JSON that was used.
- **Init time > 1.2 s** — the run was not warm. Re-run.
- Generated token count differs between arms being compared on rate.
- **`Unknown Key`** warnings from the backend config.
- Quality regression against `expected/`.

## 6. Sending results back

Send the whole `results/` directory: per-rep `--profile` JSON, `stdout_r*.txt`,
the `dialog_used.json` and `htp_backend_ext_config.json` each arm actually used,
and `MANIFEST.txt` (which records what ran and what was skipped).

```sh
# on the device
cd /data/local/tmp && tar czf results_$(date +%Y%m%d).tar.gz results/
adb pull /data/local/tmp/results_YYYYMMDD.tar.gz .
```

### Uploading to the HF repo

Requires a token with **write** access to
`vinccniv/sa8797p-qwen3-w8a16-bundles`. If you do not have one, use the existing
exchange channel instead — this is a convenience, not a requirement.

```sh
export HF_TOKEN=hf_...
export HTTPS_PROXY=http://127.0.0.1:17890   # build-side only; skip if you route directly

hf upload vinccniv/sa8797p-qwen3-w8a16-bundles \
    results_YYYYMMDD.tar.gz \
    2026-08-14-gqafix/results/results_YYYYMMDD.tar.gz
```

Two hazards, both of which have bitten this repo:

- **Never use `hf upload-large-folder` (or a folder-wide `hf upload`) against
  this repo.** Those code paths create/update the repo with their own default
  settings and have silently changed its **visibility** and clobbered the hub
  README. Upload single files, to explicit paths. The build side uses
  `scripts/util/hf_upload_file.py`, which does one `upload_file` commit and
  never reads or writes visibility.
- **This repo's visibility is toggled deliberately.** If a download 404s while
  `hf auth whoami` succeeds, it is private at that moment — ask, and it will be
  opened or your account added. Do not try to change it yourself.

Also still wanted from the 2026-08-13 session, if they still exist:
`test2_decode_profile/qnn_profile_r{1,2,3}_viewer.txt`. Only the narrative report
reached the build side, so there is no per-op baseline to diff the new profile
against.

## 7. Build provenance

- **W8A16**: INT8 per-channel symmetric weights, FP16 activations, AIMET 2.36
  PTQ. `embed_tokens`, final norm, `lm_head` and K/V-proj outputs stay FP16.
  `lm_head` is deliberately **not** quantised — `--quant-head` measured **−14%**
  under LADE (9.3 vs 10.8 tok/s) because it costs ~10% n-gram acceptance.
- **Attention**: grouped GQA. Q reshaped to `[1,8,2·AR,128]` and matmul'd against
  the un-replicated `[1,8,...]` cache; zero `Expand` / `repeat_kv` ops.
- **Topology**: 3 graphs, one encodings lineage so KV quant params are
  byte-identical across graphs (mixed encodings are a fatal Genie load error).
  `prefill` AR=128 CL=1152 past-KV all-position logits · `decode` AR=1 CL=1152 ·
  `verify32` AR=32 CL=1152.
- **Build config**: `O:3`, `vtcm_mb:16`, `hvx_threads:4`, `soc_model:0`, unsigned
  PD, weight sharing on.

Gates passed before shipping — all device-free:

| Gate | Result |
|---|---|
| Grouped-GQA numerical equivalence vs the replicating form (float64) | max \|Δ\| **6e-16**; **bit-identical** for decode. Across AR=1 / 32 / 128, with and without past, fused and unfused |
| Grouped-GQA ONNX wrapper vs HF logits | max \|Δ\| **2.12e-05** |
| ONNX prefill argmax vs HF | identical |
| 8-step greedy decode chain vs HF `generate` | token-identical |
| `parity_ladekv_read.py` — qualla's exact feed pattern | **6/6** (4 single-chunk + 2 chunked through growing past-KV) |
| `lint_gqa_ops.py` topology assertion, per DLC | **0** replication ops on all four exported graphs; attention MatMuls batched over 8 KV heads |
| AIMET `--eval` last-token argmax vs FP32 | 3/4 reference prompts (the project's standing reference) |
| ctx-bin graph names / weight sharing / spill | verified post-generation — see §2 |

## 8. Related

| Path | What |
|---|---|
| [`../README.md`](../README.md) | the 2026-08-14 drop landing page, all 8 bundles |
| [`../kit/runsheet.md`](../kit/runsheet.md) | all seven priorities, in order |
| [`../kit/decision_table.md`](../kit/decision_table.md) | every outcome's pre-agreed meaning |
| [`../profiling/`](../profiling) | priority 1 — the decisive decode-only cycle profile |
| `docs/DEVICE_TEAM_EXCHANGE_2026-08-14.md` | why the 74.7% attribution changed |
| `docs/NOTES-htp-config-keys.md` | which HTP backend keys are real |
