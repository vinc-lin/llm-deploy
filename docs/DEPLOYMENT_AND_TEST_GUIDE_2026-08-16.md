# SA8797P Qwen3-0.6B — Deployment and Test Guide
## 2026-08-16 regime drop · self-contained

**Everything needed to deploy, run, measure and report is in this one document.**
The kit files (`kit-v2/`) are the scripted form of §4–§6; this document is the
authority if they disagree.

| | |
|---|---|
| Target | SA8797P (nordy / Gen5), Hexagon v81 HTP, Android GVM, unsigned PD, 16 MB VTCM |
| Runtime | QAIRT 2.48.40.260702 · QNN API v2.37.0 · libGenie 1.19.0 (built to match exactly) |
| Model | Qwen3-0.6B, W8A16 (INT8 per-channel weights, FP16 activations), grouped-GQA attention |
| Current baseline | **44.707 ± 0.030 tok/s** basic mode, `gqafix_ladekv` |
| Session length | ~2 h for everything; **~10 min for the arm that decides the plan** (§4.2) |

---

## 1. What this drop is for, in one page

The GQA KV-replication fix shipped and measured **6.54×**: 6.836 → 44.707 tok/s
on a like-for-like control. Basic mode beats LADE (31.342), so **basic is the
ship configuration** and LADE is parked.

What is *not* known is **why 44.707 is the number**. Three models remain
admissible and they imply completely different next years of work
(`REFERENCE.md` §8.11):

| Model | For | Against |
|---|---|---|
| **Compute-bound** | Predicted the post-fix step **out-of-sample**: 88.2M cycles ÷ 4 HVX @ ~1 GHz ≈ 22.1 ms vs **22.37 measured**, written before the fix shipped. The divisor of 4 is now confirmed by reading `numHvxThreads` out of the ctx-bin | Clock is invisible under the GVM, so threads × clock is one product with two unknowns — 8 threads @ 0.5 GHz fits identically |
| **Byte-bound** | Simple; post-fix ~43 GB/s is at least plausible | Predicted 18.1 tok/s pre-fix; reality was 44.7. **And the fix removed ~0 bytes yet gave 6.5×** |
| **Access-pattern fragmentation** | The ~220 µs RPC / 30–60 µs inter-op costs are real and measured | Those costs survive the GQA fix unchanged, so they cannot explain a 6.5× change |

**The byte measurement is already done, and it is the strongest single result.**
Pre-fix and post-fix decode graphs read **byte-identical** DDR traffic —
`read_total_bytes = 961,130,496`, `write_total_bytes = 419,840`, both sides. The
replication ops moved ~264 MB of intermediate tensor per step but **never
reached DDR**; they were VTCM-resident, and `spill_bytes`/`fill_bytes` are 0.

Identical bytes, 6.54× the throughput. A byte-bound step would have taken the
same time, so **the pre-fix regime was compute-bound.**

⚠️ Note also that "effective bandwidth rose from ~17.5 to ~43 GB/s" is
**circular** if the byte count did not change — with bytes constant it is a
restatement of "the step got 6.5× faster", not an explanation of why. Do not
quote it as evidence.

That does *not* settle the **post-fix** regime — at 43.0 GB/s we are near the
streaming ceiling, so DDR could bind *now* even though it demonstrably did not
before. §4.2 is the arm that settles it, because it is the only one that varies
compute while holding bytes exactly constant.

---

## 2. ⚠️ The one rule, and why it exists

> **Never compare a decode rate from a BLENDED bundle against one from a PURE
> bundle.**

A bundle containing a graph whose **AR equals its CL** (a "bertcache" prefill)
makes Genie keep generating *through that graph*, re-processing the whole window
once per token, until the KV cache passes AR — only then does the AR-1 decode
graph take over (`kvmanager.cpp:421-429`). Its tok/s is a time-weighted blend of
two rates, and **the blend runs fast**, so the failure mode is a confident wrong
number, not an obviously broken one.

That is what the old **11.72 tok/s** figure was:

```
2026-08-13 Test 1 Arm A, qwen3_06b_w8a16_local, 56-token prompt, 128 generated, 10,837 ms
   tokens 1..72   (KV 56 -> 128) @ 40.1 ms each  = 2,887 ms    <-- 40.1 ms is that run's own TTFT
   tokens 73..128 (56 tokens)                    = 7,950 ms
                                    7,950 / 56   =  142.0 ms per AR-1 step
                          vs 146.3 ms measured independently on 2026-08-15  -> 3% agreement
```

Three previously-reported findings were artifacts of quoting that blend as a
decode rate, and **all three are withdrawn**: the "~75% build gap between `local`
and `ladekv`", "LADE is −22% on the technical prompt", and "our builds are +51%
faster than the device team's". The honest pre-fix AR-1 rate is **6.836 tok/s**.

**Every bundle in this drop is topologically pure** and was gated with
`lint_bundle_topology.py --require-pure` before shipping. From the 2026-08-14
drop, only `gqafix_ladekv` and `gqafix_pastkv2g` were ever pure — **do not run
that drop's priorities 4 or 5.**

To classify anything you add later:

```sh
python3 lint_bundle_topology.py <ctx-bin>      # prints "pure" or "BLENDED"
```

---

## 3. Deployment

### 3.1 Prerequisites on the device

Flat layout — everything in one directory, no `lib/` subdirectory,
`LD_LIBRARY_PATH=.`, and `ADSP_LIBRARY_PATH` is **not** required for
`genie-t2t-run` (it *is* required for `qnn-net-run`, §5).

Each bundle already contains the 7 required `.so` files. If any one is missing
you get error **14001**, which does *not* mean "library not found" — it means
invalid DSP performance-infrastructure configuration.

### 3.2 `/data` runs 98–99% full — plan for it

```sh
# BEFORE anything else: rescue the previous session's results
adb pull /data/local/tmp/results ./results-2026-08-15

# then clear space
adb shell 'cd /data/local/tmp && rm -rf qwen3_06b_w8a16_gqafix_* && df -h /data'
```

Each bundle is ~925 MB compressed and ~1.2 GB extracted. **Push, extract, and
delete the tarball one bundle at a time** — do not push all eight first.

### 3.3 Push and extract

```sh
adb push qwen3_06b_w8a16_gqafix_hvx8_ladekv.tar.gz /data/local/tmp/
adb shell 'cd /data/local/tmp && tar xzf qwen3_06b_w8a16_gqafix_hvx8_ladekv.tar.gz \
           && rm qwen3_06b_w8a16_gqafix_hvx8_ladekv.tar.gz'
```

Pushes >500 MB can trigger USB disconnects; `adb reconnect` restores it.

### 3.4 Smoke test one bundle before measuring anything

```sh
adb shell 'cd /data/local/tmp/qwen3_06b_w8a16_gqafix_hvx8_ladekv && \
  LD_LIBRARY_PATH=. ./genie-t2t-run -c genie_dialog_basic.json \
  --prompt_file /data/local/tmp/kit-v2/prompts/technical.txt'
```

Expect coherent English and a `TGR` around 44 tok/s. If it SIGSEGVs (exit 139)
or emits `!!!!!`, stop and report — do not proceed to measurement.

---

## 4. Testing procedure

### 4.1 Protocol — this changed, and it is not optional

The 2026-08-15 session measured **23.43 / 44.54 / 29.34 tok/s on one binary**.
That spread is larger than every effect this drop chases, so 3 reps cannot
resolve anything.

| Rule | Value |
|---|---|
| **Reps** | **5 per arm** (8 for §4.5), report the **median** and **every raw value** |
| Warm-up | 1 discarded run per bundle — cold init is 1.8–2.0 s vs ~800 ms warm |
| Sampling | greedy: `temp=0`, `top-k=1`, `top-p=1.0`, `seed=42` |
| Prompt | `prompts/technical.txt` (56 tokens) unless stated |
| Backend | `perf_profile: llm_decode_burst`, `rpc_polling_time: 9999` — **change nothing else** |
| Cool-down | **30 s between arms**, temperature logged before and after |
| Order | run `p0_rebaseline` **first**; every delta is computed against it |

**Required precision.** The effects being separated are ~5–26%, i.e. ~2–12 tok/s
at this baseline, and the smallest decision threshold in §6 is **+5%** (~2.2
tok/s). For an arm to decide anything its **5-rep spread must be under ±2 tok/s
(±4.5%)**. If it is not, that arm decided nothing — re-run it, do not interpret
it. This is the single most likely way this session produces nothing: last time
one binary spanned 23.43–44.54 tok/s.

### 4.2 ⭐ Priority 1 — the orthogonal test (~10 min, and it is the cheapest)

| Arm | Bundle | Byte model | Compute model |
|---|---|---:|---:|
| `p4_hvx8` | `qwen3_06b_w8a16_gqafix_hvx8_ladekv` | **0.0%, by construction** | up to large |

**This is the only proposed experiment that is orthogonal by construction.** It
is the *same DLCs* as the baseline with `hvx_threads: 8` set at **build** time —
so it changes compute capacity while holding DDR bytes *exactly* constant. The
byte model predicts exactly 0.0%; anything above the rep spread falsifies it
outright.

Verified device-free, two independent ways:

- **Compiled-value readback**: `numHvxThreads` reads back as **8** out of the
  finalized binary (every other bundle in this drop reads 4). The knob bound —
  this is not inferred from file size.
- **Byte accounting**: both it and the baseline report `read_total_bytes =
  961,130,496`, `write_total_bytes = 419,840`. Identical.

> ### Compare against `p0_rebaseline` — there is no separate control arm
> `qwen3_06b_w8a16_gqafix_ctrl_ladekv` ships in this drop but is **byte-identical
> to `qwen3_06b_w8a16_gqafix_ladekv`** (md5 `9c6024ad5b141137fbe22f3a4972eb96` on
> both ctx-bins). ctx-bin generation is deterministic, so rebuilding the baseline
> config reproduces the baseline bin exactly. **Running it as a separate arm
> would burn ~10 minutes measuring the same binary twice.** It is included only
> so the drop is self-contained; skip it.

### 4.3 ⭐ Priority 2 — the CL arm (~15 min)

| Arm | Bundle | Dialog | Byte model | Compute model |
|---|---|---|---:|---:|
| `p4_cl512_ladekv` | `qwen3_06b_w8a16_gqafix_cl512_ladekv` | `genie_dialog_basic.json` | +10.1% | **+26.0%** |

Changes both bytes and cycles, but by *different ratios*, so it discriminates by
magnitude rather than by sign. Its saving is activation traffic, which — unlike
the W8 head — cannot be re-materialized away at context-prepare time.

### 4.3b Priority 3 — the W8 head (optional; it is confounded)

| Arm | Bundle | Dialog | Byte model | Compute model |
|---|---|---|---:|---:|
| `p4_qh_ladekv` | `qwen3_06b_w8a16_gqafix_qh_ladekv` | `genie_dialog_basic.json` | +17.9% | +3.6% |

> ⚠️ **Demoted from priority 1 — this arm confounds the question it was meant to
> answer.** It changes DDR bytes *and* is independently suspected of never
> reaching the device. Two lines of evidence:
>
> - Its ctx-bin shrank only **8.4 MB** although the converter credits −146.1 MB,
>   reproducing the same anomaly the 2026-08-12 build showed (DLC −151.3 MB,
>   ctx-bin −12.5 MB). The likely cause is HTP re-materializing the INT8 head to
>   16 bits at prepare time, since the `FullyConnected`'s input and output are
>   both `Float_16`.
> - **On-device corroboration:** decode PD footprint disagrees 1.8× between two
>   near-identical bins — ~167 MB (ladekv) vs 304 MB (qh) — while prefill and
>   verify32 agree within a percent. 167 + 144 = 311 ≈ 304, so the ~144 MB looks
>   real and *resident on the device*.
>
> Expect **≈0%**. That is a valid, informative outcome (§6.1) — it closes a
> two-session-old question — but it is not a discriminator. Run it after the
> first two, or not at all if time is short.

> ⚠️ **Every bundle's `htp_backend_ext_config.json` says `hvx_threads: 4`,
> including `hvx8`. That is correct — do not "fix" it.** `hvx_threads` is
> build-time only; your own 2026-08-13 Test 5 showed the runtime knob is inert
> (9.17 vs 9.18 tok/s). The 8 is baked into `hvx8`'s ctx-bin, which is why that
> bin is 2,277,376 bytes larger than the control. Changing the runtime value
> would turn a clean single-variable test into a two-variable one.

### 4.4 Priority 4 — the decode-only cycle profile

**Uses `qnn-net-run`, not `genie-t2t-run`.** Not in `run_all.sh`.

Last session this was skipped, reported as "pre-fix format" profiling inputs.
**That diagnosis was wrong** — the pre-fix and post-fix decode graphs have
byte-identical input contracts (60 inputs, same names, shapes, dtypes), and the
shipped inputs score **60/60** against the gqafix bin. `position_ids_cos` is 128
bytes because it is `[1,1,64]` fp16; 128 B is correct.

So the real cause is unknown. **Verify first:**

> **Where the files are.** `verify_profile_inputs.py` and
> `decode_profile_inputs.tar.gz` ship in *this* drop. The decode-only ctx-bin
> (`qwen3-0.6b-w8a16-gqafix-decodeonly_ctx.bin`, ~1.07 GB) and
> `htp_config_decodeonly.json` are **not duplicated here** — they are unchanged
> from `2026-08-14-gqafix/profiling/`. Pull them from there, or reuse the copy
> already on the device.

```sh
cd /data/local/tmp/profiling
tar xzf decode_profile_inputs.tar.gz                    # -> ar1_decode/

python3 verify_profile_inputs.py \
    --bin qwen3-0.6b-w8a16-gqafix-decodeonly_ctx.bin \
    --inputs ar1_decode/inputs                          # expect 60/60

cd ar1_decode
export ADSP_LIBRARY_PATH=.:/vendor/lib/rfsa/adsp
LD_LIBRARY_PATH=. qnn-net-run \
    --backend libQnnHtp.so \
    --retrieve_context ../qwen3-0.6b-w8a16-gqafix-decodeonly_ctx.bin \
    --input_list input_list.txt --output_dir out_gqafix \
    --profiling_level detailed --config_file ../htp_config_decodeonly.json

qnn-profile-viewer --input_log out_gqafix/qnn-profiling-data_0.log > profile_gqafix_viewer.txt
```

If the verifier passes and `qnn-net-run` still fails, the cause is environmental.
**Report which of these four it was** — that is now more useful than the profile:

1. the `--retrieve_context` path does not resolve after extraction
2. `ADSP_LIBRARY_PATH` unset
3. a silently truncated extract (`/data` is 98–99% full; package + bin ≈ 1.1 GB)
4. the two `Unknown Key` warnings for `memory.extended_udma` and
   `graph_configs_extra.sparse_weights_compression` — these are **expected here
   and are not the error.** That config is kept byte-identical to the one behind
   the 350,302,972-cycle baseline on purpose, so the comparison stays valid

3 reps. Expect **~88M aggregate cycles** (down from 350,302,972) and **zero
`Expand` ops**.

⚠️ This profile **cannot settle byte-vs-compute on its own**: at 4 HVX threads
~88M reads compute-bound, at 8 it reads byte-bound, and the runtime reports 8
while every build compiles for 4. §4.2 is what disambiguates it.

### 4.5 Priority 5 — the variance question (8 reps)

| Arm | Bundle |
|---|---|
| `p2_pastkv2g_var` | `qwen3_06b_w8a16_gqafix_pastkv2g` (from the 2026-08-14 drop) |

8 reps, temperature logged either side. The 23.43 / 44.54 / 29.34 spread
currently bounds the resolution of every measurement either side can take. Init
time tracked it weakly (873 / 811 / 854 ms — the fastest rep also initialised
fastest), which is suggestive of thermal or DVFS state.

### 4.6 Priority 6 — the cheap knobs (~5 min each)

Compare each against `p0_rebaseline` (see §4.2 — `p5_ctrl` is a byte-identical copy of it and is not worth a separate arm).

| Arm | Bundle | Tests | Offline evidence |
|---|---|---|---|
| `p6_udma` | `..._gqafix_udma_ladekv` | `extended_udma`, a v81-and-above feature | binary +212,992 B — **consumed; its first real build ever** |
| `p6_socmodel72` | `..._gqafix_socmodel72_ladekv` | `soc_model`/`soc_id` 72 | binary +249,856 B — consumed |
| `p6_dlbc` | `..._gqafix_dlbc_ladekv` | activation DLBC | binary **identical** — unproven |
| `p6_wpack` | `..._gqafix_wpack_ladekv` | `weights_packing`, never tried | binary **identical** — unproven |

`dlbc` and `wpack` produced byte-identical binaries offline. Run only if time
allows.

### 4.7 Priority 7 — prompt distribution

| Arm | Prompt | Generated tokens |
|---|---|---|
| `p3_basic_structured` | `prompts/structured.txt` | ~150–250 |
| `p3_basic_simple` | `prompts/simple.txt` | **~10–30 ⚠️** |

⚠️ `simple` generates too few tokens for a trustworthy rate — it is dominated by
prefill and first-token latency. **Indicative only**; do not average it into a
distribution claim or quote it as a decode rate. `structured` carries that claim.

### 4.8 Unattended run

```sh
cd /data/local/tmp
sh kit-v2/run_all.sh results-2026-08-16
```

Runs §4.2, §4.3, §4.3b, §4.5, §4.6, §4.7 in priority order with the correct rep counts
and cool-downs. Skips any absent bundle. §4.4 must be run by hand.

---

## 5. Required result metrics

### 5.1 Per rep — record all of these

| Field | Source | Notes |
|---|---|---|
| `TGR` (tok/s) | `--profile` JSON | **the headline** |
| `TTFT` (ms) | `--profile` JSON | prefill start → first token |
| `init` (ms) | `--profile` JSON | **never compare against TTFT** — they differ by ~800 ms and this has already produced one phantom regression |
| `PPR` (tok/s) | `--profile` JSON | prompt processing rate |
| generated tokens | stdout | needed to sanity-check TGR |
| generation time (ms) | `--profile` JSON | |
| exit code | shell | 139 = SIGSEGV — record and continue, do not abort the session |

### 5.2 Per arm — the reporting unit

| Field | Required | Acceptance |
|---|---|---|
| **median TGR** | ✅ | this is what gets compared |
| **all 5 raw TGR values** | ✅ | not just the mean |
| **spread** = (max − min) / median | ✅ | **must be < 4.5%**, else the arm decided nothing |
| temperature before / after | ✅ | `/sys/class/thermal/thermal_zone*/temp` |
| median TTFT, median init | ✅ | |
| Δ vs its control, in % | ✅ | **every arm vs `p0_rebaseline`** — there is no separate control bundle (§4.2) |
| quality verdict | ✅ | §5.3 |
| dialog + backend config actually used | ✅ | `run_all.sh` copies these automatically |

### 5.3 Quality gate — before trusting any speed number

Diff the first ~30 tokens of each arm's `stdout_r1.txt` against `expected/`.
**Early divergence is the signal; eventual divergence is normal for INT8.**

Flag an arm if it diverges in the first few tokens, or degenerates into
repetition. W8A16 bugs historically produce *fluent but wrong* output rather than
obvious garbage — the 2026-08-15 `gqafix_hybrid` arm looped on `"and parallel,
and parallel…"` forever.

> **Two arms legitimately differ from the references:**
> - **`qh`** has an INT8 `lm_head`. Quality is unchanged at 0.6B greedy, but
>   token-exactness is **not** guaranteed. Judge on coherence; do not report a
>   wording difference as a regression.
> - **`cl512`** has `context.size` 512 — identical for these prompts, divergent
>   only past 512 generated tokens.
>
> Every other arm uses **byte-identical DLCs** to the baseline, so those must
> match token-for-token. A divergence there is a real finding.

### 5.4 Results template

```
ARM: p4_cl512_ladekv          BUNDLE: qwen3_06b_w8a16_gqafix_cl512_ladekv
TOPOLOGY: pure                DIALOG: genie_dialog_basic.json
PROMPT: technical.txt (56 tok)

temp_before: <...>            temp_after: <...>
TGR raw:  __.___  __.___  __.___  __.___  __.___
TGR median: __.___    spread: __._%   (must be < 4.5%)
TTFT median: ___._ ms   init median: ___ ms   PPR median: ____ tok/s
Δ vs p0_rebaseline: +__._%
quality: PASS / FAIL (first 30 tokens vs expected/technical.txt)
notes:
```

### 5.5 What to send back

The whole `results-2026-08-16/` directory — per-rep `--profile` JSON, stdout, and
the dialog + backend configs each arm used. `MANIFEST.txt` records what ran and
what was skipped. Plus the §4.4 viewer outputs and raw `.log` files, and the
temperature log.

---

## 6. What each outcome means — agreed in advance

No result below needs a round-trip. If you hit a state that is not here, record
it and stop — that is a finding too.

### 6.0 Gates that outrank everything else

| Observation | Action |
|---|---|
| an arm's spread > the delta being measured | **Undecided.** Re-run under §4.1. Do not interpret |
| a blended bundle's tok/s appears in a comparison | **Reject the comparison** (§2) |
| `p0_rebaseline` median is not ≈ 44.7 | Something environmental changed — report it; the session is not comparable to 2026-08-15 until explained |
| any arm's first ~30 tokens diverge (except per §5.3) | **Quality failure — the speed number is meaningless.** Report the divergence, not the tok/s |

### 6.1 The arms, in priority order

**`hvx8` (§4.2) is the decisive one** — it is the only arm that varies compute
while holding DDR bytes exactly constant, so it needs no assumption about which
model is right.

| `p4_hvx8` vs `p0_rebaseline` | Verdict | Consequence |
|---|---|---|
| **> +5%** | **Byte model falsified outright** — zero DDR bytes changed, so a byte-bound step could not have moved | Adopt `hvx_threads: 8` in every build immediately; it is free. Compute-bound is confirmed; the CL ladder and attention-side compute become the workstream. Re-read §4.4's cycle count at 8 threads, not 4 |
| within the rep spread | Build-time 4 was not binding, or 8 units were already in use. **Does not confirm byte-bound** — it only removes one refutation | Keep 4. Fall through to `cl512` |
| **negative beyond the spread** | Thread oversubscription on a single HTP | Keep 4 and record — first evidence that HVX scheduling is contended |

Then `cl512`, which discriminates by magnitude rather than sign:

| `p4_cl512_ladekv` | Reading |
|---|---|
| **≥ +19%** | Consistent with compute-bound (predicted +26.0%). Build `cl768`, reopen `cl256` for short-context products |
| ~+10% | Consistent with byte-bound (predicted +10.1%). KV signed-INT8 and weight-stream compression lead; CL stays a product knob |
| between | Mixed regime — do not guess a ladder; re-derive it from §4.4's per-op table |
| **< +5%** | **Neither model holds.** Something outside both binds: per-op dispatch (~220 µs RPC × op count), DVFS, or whatever drives the rep variance. §4.4 becomes the only admissible next input, and "is `llm_decode_burst` actually the maximum HTP/DDR clock?" escalates to the platform team |

And `qh`, which is confounded and expected to be a null:

| `p4_qh_ladekv` | Reading |
|---|---|
| **≈ 0%** | **The expected result.** Confirms the head's saving never reaches the device — closing a question open since 2026-08-12, and corroborating the 1.8× decode PD-footprint anomaly. Park `qh` permanently |
| ≈ +18% | The converter accounting *is* what the device streams. Surprising; would reopen the byte model and make the PD-footprint anomaly need another explanation |
| ≈ +4% | Compute-bound, and the head does reach the device | 

### 6.2 The null test — see §6.1

`hvx8` is now priority 1 and its outcomes are tabulated at the top of §6.1.
Compare it against `p0_rebaseline`, not against a separate control.

### 6.3 The cycle profile

| Aggregate cycles | Meaning |
|---|---|
| **~88M, zero `Expand`** | Fix confirmed structurally. Combine with §6.2: at 4 threads ~99% DSP-busy (compute-bound); at 8, ~50% idle (byte-bound) |
| ~350M | The fix did **not** reach this binary — a build defect on our side. Send the `qnn-context-binary-utility` dump and stop |
| between | Partial — some graphs kept the old attention |
| still cannot run | Report **which** of §4.4's four candidates. Do not spend >15 min |

**The new top-20 is as valuable as the total** — with replication gone, whatever
now sits at the top is the next round's target.

### 6.4 The knobs

| Observation | Action |
|---|---|
| any knob ≥ +5% vs `p0_rebaseline` | Adopt immediately — zero build cost, zero risk |
| `udma` ≥ +5% | Fold into every future build **and** the 4B work — v81-and-above feature, never enabled before |
| `socmodel72` ≥ +5% | Adopt as the default `soc_model` for all builds including the ViT and 4B tower |
| `dlbc` / `wpack` flat | Expected — both were byte-identical offline. Record as tried; do not re-test |

### 6.5 The variance question

| Observation | Action |
|---|---|
| spread tracks temperature | Thermal. Every future protocol needs mandatory cool-down, and 2026-08-15's numbers need re-reading |
| spread does **not** track temperature | More serious — something in the 2-graph layout or the scheduler is non-deterministic. Escalate: it bounds every measurement we can take |
| spread ≤ 5% over 8 reps | Transient. Note it, keep 5 reps anyway |

---

## 7. Known gotchas

| Symptom | Cause / action |
|---|---|
| exit **139** (SIGSEGV) on a lade dialog | `type: "lade"` + `max-num-tokens` is fatal. No bundle here ships a lade dialog; if you add one, drop `max-num-tokens` |
| exit 139 on a 1-token prompt under lade | `rand() % 0` in qualla's warmup — unconditional bug. Lade prompts must tokenize to ≥ 2 tokens |
| error **14001** | Missing one of the 7 `.so` files, malformed `htp_backend_ext_config.json`, graph-switching without `use-mmap`, or a `graph_names` mismatch. Not "library not found" |
| error **5005** | `vtcm_mb > 16` under unsigned PD, or multi-core. Genie creates a single-core device; there is no JSON override |
| error **0xc26** | Quantized `embed_tokens` (Gather has no INT8/16 path), or an unsupported dtype+kernel combination |
| `Context Size was exceeded`, looping output | Greedy with no `max-num-tokens`, or a wrong chat template. Use the shipped prompts |
| `Unknown Key` warnings in §4.4 | **Expected there** — see §4.4 item 4 |
| Output is `!!!!!` | All-position-logits contract violated. Stop; this is a build defect |
| tok/s looks great but the bundle is 1.3 GB | It is a **blended** bertcache bin (§2). The number is not a decode rate |

---

## 8. Provenance — how these numbers were produced

Device-free, on QAIRT 2.48.40.260702, gated before shipping:

- **Topology**: every bundle passed `lint_bundle_topology.py --require-pure`
- **Numerics**: `parity_ladekv_read.py` **6/6** on the two rebuilt towers,
  including two chunked prompts (n=129, n=200), replaying qualla's exact feed
  pattern
- **Head dtype** (`qh`): `sFxp_8` verified on **all three** graphs against a
  `Float_16` baseline
- **Byte accounting**: `read_total_bytes` read from each build's own
  `qnn-context-binary-generator` DDR summary, not estimated
- **Knob consumption**: each variant's ctx-bin diffed against a control built
  through the identical config path
- **Weight sharing**: ~1.08 GB per bin (an unshared build is ~1.5 GB); spill 0
  on every graph

Baseline constants used throughout: decode `read_total_bytes` **961,130,496**,
residual DSP cycles **88,225,159** (2026-08-13 category table minus the 261.8M
replication ops), step time **22.37 ms**.

## 9. Still outstanding from 2026-08-14

1. **Test 2 raw artifacts** — `test2_decode_profile/qnn_profile_r{1,2,3}_viewer.txt`.
   Never received, so there is no per-op baseline to diff §4.4 against.
2. **The 7.79 tok/s build** — binary, converter command lines, build-time HTP
   config, `qnn-context-binary-utility` dump, dialog JSON. Newly interesting:
   correctly stated our pre-fix AR-1 rate is 6.84, so that build may genuinely be
   faster. Also: is it topologically pure? A PPR of 7.45 tok/s on a 6-token
   prompt is decode-speed prefill and does not look like anything we build.
3. **`/data/local/tmp/results/`** — pull before anything else (§3.2).
