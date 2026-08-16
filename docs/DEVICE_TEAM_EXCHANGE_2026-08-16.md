# Device-team exchange — 2026-08-16

**To:** SA8797P device team
**From:** llm-deploy build side (device-free)
**Re:** the GQA fix landed at 6.54×; a correction that withdraws part of the
2026-08-13 report; why six of the eight bundles we sent you cannot be measured
as shipped; and kit v2.

First: **thank you for the 2026-08-15 run.** The A/B you chose — post-fix
`gqafix_ladekv` basic against the *pre-fix* `ladekv` basic — is exactly the
right control, and it is the measurement that made everything below findable.
Had you compared against the `local` bundle instead, as the runsheet's own
priority-2 arm invited, none of this would have surfaced.

---

## 1. The result

| Arm (one binary) | tok/s |
|---|---:|
| `gqafix_ladekv` basic | **44.707 ± 0.030** (22.37 ms/step) |
| `gqafix_ladekv` LADE | 31.342 (acceptance 1.61 tok/iter) |
| pre-fix `ladekv` basic — your P3 control | 6.836 (146.3 ms/step) |

**6.54×.** Basic is the ship configuration. We agree with your recommendations 1
and 2 and have parked LADE.

One refinement on *why* LADE lost, because it changes what would revive it. The
fix sped decode 6.54× but `verify32` only 3.50× (180 → 51.4 ms), since
replication cost is AR-independent and so dominated the AR-1 graph far more. So
speculation's break-even moved from ~1.2 to **2.30 accepted tokens per call**,
against ~1.6 measured. LADE did not get worse — decode got better faster. A
learned draft head would need **+43% acceptance just to reach parity**.

## 2. Correction: 11.72 tok/s was never a decode rate, and three findings go with it

This is the important part of this message.

`qwen3_06b_w8a16_local` is a **bertcache** bundle — its prefill graph has
AR = CL = 128. In that topology Genie keeps generating *through the prefill
graph*, re-processing the whole 128-wide window once per token, until the KV
cache passes 128; only then does the AR-1 decode graph take over
(`kvmanager.cpp:421-429`). So your Test 1 Arm A's 128 generated tokens did not
all run at one rate:

```
prompt 56 tokens, 128 generated, 10,837 ms
  tokens 1..72   (KV 56 -> 128) @ 40.1 ms  = 2,887 ms   <-- 40.1 ms is that run's own TTFT
  tokens 73..128 (56 tokens)               = 7,950 ms
                                7,950 / 56 =  142.0 ms per AR-1 step
```

**142.0 ms against the 146.3 ms you measured on 2026-08-15** for the pre-fix
`ladekv` bin — a 3% agreement between two numbers taken on different days, on
different bundles, by different means. The blend model closes.

Consequences, all on our side of the ledger, not yours:

- **"LADE is −22% on the technical prompt" (Test 1) is withdrawn.** It compared
  LADE on `ladekv` against blended basic on `local`. Like-for-like on one bin,
  pre-fix LADE was 9.18 vs 6.84 = **+34%**. LADE's loss is real but only
  post-fix, and only measurable from your 08-15 data.
- **"~75% build gap between `local` and `ladekv`" (§6.3) does not exist.** The
  two decode graphs are structurally identical and share weights within 4 MB.
  Please disregard recommendation 2; there is nothing to investigate.
- **"Our builds are +51% faster than yours" (Test 3) is withdrawn.** Correctly
  stated our pre-fix AR-1 rate is 6.84, i.e. ~12% *slower* than your 7.79. We
  have reopened this as an open question rather than re-inverting it — see §5.
- **Your Test 2, the op-level cycle profile, is unaffected and remains the most
  valuable measurement anyone has taken on this model.** It ran under
  `qnn-net-run` on a single-graph decode-only bin, so no graph selection was
  involved.

## 3. Six of the eight bundles we sent you cannot answer the question they were sent for

`gqafix_local`, `gqafix_qh`, `gqafix_cl512`, `gqafix_dlbc`, `gqafix_udma` and
`gqafix_hybrid` all carry the CL=128 bertcache prefill — they are the ~1.32 GB
ones. Their basic-mode tok/s is therefore blended and **not comparable to
44.707**, and the blend runs *fast*, so the failure mode is a confident wrong
number rather than an obviously broken one.

**That is the whole of priorities 4 and 5 in `kit/runsheet.md`. Please do not
run them.** Your decision to skip P4 as "unnecessary" happened to avoid a
measurement trap we had built for you, and we are sorry for it.

Only `gqafix_ladekv` and `gqafix_pastkv2g` were pure — and note that
`pastkv2g`, the one variant arm you did run, is also the one whose best rep
(44.54) lands on the baseline exactly.

We have rebuilt every variant on the pure 3-graph past-KV topology for kit v2,
and added a build-side gate (`lint_bundle_topology.py`) so this class cannot
recur.

## 4. P1: our stated reason for the failure was wrong

Your §5 records that the decode-only cycle profile could not run because the
shipped inputs were "pre-fix format with 128-dim KV and 128-byte
`position_ids`". We accepted that and wrote it into our own notes. It is not
possible:

- The pre-fix and post-fix decode-only ctx-bins have **byte-identical input
  contracts** — 60 inputs, same names, shapes and dtypes. The GQA fix is
  graph-internal and we froze KV I/O by design, so one input set feeds both.
- The shipped `ar1_decode/inputs` scores **60/60** against the gqafix bin.
  Including the file singled out: `position_ids_cos` is 128 bytes because it is
  `[1,1,64]` fp16 — 128 B is *correct*, and "64-dim" describes `position_ids`,
  not the KV cache (`[1,8,128,1151]` in both bins).

So the real cause is still unknown, and regenerating the inputs would have
fixed nothing. Kit v2 ships `verify_profile_inputs.py`, which checks the package
against the bin's own declared input list and names the exact mismatch — or
confirms there is none and points at four environmental candidates
(`--retrieve_context` path resolution, `ADSP_LIBRARY_PATH`, a silently truncated
extract on a `/data` that runs 98–99% full, and the two **expected** `Unknown
Key` warnings being read as the error).

**Whatever P1 fails on next time, please report which of those four it was.**
That is more useful to us than the profile itself at this point.

## 5. What kit v2 is actually deciding, and what we need

Post-fix, two models fit your 22.37 ms/step **exactly**, and we cannot tell them
apart:

| | fit |
|---|---|
| Byte-bound: 961 MB ÷ 22.37 ms = 43.0 GB/s | 88% of your own 49 GB/s excl-wait ceiling |
| Compute-bound: 88.2M residual cycles ÷ 4 HVX @ ~1 GHz = 22.06 ms | 1.4% |

They imply completely different next years of work. They also predict **opposite
orderings** for two cheap arms, which is what kit v2's priority 4 exploits:

| Arm | byte model | compute model |
|---|---:|---:|
| W8 `lm_head` | **+17.9%** | +3.6% |
| `cl512` (`context.size` 512) | +10.1% | **+26.0%** |

Plus a free falsification test: `hvx_threads: 8` changes **zero DDR bytes** by
construction, so the byte model predicts exactly 0.0%. (We confirmed the
build-time knob is genuinely consumed — the binary is 2,277,376 bytes larger.
Your Test 5 correctly showed the *runtime* knob is inert; they are different
things.)

**Protocol change, and it is not optional.** Your `pastkv2g` spread — 23.43 /
44.54 / 29.34 tok/s on one binary — is larger than every effect we are chasing.
Kit v2 uses **5 reps, median reported, every raw value kept**, a 30 s cool-down
between arms, and temperature logged either side. Priority 2 spends 8 reps
isolating that variance, because it currently bounds the resolution of every
measurement either of us can take.

### Still outstanding from 2026-08-14

1. **The Test 2 raw artifacts** — `test2_decode_profile/qnn_profile_r{1,2,3}_viewer.txt`.
   We never received them, so we still have no per-op baseline to diff the new
   profile against, only your narrative table.
2. **Your 7.79 tok/s build** — binary, converter command lines, build-time HTP
   config, `qnn-context-binary-utility` dump, dialog JSON. §2 makes this
   interesting again: our honest pre-fix AR-1 is 6.84, so your build may
   genuinely be faster. We would also like to know whether *your* bundle is
   topologically pure, since your reported PPR of 7.45 tok/s on a 6-token prompt
   is decode-speed prefill and does not look like anything we build.
3. **Pull `/data/local/tmp/results/` off the device before anything else.** Your
   `/data` runs 98–99% full and the 08-15 per-op record is one cleanup away from
   being gone permanently.

## 6. If you think §2 is wrong

Say so before running kit v2. That correction is new, it is doing a great deal
of work in the session's design, and it rests on a graph-selection rule we read
out of the SDK sources plus one arithmetic reconciliation. We would rather
re-argue it now than build another session on it.
