# Test O — the variable matrix, and the session designed to close it

**Status:** ready to run · **Opened:** 2026-08-22 · **Needs:** no rebuild — the
kit is configs, two token files already on the board, and one runner script.
**Board time ≈ 45 min, +25 min if a knob passes (then the session ENDS the
project).** Execution detail: **`DEVICE_SESSION_PROTOCOL_O.md`**.

**Intent: this is the final exploratory device session for Qwen3-VL.** Every
branch of the decision tree below ends in one of exactly two places: the model
works end-to-end (Stage O6, same session), or the defect is pinned to a named
Genie-internal mechanism with a complete, self-contained evidence package for
Qualcomm. Either way, no further *exploratory* sessions — at most one
confirmation run after an external fix.

---

## 1. The variable inventory — every candidate, with its status

The defect, precisely: on the real 4B, **prefill produces the correct token for
any content** (Test N, 6/6, two prompts × four lengths, incl. EOS) and **the
first decode step is wrong, content-independently**. The decode *graphs* are
correct with host-supplied inputs, recurrence included (Test J A1). So the
fault is in what Genie does between prefill and the decode graph's output.

### 1a. Eliminated — with the evidence, so nobody re-tests these

| # | variable | eliminated by |
|---|---|---|
| E1 | model weights / quantization / encodings | host parity 20/20 vs HF; `host_generate_check` coherent; Test J graphs 3/3 on device |
| E2 | graph topology (GQA, all-position logits, past-KV prefill) | lint gates 0 replication ops; Test J/G on device |
| E3 | `inputs_embeds` dtype (the fp16 pad bug) | shipping bin is `uFxp_16`, gate PASS; N1a works through prefill |
| E4 | prompt form (raw vs templated) | Test I, confirmed twice |
| E5 | tokenizer | N1a used `-tok` — bypassed entirely, counts verified |
| E6 | prefill-time LUT lookup | N1b (`-e`) — same behaviour with the LUT bypassed |
| E7 | prefill→decode KV **width relay** (2048→2175) as sole cause | the 0.6B single-bin makes the same relay (1024→1151) and works; and N1a P21 shows a prefill containing the generated token continues correctly |
| E8 | the ctx-bin **split at prefill** | N1a: prefill crosses `prefill_0 → prefill_1` correctly at 6 lengths |
| E9 | prompt content / length / specific token ids | N1a: content-independent by construction |
| E10 | sampler randomness | greedy, temp 0, top-k 1, seed fixed; deterministic wrong answer |
| E11 | libGenie / SDK version, board build | constants across every session; the 0.6B works on the same stack |
| E12 | `context.size` convention | matches the working 0.6B's convention (size = CTX, graphs = CTX+CL) |

### 1b. Live — each one named, with the stage that tests it and the signature that convicts it

The decode step consumes five things and produces one. Each is a variable:

| # | variable | what could be wrong | tested by | convicting signature |
|---|---|---|---|---|
| **L1** | **embedding of the generated token** (decode-time LUT read) | wrong row / stride / offset at decode | **O1** KV diff | position-N **V differs** (V = v_proj(x): no rope, so V wrong ⇒ the input x was wrong) |
| **L2** | **position / rope advance** | off-by-one (e.g. using `n-prompt`=21 instead of 20) | **O1** KV diff | position-N **V identical, K differs** (only K carries rope) |
| **L3** | **attention mask** for the decode step | wrong valid span / slot | **O1** by elimination | K **and** V identical, token still wrong — mask affects the *output*, not the new token's KV |
| **L4** | **old-KV integrity** after Genie's internal relay | relay corrupts committed positions | **O1** host-ref compare | dump positions 0..19 vs the host-built cache for the same prompt (the Test J kit's own reference) |
| **L5** | **the split handoff at decode** (`decode_0 →[1,1,2560]→ decode_1`) | variant/buffer selection between bins | **O1** elimination + **O2** knobs | K, V identical through **both shards** (shard-1 V at pos N = v_proj of the handoff tensor — if it matches, the handoff bytes were right) |
| **L6** | **logits buffer read** | reading the wrong row/variant of the output buffer (the exact class of correction #1) | **O1** full elimination | *everything* KV-side identical and the token still wrong ⇒ the fault is after the graphs — logits read or sampler feed |
| **L7** | **engine knobs** — `enable-graph-switching`, `poll`, `allow-async-init`, `mmap-budget`, `use-mmap` | any of five untested values changes decode's buffer/graph routing on a 2-bin model | **O2** sweep, one knob per run | a run whose output is `4` then STOP |
| **L8** | Test M's proxy result (0.6B split, failed at token 0) | confounded by 4 knobs vs the 4B | **O3** | with the 4B's block: reproduces the 4B signature ⇒ real reproduction; clean ⇒ the knobs were Test M's whole failure |

The power of O1 is that **one measurement reads out L1–L6 at once**: the KV a
decode step writes for token X is a pure function of its embedding, position and
the prior cache — and we hold a prefill-written copy of the *same* token at the
*same* position to compare against, plus the host-built cache as absolute
reference. Bytes cannot lie about which input was wrong.

---

## 2. The stages

### O1 — the KV dumps (~10 min device + host analysis) — THE READOUT

Four `--save` runs with the `o1_save` config (`max-num-tokens: 1`, no hand
edits), two pairs:

| dump | prompt | positions | position of interest |
|---|---|---:|---|
| `state_p20` | 2+2, 20 tok | 21 | **20** = KV of `4`, **decode-written** |
| `state_p21` | 2+2 + `4`, 21 tok | 22 | **20** = KV of `4`, **prefill-written** |
| `state_w18` | weather, 18 tok | 19 | **18** = KV of `Mountain`, **decode-written** |
| `state_w19` | weather + `Mountain`, 19 tok | 20 | **18** = same, **prefill-written** |

Two independent pairs → the signature is measured twice. Transfer the four
`kv-cache.primary.qnn-htp` files (~3 MB each);
`parse_genie_kv_dump.py --diff` prints the per-position K/V split and fp16
magnitudes. The format is fully decoded (592 B header + 72 tensors × n_pos ×
2048 B — correction #40), so analysis is immediate.

⚠ Interpretation note: prefill and decode may legitimately differ by
quantization jitter (~1e-3 in fp16). A *wrong input* produces O(1) differences
across whole tensors. The parser prints max|Δ| so the two are unmistakable.

### O2 — the 4B engine-knob sweep (~15 min) — THE ONLY BRANCH THAT FIXES IT TODAY

Seven pre-built configs (`testo/configs/`), **one knob per run** plus a control
and an all-at-once, each run = the same 20 exact token ids, verdict printed by
`run_o2_sweep.sh`:

| run | change vs shipping |
|---|---|
| `o2a_ctrl` | none — **must FAIL**, or the sweep is uninterpretable |
| `o2b_gswitch` | `enable-graph-switching: true` — **the prime candidate**: it is the machinery that routes execution between ctx-bins, it is on in every working 0.6B config and absent in the 4B's |
| `o2c_poll` | `poll: true` |
| `o2d_async` | `allow-async-init: true` |
| `o2e_mmapb` | `mmap-budget: 25` |
| `o2f_nommap` | `use-mmap: false` |
| `o2g_all` | all of the working 0.6B's knobs at once |

PASS = `4` then stop. Any PASS → confirm on the weather ladder → **go directly
to O6.**

### O3 — Test M, deconfounded (~5 min)

Two pre-built lutsplit configs: `o3a_4bknobs` (the 4B's exact block) and
`o3b_nogswitch` (only graph-switching removed). Same two prompts as Test M,
scored against the same HF strings.

| o3a result | meaning |
|---|---|
| first token right, then garbage | **the 4B's signature reproduced at 0.6B** — the split is convicted, and the bisect moves to the host |
| still wrong from token 0 | Test M's failure is knob-independent — a real prefill-path problem in the 2-bin 0.6B, different from the 4B's defect; park it, it is not on the 4B's critical path |
| fully correct | Test M's failure was **entirely the knobs** — the split is clean at 0.6B, and L5 lives or dies on O1's readout |

### O4 — restore probes (~5 min, opportunistic)

`--restore` each saved state and let generation continue one step
(`o2a_ctrl` config): restoring `state_w19` should continue with ` weather`
(9104). Also: **keep all four state directories** — if O1's diff convicts the
KV write, the build side will return a *patched* state (prefill-written bytes
spliced into the decode dump) to restore and continue, which converts the
diagnosis into a demonstrated fix-point. The dump format has no checksum field
(`dialog.json` carries only counters), so this is expected to work.

### O5 — evidence completion (~5 min)

Only what the escalation package still lacks: `adb logcat -d > o5_logcat.txt`
after a failing run, and one failing run with maximum Genie verbosity if any
env knob exists (capture the attempt verbatim even if it does nothing).

### O6 — THE END-TO-END RUN (~25 min, only if O2 found a passing knob)

`RUNBOOK_e2e_qwen3vl_4b.md` with the passing config: templated 2+2 → `4` stop;
sample-image caption; six photograph captions; timing. **Four passes = the
project's goal is met, this session.**

---

## 3. The decision tree — every leaf is terminal

```
O2 sweep: any PASS?
├─ YES → confirm, run O6 → ✅ DONE (deliver: config change, no rebuild)
└─ NO → O1 diff, position-of-interest:
    ├─ V differs             → L1: decode-time embedding read is wrong
    ├─ V same, K differs     → L2: decode position/rope is wrong (off-by-one class)
    ├─ K,V same, shard-1 V differs → L5: the decode handoff corrupts the boundary tensor
    ├─ all same, host-ref mismatch at 0..N-1 → L4: the internal KV relay corrupts old positions
    └─ all same, host-ref clean → L3/L6: mask or logits-buffer read — Genie-internal,
                                   not observable from outside
    → in every leaf: the named mechanism + both dump pairs + the O2 matrix +
      O3's resolution = the ESCALATION PACKAGE. File it with Qualcomm;
      one confirmation session after their fix.
```

No leaf loops back to "run another exploratory session". That is the design
goal of this plan.

## 4. What to have ready before the session

* the four N-session artefacts already on the board (`testn/` token files)
* `testo/` pushed (10 configs + `run_o2_sweep.sh`)
* the Test M bundle still on the board for O3
* ~20 MB free for the four state dirs

Full command sequences, capture lists and the results template:
**`DEVICE_SESSION_PROTOCOL_O.md`**.
