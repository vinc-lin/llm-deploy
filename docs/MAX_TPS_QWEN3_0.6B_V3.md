# MAX TPS Qwen3-0.6B on SA8797P — Plan V3 (device-free execution)

**Date:** 2026-08-14 · **Supersedes:** `MAX_TPS_QWEN3_0.6B_V2.md` **§3–§5 (sequencing only)**.
V2 §0–§2 (measured baselines, revised performance model, ctx-bin forensics) remain the
analytical basis and are not restated here.
**Constraint driving this revision:** convenient on-device testing is not available. All
measurement goes through the device team, asynchronously, via HF bundle deployment.

---

## 0. The inversion principle

V2 sequences *measure → decide → build*: A1 gates C, B7 gates F/E4/doc-edits, A6 gates D2.
Under the no-device constraint every one of those gates is a stall of unknown length.

V3 inverts it: **local compute and disk are cheap; device-hours are the scarce resource.
Build every branch of the decision tree locally, validate with local gates, and compress all
measurement into one pre-scripted session kit that cannot require a second round-trip.**
A wasted build costs disk. A wasted device session costs weeks.

Consequences:
- Builds that V2 made contingent (C1, C2's clean qh bin, F variants) are built unconditionally.
- Where V2 defers an implementation choice to iteration (B3 rank-3 vs rank-5), V3 builds both.
- The plan's deliverable is not a measurement — it is **everything on HF, ready to run**,
  plus a decision table that maps every possible session outcome to a pre-committed action.

---

## 1. Phase 0 — Ship-safety and the async channel (~half day)

V2 P0 unchanged, with one promotion:

| # | Task | Change vs V2 |
|---|---|---|
| P0.1 | Fix demo dialog SIGSEGV (drop `max-num-tokens` from lade demo), re-upload configs via single-file `HfApi().upload_file` commits, add bundle-gate check | unchanged — only user-visible fix available today; every shipped demo currently exits 139 |
| P0.2 | Redact report (device serial, jump-host access string) before any commit | unchanged |
| P0.3 | Commit report + `docs/test_artifacts/measurement_2026-08-13/` | unchanged — **now also feeds B1 (§2)** |
| P0.4 | Device-team exchange | **promoted to critical path.** The device team is the only measurement channel. `docs/DEVICE_MEASUREMENT_REQUEST_2026-08-13.md` shows the async-request pattern works; Phase 5 hands them a self-running kit. Open the thread now: send our 11.72, request their 7.79-run artifacts, and announce the coming kit |
| P0.5 | Doc corrections queue — still gated on B7 confirmation | unchanged (B7 may now arrive via simulator, §2) |
| P0.6 | Supplier-analysis publication gate | unchanged |

**Exit gate:** fixed configs live on HF; device team thread open.

---

## 2. Phase 1 — Kill uncertainty with paper, not silicon (1–2 desk days)

Extract everything the existing artifacts can still tell us before building anything.

| # | Item | Decides | Protocol |
|---|---|---|---|
| V3-1.1 | **B1** — grep 08-13 profiler artifacts for top-op output dims | Confirms/refutes GQA-replication hypothesis pre-build | expect `[1,8,2,*,*]`, not `[1,8,1,1152]` |
| V3-1.2 | **B2** — converter recon | Flag-only fix vs export rewrite | 2.48 IR optimizer passes, masked-softmax-style matchers |
| V3-1.3 | **Simulator recon** (new) | Whether B7's cycle experiment can move on-desk entirely | Run today's shipped `decode.dlc`/ctx-bin under x86 HTP simulation (`qnn-net-run`, detailed profiling, v81 target). **Acceptance: aggregate ≈ 350.3M cycles within tolerance AND the 56-op replication signature reproduced.** Pass → B7a is a desk experiment. Fail → drop it; no partial credit for a proxy that can't reproduce a known baseline |
| V3-1.4 | E2 config-key audit | `sparse_weights_compression` / `extended_udma` consumed or ignored | info.json + SDK docs |
| V3-1.5 | D4 desk read | CPU-draft (`QnnGenAiTransformer` GGUF) as spec-decode secondary — plausible or dead | `Genie/tutorials/dialog/*/kvshare/` + spd docs, 1 h, no build |

The simulator recon is the single highest-leverage new item in V3: it is the only path that
could turn "the single decisive experiment" (V2 §5) into a desk result. Worth one day before
committing to the Phase-3 build fan-out — but do not let it block Phase 2, which is justified
by B1 alone.

**Exit gate:** GQA hypothesis confirmed from raw artifacts (V3-1.1). If refuted → stop, re-plan;
everything downstream assumes it.

---

## 3. Phase 2 — Workstream B in full (the trunk)

B3→B6 are device-free by design (V2 §3-B). V3 modifications:

| # | Step | V3 delta |
|---|---|---|
| V3-2.1 | B3 export rewrite | **Build both formulations**: rank-3 grouped batched MatMul (`[8, 2·AR, 128] @ [8,128,CL]`) AND rank-5 broadcast MatMul (`[1,8,2,AR,128] @ [1,8,1,128,CL]`) if the converter accepts it. V2 treats rank-5 as an iterate-later fallback; with no device to iterate against, ship both to the session and let it pick. HF Q-head ordering (contiguous per KV head) preserved in both |
| V3-2.2 | B4 encodings | unchanged — weight names untouched, adopt-encodings path, check for orphaned Expand-output activation encodings |
| V3-2.3 | B5 local gates | unchanged suite (ONNX parity vs HF argmax all prompts → `quantize_aimet.py --eval` ≥3/4 → `parity_ladekv_read.py` → `qnn-context-binary-utility` graph-name check) **plus one new hard gate: `qairt-dlc-info` op-count diff proving all 56 Expand+Reshape pairs are gone.** The gate suite is the entire safety net now — nothing may be skipped, because no device run will catch what slips through. Read `docs/NOTES-genie-io.md` first (topology change ⇒ mandatory) |
| V3-2.4 | B6 build | 3-graph ladekv + 2-graph basic per formulation. Same lineage rule (`--export-decode` / `--adopt-encodings`); KV I/O shapes frozen (`past_key_i [1,8,128,1151]`, `past_value_i [1,8,1151,128]`) |

**Exit gate:** all B5 gates green on at least one formulation, both if both convert.

---

## 4. Phase 3 — Pre-build every contingency leaf

The departure from V2: A1's answer is unknown and unknowable until the session, and §1.4's
bounding question likewise — so build **both** branches of each.

| # | Leaf | Covers | Notes |
|---|---|---|---|
| V3-3.1 | Post-B 3-graph ladekv + 2-graph basic | trunk (from Phase 2) | suffix `_gqafix_ladekv` / `_gqafix_local` |
| V3-3.2 | **Post-B W8-head 2-graph bin** | byte-bound branch — head is 311 of 751 MB (41% of stream) | `--quant-head --keep-head-weight`; verify `qairt-dlc-info \| grep lm_head.weight` → `sFxp_8`. Own encodings lineage (head encoding differs) — the cross-graph rule requires per-bundle internal consistency, which this satisfies. Basic-mode candidate only; the −14% LADE acceptance regression stands |
| V3-3.3 | **CL=512 decode/verify variants** (F) | byte-bound branch — KV read 132 → 59 MB | new conversions, same post-B lineage |
| V3-3.4 | C1's 2-graph past-KV bin (CL=1152 prefill + decode) | "A1 slow" branch — isolates graph-count from switching | |
| V3-3.5 | C3 hybrid prefill bin (bertcache CL=128 + past-KV CL=1152 + decode) | TTFT product win (186 → ~40 ms), independent of the decode question | **basic-mode only, never lade** (AR==CL rule) |
| V3-3.6 | E1 `"dlbc": 1` variant | free ctx-bin rebuild | plain DLBC only; `dlbc_weights` incompatible with weight sharing |

Build discipline (this phase is exactly the workload class that produced the three 08-12 vhdx
crashes):
- `disk_guard` sized per step before every multi-GB operation; delete intermediates between
  builds; `du -h` the vhdx, not `ls`.
- **Graph-name trap:** convert straight to final filenames — never rename a DLC after
  conversion. `qnn-context-binary-utility --json_file` check on every bin before bundling.
- Never ship two graphs with the same (AR, CL) in one bin; no AR==CL graph in any lade bundle.

**Exit gate:** every leaf passes the full B5 gate suite locally.

---

## 5. Phase 4 — The decisive session kit (runnable without us)

A self-contained package the device team executes asynchronously. Contents:

1. **Bundles** — everything from Phase 3 (uploaded in Phase 5).
2. **`run_all.sh`** — one arm per numbered run: warm, 3 reps, greedy, both the 56-token
   technical prompt and one simple prompt (plus one repetitive-structured prompt for the
   LADE arms — the A6-lite acceptance map).
3. **`runsheet.md`** — priority-ordered so a truncated session still decides the big questions:

| Priority | Run | Decides |
|---|---|---|
| 1 | **B7a** — decode-only post-B bin, `qnn-net-run --profiling_level detailed` | 350M → ~90M cycles? **Skip if V3-1.3 simulator proxy validated** |
| 2 | **B7b** — post-B tok/s: `_gqafix_local` basic, `_gqafix_ladekv` basic + LADE | §1.4 bounding model + the headline number; the LADE arm doubles as D1 |
| 3 | **A1** — basic on plain (pre-B) `qwen3_06b_w8a16_ladekv` | splits qh regression from 3-graph penalty (V2 §2.2), retroactively |
| 4 | W8-head bin (V3-3.2), CL=512 bin (V3-3.3) | picks the ship configuration if byte-bound |
| 5 | A2 (spill-fill 0→640 MB runtime edit), A3 fused, A5 hvx8, E1 dlbc, A7 switching toggle | trimming |
| 6 | C3 hybrid prefill TTFT measurement | product metric, not decode TPS |

4. **`decision_table.md`** — every outcome maps to a pre-committed action; nobody improvises
   on-device and no result needs a second round-trip:

| Observation | Pre-committed action |
|---|---|
| B7b basic ≥ 15 tok/s | §1.4 compute-bound confirmed; `_gqafix` is the ship base; compare priority-4 arms to pick final config |
| B7b basic ≈ 11.7–12.5 | hidden byte floor partially real; W8-head + CL=512 arms decide where the bytes are; F becomes the lead workstream |
| A1 ≈ 11.7 | "75% build gap" was the qh head; close C; C1 bin (V3-3.4) shelved unrun |
| A1 ≈ 6.7 | real switching/graph-count penalty; run A7 toggle + C1 bin in the same session |
| W8-head ≥ neutral vs `_gqafix_local` | adopt for basic-mode products (LADE keeps FP16 head) |
| CL=512 ≥ +8% | add F products; consider CL=768 in a later build |
| post-B LADE multiplier < 1.2 on technical prompts | basic is the default ship; LADE reserved for favorable-workload products; D2/D3 stay parked |

5. **Expected-output references** — argmax continuations per prompt per bundle (from
   `parity_ladekv_read.py` runs) so the team can flag quality regressions without judgment calls.

**Kit hygiene:** the kit's docs go to the same HF repo — P0.2's redaction class applies
(no serials, no ssh strings, no terminal chrome in any kit file).

**Exit gate:** kit dry-run locally (scripts parse, paths resolve, prompts load).

---

## 6. Phase 5 — Upload to Hugging Face (final step: the deployment handoff)

Deployment to the device goes exclusively through HF — the plan ends when everything is on
the hub and the device team is pointed at it.

| # | Step | Detail |
|---|---|---|
| V3-5.1 | Pre-flight | `HfApi().repo_info("vinccniv/sa8797p-qwen3-w8a16-bundles").private` — **read visibility live, record it. Do not change it.** If the upload flips it as a side effect: report and stop (four prior incidents) |
| V3-5.2 | Manifest | one tarball per bundle (V3-3.1…3.6, both B3 formulations), the decode-only profiling bin + HTP config for B7a, and the kit (`run_all.sh`, `runsheet.md`, `decision_table.md`, prompts, expected outputs) |
| V3-5.3 | Upload | `scripts/util/hf_upload_watchdog.sh` with `SOCKET_CHECKS=999999` (proxy `http://127.0.0.1:17890` drops long streams; the socket detector false-positives through it). Budget commits: well under 128/hour — tarballs keep the count low. "Hung" commit with blobs pre-uploaded = 429; recover with spaced single-file commits after ~1 h |
| V3-5.4 | Verify | `HfApi().list_repo_files` against the manifest; re-check visibility unchanged; spot-download one tarball and checksum |
| V3-5.5 | Handoff | send the device team (P0.4 thread) the repo path, `runsheet.md` link, and the decision table. **This message is the plan's terminal action** — everything after it is their session and our pre-committed responses |

---

## 7. Refined ceiling (records the V3 analysis; V2 §5 ladder otherwise unchanged)

V2's ~21 tok/s "hard ceiling" (751 MB @ 16 GB/s) zeroes four terms and is not reachable:

1. **KV read is irremovable:** the 132 MB/step un-replicated cache read survives B (B removes
   only the 264+264 MB replication write/re-read). Real floor is **883 MB → ~18.1 tok/s** at
   the same bandwidth.
2. **16 GB/s is biased by the removed traffic:** measured on a mix dominated by the friendly
   sequential replication copy; the surviving mix (INT8 weight tiles + strided KV gather) will
   run slower.
3. **~22 ms of post-B compute must overlap 47 ms of streaming:** the ceiling assumes 100%
   overlap; GEMMs wait on tiles, GEMV waits on KV, `lm_head` is a serial tail.
4. **A 64 ms unexplained term exists:** 3-graph decode = 149 ms with identical weights; the
   ceiling model has no slot for it (A1/A7/C1 diagnose it).

Restoring these terms lands at 13–18 — V2's central projection **is already ceiling-minus-
these**. The LADE ~35 is weaker still: post-B, verify32's cost regains AR-scaling, so the
1.68× multiplier should fall. What raises the bound is not more B — it is the two byte terms
B cannot reach: **F (CL=512: 132→59 MB) and the W8 head (751→595 MB)** — together ~654 MB
⇒ ~24 tok/s of headroom. Hence both are unconditional Phase-3 builds.

## 8. What we explicitly do NOT do

- **D2/D3 LADE sweeps** — pure device-side tuning, zero local signal, unbounded session time.
  Parked until a post-B base exists and the decision table triggers them.
- **E4 and further E micro-levers** — contingent on measurements we cannot take; E1 (free
  rebuild) is already in Phase 3.
- **Edit REFERENCE.md / the report's numbers** — P0.5's gate stands: not before B7 (device or
  validated simulator) confirms.

## 9. Risk register (delta from V2 §6)

| Risk | Mitigation |
|---|---|
| Simulator proxy passes acceptance but diverges from device on the *post-B* graph | treat simulator B7a as provisional; B7b (tok/s, priority 2) independently decides §1.4 on device |
| Session runs without us and hits an unscripted state | decision table covers all branches; runsheet orders by priority; kit dry-run gate |
| Phase-3 fan-out exhausts disk (vhdx → C: → SIGBUS VM crash) | `disk_guard` per step; delete intermediates between builds; `du -h` the vhdx |
| Local gates pass but device output is garbage (the class B5 can't catch) | expected-output references in the kit make quality regressions visible to the device team without our judgment |
| Upload flips repo visibility | read live pre/post (V3-5.1/5.4); report and stop, never "restore" from memory |
| 429 on commit phase | tarball-level commits; watchdog; spaced single-file recovery after ~1 h |
| Kit leaks identifiers into a repo scrubbed twice | P0.2 redaction class applied to every kit file before upload (Phase-4 hygiene gate) |

## 10. Open questions ledger (carried from V2 §7, plus)

1–7. unchanged from V2.
8. Does x86 HTP simulation reproduce the 350.3M-cycle decode baseline? (V3-1.3 — if yes, B7a
   is a desk experiment and the projection ladder de-risks without any device time)
9. Post-B, where do the bytes actually bind — weights, KV, or the unexplained 64 ms term?
   (decision table rows 1–2/4 resolve this in one session)
