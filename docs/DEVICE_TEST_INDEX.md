# Device test index

Every device session in one table: what it asked, what it settled, where its
artefacts are, and — the part that costs a session when it is wrong — **which
bundle is current and which is superseded**.

*Updated 2026-08-22. When a session finishes, add its row and move the previous
"NEXT" marker.*

---

## Start here

| | |
|---|---|
| **Next test** | **(1) send the two `state_p*.bin` KV dumps** — `parse_genie_kv_dump.py --diff` measures the decode step's KV write directly, no device time. **(2) re-run Test M with the 4B's `QnnHtp` block** — config-only; Test M's config differs in 4 knobs incl. `enable-graph-switching: true` on a 2-bin model. Prior plan: **Test N** — `TEST_N_final_e2e.md` + `DEVICE_SESSION_PROTOCOL_N.md`. The complete remaining-work session (~60 min): **N1 probes the real 4B directly** (token ladder via `-tok`, `-e`, `--save` state dump — no build, runs on the bundle already on the board), with Tests **L** and **M** folded in as stages N2/N3, the image first-word grid as N4, timing completion as N5, and the post-fix e2e confirmation as N6. Kit: `qwen3vl_4b_testn_session/` (VL repo). If time is short: **N1a P21 is one command and the sharpest single measurement available** |
| **The one open defect** | Genie's decode-step machinery on the 4B. **Device-verified content-independent** (Test N N1a: prefill correct at 6/6 rungs, 2 prompts × 4 lengths, incl. EOS) |
| **Leading suspect** | the two-ctx-bin split **at decode specifically** (`decode_0 → [1,1,2560] → decode_1`). ⚠ Not established: N1a proves the *prefill* handoff across the same split is correct on the 4B, and Test M's 0.6B failure is at token 0 — a different signature, and confounded by 4 engine knobs (correction #41) |
| **What is already proven on hardware** | image input · ViT · splice · prefill numerics · decode graphs incl. recurrence · the image path end-to-end on 7/7 images |
| **Current 4B bundle** | `qwen3vl_v5_session/03_vl4b_v5/` — shard 0 **`f031e3a7563bf16f2d5ca98a71b357f6`** |

---

## Sessions

| test | date | the question | verdict | doc · report |
|---|---|---|---|---|
| **v3 gate** | 08-17 | does the GQA-fixed VL text tower pass every host gate? | ✅ pass; 0 replication ops on all 4 DLCs | `reports/qwen3vl-v3-gate-results.md` |
| **v4** | 08-18 | does the fp32 image blob fix the SIGSEGV? | ✅ **image path works** — 3 clean runs + 6 photos | `reports/qwen3vl-4b-v4-device-test-results-2026-08-15.md` |
| **F** | 08-19 | is the 1.39× boundary gain caused by the sink CONDITION or the ROW INDEX? | CONDITION — but ⚠ later shown to be an artefact of a raw-prompt-shaped probe | `TEST_F_*` |
| **G** | 08-20 | does the boundary hold on realistic chat-templated input? | ✅ clean; **shard 0 exonerated** | `reports/qwen3vl-4b-testg-device-results-2026-08-20.md` |
| **H** | 08-20 | cross-chunk prefill, never once run on device | shipped; **folded into J as stage A2** | `TEST_H_multichunk_prefill.md` |
| **I** | 08-21 | templated vs raw prompt | ✅ **root cause confirmed** — templated first token correct, raw wrong | `reports/qwen3vl-4b-testi-device-results-2026-08-21.md` |
| **J** | 08-21 | is the decode *graph* right, incl. the recurrence? | ✅ **graphs cleared** (`j0`→151645, `j1`→9104, `j2`→4344) ⇒ the fault is **Genie's decode-step feed** | `TEST_J_decode_step.md` · `reports/…testj…md` |
| **K** | 08-21 | LUT feed vs split · image one-word · 4B timing | ✅ **image path 7/7** · ⚠ **K1 VOID** (ran a FLOAT_16 bin) · ✅ first timing | `TEST_K_lut_vs_split.md` · `reports/…testk…md` |
| **L** | **open** | does the ctx-bin reproduce its ONNX, or is it Genie? | — | `TEST_L_ctxbin_vs_genie.md` |
| **M** | **open** | does the 2-ctx-bin **split** break decode at 0.6B? | — | `TEST_M_split_reproduction.md` |
| **N** | 08-22 | the full remaining-work session: decode bisect **on the real 4B** + L + M + image grid + timing | ✅ **N1a 6/6 on the real 4B** ⇒ the defect is decode-step machinery, **content-independent**; N1b exonerates the prefill feed; N1c's KV dump is **complete raw KV** (147,456 B/pos, GQA — correction #40). ⚠ Its "split is the root cause" verdict is **not supported** — it contradicts N1a (correction #41) | `reports/qwen3vl-4b-testn-device-results-2026-08-22.md` · `TEST_N_final_e2e.md` · `DEVICE_SESSION_PROTOCOL_N.md` |

### Dead ends, so nobody re-runs them

| ✝ | why |
|---|---|
| prefill→decode KV width handoff | the shipping 0.6B makes the same 1024→1151 change and works (correction #36) |
| MRoPE decode advance | gated on `m_visionParam.size() > 0`; a text-only run is plain rope |
| shard 1 decode "never tested" | it was — Test G `r3_decodectx`, both shards, 1/1 |
| `c1` cross-chunk as the image-path cause | K2 disproves it: all 7 image prompts are 3 chunks, successes and failures alike |

---

## Which bundle is current

⚠ **Three sessions have now been damaged by running a superseded artefact.**
Check the md5 before anything else; a failed precondition is a result.

### Qwen3-VL-4B — `vinccniv/sa8797p-qwen3vl-4b-bundles`

**Cleaned 2026-08-22: 1581 → 573 files, 36.6 GB → 9.8 GB.** Eight superseded
folders removed, so what remains is what you should run. The removals are still
in the repo's git history if a past artefact is ever needed.

| folder | status |
|---|---|
| `qwen3vl_v5_session/03_vl4b_v5/` | ✅ **CURRENT.** Complete standalone pipeline bundle — verified to carry both ctx-bins, `genie-app`, the LUT, tokenizer, image blobs and segment files. Shard 0 `f031e3a7…` |
| `qwen3vl_v5_session/01_probe_06b_fp16in/`, `02_probe_06b_u16in/` | kept as **evidence for correction #39** — the two arms of the LUT-probe dtype A/B. `01_` is the FLOAT_16 arm that Test K accidentally ran |
| `qwen3vl_4b_testj_session/` | ✅ current kit — carries `testj/` **and its own copy of `testh/`** (67 files) |
| `qwen3vl_4b_testi_session/` | ✅ **kept: still live.** Its `testi/prompt_*_templated.txt` are what `DEVICE_SESSION_PROTOCOL.md` §3/§5 and the runbook feed to `genie-t2t-run` |
| `qwen3vl_4b_testk_session/` | ✅ current — the one-word `prompt_seg2_*` files |
| `qwen3vl_4b_testl_session/` | ✅ current — Test L docs (kit lives with the bundle in the 0.6B repo) |
| `qwen3vl_4b_testn_session/` | ✅ **current — the Test N package**: master plan, protocol, results template, collector, and the `testn/` kit (6 token-ladder files + the `-e` embedding blob). Merges into the v5 folder on device |
| ~~`qwen3vl_4b_e2e_pipeline`, `_v2`, `_v3`, `_v4`~~ | **deleted** — superseded by `03_vl4b_v5` |
| ~~`qwen3vl_4b_e2e_pipeline_v5`~~ | **deleted** — it shipped a stale shard 0 (`065056ba…`) and every doc that mentioned it did so to warn you off. Removing it removes the trap |
| ~~`qwen3vl_4b_testf/g/h_session`~~ | **deleted** — F and G are settled and written up in `reports/`; H ships inside `testj_session/` |

### Qwen3-0.6B — `vinccniv/sa8797p-qwen3-w8a16-bundles`

**Cleaned 2026-08-22: 108 → 94 files, 34.6 GB → 20.6 GB.** Removed: the whole
pre-GQA-fix bundle lineage (10 tarballs — their measurements live in `reports/`
and `REFERENCE.md` §6, and the bundles are deterministic rebuilds from
`BUILD_GUIDE.md`), the 1.7B baseline the repo's own README marks *"do not
test — still carries the broken prefill"*, and the root `profiling/` folder,
whose decode-only bin was the **pre-fix** one and whose profile inputs are
duplicated in both dated drops.

| folder | status |
|---|---|
| `qwen3_06b_lutprobe/` | ✅ **CURRENT** (Test L). Bin **`9720e46e…`**. ⚠ The Test K version (`880a6abd…`) was **FLOAT_16** and fails `lint_embedding_dtype.py` |
| `qwen3_06b_lutsplit/` | ✅ **CURRENT** (Test M). The 2-shard 0.6B. Shards **`1f4dcd44…`** / **`11cabce4…`**. ⚠ Holds only what is new — copy the LUT, tokenizer, runner and libs across from `qwen3_06b_lutprobe/` |
| `2026-08-14-gqafix/bundles/qwen3_06b_w8a16_gqafix_ladekv.tar.gz` | ✅ **live** — the 44.707 tok/s reference build (**CTX 1152**) |
| `2026-08-16-regime/bundles/qwen3_06b_w8a16_gqafix_cl512_ladekv.tar.gz` | ✅ **live** — the **CTX-640** build, the shape-matched control for the LUT probe (correction #38) |
| the other 14 bundles in those two drops | the completed **post-fix perf sweep** (dlbc, hybrid, pastkv2g, qh, udma, hvx8, socmodel72, wpack, ctrl…). Spent: every number is recorded in `REFERENCE.md` §6/§8, and several bins are cited there by name (e.g. `gqafix_qh_ladekv` at 95% pooled). Kept deliberately — ~13 GB, and the only remaining bulk. Prune if a re-verification of §8.2 is never expected |
| `docs/` | the device team's context copies. Tiny; some are dated snapshots |

---

## Running a session

Three documents, in order: this index → the test's own doc → its
`RESULTS_TEMPLATE.md`. `DEVICE_SESSION_PROTOCOL.md` is the long-form execution
guide written for Test J; the shorter tests since then are self-contained.

**Preconditions first, always.** Every test doc opens with md5s. Record them
before anything else — every later number is uninterpretable without them, and a
failed precondition is worth more than a run on the wrong bytes.

**Three rules that make a report usable:**

1. **Verbatim beats summary.** `ention ably ance` and `aged aged aged` are
   different findings. Paste the characters.
2. **Say what you actually ran** — skipped stages, changed paths, retries,
   failures. An unexplained gap costs a round-trip.
3. **Separate observation from interpretation.** Both are welcome; label which is
   which. Test K's report inverted a verdict by interpreting without a reference.

**Score against a reference, not against intuition.** Twice now a correct result
has been read as a failure: the 0.6B's greedy repetition is what HF fp32 does,
and `S` + garbage is the *correct* first token of `Sunny` plus a known decode
bug. If a test doc gives expected token ids, use them.

### Runtime facts that keep biting

| | |
|---|---|
| `--max-num-tokens` | **not a flag.** It is `dialog.max-num-tokens` in the config (`Dialog.cpp:2493`). Without a cap the run ends in `Context Size was exceeded` and **no profile is written** |
| `--profile` | **refuses an output file that already exists.** Fresh name per run |
| `-tok` / `--tokens_file` | exists — feed explicit ids to take the tokenizer out of the experiment |
| images | must be `*_fp32.raw`, **6,295,552 bytes**. A `*_u16.raw` is a guaranteed `SIGSEGV` |
| `genie-app` prompts | `node set textFile`, never `node set text` — script strings never unescape `\n` |
| debug dumps | unreachable: `Engine.cpp` whitelists config keys and throws `Unknown QnnHtp config key` |
