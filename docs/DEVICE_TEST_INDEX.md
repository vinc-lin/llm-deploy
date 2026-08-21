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
| **Next test** | **Test L** — `TEST_L_ctxbin_vs_genie.md`. ~15 min. Bundle `qwen3_06b_lutprobe/` (0.6B repo), **md5 `9720e46e…`** |
| **The one open defect** | Genie's decode-step feed on the 4B: prefill right, decode step 1 wrong |
| **Sole live suspect** | the two-ctx-bin split (shard 0 → shard 1, `[1,1,2560]`, every decode step) |
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

| folder | status |
|---|---|
| `qwen3vl_v5_session/03_vl4b_v5/` | ✅ **CURRENT.** Complete pipeline bundle. Shard 0 `f031e3a7…` |
| `qwen3vl_4b_testj_session/` | ✅ current kit (Test J: `testj/` + `testh/`) |
| `qwen3vl_4b_testk_session/` | ✅ current — incl. the one-word `prompt_seg2_*` files |
| `qwen3vl_4b_e2e_pipeline_v5/` | ⚠ **shipped a stale shard 0** (`065056ba…`, pre-`uFxp_16`). Replaced on the Hub, but prefer `qwen3vl_v5_session/03_vl4b_v5/` |
| `qwen3vl_4b_e2e_pipeline` … `_v4` | superseded — history only |
| `qwen3vl_4b_testf/g/h/i_session` | superseded by J and K; kept for provenance |

### Qwen3-0.6B — `vinccniv/sa8797p-qwen3-w8a16-bundles`

| folder | status |
|---|---|
| `qwen3_06b_lutprobe/` | ✅ **CURRENT, rebuilt 2026-08-22.** Bin **`9720e46e…`**. ⚠ The Test K version (`880a6abd…`) was **FLOAT_16** and fails `lint_embedding_dtype.py` |
| `2026-08-14-gqafix/bundles/qwen3_06b_w8a16_gqafix_ladekv.tar.gz` | ✅ the 44.707 tok/s reference build (**CTX 1152**) |
| `2026-08-16-regime/bundles/qwen3_06b_w8a16_gqafix_cl512_ladekv.tar.gz` | ✅ the **CTX-640** build — the shape-matched control for the LUT probe (correction #38) |
| top-level `qwen3_06b_w8a16_*.tar.gz` | pre-GQA-fix lineage. Historical; **never quote their tok/s as current** |

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
