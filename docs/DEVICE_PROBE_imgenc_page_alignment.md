# ImageEncoder `setData` crash — probe results and corrected analysis

**Bundle:** `qwen3vl_4b_e2e_pipeline_v3`
**Last updated:** 2026-08-17, after the +1-byte page-alignment probe
**Status:** blocker unresolved; **padding is now ruled out structurally**, one
cheap test remains before vendor escalation

> **This document supersedes its own first version.** That version proposed a
> 1-byte pad as a likely fix. The probe was run and **the fix failed**. What
> follows is the corrected analysis, including what the failure taught us —
> which is more useful than the original hypothesis would have been.

---

## 0. Where this stands

| | |
|---|---|
| Symptom | `SIGSEGV (SEGV_ACCERR)` on `node set image`, inside `GenieNode_setData` |
| Runtime | libGenie 1.19.0, BuildId `f6899695c925325c` |
| Tried | unpadded (v2), +4096 (v3), +1 byte (probe) — **all three crash** |
| Established | the fault has landed at exactly `user_buffer + roundDown(file_size, page)` in all three runs |
| Two live theories | **A:** one-past-the-end of a page-granular buffer → no padding can ever work (§3). **C:** a fixed `tensor + 4096` overshoot → a 771-page blob FIXES it (§4) |
| Decides between them | **one 30-second test, §4 — run it before believing §3** |
| Not implicated | the GQA text tower, the ViT ctx-bin, the bundle layout |

---

## 1. The three runs

| run | file size | `roundDown(size, 4096)` | observed fault offset | crash |
|---|---|---|---|---|
| v2 | 3,145,728 (768 pages) | `0x300000` | `0x300000` | yes |
| v3 | 3,149,824 (769 pages) | `0x301000` | `0x301000` | yes |
| probe | 3,149,825 (769 pages + 1 B) | `0x301000` | `0x301000` | yes |

The probe appended a single byte, expecting the allocator to round up to a
whole extra page and leave 4,095 bytes of slack in front of the guard page. It
did not. **The allocation did not change at all.**

### Probe tombstone (`tombstone_21`)

```
signal: SIGSEGV, code: SEGV_ACCERR
Scudo secondary: 0x76eadf9000 - 0x76eb0fbfff   (0x302000 = 3,153,920 B)
  ├── header:       0x76eadf9000 - 0x76eadf9fff  (0x1000, 1 page)
  ├── user buffer:  0x76eadfa000 - 0x76eb0fafff  (0x301000 = 3,149,824 B)
  └── guard page:   0x76eb0fb000 - 0x76eb0fcfff  (0x2000)
                         ▲ fault address
x20 = 0x180000            backtrace #08: GenieNode_setData+572
```

Cross-check: `user (0x76eadfa000) + 0x301000 = 0x76eb0fb000` = guard start =
fault address. All three are the same number.

---

## 2. The rule that fits every run

```
user_buffer_size = roundDown(file_size, 4096)
fault address    = user_buffer + user_buffer_size
```

Verified against all three runs in §1 — it is the only rule tested so far that
fits all of them.

The buffer **is** derived from the input file. The step that was missed in
every prior analysis, including the first version of this document, is that it
is **truncated to whole pages** first. That single fact explains all three
outcomes:

- **+1 byte changed nothing** — `roundDown` discarded it before it could have
  any effect.
- **+4096 moved the fault by one page** — a whole page survives truncation.
- **The guard page is always flush against the data** — a page-granular buffer
  ends exactly on a page boundary by construction, so the allocator places the
  guard immediately after, every time.

---

## 3. Why, under theory A, no amount of padding can fix this

⚠ **This section is conditional on theory A** (the access is one-past-the-end
of the buffer itself). Theory C — a fixed `tensor + 4096` overshoot that every
run so far has happened to sit exactly at — survives all three data points and
is *fixed* by a larger blob. §4's test separates them; do not act on this
section until it has run. A reader who stops here would skip the one cheap
thing that might clear the blocker.

Under theory A, the conclusion is stronger than "padding didn't help":

Slack only appears when a buffer ends *partway* through its last page. A buffer
whose size is always a whole number of pages **never** does. So:

```
  any file size  ──roundDown──▶  whole pages  ──▶  ends on a page boundary
                                                    ──▶  guard page immediately after
                                                    ──▶  overrun always faults
```

| pad | file | buffer (page-granular) | slack before guard |
|---|---|---|---|
| +0 | 3,145,728 | 3,145,728 | **0** |
| +1 | 3,145,729 | 3,145,728 | **0** |
| +64 | 3,145,792 | 3,145,728 | **0** |
| +4096 | 3,149,824 | 3,149,824 | **0** |
| +1 MB | 4,194,304 | 4,194,304 | **0** |

The slack column is zero for every possible input. The original hypothesis in
this document assumed a byte-granular buffer, where non-page padding would
have produced 4,095 bytes of slack. That assumption was wrong.

---

## 4. The one test left — and under theory C it IS the fix

The suspicious coincidence: the observed buffer is `tensor_size + exactly one
page` (3,145,728 + 4,096 = 3,149,824), and **every input tested so far has put
the buffer end exactly at that address**. So a fixed overshoot of the *tensor*
by 4,096 bytes (theory C) predicts the same three crashes as a one-past-the-end
of the *buffer* (theory A). They have never been separated.

- **Theory A** — access at `buffer_end`. Fault follows the buffer wherever it
  goes. Not host-fixable (§3).
- **Theory C** — access at `tensor + 4096` = `user + 0x301000`, fixed. Any
  buffer bigger than that contains the access harmlessly. **Host-fixable by
  re-cutting the blobs at 771 pages.**

Scorekeeping honesty: A fits all three runs; C mispredicts v2's fault address
by one page — but v2's tombstone is the least reliable of the three (no memory
map; its stated allocation is inconsistent with the header-page layout the
probe tombstone shows). Call it **~1-in-4 that C is right**. Thirty seconds
for a 25% chance of clearing the blocker outright.

Set the file to a size the two theories disagree about — exactly 771 pages,
which gives 8,192 bytes of headroom past `tensor + 4096`:

```bash
truncate -s 3158016 sample_image.raw     # 3,158,016 B = 771 pages exactly
ls -l sample_image.raw                   # must read 3158016
LD_LIBRARY_PATH=. ./genie-app -s genie_pipeline_qwen3vl.script
```

| outcome | verdict | next action |
|---|---|---|
| **no crash — you get a caption** | **Theory C** — fixed `tensor+4096` overshoot, contained by the larger buffer | **Blocker cleared.** We re-cut every blob at 771 pages (v3.1, ~1 h) and the full image path opens. Run the kit while you have the board |
| crash at `user + 0x303000` (moved with the buffer) | **Theory A** — one-past-the-end, padding dead across a three-point spread | Escalate with the precise sizing rule; proceed to `qnn-net-run` bisection |
| crash at `user + 0x301000` (unchanged) | buffer truly capped at `tensor + 1 page` regardless of file | Redirects again — the cap itself becomes the escalation question |

`truncate` zero-fills, so the first 3,145,728 bytes — everything the tensor
reads — are untouched. Revert with `truncate -s 3149824 sample_image.raw`.

One command; either it clears the blocker or it removes the last ambiguity
from the vendor report.

---

## 5. Corrections to earlier analyses

### 5a. To the first version of this document (mine)

| claimed | correction |
|---|---|
| The buffer is sized from the file **byte-for-byte** | It is sized from the file **rounded down to whole pages**. |
| Padding by a non-page amount yields 4,095 B of slack | It yields **zero** — the pad is truncated away. |
| The +1-byte probe should fix the crash | It did not. Hypothesis falsified by direct test. |

The probe was still worth running: it cost one command and it converted
"padding might work" into "padding provably cannot work", which is the useful
half of §3.

### 5b. To `V3_IMGENCODER_FAILURE_ANALYSIS.md` (branch-D report)

| claimed | correction |
|---|---|
| "The allocation is **not driven by file size**" | It is — the +1 byte was absorbed by page truncation, not ignored. §2's rule fits v2, v3 and the probe. |
| "Destination overrun, **not host-fixable**" | The right conclusion for the wrong reason. Padding is dead because the buffer is page-granular (§3), not because the buffer is unrelated to the file. |
| `x20 = 0x180000` "exactly matches a float16 tensor of **1 × 3 × 512 × 512**" | **Off by 2×.** `1×3×512×512` = 786,432 elements = 1,572,864 B as float16. The real tensor is `pixel_values [1024, 1536] UFIXED_POINT_16` = 1,572,864 elements = 3,145,728 B — 1024 patches (a 32×32 grid) × 1536 features. The element-count reading of `x20` is correct; only the shape and dtype attribution are wrong. |

That last one matters operationally: left uncorrected it sends the developer
hunting for a `1×3×512×512` fp16 conversion path that does not exist in this
graph. The dtype is load-bearing too — a stock Genie pipeline **cannot** drive
an fp16 image encoder at all (`setupInputFP16` is an empty stub that discards
the blob and returns success), which is why this tower ships `UFIXED_POINT_16`.

---

## 6. Escalation package

Everything below is established and reproducible:

1. **Reproduction:** load the pipeline, `node set image` with any `.raw` of any
   size → `SIGSEGV (SEGV_ACCERR)` at `user_buffer + roundDown(file_size, 4096)`.
2. **Runtime:** libGenie 1.19.0, BuildId `f6899695c925325c`, QAIRT 2.48.40,
   Hexagon v81, Android 16, unsigned PD.
3. **Fault site:** `pc 0x646e84` in `libGenie.so` (stripped internal symbol).
   `GenieNode_setData+572` is **frame #08** — the fault is eight frames deeper.
4. **Allocation:** `[anon:scudo:secondary]`, i.e. an ordinary heap allocation
   with a `memcpy`-class access, **not** DMA or ION.
5. **Bisection:** three input sizes (768 pages, 769 pages, 769 pages + 1 byte);
   the fault tracks `roundDown(file_size, page)` exactly, so it is not an
   input-validation or blob-content problem.
6. **Tombstones:** `tombstone_27` (v3), `tombstone_21` (probe).
7. **Not the graph:** the ViT ctx-bin loads and allocates cleanly; the fault is
   in the runtime's input-feeding path, before graph execution.

---

## 7. Investigation, in priority order

**7.1 Resolve `pc 0x646e84` against an unstripped `libGenie.so`.** Highest
value by a distance. Every analysis so far — mine and the device team's — is
inference around a symbol nobody can see, and both of us have now been wrong
once. `GenieNode_setData+572` is not the faulting instruction.

**7.2 Run the ViT standalone through `qnn-net-run`, bypassing Genie entirely.**
This is the strongest bisection available and it has never been done. If the
ViT ctx-bin executes correctly outside Genie, the defect is isolated to Genie's
input path and "the graph runs fine outside your runtime" is a much harder
claim for a vendor to deflect. `qnn-net-run` is not in this bundle; it comes
from the QAIRT SDK.

**7.3 Test a newer libGenie.** 1.19.0 is what fails. Check later QAIRT 2.48.x
hotfixes and 2.49.x for changes to the ImageEncoder input path. Swap **only**
the runtime library (and `genie-app` if ABI requires) — do not rebuild any
ctx-bin; all three are valid and load.

**7.4 Determine what fixes the buffer at `0x301000` = tensor + 1 page.** §4
decides whether this is even the right question.

---

## 8. Do not change

- **The GQA text tower.** Both ctx-bins load and execute on device — 0
  replication ops, attention MatMul batch dim 8 (was 32), converter DDR reads
  down 18–38% per graph. This is the first on-silicon confirmation that the
  rebuilt split tower works, and it is independent of this blocker.
- **The ViT ctx-bin.** Loads cleanly. Not implicated.
- **The image blob encoding.** `UFIXED_POINT_16` is correct and required.
- **The blob sizes — until §4 has run.** Under theory A padding cannot help
  (§3); under theory C the 771-page size is the fix. §4 decides which.
- **The bundle layout, runtime libraries, configs, scripts, tokenizer.** All
  validated, MD5s match.

---

## 9. Available regardless of this blocker

The **text-only measurement** touches none of the image path:

```bash
LD_LIBRARY_PATH=. ./genie-t2t-run -c genie_dialog_qwen3vl_4b.json \
    -p "What is 2+2? Answer with one number." \
    --profile qwen3vl_4b_text_profile.json
```

No 4B two-shard W8A16 text tower has ever had its tok/s or TTFT measured on
this silicon. That number stands on its own whichever way this resolves. See
`OPERATOR_GUIDE.md` §5a for the flag semantics and §6 for the metric
definitions and the results table to fill in.
