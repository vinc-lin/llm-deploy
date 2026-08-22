# Device session runbook — Qwen3-VL-4B v4

**Bundle:** `qwen3vl_4b_e2e_pipeline_v4`
**Written:** 2026-08-18, after root-causing the v3 ImageEncoder crash
**Board time:** ~25 min if V2 clears; ~35 min if it does not
**Companions:** `V4_CHANGES.md` (why v4 differs), `OPERATOR_GUIDE.md` (metric
definitions, triage, limitations), `TEST_IMAGES.md` (expected kit captions)

This document is the ordered plan: what to run, in what order, what each
outcome means, and what to collect. Every command is repeated here so you can
run the session from this file alone.

---

## 0. The session at a glance

| # | Test | Time | What it produces |
|---|---|---|---|
| **V1** | Text-only (`genie-t2t-run`) | ~6 min | Retest of the v3 garbage output **and** the first tok/s for a 4B two-shard W8A16 tower. Cannot be blocked by the image path |
| **V2** | Image pipeline, sample image | ~4 min | **The headline:** did the v3 SIGSEGV clear |
| **V3** | Weather kit, 6 images | ~10 min | Real-photograph captions — only informative if V1 passed |
| **V4** | Decode-only fallback | ~5 min | Only if V2 fails at **load** (not if it crashes at `setData`) |
| **V5** | Newer libGenie | ~5 min | Only if V2 still SIGSEGVs **and** a later QAIRT SDK is on hand |

**Run V1 first, and read its result before judging any caption.** If the text
tower is still emitting garbage, V2 and V3 will produce garbage captions even
with a perfectly working image path — in that case V2 is judged on
**crash-or-no-crash alone**. Judging a caption without V1's result will produce
a wrong conclusion about the image path.

### Setup (once)

```bash
adb push qwen3vl_4b_e2e_pipeline_v4 /data/local/tmp/qwen3vl_v4
adb shell
cd /data/local/tmp/qwen3vl_v4
chmod +x genie-app genie-t2t-run qnn-net-run
export LD_LIBRARY_PATH=.

ls -l sample_image_fp32.raw              # MUST read 6295552
# stale v3 blobs must not exist -- this must print NOTHING
find . -maxdepth 1 -name '*.raw' ! -name '*_fp32.raw' ! -name '*_u16.raw'
```

**That second check is not optional.** v4's blobs are float32 and named
`*_fp32.raw`; a leftover v3-era `sample_image.raw` in the directory is the one
way to reproduce the old crash with a correct bundle and conclude the fix
failed. If any v3 blob is present, delete it before starting.

If you reused v3's ctx-bins rather than re-downloading, verify them against the
md5s in `V4_CHANGES.md` §4 first.

---

## V1 — Text only (run FIRST, always)

No image involved: only the two text ctx-bins, which are known to load and
execute. In the v3 session this produced **garbage output** — that is what this
retest is for. v4 sets `bos-token: -1`, removing a spurious `<|endoftext|>`
that was being prepended to every prompt.

```bash
./genie-t2t-run -c genie_dialog_qwen3vl_4b.json \
    -p "What is 2+2? Answer with one number." \
    --profile v1_short_profile.json

./genie-t2t-run -c genie_dialog_qwen3vl_4b.json \
    -p "Describe, in three or four sentences, what makes mountain weather change quickly." \
    --profile v1_long_profile.json
```

Run the short prompt **three times** (cold + 2 warm), the long one once.
`--profile` is the only reliable way to capture timing: on Android, console
logging goes to logcat rather than stdout.

| Outcome | Meaning | Do next |
|---|---|---|
| **Coherent answers** ("4"; a sensible paragraph) | The text tower is healthy; `bos-token` was the defect. **Captions become judgeable** | V2, judging the caption normally |
| **Garbage / repetition** | The text defect is deeper than the config. Not fatal to the session | V2, judging **only** on crash-or-no-crash. Do not change configs chasing it |

**Collect:** all four profile JSONs verbatim, plus the generated text — **even
if it is nonsense**, which is itself the measurement. Report cold and warm
separately; never average. Note that the **timing numbers are valid regardless
of output quality**, so this test produces its headline number either way.

Nothing to revert.

---

## V2 — Image pipeline, sample image (the headline test)

The exact sequence that crashed in v3, with only the blob format changed.

```bash
./genie-app -s genie_pipeline_qwen3vl.script 2>&1 | tee v2_e2e.log
echo "exit=$?"
```

Then, if it completed, two warm runs for timing:

```bash
for i in 1 2; do echo "== warm $i"; time ./genie-app -s genie_pipeline_qwen3vl.script; done 2>&1 | tee -a v2_e2e.log
```

| Outcome | Meaning | Do next |
|---|---|---|
| **No crash, coherent caption** | **End-to-end achieved.** Expected ≈ *"A red circle and a blue square are positioned side by side on a white background"* — semantic match is the bar, not wording | V3 |
| **No crash, garbage caption** | **The image blocker is cleared** — a major result. The remaining fault is V1's text defect, not the image path | V3 (captions will also be garbage; run it anyway for the crash/perf data), then report both |
| **`ShapeError` / fails at load** | Not the setData bug — a different problem | Capture the log + `*.info.json`, go to **V4** |
| **SIGSEGV at `node set image`** | Either a stale blob, or a new mechanism | Re-check `ls -l *.raw` per §0. If the fed file really is 6,295,552 B, capture the tombstone and go to **V5** if available, else **stop** |

**Collect:** `v2_e2e.log`, exit code, the caption verbatim, timing for cold +
2 warm, and on any crash the newest file in `/data/tombstones/` — the **fault
address line** is the single most important line in it.

**One observation doubles as an experiment:** the 273-token prompt should be
consumed by **3** prefill calls (128+128+17), not 273 decode steps. A short
TTFT (seconds) means prefill was selected; a very long one (tens of seconds)
means it silently fell back to all-decode. Report which — this has never been
observed on device.

---

## V3 — Weather / road kit

Six real photographs covering the deployment's scenes: rain, fog, snow, clear,
overcast, traffic.

```bash
for s in wx_*.script; do echo "== $s"; ./genie-app -s "$s"; done 2>&1 | tee v3_kit.log
```

Run this **even if V1 produced garbage** — it still exercises six different
images through the full path and will surface any crash that only a real
photograph triggers.

**Judging** (only meaningful if V1 passed): `TEST_IMAGES.md` carries two
references per image. Compare against the **device-faithful** column, not the
HF one — the HF column is context only and the device cannot reach it by
design (deepstack is zeroed). The bar is semantic: weather and scene contents
right. Fluent text describing a *different* scene is a failure.

All six images were checked against the ViT's calibrated input range and clip
by at most 1.00 LSB, so none is an out-of-domain input.

**Collect:** `v3_kit.log` and all six captions as text.

---

## V4 — Decode-only fallback (only if V2 failed at LOAD)

Do **not** run this for a `setData` SIGSEGV — it changes nothing about the
image input path. It is only for a load-time failure.

```bash
./genie-app -s genie_pipeline_qwen3vl_decodeonly.script 2>&1 | tee v4_fallback.log
echo "exit=$?"
```

One file swap; it filters the prefill graphs out before they become variants,
so nothing shape-validates them. Cost: no prefill — every prompt token goes
through the AR=1 decode graph, so expect a much longer TTFT. Output should be
unchanged.

**This has never run on device** and carries one unverified risk: the same
graph-name list is handed to both contexts, and whether HTP tolerates an
enable-graphs name absent from a given binary is sealed inside `libQnnHtp`. If
it also fails, **report the error text and stop swapping configs.**

**Collect:** `v4_fallback.log`, exit code, and the same metrics as V2 so the
prefill benefit is measurable rather than assumed.

---

## V5 — Newer libGenie (only if V2 still SIGSEGVs)

Only worth doing if a later QAIRT runtime is available. The v3 crash lived in
`libGenie.so 1.19.0` (BuildId `f6899695c925325c`); v4 fixes it from the host
side, so reaching this step means the host-side fix did **not** work and a
runtime difference is the next variable.

```bash
cp libGenie.so libGenie.so.bak
# push the newer libGenie.so (and genie-app from the SAME SDK if load fails
# with a symbol/ABI error), then:
./genie-app -s genie_pipeline_qwen3vl.script 2>&1 | tee v5_newlib.log
```

Swap **only** the runtime — do not rebuild or replace any `.bin`; all three
ctx-bins are valid and load. Revert with the `.bak` copy.

---

## What to send back (all of it, even for failures)

1. **V1:** the four profile JSONs + all generated text, garbage included.
2. **V2:** `v2_e2e.log`, exit code, the caption, cold/warm timings, the
   prefill-call observation, and the tombstone + **fault address** if it crashed.
3. **V3:** `v3_kit.log` and all six captions as text.
4. **V4/V5** if reached: the logs, exit codes, and the exact error text.
5. The filled-in metrics table from `OPERATOR_GUIDE.md` §5.
6. `ls -l *.raw` from the directory you actually ran in.
7. The `*.info.json` files from the bundle you actually pushed.
8. `adb logcat -d` captured around any crash.

Screen photographs are fine; they get transcribed and the Markdown becomes the
record.

---

## Fixed reference points

- **All three ctx-bins are byte-identical to v3 and are not suspect.** Do not
  rebuild anything during this session; v4 changed only configs, scripts and
  image blobs.
- The pipeline input is `*_fp32.raw` at **6,295,552** bytes. `*_u16.raw` files
  (3,145,728 B) are `qnn-net-run` inputs only and will crash `genie-app`.
- v3 crash signature, for comparison: `SIGSEGV (SEGV_ACCERR)`, fault at the end
  of a `[anon:scudo:secondary]` allocation, `GenieNode_setData+572` at frame
  #08, faulting pc `0x646e84` in libGenie.
- The v3 session's own results that still stand: both text ctx-bins load and
  execute; the ViT ctx-bin runs correctly under `qnn-net-run` with all four
  outputs at 1,310,720 bytes each.
