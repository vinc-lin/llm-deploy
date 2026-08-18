# Device session runbook — Qwen3-VL-4B v3, the unblocking batch

> **SUPERSEDED IN PART (v4).** The crash this document investigates is
> resolved: Genie's image staging interprets the input file as float32
> (nsp-image-model.cpp:501-524, `embedding-datatype` default), so the
> UFixed16 blobs v2/v3 shipped were read at 2x their size — a ~3 MB
> over-read, which is why padding and page-alignment probes could not
> help. v4 ships float32 blobs. Kept as the diagnostic record; do not
> run its probes. Current instructions: `V4_CHANGES.md`.

**Bundle:** `qwen3vl_4b_e2e_pipeline_v3` (already on the device from the last session)
**Written:** 2026-08-18, after the +1-byte probe result
**Extra file this session:** `qnn-net-run` (in the same HF folder — needed only for T5)
**Board time:** ~20 min if the blocker clears at T2/T3; ~40 min if it does not

This document is standalone: every command, every expected output, every
decision is here. Companions for depth: `IMAGE_PROBE.md` (why T2 works the way
it does), `OPERATOR_GUIDE.md` (metric definitions), `TEST_IMAGES.md` (expected
kit captions).

---

## 0. The session at a glance

Run in THIS order. T1 is guaranteed value regardless of everything else. T2 and
T3 are the two cheap shots at clearing the image blocker; each takes under a
minute. T4 runs only if one of them clears it; T5 only if neither does.

| # | Test | Time | What it produces |
|---|---|---|---|
| T1 | Text-only perf (`genie-t2t-run`) | ~5 min | **First-ever tok/s + TTFT for a 4B two-shard W8A16 tower.** Cannot be blocked by the image bug |
| T2 | 771-page truncate probe | ~1 min | ~25% chance: **image blocker cleared**. Otherwise: the decisive data point for the vendor report |
| T3 | `use-mmap: true` flip | ~2 min | Second independent shot at bypassing the crashing code path |
| T4 | Full e2e + kit + metrics | ~15 min | **End-to-end achieved** — only if T2 or T3 cleared the blocker |
| T5 | `qnn-net-run` standalone ViT | ~10 min | Proves the ViT graph executes outside Genie — the strongest piece of the escalation package |

Between tests, **revert what the test changed** (each section says how) so
results are never confounded.

Setup for every test:

```bash
adb shell
cd /data/local/tmp/qwen3vl_4b_e2e_pipeline_v3
export LD_LIBRARY_PATH=.
```

---

## T1 — Text-only performance (run FIRST, always)

No image involved: this exercises only the two text ctx-bins, which already
load and execute. It is the one measurement guaranteed to produce a result.

```bash
./genie-t2t-run -c genie_dialog_qwen3vl_4b.json \
    -p "What is 2+2? Answer with one number." \
    --profile t1_short_profile.json

./genie-t2t-run -c genie_dialog_qwen3vl_4b.json \
    -p "Describe, in three or four sentences, what makes mountain weather change quickly." \
    --profile t1_long_profile.json
```

Run the short prompt **three times** (cold + 2 warm) and the long one once.
`--profile` writes the numbers to a JSON — on Android, console logging goes to
logcat, so the profile file is the only reliable way to capture timing.

**Collect:** all four `t1_*_profile.json` files, verbatim, plus the generated
text. Report cold and warm separately; never average them.

**Expected:** sensible answers ("4"; a coherent weather paragraph). There is
deliberately no predicted tok/s — no 4B tower has ever been measured on this
silicon, and this run creates the first data point.

Nothing to revert.

---

## T2 — The 771-page truncate probe (~25% chance this IS the fix)

**Why.** All three crashes so far happened with the input buffer ending at
exactly `tensor_size + 4096` bytes. Two theories fit that: (A) the code
touches one byte past the end of the buffer, wherever it ends — unfixable from
the host; (C) the code touches a *fixed* address 4,096 bytes past the tensor,
and every test so far coincidentally put the buffer end exactly there. Under C,
a buffer any bigger than that contains the access harmlessly. This test makes
the file 771 pages — 8,192 bytes past the fixed address — separating the
theories for the first time. Full derivation: `IMAGE_PROBE.md` §4.

```bash
cp sample_image.raw sample_image.raw.bak
truncate -s 3158016 sample_image.raw
ls -l sample_image.raw          # MUST read 3158016 — if not, stop here
./genie-app -s genie_pipeline_qwen3vl.script 2>&1 | tee t2_truncate.log
echo "exit=$?"
```

`truncate` zero-fills; the first 3,145,728 bytes — everything the tensor
reads — are untouched.

| Outcome | Meaning | Do next |
|---|---|---|
| **A caption appears** | Blocker **cleared** (theory C). Expected text ≈ *"A red circle and a blue square are positioned side by side on a white background."* — semantic match is the bar, not exact wording | Enlarge the kit blobs the same way (below), then go to **T4**. Skip T3 |
| Crash, fault at `base + 0x303000` | Theory A confirmed — padding is conclusively dead | Pull the tombstone, revert (below), go to **T3** |
| Crash, fault at `base + 0x301000` (unchanged) | The buffer is hard-capped at tensor+1page — a third behaviour | Pull the tombstone, revert, go to **T3** |

If it cleared, enlarge all six kit blobs before T4:

```bash
for f in wx_*.raw; do cp "$f" "$f.bak"; truncate -s 3158016 "$f"; done
ls -l wx_*.raw                   # each must read 3158016
```

**Revert** (only if it crashed): `mv sample_image.raw.bak sample_image.raw`

**Collect either way:** `t2_truncate.log`, the exit code, and on crash the
newest file in `/data/tombstones/` — the **fault address line** is the single
most important byte of this whole session.

---

## T3 — The `use-mmap` flip (skip if T2 cleared the blocker)

**Why.** The crash is in libGenie's heap-copy input path. The image-encoder
config ships `"use-mmap": false`; flipping it to `true` makes the runtime map
the file instead, which is a **different code path** that may never execute
the faulting instruction. Untested until now.

The key lives in `genie_image_encoder_qwen3vl.json` under
`image-encoder.engine.backend.QnnHtp`:

```bash
cp genie_image_encoder_qwen3vl.json genie_image_encoder_qwen3vl.json.bak
sed -i 's/"use-mmap": false/"use-mmap": true/' genie_image_encoder_qwen3vl.json
grep use-mmap genie_image_encoder_qwen3vl.json      # must show: true
./genie-app -s genie_pipeline_qwen3vl.script 2>&1 | tee t3_mmap.log
echo "exit=$?"
```

Run this with the **original** 3,149,824-byte `sample_image.raw` (i.e. after
T2's revert), so the two variables are never changed at once.

| Outcome | Meaning | Do next |
|---|---|---|
| **A caption appears** | The mmap path bypasses the bug. Blocker cleared | Keep the config change, go to **T4** |
| Same crash | The mmap setting does not reach the faulting path | Revert, go to **T5** |
| A *different* error (message, not SIGSEGV) | The mmap path exists but wants something else | Capture the exact message — that is itself a lead — revert, go to **T5** |

**Revert** (if it did not clear):
`mv genie_image_encoder_qwen3vl.json.bak genie_image_encoder_qwen3vl.json`

**Collect:** `t3_mmap.log`, exit code, tombstone if any.

---

## T4 — Full end-to-end (only after T2 or T3 cleared the blocker)

This is the finish line: image → ViT → caption on silicon, plus the numbers.

```bash
# 1. the sample scene, three runs (1 cold + 2 warm), timed
for i in 1 2 3; do
  echo "== run $i"; time ./genie-app -s genie_pipeline_qwen3vl.script
done | tee t4_e2e.log

# 2. the six real photographs
for s in wx_*.script; do
  echo "== $s"; ./genie-app -s "$s"
done 2>&1 | tee t4_kit.log
```

**Judging captions:** `TEST_IMAGES.md` carries two references per image —
compare against the **device-faithful** column, not the HF one. The bar is
semantic (weather + scene contents right); wording will differ, because the
references are fp32 and the device is W8A16. Fluent text describing a
*different* image is a failure.

**Metrics** (definitions in `OPERATOR_GUIDE.md` §6 — use them verbatim):
init, TTFT, decode tok/s, total wall, peak RSS; cold and warm separate.

**One observation doubles as an experiment:** the 273-token prompt should be
consumed by **3** prefill calls (128+128+17), not 273 decode steps. A short
TTFT (seconds) means prefill was selected; a very long one (tens of seconds)
means it silently fell back to all-decode. Either way, report which — this has
never been observed on device.

---

## T5 — Standalone ViT through `qnn-net-run` (only if T2 and T3 both failed)

**Why.** Every crash so far is in *Genie's* input path, before the graph runs.
`qnn-net-run` executes the same ctx-bin with no Genie involved. If the ViT
runs clean here, the defect is isolated to libGenie with the graph exonerated —
which turns the vendor report from "your library crashes" into "your library
crashes and here is proof the workload is fine," a much harder claim to
deflect.

**One-time setup** — `qnn-net-run` is NOT in the original bundle; it is in the
same HF folder as this document:

```bash
# host:
adb push qnn-net-run /data/local/tmp/qwen3vl_4b_e2e_pipeline_v3/
adb shell chmod +x /data/local/tmp/qwen3vl_4b_e2e_pipeline_v3/qnn-net-run
```

On the device, build the exact-size input and the config (the tool needs the
tensor's exact 3,145,728 bytes — strip the pad; and it needs the same HTP
settings Genie uses, reusing the bundle's own `htp_backend_ext_config_vit.json`,
which already declares the unsigned PD session):

```bash
cd /data/local/tmp/qwen3vl_4b_e2e_pipeline_v3
head -c 3145728 sample_image.raw > vit_input.raw
ls -l vit_input.raw                     # MUST read 3145728

echo "pixel_values:=vit_input.raw" > vit_input_list.txt

cat > netrun_htp_config.json <<'EOF'
{
  "backend_extensions": {
    "shared_library_path": "libQnnHtpNetRunExtensions.so",
    "config_file_path": "htp_backend_ext_config_vit.json"
  }
}
EOF

./qnn-net-run \
    --retrieve_context qwen3vl-4b-vit-w8a16_ctx.bin \
    --backend libQnnHtp.so \
    --input_list vit_input_list.txt \
    --config_file netrun_htp_config.json \
    --use_native_input_files --use_native_output_files \
    --output_dir t5_vit_out 2>&1 | tee t5_netrun.log
echo "exit=$?"
```

The two `--use_native_*` flags tell the tool the input file is already the
tensor's own quantized `uint16` layout (and to write outputs the same way)
rather than float32 to be converted. **If your build rejects those exact flag
names, run `./qnn-net-run --help` and use whatever it calls its native/raw
input and output file options — do not feed the blob as float.**

**Success looks like:** exit 0 and four files under `t5_vit_out/Result_0/`,
each exactly **1,310,720 bytes** (256×2560 `uint16`): `image_features.raw` and
`deepstack_visual_embed_{0,1,2}.raw`.

```bash
ls -l t5_vit_out/Result_0/
```

**Pull all four output files and send them back** — the host has the gated
fp32 reference for this exact image and will verify the numerics offline
(cosine per output; host quantsim measured ≥0.9975, and this run is the first
check of that number on silicon).

| Outcome | Meaning |
|---|---|
| Runs clean, 4 outputs, sizes right | ViT graph proven on silicon; defect isolated to libGenie's input path. Escalation package complete |
| Crashes the same way | The fault is below Genie after all — a major redirect; the tombstone from THIS crash becomes the key artifact |
| Different error | Capture verbatim; likely a tool-flag issue, not the graph — check `--help` per above |

---

## What to send back (all of it, even for failures)

1. **T1:** the four profile JSONs + generated text.
2. **T2:** `t2_truncate.log`, exit code, `ls -l sample_image.raw` output,
   tombstone + **fault address** if it crashed.
3. **T3:** `t3_mmap.log`, exit code, the `grep use-mmap` line, tombstone if any.
4. **T4** (if reached): `t4_e2e.log`, `t4_kit.log`, all seven captions as text,
   the metrics table from `OPERATOR_GUIDE.md` §6, and the prefill-selection
   observation.
5. **T5** (if reached): `t5_netrun.log`, `ls -l t5_vit_out/Result_0/`, and the
   four output `.raw` files.
6. `adb logcat -d` captured around any crash.

Screen photographs are fine; they get transcribed and the Markdown becomes the
record.

---

## Appendix — optional T6, only if a newer QAIRT SDK is on hand

The crash lives in `libGenie.so 1.19.0` (BuildId `f6899695c925325c`). If a
later QAIRT 2.48.x hotfix or 2.49.x is available, its runtime may have
rewritten the faulting path:

```bash
cp libGenie.so libGenie.so.bak
# push the newer libGenie.so (and genie-app from the SAME SDK if load fails
# with a symbol/ABI error) into the bundle dir, then:
./genie-app -s genie_pipeline_qwen3vl.script
```

Swap **only** the runtime — do not rebuild or replace any `.bin`; all three
ctx-bins are valid and load. If the newer runtime clears the crash, that plus
the T5 result is the whole story for Qualcomm. Revert with the `.bak` copies.

---

## Fixed reference points

- All three ctx-bins load; the text tower generates correctly — do not rebuild
  anything while chasing the image bug.
- The image blob's first 3,145,728 bytes are the tensor payload; everything
  after is inert padding. No test above changes the payload.
- Crash signature to compare against: `SIGSEGV (SEGV_ACCERR)`, fault at the
  end of a `[anon:scudo:secondary]` allocation, `GenieNode_setData+572` at
  frame #08, faulting pc `0x646e84` in libGenie.
