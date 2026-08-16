# Qwen3-VL-4B v3 — GQA-fixed tower + padded blobs, one-batch device kit — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development
> (recommended) or superpowers:executing-plans to implement this plan task-by-task.
> Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Finish every device-free work item so that ONE device session can
untar bundle v3, run `genie-app` on an image, and get a valid caption — with
the two known defects (ImageEncoder SIGSEGV, un-grouped GQA attention) fixed
before any hardware is touched.

**Architecture:** Two independent fixes converge into one bundle. (1) The VL
text tower ships with 36 KV-replication ops per shard because
`ExportQwen3.from_hf_vl_text` never grew a `grouped_gqa` parameter — the exact
defect whose removal gave 0.6B its 6.54× speedup. Wire the flag through the
three call sites, re-quantize on tank (fresh lineage, calib npz reused),
re-split, re-convert, re-gate. (2) The device-blocking
`SIGSEGV (SEGV_ACCERR)` at `GenieNode_setData+572` is a one-byte over-read into
a Scudo guard page on a heap buffer sized from the blob's `fileSize`
(runbook §1.2b) — pad every image blob by 4 KB at generation time and teach the
bundle lint the new size. Everything is re-gated (`parity_e2e_vl.py` full,
`lint_gqa_ops.py` on all four DLCs — the gate that was never pointed at this
tower — `lint_pipeline_bundle.py`, `genie_load_check.py`), then shipped as
`qwen3vl_4b_e2e_pipeline_v3/` on HF with a single-session DEVICE_TEST.

**Tech Stack:** tank (44 cores / 125 GB / 326 GB free) for AIMET quant +
conversion + ctx-bins; local WSL (180 GB free on C:) for code, fp32 exports,
the e2e gate, kit, bundling, HF upload via proxy watchdog. `qwen3-deploy` env,
QAIRT 2.48.40, `git push ssh://tank/...` sync.

---

## Ground truth this plan is built on (verified 2026-08-17, not assumed)

### The GQA defect, at DLC level

```
$ lint_gqa_ops.py /home/vinc/llm-local/work/dlc/qwen3vl-4b-w8a16-splitkv/decode_0.dlc
FAIL  replication ops (Eltwise_Binary MULTIPLY, /Expand): 36  (expected 0)
      attention MatMuls: 36 (expected 36), batch dim ['32'] (expected 8)
```

Cause is structural, not a missed flag: `quantize_aimet.py` HAS `--grouped-gqa`
(line 289) and its non-VL branch passes it (line 404), but the VL branch (line
399-401) calls `from_hf_vl_text(...)` which has **no `grouped_gqa` parameter**
(modeling_export.py:337-338) and therefore builds `ExportQwen3` with the
default `grouped_gqa=False`. `export_qwen3vl_text.py` (the fp32 export the e2e
gate runs on) has no flag either. VL config: 32 Q heads / 8 KV heads = 4:1
(0.6B was 2:1), 36 layers. The IO contract is untouched by the fix — KV stays
8 heads (`io_spec` uses `n_kv`); only the two attention MatMuls' batch dim
changes (modeling_export.py:113-138), so configs and graph names stay as v2.

### The SIGSEGV fix

All shipped `.raw` blobs are exactly 3,145,728 B (= 0x300000 = fault offset).
`docs/DEVICE_TEST_qwen3vl_imgenc_sigsegv.md` §1.2b resolves the mechanism
off-device (caller-sized heap alloc + memcpy, over-read into the guard page);
§4 predicts a padded blob (3,149,824 B) succeeds; §6 branch C's remediation is
"re-cut the bundle with padded blobs". Padding is safe: the runtime consumes
only the first 0x300000 bytes. `lint_pipeline_bundle.py` currently hard-fails
any `.raw` ≠ 3,145,728 B (lines 474, 1182) — it must learn the padded size or
it will reject the fix.

### Inventory

- **tank** `~/llm-local/work/`: `quant/qwen3vl-4b-w8a16-{prefill,decode,
  decode-unified,prefillkv128,prefillkv128-unified,split-enc}` +
  `calib-ar128.npz`; `onnx/qwen3vl-4b-{aimet-split,aimet-splitkv,text,
  text-split,vit}`; `dlc/` and `ctxbin/` for `-split` and `-splitkv`.
  326 GB free. `~/llm-deploy` in sync with local main (`850126a`).
- **local** `~/llm-local/`: `work/ctxbin/qwen3vl-4b-w8a16-splitkv/`,
  `work/onnx/qwen3vl-4b-{text,text-split,aimet-split,aimet-splitkv,vit}`,
  `work/kit/` (6 wx_* images + `device_captions.json`),
  `work/kit-candidates/selection.json`, ViT ctx-bin
  `work/ctxbin/qwen3vl-4b-vit-w8a16/`. C: 180 GB free.
  ⚠ **The shipped v2 tree is `hf-staging-v2/qwen3vl_4b_e2e_pipeline_v2/`
  (58 files, matches HF), NOT `bundles/qwen3vl_4b_e2e_pipeline/`** — the
  latter is an older 31-file build dir with no test kit and no decode-only
  fallback. Use the staging path for every v2 comparison (lint mutation
  tests, file-count diffs, caption baselines).
- **HF** `vinccniv/sa8797p-qwen3vl-4b-bundles`: currently **public** (report
  only — NEVER change visibility); layout `qwen3vl_4b_e2e_pipeline/` (16
  files, v1) + `qwen3vl_4b_e2e_pipeline_v2/` (58 files). v3 goes to
  `qwen3vl_4b_e2e_pipeline_v3/`.
- ViT tower, LUT, tokenizer, configs, runtime .so set: **unchanged from v2**.
  Graph names (`prefill_0/1`, `decode_0/1`), ctx-bin filenames
  (`qwen3vl-4b-w8a16_{1,2}_of_2.bin`) and all node configs stay byte-relevant
  — do not rename anything.

### What is deliberately NOT in scope

- No ViT rebuild, no config/topology changes, no LADE, no `--quant-head`.
- The strategy-loop "does the device actually select prefill for a 273-token
  prompt" question is device-only; it stays a documented unknown.
- The Stage-2 text-only bundle (`qwen3vl_4b_text_w8a16`) is not re-cut; the
  v3 e2e bundle gains `genie_dialog_qwen3vl_4b.json` so `genie-t2t-run` works
  in-bundle instead.

---

## Phase 0 — Sync, guards, coordination (local, ~10 min)

### Task 0.1: Branch, sync, space

- [ ] **Step 1:** `cd /mnt/x/code/llm-deploy && git status --short` — expect
  clean (untracked `reports/` drops are fine). `git log --oneline -1 origin/main`
  vs `main`; if main is ahead, `git push origin main` first.
- [ ] **Step 2:** `git checkout -b qwen3vl-v3-gqa`
- [ ] **Step 3:** Disk guards: `df -h /mnt/c` (need ≥ 40 GB; have ~180),
  `ssh tank df -h /home` (need ≥ 80 GB; have ~326). Source `scripts/env.sh`
  and run `disk_guard 20`.
- [ ] **Step 4:** Coordination check (docs/NOTES-coordination.md):
  `$PY_DEPLOY scripts/util/coord.py who` locally AND
  `ssh tank '~/llm-local/envs/qwen3-deploy/bin/python ~/llm-deploy/scripts/util/coord.py who'`
  (adjust python path to tank's env if it differs — `scripts/env.sh` on tank
  resolves it). If another session claims the VL tower, STOP and report.

---

## Phase 1 — Wire `grouped_gqa` through the VL export chain (local, ~1 h, TDD)

**Files:**
- Create: `scripts/validate/test_vl_grouped_gqa.py`
- Modify: `scripts/export/modeling_export.py:337-364`
- Modify: `scripts/quant/quantize_aimet.py:398-401`
- Modify: `scripts/export/export_qwen3vl_text.py` (argparse + `from_hf_vl_text` call)

### Task 1.1: Failing test — tiny fake VL model, no 4B weights needed

- [ ] **Step 1: Write the test.** Create `scripts/validate/test_vl_grouped_gqa.py`:

```python
#!/usr/bin/env python
"""Prove from_hf_vl_text actually forwards grouped_gqa.

The VL chain shipped v2 with 36 replication ops per shard because this factory
silently dropped the flag (the quantizer HAS --grouped-gqa; the VL branch
could not pass it anywhere). This test runs in seconds on a tiny random tower:
no 4B checkpoint, no AIMET.
"""
import sys
import tempfile
from pathlib import Path
from types import SimpleNamespace

import onnx
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "export"))
from modeling_export import ExportQwen3, causal_mask, rope_tables  # noqa: E402

H, LAYERS, HEADS, KV, HD, VOCAB, INTER, NDEEP, S = 64, 2, 8, 2, 8, 512, 128, 3, 16

CFG = SimpleNamespace(hidden_size=H, num_hidden_layers=LAYERS,
                      num_attention_heads=HEADS, num_key_value_heads=KV,
                      head_dim=HD, vocab_size=VOCAB, intermediate_size=INTER,
                      rms_norm_eps=1e-6)


class FakeVL:
    """Just enough of Qwen3VLForConditionalGeneration for from_hf_vl_text."""
    def __init__(self, sd):
        self.config = SimpleNamespace(text_config=CFG)
        self._sd = sd

    def state_dict(self):
        return self._sd


def make_state_dict():
    torch.manual_seed(0)
    p = "model.language_model."
    sd = {p + "embed_tokens.weight": torch.randn(VOCAB, H) * 0.02,
          p + "norm.weight": torch.ones(H)}
    for i in range(LAYERS):
        s = f"{p}layers.{i}."
        sd[s + "input_layernorm.weight"] = torch.ones(H)
        sd[s + "post_attention_layernorm.weight"] = torch.ones(H)
        sd[s + "self_attn.q_norm.weight"] = torch.ones(HD)
        sd[s + "self_attn.k_norm.weight"] = torch.ones(HD)
        sd[s + "self_attn.q_proj.weight"] = torch.randn(HEADS * HD, H) * 0.02
        sd[s + "self_attn.k_proj.weight"] = torch.randn(KV * HD, H) * 0.02
        sd[s + "self_attn.v_proj.weight"] = torch.randn(KV * HD, H) * 0.02
        sd[s + "self_attn.o_proj.weight"] = torch.randn(H, HEADS * HD) * 0.02
        sd[s + "mlp.gate_proj.weight"] = torch.randn(INTER, H) * 0.02
        sd[s + "mlp.up_proj.weight"] = torch.randn(INTER, H) * 0.02
        sd[s + "mlp.down_proj.weight"] = torch.randn(H, INTER) * 0.02
    return sd


def export_and_count_expand(model):
    embeds = torch.zeros(1, 1, S, H)
    mask = causal_mask(S, S)
    cos, sin = rope_tables(torch.arange(S), HD, 10000.0)
    deep = [torch.zeros(1, 1, S, H) for _ in range(NDEEP)]
    with tempfile.TemporaryDirectory() as d, torch.no_grad():
        path = Path(d) / "m.onnx"
        torch.onnx.export(model, (embeds, mask, cos, sin, *deep), str(path),
                          opset_version=17, dynamo=False)
        g = onnx.load(str(path)).graph
        return sum(1 for n in g.node if n.op_type == "Expand")


def main():
    sd = make_state_dict()
    rep = ExportQwen3.from_hf_vl_text(FakeVL(sd), use_past=False,
                                      n_deepstack=NDEEP)
    grp = ExportQwen3.from_hf_vl_text(FakeVL(sd), use_past=False,
                                      n_deepstack=NDEEP, grouped_gqa=True)

    # 1. numerics: the grouped form is the same math, reshaped
    torch.manual_seed(1)
    embeds = torch.randn(1, 1, S, H) * 0.02
    mask = causal_mask(S, S)
    cos, sin = rope_tables(torch.arange(S), HD, 10000.0)
    deep = [torch.randn(1, 1, S, H) * 0.02 for _ in range(NDEEP)]
    with torch.no_grad():
        lo = rep(embeds, mask, cos, sin, *deep)[0]
        lg = grp(embeds, mask, cos, sin, *deep)[0]
    d = (lo - lg).abs().max().item()
    assert d < 1e-5, f"grouped vs replicating logits diverge: max|d|={d:.3e}"

    # 2. topology: grouped export has ZERO Expand nodes; replicating has 2/layer
    n_rep, n_grp = export_and_count_expand(rep), export_and_count_expand(grp)
    assert n_rep == 2 * LAYERS, f"control: expected {2*LAYERS} Expand, got {n_rep}"
    assert n_grp == 0, f"grouped export still has {n_grp} Expand nodes"
    print(f"OK  max|dlogits|={d:.3e}  Expand: replicating={n_rep} grouped=0")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Run it — must FAIL on the factory signature.**
  `$PY_DEPLOY scripts/validate/test_vl_grouped_gqa.py`
  Expected: `TypeError: from_hf_vl_text() got an unexpected keyword argument
  'grouped_gqa'`.

### Task 1.2: The three-line wiring fix

- [ ] **Step 1:** `scripts/export/modeling_export.py` — extend the factory
  signature (line 337-338) and forward the flag (line 362-364):

```python
    @staticmethod
    def from_hf_vl_text(hf_vl, fuse_gate_up=False, fuse_qkv=False, use_past=True,
                        logits_last_only=False, n_deepstack=3, layer_range=None,
                        grouped_gqa=False):
```

```python
        m = ExportQwen3(cfg, fuse_gate_up, fuse_qkv, use_past, logits_last_only,
                        input_embeds=True, n_deepstack=n_deepstack,
                        n_layers=end - start, is_first=is_first, is_last=is_last,
                        grouped_gqa=grouped_gqa)
```

- [ ] **Step 2:** `scripts/quant/quantize_aimet.py` — the VL branch of
  `build_wrapper` (line 398-401) forwards the existing flag:

```python
        if args.vl_text:
            return ExportQwen3.from_hf_vl_text(
                hf, args.fuse_gate_up, args.fuse_qkv, use_past=use_past,
                logits_last_only=False, n_deepstack=args.n_deepstack,
                grouped_gqa=args.grouped_gqa)
```

- [ ] **Step 3:** `scripts/export/export_qwen3vl_text.py` — add the flag to
  `main()`'s argparse (after `--parity-check`, line 248-249):

```python
    ap.add_argument("--grouped-gqa", action="store_true",
                    help="batch the attention MatMuls over the 8 KV heads "
                         "instead of replicating them to 32 (4x). MANDATORY "
                         "for any shipping VL build; lint_gqa_ops.py gates it.")
```

  and forward it at the `from_hf_vl_text` call (line 309-312):

```python
        model = ExportQwen3.from_hf_vl_text(
            hf, use_past=True, logits_last_only=False,
            n_deepstack=args.n_deepstack if s == 0 else 0,
            layer_range=(s, e) if split else None,
            grouped_gqa=args.grouped_gqa)
```

- [ ] **Step 4: Run the test — must PASS.**
  `$PY_DEPLOY scripts/validate/test_vl_grouped_gqa.py`
  Expected: `OK  max|dlogits|=...e-06  Expand: replicating=4 grouped=0`
- [ ] **Step 5: Regression-check the 0.6B path is untouched:**
  `$PY_DEPLOY -c "import sys; sys.path.insert(0,'scripts/export'); import modeling_export; print('import ok')"`
  and `git diff --stat` shows exactly the three files + the new test.
- [ ] **Step 6: Commit.**

```bash
git add scripts/export/modeling_export.py scripts/quant/quantize_aimet.py \
        scripts/export/export_qwen3vl_text.py scripts/validate/test_vl_grouped_gqa.py
git commit -m "fix(vl): wire --grouped-gqa through from_hf_vl_text and both VL exporters"
```

---

## Phase 2 — 4 KB blob padding + lint update (local, ~1 h)

**Files:**
- Modify: `scripts/pipeline/preprocess_image.py:41,156-166`
- Modify: `scripts/pipeline/build_test_kit.py` (raw write + sidecar)
- Modify: `scripts/validate/lint_pipeline_bundle.py:93,473-479,1182-1183`

### Task 2.1: Pad at generation time

- [ ] **Step 1:** `preprocess_image.py` — below `RAW_BYTES` (line 41) add:

```python
# +4 KB of zero padding on every shipped blob. GenieNode_setData allocates
# exactly fileSize bytes and a consumer over-reads one byte past 0x300000 into
# Scudo's guard page -> SIGSEGV (SEGV_ACCERR) on device, 2026-08-15. The
# runtime consumes only the first RAW_BYTES; the padding just moves the guard
# page out of reach. See docs/DEVICE_TEST_qwen3vl_imgenc_sigsegv.md §1.2b/§4.
PAD_BYTES = 4096
```

  and change the write (lines 156-159) to:

```python
    out = Path(args.out)
    out.write_bytes(q.tobytes() + b"\x00" * PAD_BYTES)
    got = out.stat().st_size
    assert got == RAW_BYTES + PAD_BYTES, f"{got} bytes != {RAW_BYTES}+{PAD_BYTES}"
```

  and add `"pad_bytes": PAD_BYTES,` to the sidecar dict (keep `"bytes":
  RAW_BYTES` — it describes the tensor payload).
- [ ] **Step 2:** `build_test_kit.py` — extend the import to include
  `PAD_BYTES`, change the blob write to:

```python
        raw.write_bytes(q.tobytes() + b"\x00" * PAD_BYTES)
        assert raw.stat().st_size == RAW_BYTES + PAD_BYTES, raw.stat().st_size
```

  and add `"pad_bytes": PAD_BYTES,` to its sidecar dict too.
- [ ] **Step 3: Verify on a real image (encoder unchanged, local ViT bin):**

```bash
$PY_DEPLOY scripts/pipeline/preprocess_image.py \
    --model $LLMDEPLOY_DATA/models/Qwen3-VL-4B-Instruct \
    --image $LLMDEPLOY_DATA/bundles/qwen3vl_4b_e2e_pipeline/sample_image.png \
    --out /tmp/claude-1000/-mnt-x-code-llm-deploy/0db36e5c-7da7-4885-9c25-a3950f13c354/scratchpad/pad_test.raw \
    --encodings $LLMDEPLOY_DATA/bundles/qwen3vl_4b_e2e_pipeline/qwen3vl-4b-vit-w8a16_ctx.info.json
stat -c%s /tmp/claude-1000/-mnt-x-code-llm-deploy/0db36e5c-7da7-4885-9c25-a3950f13c354/scratchpad/pad_test.raw   # must print 3149824
cmp -n 3145728 /tmp/claude-1000/-mnt-x-code-llm-deploy/0db36e5c-7da7-4885-9c25-a3950f13c354/scratchpad/pad_test.raw \
    $LLMDEPLOY_DATA/bundles/qwen3vl_4b_e2e_pipeline/sample_image.raw   # payload byte-identical to v2
```

  (Payload identity holds because the ViT encoding is unchanged; if `cmp`
  differs, STOP — the encoding drifted and the kit references are stale.)

### Task 2.2: Teach the bundle lint the padded size

- [ ] **Step 1:** `lint_pipeline_bundle.py` — under `RAW_BYTES` (line 93) add
  `PAD_BYTES = 4096` with a one-line pointer to the runbook. In
  `check_sample_image` (line 473-479):

```python
    got = raw.stat().st_size
    if got != RAW_BYTES + PAD_BYTES:
        rep.fail(f"{raw_name}: {got} bytes != {RAW_BYTES}+{PAD_BYTES} "
                 "(payload + the 4 KB guard-page padding; an unpadded blob "
                 "re-ships the GenieNode_setData SIGSEGV — imgenc runbook §4)")
    else:
        rep.ok(f"{raw_name}: {got} bytes == payload+{PAD_BYTES} pad")
```

  and in `check_kit` (line 1182-1183) the same expectation:

```python
        if blob.is_file() and blob.stat().st_size != RAW_BYTES + PAD_BYTES:
            rep.fail(f"{blob.name}: {blob.stat().st_size} bytes != "
                     f"{RAW_BYTES}+{PAD_BYTES} (unpadded blob re-ships the "
                     "setData SIGSEGV)")
```

- [ ] **Step 2: Mutation test — the lint must now REJECT v2:**

```bash
$PY_DEPLOY scripts/validate/lint_pipeline_bundle.py \
    --bundle $LLMDEPLOY_DATA/bundles/qwen3vl_4b_e2e_pipeline \
    --model $LLMDEPLOY_DATA/models/Qwen3-VL-4B-Instruct 2>&1 | grep -c "3145728"
```

  Expected: ≥ 7 failures (sample_image.raw + 6 kit blobs, all unpadded). A
  lint that still passes v2 is vacuous — do not proceed until it fails.
- [ ] **Step 3: Commit.**

```bash
git add scripts/pipeline/preprocess_image.py scripts/pipeline/build_test_kit.py \
        scripts/validate/lint_pipeline_bundle.py
git commit -m "fix(vl): pad every image blob +4KB past the setData over-read; lint enforces it"
```

### Task 2.3: Make the GQA gate automatic in the VL ctx-bin build

**Why this is a task and not a doc line.** v2 shipped 36 replication ops per
shard even though `lint_gqa_ops.py` already existed — the gate was never
pointed at this tower. `CLAUDE.md` prose is the mitigation that already
failed once. `vl_text_ctxbin_split.sh` is where the other structural gates
live (shape readback, HTP-config bind, shared-weights floor, Genie load sim),
so the topology assertion belongs there too, running on the DLCs it just
converted, before any ctx-bin is generated.

**Files:** Modify `scripts/build/vl_text_ctxbin_split.sh` (after the four
`convert` calls and `ls -lh "$DLC"`, before the ctx-bin generation loop).

- [ ] **Step 1:** Insert the gate:

```bash
# GQA topology gate. The v2 tower shipped with 36 KV-replication ops per shard
# (4:1 head ratio) because --grouped-gqa could not reach the VL exporters at
# all; nothing downstream notices -- the build succeeds, parity passes (the two
# forms are numerically identical) and the ctx-bin loads. The only symptom is a
# slow device. Assert it here, on the DLCs just converted.
#
# --layers is PER SHARD (each holds $((LAYERS / 2)) of the 36), and the default
# is 28 (the 0.6B tower): omitting it demands 56 MatMuls and fails a correct
# shard. GQA_EXPECT=replicating deliberately builds the old topology.
echo "== GQA topology gate (grouped attention, per-shard) =="
GQA_FLAGS=(--layers $((LAYERS / 2)) --n-kv "$NKV")
[ "${GQA_EXPECT:-grouped}" = "replicating" ] && GQA_FLAGS+=(--expect-replicating)
PATH="$(dirname "$PY_QAIRT"):$PATH" \
$PY_DEPLOY "$LLMDEPLOY_ROOT/scripts/validate/lint_gqa_ops.py" "${GQA_FLAGS[@]}" \
    "$DLC"/prefill_0.dlc "$DLC"/decode_0.dlc "$DLC"/prefill_1.dlc "$DLC"/decode_1.dlc
```

  `set -euo pipefail` is already in force, so a nonzero exit aborts the build
  before a bad ctx-bin exists — same failure discipline as the readback gate.
  `PATH` is prefixed because `lint_gqa_ops.py` shells out to `qairt-dlc-info`,
  whose system python lacks numpy on tank.

- [ ] **Step 2: Prove the gate fires both ways, using the DLCs already on
  disk** (no rebuild needed — the OLD un-grouped splitkv DLCs are the perfect
  negative control, and this also confirms the `--layers 18` arithmetic):

```bash
source scripts/env.sh && export PATH="$(dirname "$PY_QAIRT"):$PATH"
# negative control: old tower, grouped expectation -> must FAIL (36 repl ops, batch 32)
$PY_DEPLOY scripts/validate/lint_gqa_ops.py --layers 18 --n-kv 8 \
    $LLMDEPLOY_DATA/work/dlc/qwen3vl-4b-w8a16-splitkv/decode_0.dlc; echo "exit=$?"
# same file, replicating expectation -> must PASS, proving --layers 18 is the
# right arithmetic and the earlier FAIL was topology, not a miscount
$PY_DEPLOY scripts/validate/lint_gqa_ops.py --layers 18 --n-kv 8 --expect-replicating \
    $LLMDEPLOY_DATA/work/dlc/qwen3vl-4b-w8a16-splitkv/decode_0.dlc; echo "exit=$?"
```

  Expect exit 1 then exit 0. If the second also fails, the `--layers`/`--n-kv`
  values are wrong for this tower — fix them before wiring the gate, or Phase
  3.6 will report a false failure on a good build.
- [ ] **Step 3: Commit.**

```bash
git add scripts/build/vl_text_ctxbin_split.sh
git commit -m "gate(vl): assert grouped attention on all four DLCs at ctx-bin build time"
```

---

## Phase 3 — Rebuild the text tower with grouped GQA (tank, ~1 day wall)

New quant lineage (`-gqa` names). The grouped graph has different attention
tensors, so encodings CANNOT be adopted from v2 — full recalibration, calib
npz reused. Ctx-bin FILENAMES stay `qwen3vl-4b-w8a16_{1,2}_of_2.bin` (NAME
arg unchanged); only the work-dir names carry `-gqa`.

### Task 3.1: Sync the branch to tank

- [ ] **Step 1:** `git push ssh://tank/home/vinc/llm-deploy qwen3vl-v3-gqa:refs/heads/qwen3vl-v3-gqa`
- [ ] **Step 2:** `ssh tank 'cd ~/llm-deploy && git checkout qwen3vl-v3-gqa && git log --oneline -2'`
  — must show the Phase 1+2 commits. (A stale tank tree silently ships old
  attention — this is the exact failure `lint_gqa_ops.py` exists for.)
- [ ] **Step 3:** Smoke the wiring ON TANK (same tiny test, no GPU needed):
  `ssh tank 'cd ~/llm-deploy && source scripts/env.sh && $PY_DEPLOY scripts/validate/test_vl_grouped_gqa.py'`
  Expected: `OK ... grouped=0`.
- [ ] **Step 4:** Cross-check against the v2 run. `~/.bash_history` on tank is
  **empty** — the authoritative record is `~/llm-local/logs/` (`splitkv.log`,
  `prefillkv-export.log`, `dsp-unify.log`, `fp32-export.log`, `e2e-gate.log`,
  `kit-captions.log`, all 2026-08-15). Each ends with the `/usr/bin/time -v`
  "Command being timed:" line giving the verbatim invocation. Diff those
  against the commands below; if v2 used a flag this plan lacks (or vice
  versa), STOP and reconcile before burning tank hours.

⚠ **Scripts are not executable on tank.** The local repo lives on a
Windows-backed mount where every file reads 0777, so the exec bit never enters
git; on tank the same files land 0664 and `scripts/build/foo.sh` fails with
`Permission denied` (exit 126). **Invoke every build script as `bash
scripts/build/foo.sh ...`.** All tank commands in this plan already do.

**v2 reference wall-times** (same host, same model, for sanity-checking
progress): past-KV prefill export **38 min**; split + 4 conversions + 2
ctx-bins + load-sim **~74 min** (07:05→08:19). The full `vl_text_build.sh`
(calibration + prefill quant + decode export) is the long pole and has no v2
log here — expect it to dominate.

### Task 3.2: Quantize (calibrate + prefill + decode) — the long step

- [ ] **Step 1:** Launch under nohup (survives ssh drops); `QUANT_DEVICE=cpu`
  is automatic on tank:

```bash
ssh tank 'cd ~/llm-deploy && source scripts/env.sh && mkdir -p ~/llm-local/logs && \
  nohup scripts/build/vl_text_build.sh qwen3vl-4b-w8a16-gqa 128 2048 --grouped-gqa \
  > ~/llm-local/logs/vl_gqa_quant.log 2>&1 & echo started'
```

  This reuses `qwen3vl-4b-calib-ar128.npz`, runs prefill quant with `--eval`,
  then the decode export on the SAME encodings lineage, then filter + rename.
  Expected wall: several hours (poll, don't sit on it — Phase 4.1 and Phase 5
  prep run locally in parallel).
- [ ] **Step 2:** Poll `tail -5 ~/llm-local/logs/vl_gqa_quant.log` until
  `QUANTIZATION COMPLETE`. **Gate: the `--eval` line must read 4/4** (v2's
  bar). 3/4 = STOP and report (grouped quantsim interaction — do not ship a
  quality regression to fix a speed bug). Record the eval line.
- [ ] **Step 3:** Sanity-read the exported decode ONNX input list from the log
  (36 layers, `past_key_35_in [1,8,128,2175]`, deepstack present).

### Task 3.3: Past-KV prefill export on the new lineage

- [ ] **Step 1:** (tank) Mirror the v2 prefillkv recipe with the new lineage +
  flag:

```bash
ssh tank 'cd ~/llm-deploy && source scripts/env.sh && \
  nohup $PY_DEPLOY scripts/quant/quantize_aimet.py \
    --model $LLMDEPLOY_DATA/models/Qwen3-VL-4B-Instruct \
    --cl-prefill 128 --ctx 2048 --decode-ar 128 \
    --export-decode $LLMDEPLOY_DATA/work/quant/qwen3vl-4b-w8a16-gqa-prefill \
    --out $LLMDEPLOY_DATA/work/quant/qwen3vl-4b-w8a16-gqa-prefillkv128 \
    --vl-text --n-deepstack 3 \
    --vl-calib $LLMDEPLOY_DATA/work/quant/qwen3vl-4b-calib-ar128.npz \
    --device cpu --grouped-gqa \
    > ~/llm-local/logs/vl_gqa_prefillkv.log 2>&1 & echo started'
```

- [ ] **Step 2:** (tank) Canonical rename with past-KV names:

```bash
$PY_DEPLOY scripts/quant/rename_aimet_io.py \
    --model $LLMDEPLOY_DATA/work/quant/qwen3vl-4b-w8a16-gqa-prefillkv128/model.onnx \
    --encodings $LLMDEPLOY_DATA/work/quant/qwen3vl-4b-w8a16-gqa-prefill/model_filtered.encodings \
    --layers 36 --vl-text --n-deepstack 3 --with-past
```

- [ ] **Step 3:** (tank) Deepstack rename, prefill only (`_p` suffix — the
  shared-buffer memset hazard):

```bash
$PY_DEPLOY scripts/quant/rename_deepstack_inputs.py \
    --model $LLMDEPLOY_DATA/work/quant/qwen3vl-4b-w8a16-gqa-prefillkv128/model_renamed.onnx \
    --n-deepstack 3 --suffix _p
```

  **Confirmed against the v2 run** (`~/llm-local/logs/dsp-unify.log`, tank,
  2026-08-15): this writes a NEW file `model_renamed_dsp.onnx` beside the
  input — it is not in-place. Expected output lines: `no encodings reference
  [...] (float inputs, as expected)`, `onnx.checker clean`, `renamed 3
  deepstack inputs`. That `_dsp` filename is what Task 3.4 must consume.
- [ ] **Step 4:** Verify the prefillkv export's IO from the rename output:
  `attention_mask [1,128,2176]`, `past_key_0_in [1,8,128,2048]`,
  `logits [1,128,151936]`, `deepstack_visual_embed_{0,1,2}_p`.

### Task 3.4: Weight unification + hash gate

⚠ **`unify_pair_weights.py` takes FILE paths, not directories** — its own
usage line is `--prefill <prefill/model_renamed.onnx> --decode
<decode/model_renamed.onnx> --out <decode_unified.onnx>`. Passing directories
fails. Confirmed against the v2 run (`~/llm-local/logs/dsp-unify.log`).

- [ ] **Step 1:** (tank) Unify decode onto the prefill lineage weights:

```bash
Q=$LLMDEPLOY_DATA/work/quant
mkdir -p $Q/qwen3vl-4b-w8a16-gqa-decode-unified
$PY_DEPLOY scripts/quant/unify_pair_weights.py \
    --prefill $Q/qwen3vl-4b-w8a16-gqa-prefill/model_renamed.onnx \
    --decode  $Q/qwen3vl-4b-w8a16-gqa-decode/model_renamed.onnx \
    --out     $Q/qwen3vl-4b-w8a16-gqa-decode-unified/model_renamed.onnx
```

- [ ] **Step 2:** (tank) Same for prefillkv — note the `--decode` input is the
  **`_dsp`** file from Task 3.3 Step 3, not `model_renamed.onnx`. Getting this
  wrong silently unifies the wrong graph and loses the deepstack rename:

```bash
mkdir -p $Q/qwen3vl-4b-w8a16-gqa-prefillkv128-unified
$PY_DEPLOY scripts/quant/unify_pair_weights.py \
    --prefill $Q/qwen3vl-4b-w8a16-gqa-prefill/model_renamed.onnx \
    --decode  $Q/qwen3vl-4b-w8a16-gqa-prefillkv128/model_renamed_dsp.onnx \
    --out     $Q/qwen3vl-4b-w8a16-gqa-prefillkv128-unified/model_renamed.onnx
```

- [ ] **Step 3:** Hash gate — the script reports it itself. v2's reference
  output was `274/398 shared initializers differ before unification` then
  `replaced 398 shared initializers; 0 left untouched`, followed by per-tensor
  `MATCH=True` samples. **Require `0 left untouched` and every sampled
  `MATCH=True`.** A nonzero "left untouched" means the two graphs disagree on
  dtype/dims for some tensor — investigate, do not proceed. (This is what makes
  ctx-bin weight pooling work; without it v2 measured 7.7 GB instead of 4.3 GB.)

### Task 3.5: Split ONNX + encodings

⚠ **v2 did NOT split its decode graphs — it symlinked them** from the Stage-2
`qwen3vl-4b-aimet-split/decode_{0,1}` build, because only the prefill was being
re-exported (`~/llm-local/logs/splitkv.log`, tank, 2026-08-15, "link the decode
chunks (same lineage, unchanged since Stage 2)"). **For v3 that shortcut is
forbidden**: the decode graph is exactly what carries the grouped attention
being fixed, so it must be a real split of the new unified decode. If any
`decode_*.onnx` under the v3 split dir is a symlink, the build is wrong.

- [ ] **Step 1:** (tank) Split both unified graphs at layer 18. The seam name
  is **confirmed** as `/layers.17/Add_1_output_0` (v2 log: `seam
  /layers.17/Add_1_output_0 typed as [1, 128, 2560]`):

```bash
$PY_DEPLOY scripts/quant/split_aimet_onnx.py \
    --onnx $LLMDEPLOY_DATA/work/quant/qwen3vl-4b-w8a16-gqa-prefillkv128-unified/model_renamed.onnx \
    --seam '/layers.17/Add_1_output_0' --split-at 18 --layers 36 --with-past \
    --out-dir $LLMDEPLOY_DATA/work/onnx/qwen3vl-4b-gqa-aimet-splitkv --tag prefill
$PY_DEPLOY scripts/quant/split_aimet_onnx.py \
    --onnx $LLMDEPLOY_DATA/work/quant/qwen3vl-4b-w8a16-gqa-decode-unified/model_renamed.onnx \
    --seam '/layers.17/Add_1_output_0' --split-at 18 --layers 36 --with-past \
    --out-dir $LLMDEPLOY_DATA/work/onnx/qwen3vl-4b-gqa-aimet-splitkv --tag decode
```

  Result: `{prefill_0,prefill_1,decode_0,decode_1}/<g>.onnx`, each in its own
  subdir (external-data filename collision rule).
- [ ] **Step 2:** (tank) Split the encodings (one calibration, two chunks;
  `--renumber` OFF — extract_model preserves global names):

```bash
$PY_DEPLOY scripts/quant/split_encodings.py \
    --encodings $LLMDEPLOY_DATA/work/quant/qwen3vl-4b-w8a16-gqa-prefill/model_filtered_renamed.encodings \
    --split-at 18 --layers 36 \
    --out-dir $LLMDEPLOY_DATA/work/quant/qwen3vl-4b-w8a16-gqa-split-enc
```

- [ ] **Step 3:** Split parity (bit-identity, chunked vs whole):
  `$PY_DEPLOY scripts/validate/parity_vl_text_split.py --model $LLMDEPLOY_DATA/models/Qwen3-VL-4B-Instruct --split-at 18`
  (run on tank — check its other args with `--help` and point it at the new
  gqa split; the bar is max|d| = 0.0 on logits and all 72 KV outs, same as v2).

### Task 3.6: Convert + ctx-bins + readback gates

- [ ] **Step 1:** (tank) The whole conversion + generation + shape/HTP/shared
  readback + Genie load simulation is one gated script; only the dirs are new:

```bash
ssh tank 'cd ~/llm-deploy && source scripts/env.sh && \
  ONNX=$LLMDEPLOY_DATA/work/onnx/qwen3vl-4b-gqa-aimet-splitkv \
  ENCDIR=$LLMDEPLOY_DATA/work/quant/qwen3vl-4b-w8a16-gqa-split-enc \
  DLC=$LLMDEPLOY_DATA/work/dlc/qwen3vl-4b-w8a16-gqa-splitkv \
  CTXBIN=$LLMDEPLOY_DATA/work/ctxbin/qwen3vl-4b-w8a16-gqa-splitkv \
  PREFILL_PAST=2048 DSP_SUFFIX=_p \
  nohup scripts/build/vl_text_ctxbin_split.sh qwen3vl-4b-w8a16 128 2048 18 \
  > ~/llm-local/logs/vl_gqa_ctxbin.log 2>&1 & echo started'
```

  NAME stays `qwen3vl-4b-w8a16` so the .bin filenames match the node configs.
  The script's own gates must all pass: graph names, mask `[1,128,2176]` /
  `[1,1,2176]`, KV shapes, deepstack `_p` on prefill_0 only, O=3/vtcm 16/
  4 HVX bound, `sharedWeightsSize ≥ {1.4, 2.0}` GB, `genie_load_check.py`
  PASS. Any failure = STOP; the log names the violated contract.
- [ ] **Step 2: THE gate this plan exists for** — zero replication ops in all
  four DLCs (qairt-dlc-info needs the env python first on PATH on tank):

```bash
ssh tank 'cd ~/llm-deploy && source scripts/env.sh && \
  export PATH="$(dirname "$PY_QAIRT"):$PATH" && \
  $PY_DEPLOY scripts/validate/lint_gqa_ops.py --layers 18 --n-kv 8 \
    $LLMDEPLOY_DATA/work/dlc/qwen3vl-4b-w8a16-gqa-splitkv/{prefill_0,decode_0,prefill_1,decode_1}.dlc'
```

  **`--layers 18` is mandatory and is not the default.** `lint_gqa_ops.py`
  defaults `--layers 28` (the 0.6B tower) and its pass criterion includes
  `len(matmuls) == 2 * layers`. Each VL split shard holds 18 layers = 36
  attention MatMuls, so the default demands 56 and a perfectly grouped shard
  reports **FAIL** — indistinguishable at a glance from the fix not working.
  (`--n-kv 8` happens to match the default; state it anyway so the assertion
  is explicit rather than coincidental.)

  Expected: `4 DLC(s) checked, 0 failing`, attention MatMul batch dim 8.
  As a control, run the SAME command (same `--layers 18`) against the OLD
  splitkv `decode_0.dlc` on tank — it must still FAIL with 36 replication ops
  and batch dim 32. A lint that passes both, or fails both, is broken.
- [ ] **Step 3:** Record the converter's `DDR bandwidth summary` block for all
  four graphs WITH log path + date (per the `read_total_bytes` rule — never
  quote one without its build log and date).

  **The v2 (un-grouped) baseline is already measured.** Source:
  `~/llm-local/logs/splitkv.log` on tank, 2026-08-15, in convert order:

  | graph | v2 `read_total_bytes` |
  |---|---|
  | `prefill_0` | 3,637,106,688 |
  | `decode_0` | 3,609,839,616 |
  | `prefill_1` | 4,413,063,168 |
  | `decode_1` | 4,387,743,744 |

  Extract the v3 figures the same way (`grep -n "read_total_bytes\|== convert"`)
  and report the deltas per graph. Note these are *converter estimates* for a
  build that has never run on device — they bound the byte traffic, they do not
  predict tok/s.
- [ ] **Step 4:** rsync the two ctx-bin dirs (bins + info.json + configs) to
  local `$LLMDEPLOY_DATA/work/ctxbin/qwen3vl-4b-w8a16-gqa-splitkv/` (~4.5 GB;
  `disk_guard 10` first).

---

## Phase 4 — Numerical gates (**tank**, serialized after Phase 3.2, ~2.5 h)

⚠ **Host correction.** This phase was originally scoped "local, in parallel";
that is wrong on two counts. (a) The 4B fp32 export peaks around 63.5 GB
(CLAUDE.md) against local's 47 GB, and v2 ran both the export and the gate on
tank (`~/llm-local/logs/fp32-export.log`, `e2e-gate.log`, 2026-08-15).
(b) It must NOT run *concurrently* with Phase 3's quantization either: AIMET's
legacy `sim.export` holds four fp32 copies of the graph at once, so the quant
job spikes well above its steady ~34 GB RSS and a 63.5 GB export alongside it
would OOM-kill both. **Wait for `QUANTIZATION COMPLETE`, then run this.**

v2 reference wall-times on tank: fp32 export ~30 min; e2e gate **4333 s
(72 min)** for all 5 chains at n=273 prompt rows.

### Task 4.1: fp32 exports with the grouped topology (tank)

- [ ] **Step 1:** Re-export the gate's fp32 text ONNX — base pair AND the
  past-KV prefill — into a NEW dir (the old `qwen3vl-4b-text` stays as the v2
  record):

```bash
$PY_DEPLOY scripts/export/export_qwen3vl_text.py \
    --model $LLMDEPLOY_DATA/models/Qwen3-VL-4B-Instruct \
    --out $LLMDEPLOY_DATA/work/onnx/qwen3vl-4b-text-gqa --grouped-gqa --parity-check
$PY_DEPLOY scripts/export/export_qwen3vl_text.py \
    --model $LLMDEPLOY_DATA/models/Qwen3-VL-4B-Instruct \
    --out $LLMDEPLOY_DATA/work/onnx/qwen3vl-4b-text-gqa --grouped-gqa \
    --prefill-past 2048
```

  (`disk_guard 40` first — ~31 GB of external data per run; the second run adds
  `prefillkv/` beside `prefill/` and `decode/`.) `--parity-check` gives the
  wrapper-vs-HF max|dlogits| gate for free on the grouped topology; it is
  rejected for split exports, which is fine here since these are unsplit.
  v2's export produced exactly these graph dirs: `prefillkv` (S=128,
  past=2048), `decode` (S=1, past=2175), `prefill` (S=128, past=0).
- [ ] **Step 2:** Structural check on the real export: zero Expand nodes.

```bash
$PY_DEPLOY - <<'EOF'
import onnx
for g in ("prefill", "decode", "prefillkv"):
    p = f"/home/vinc/llm-local/work/onnx/qwen3vl-4b-text-gqa/{g}/{g}.onnx"
    n = sum(1 for x in onnx.load(p, load_external_data=False).graph.node
            if x.op_type == "Expand")
    print(g, "Expand:", n); assert n == 0, (g, n)
EOF
```

### Task 4.2: Full e2e gate — no `--chains` filter

- [ ] **Step 1:**

```bash
$PY_DEPLOY scripts/validate/parity_e2e_vl.py \
    --model $LLMDEPLOY_DATA/models/Qwen3-VL-4B-Instruct \
    --vit-onnx $LLMDEPLOY_DATA/work/onnx/qwen3vl-4b-vit/vit.onnx \
    --text-onnx $LLMDEPLOY_DATA/work/onnx/qwen3vl-4b-text-gqa \
    --caption-out $LLMDEPLOY_DATA/work/kit/captions/sample_image_v3.json
```

  (prefillkv defaults to `<text-onnx>/prefillkv/prefillkv.onnx`.) Running the
  full chain list is mandatory — a `--chains` subset skips the mutation checks
  (CLAUDE.md).

  **The gate has 6 chains and only 3 are gated token-exact.** Read from the
  source (`ALL_CHAINS`, `GATED_EXACT`, `CHAIN2_MIN_AGREE` in
  `parity_e2e_vl.py`), with v2's measured results
  (`~/llm-local/logs/e2e-gate.log`, tank, 2026-08-15):

  | chain | bar | v2 result |
  |---|---|---|
  | `chain0-alldecode` | token-exact | 20/20 |
  | `chain0b-prefillkv` — **the device path** | token-exact | 20/20 |
  | `chain1-hf-vit` | token-exact | 20/20 |
  | `chain2-onnx-vit` | **≥75% step agreement**, text for human review | 20/20 |
  | `tierA-zero-deep` | **not gated** — historical bertcache reference | 0/20 |
  | `tierB-prefillkv-zero-deep` | **not gated** | (post-dates v2's gate run) |

  ⚠ **A 0/20 on the two tier chains is EXPECTED, not a failure.** They feed
  zeroed deepstack because a stock Genie pipeline has no deepstack path; the
  wording legitimately differs from HF and that gap *is* the defined
  degradation. Never "fix" it. Equally, do not read `chain2`'s floor as
  token-exact — it carries the ViT's own fp32 trace delta by design.

  ⚠ **`tierB-prefillkv-zero-deep` — not tierA — is the caption the device
  actually produces.** It is chain0b's real chunk plan (three AR=128 prefill
  calls, `n_process` 128/128/17) with zeroed deepstack. tierA reaches a similar
  place through the *bertcache* path and is kept only as the historical
  reference; v2's gate log labels tierA as the device text because tierB did
  not exist yet, and that label is now stale. Take the v3 sample caption and
  every kit caption from **tierB**.
- [ ] **Step 2:** Record the sample-image device-faithful caption from
  `--caption-out` — it goes into DEVICE_TEST.md as THE expected result of the
  single smoke test.
- [ ] **Step 3: Commit** any gate-support edits plus a short
  `reports/qwen3vl-v3-gate-results.md` stub with: eval 4/4 line, gate chain
  results, lint_gqa 4/4-clean, DDR before/after bytes.

---

## Phase 5 — Kit refresh: padded blobs + v3 captions (local, ~1.5 h)

The ViT encoding is unchanged, so blob payloads are identical to v2 — only
the padding and the caption references change. Images/selection are reused;
nothing is re-downloaded.

### Task 5.1: Regenerate device-faithful captions on the v3 numerics

- [ ] **Step 1:** For each of the 6 kit images (stems from
  `ls $LLMDEPLOY_DATA/work/kit/wx_*.jpg`):

```bash
for j in $LLMDEPLOY_DATA/work/kit/wx_*.jpg; do
  s=$(basename "$j" .jpg)
  $PY_DEPLOY scripts/validate/parity_e2e_vl.py \
      --model $LLMDEPLOY_DATA/models/Qwen3-VL-4B-Instruct \
      --vit-onnx $LLMDEPLOY_DATA/work/onnx/qwen3vl-4b-vit/vit.onnx \
      --text-onnx $LLMDEPLOY_DATA/work/onnx/qwen3vl-4b-text-gqa \
      --image "$j" --chains tierB-prefillkv-zero-deep \
      --caption-out $LLMDEPLOY_DATA/work/kit/captions/${s}.json
done
```

  **`tierB-prefillkv-zero-deep` is the chain** — confirmed against v2's
  `~/llm-local/logs/kit-captions.log`, whose per-image output reads
  `DEVICE-FAITHFUL (past-KV prefill + zero deepstack)` and shows the three
  chunk calls (`n_process` 128/128/17). Runs on tank at **~6 min/image**, so
  ~40 min for the six. Each run prints `PASS: ... (1 chains, n=273 prompt
  rows, N generated tokens)` — that PASS refers to the chain executing, not to
  matching HF, which tierB is deliberately not gated against.
- [ ] **Step 2:** Assemble `device_captions.json` (same shape as the v2 file —
  `{stem: caption}`) from the 6 outputs; diff against v2's: wording drift is
  expected (new quant lineage), semantic drift (wrong weather/scene) = STOP
  and investigate before shipping.

### Task 5.2: Rebuild the kit (now padded)

- [ ] **Step 1:**

```bash
$PY_DEPLOY scripts/pipeline/build_test_kit.py \
    --selection $LLMDEPLOY_DATA/work/kit-candidates/selection.json \
    --model $LLMDEPLOY_DATA/models/Qwen3-VL-4B-Instruct \
    --encodings $LLMDEPLOY_DATA/bundles/qwen3vl_4b_e2e_pipeline/qwen3vl-4b-vit-w8a16_ctx.info.json \
    --script configs/genie_pipeline_qwen3vl.script \
    --captions $LLMDEPLOY_DATA/work/kit/device_captions.json \
    --out $LLMDEPLOY_DATA/work/kit
```

  (Mirror v2's exact invocation if flags differ — `--help` + the captions dir
  are the record.)
- [ ] **Step 2:** Verify: every `wx_*.raw` is 3,149,824 B
  (`stat -c'%n %s' $LLMDEPLOY_DATA/work/kit/wx_*.raw`), sidecars carry
  `"pad_bytes": 4096`, `TEST_IMAGES.md` shows the v3 captions.

---

## Phase 6 — Bundle v3 + single-session device docs (local, ~2 h)

### Task 6.1: Add the text-only perf path to the bundle

- [ ] **Step 1:** `scripts/build/vl_pipeline_bundle.sh` — add
  `genie_dialog_qwen3vl_4b.json` to the `CONFIGS` array (line 79-90), with a
  comment: `# genie-t2t-run's dialog config: the text-only tok/s+TTFT
  measurement runs from THIS bundle, no second download`. Verify the config's
  referenced files (`tokenizer.json`, `embedding_float32_lut.bin`, the two
  `.bin`s) are all bundle-root names — `grep -o '"[^"]*\.\(bin\|json\)"' configs/genie_dialog_qwen3vl_4b.json`.
- [ ] **Step 2:** Run the lint on a dry re-cut later (Task 6.3) — if the lint
  objects to the extra config, wire it in as an explicitly-known file rather
  than weakening the check.

### Task 6.2: Rewrite `docs/DEVICE_TEST_qwen3vl_e2e.md` for ONE session

Structure (this doc ships in the bundle as DEVICE_TEST.md):

- [ ] **Step 1: §1 The one command.** untar → `chmod +x genie-app` →
  `LD_LIBRARY_PATH=. ./genie-app -s genie_pipeline_qwen3vl.script` →
  expected: a caption semantically matching the recorded v3 sample caption
  (quote it verbatim from Task 4.2 Step 2). State plainly: blobs are
  pre-padded; the 2026-08-15 setData SIGSEGV is expected to be gone.
- [ ] **Step 2: §2 If it still crashes at `node set image`** — the
  discriminator, no new files needed: `head -c 3145728 sample_image.raw >
  nopad.raw` recreates the unpadded probe; run the imgonly script pair per
  `DEVICE_TEST_qwen3vl_imgenc_sigsegv.md` §3-4 and report the branch (A-D
  table). Branch D = Qualcomm escalation material; capture the tombstone.
- [ ] **Step 3: §3 Weather kit loop** — the 6 `wx_*.script` runs, diff each
  caption against TEST_IMAGES.md (bar: semantic agreement on weather + scene).
- [ ] **Step 4: §4 Performance capture** (same session, same bundle):
  text-only `genie-t2t-run -c genie_dialog_qwen3vl_4b.json -p "<prompt>"`
  (give the exact prompt string) → record tok/s + TTFT — the FIRST real 4B
  text number; then note the e2e TTFT wall-clock of §1 (273-token prompt
  through prefill_0/1 — this observation doubles as the prefill-selection
  probe: TTFT ~3-5 s = prefill selected, ~30 s+ = all-decode fallback,
  report which).
- [ ] **Step 5: §5 Fallback** — `genie_pipeline_qwen3vl_decodeonly.script`
  one-file swap, unchanged from v2, still device-unvalidated (say so).
- [ ] **Step 6:** Update `docs/DEVICE_TEST_qwen3vl_imgenc_sigsegv.md` §6
  branch C row: the re-cut already shipped in v3 — on branch C the session
  just continues with the padded bundle it already has.

### Task 6.3: Cut bundle v3

**Prerequisite already resolved (2026-08-17).** `vl_pipeline_bundle.sh` hard-
requires `$LLMDEPLOY_DATA/bundles/qwen3vl_4b_vit_fp16/htp_backend_ext_config_vit.json`,
but that directory had been reclaimed — only `qwen3vl_4b_vit_fp16.tar.gz`
survived, so the script would have aborted at its source check. Restored with
`tar xzf bundles/qwen3vl_4b_vit_fp16.tar.gz -C bundles/ qwen3vl_4b_vit_fp16/htp_backend_ext_config_vit.json`
and verified byte-identical to the copy v2 shipped. The ViT ctx-bin itself
(`work/ctxbin/qwen3vl-4b-vit-w8a16/qwen3vl-4b-vit-w8a16_ctx.bin`, 433,101,160 B)
is present. If either goes missing again, both are recoverable from
`hf-staging-v2/qwen3vl_4b_e2e_pipeline_v2/`.

- [ ] **Step 1:** `disk_guard 16`, then:

```bash
TEXT_CTX=$LLMDEPLOY_DATA/work/ctxbin/qwen3vl-4b-w8a16-gqa-splitkv \
KIT=$LLMDEPLOY_DATA/work/kit \
scripts/build/vl_pipeline_bundle.sh qwen3vl_4b_e2e_pipeline_v3
```

  (Check how v2 passed TEXT_CTX — the script reads it as the DIRECTORY holding
  the two .bin files; if the gqa-splitkv dir nests `{1,2}_of_2/`, flatten with
  hard links first exactly as v2 did.) The script re-reads graph names from
  the final bins, regenerates prompt segments, quantizes+pads
  `sample_image.raw` against the shipped ViT info.json, copies the padded kit,
  and runs the full lint — which now enforces padded sizes, load-sim on both
  configs, kit closure.
- [ ] **Step 2:** Lint PASS is the gate. Then two mutation spot-checks on a
  COPY of the bundle: truncate one `wx_*.raw` to 3,145,728 → lint fails with
  the SIGSEGV message; delete `genie_dialog_qwen3vl_4b.json` → whatever check
  covers it fires (or record that none does and it's copy-only).
- [ ] **Step 3:** `tar tzf $LLMDEPLOY_DATA/bundles/qwen3vl_4b_e2e_pipeline_v3.tar.gz | wc -l`
  — record the count; sanity-diff the file list against v2's 58 (+
  `genie_dialog_qwen3vl_4b.json`, ± nothing unexpected).

### Task 6.4: Repo docs

- [ ] **Step 1:** `CLAUDE.md` — two surgical edits: (a) the Qwen3-VL line in
  Build chain gains "`--grouped-gqa` is mandatory here too —
  `vl_text_build.sh <name> <cl> <ctx> --grouped-gqa`; the flag must ALSO reach
  the past-KV prefill export"; (b) the validation-gates VL bullet gains
  "`lint_gqa_ops.py` on all four text DLCs — v2 shipped 36 replication ops per
  shard (4:1) because the VL chain was outside this gate".
- [ ] **Step 2:** `docs/REFERENCE.md` — §0 board: new row "VL-4B v3 (GQA-fixed
  tower + padded blobs) built, awaiting device"; corrections ledger #34: the
  v2 VL text tower shipped un-grouped (evidence: the lint output, this plan);
  note the DDR before/after bytes from Task 3.6 Step 3 with log path + date.
- [ ] **Step 3:** `docs/BUILD_GUIDE.md` — the VL text recipe gains
  `--grouped-gqa` in its command lines + one line on the padded-blob contract
  (blobs are payload+4096; preprocess/build_test_kit pad automatically).
- [ ] **Step 4: Commit** (docs + bundle-script change, separate from code
  commits):

```bash
git add CLAUDE.md docs/REFERENCE.md docs/BUILD_GUIDE.md \
        docs/DEVICE_TEST_qwen3vl_e2e.md docs/DEVICE_TEST_qwen3vl_imgenc_sigsegv.md \
        scripts/build/vl_pipeline_bundle.sh
git commit -m "docs+bundle: v3 one-session device test; VL gains the GQA gate and padded-blob contract"
```

---

## Phase 7 — Ship (local, ~1-2 h upload)

### Task 7.1: Upload to HF

- [ ] **Step 1:** Stage `qwen3vl_4b_e2e_pipeline_v3/` with hard links (~6.3 GB
  tree; the two text ctx-bins ~4.5 GB are genuinely new bytes, the ViT bin +
  .so set dedup against v2's blobs server-side).
- [ ] **Step 2:** Upload via
  `SOCKET_CHECKS=999999 scripts/util/hf_upload_watchdog.sh` (proxy drops long
  streams; the socket detector false-positives through it). Target folder
  `qwen3vl_4b_e2e_pipeline_v3/` in `vinccniv/sa8797p-qwen3vl-4b-bundles`.
  Mind the 128 commits/h Hub limit — one folder commit, not per-file commits.
- [ ] **Step 3:** Verify the uploaded bytes, not the local ones:
  `list_repo_files` count matches Task 6.3 Step 3; `get_paths_info` sizes on
  the two text .bins; re-download one text info.json and re-run
  `genie_load_check.py` against it.
- [ ] **Step 4:** Read and REPORT visibility
  (`HfApi().repo_info(...).private`) — **never change it**; if the upload
  flipped it, report and stop (standing rule, four incidents).

### Task 7.2: Merge + push + tank sync

- [ ] **Step 1:** Pre-push scan on the branch: no binaries/photos staged
  (`git diff --stat main...HEAD` — scripts/docs only), no device serial, no
  files > 200 KB.
- [ ] **Step 2:** `git checkout main && git merge --no-ff qwen3vl-v3-gqa && git push origin main`
- [ ] **Step 3:** `git push ssh://tank/home/vinc/llm-deploy main` and
  `ssh tank 'cd ~/llm-deploy && git checkout main'`.

### Task 7.3: Final report to the user

- [ ] One message: eval + gate numbers, lint_gqa before/after, DDR bytes
  before/after (with log paths), bundle file count + HF URL, visibility
  status, the exact single device command + expected caption, and the ordered
  device checklist (§1-§5 of DEVICE_TEST). Explicitly restate what is still
  unknowable off-device: absolute tok/s, prefill selection, ViT-on-silicon
  accuracy, branch-D residual risk.

---

## Risk register

| Risk | Mitigation | Residual |
|---|---|---|
| AIMET quantsim on the grouped 4B graph misbehaves (0.6B proved the path; 4B VL has not) | `--eval` 4/4 gate; full e2e gate token-exact; STOP conditions | Low |
| The padding is the wrong fix (runbook branch B/D) | Padding is the runbook's own prediction from disassembly; the bundle keeps the discriminator procedure (§2) and the decodeonly fallback | Branch D = Qualcomm escalation, not host-fixable |
| Encodings lineage divergence breaks weight pooling | unify + hash gate + `MIN_SHARED_GB` floors {1.4, 2.0} in the ctxbin script | ~0 |
| Flag silently not bound in some graph (the v2 failure class) | tank tree verified at the right commit; tiny test run ON tank; `lint_gqa_ops.py` on all four FINAL DLCs + a must-still-fail control on the old DLC | ~0 |
| Caption references drift semantically under the new quant | Task 5.1 Step 2 diff gate | Low |
| Upload 429 / visibility flip | one-commit upload, watchdog, report-only visibility rule | Low |
| C:/vhdx SIGBUS | `disk_guard` sized per step; heavy steps on tank; 180 GB current headroom | Low |
| Prefill never selected on device (TTFT stays ~30 s) | Unchanged from v2 (correctness unaffected); DEVICE_TEST §4 turns the observation into data | Accepted |

## STOP conditions

- `--eval` < 4/4 on the gqa prefill quant → report, do not proceed to split.
- `lint_gqa_ops.py` nonzero on ANY of the four new DLCs → find where the flag
  dropped; never ship.
- Any e2e gate chain not token-exact, or split parity ≠ 0.0 → diagnose first.
- `sharedWeightsSize` below floor, `genie_load_check` fail, ctx-bin script
  gate fail → the log names the contract; fix, never override.
- Kit caption semantic drift → investigate before bundling.
- Any upload flips repo visibility → report and stop.
- `df -h /mnt/c` < 20 GB at any point → stop, reclaim, then continue.

## Estimates

Local code+lint (Phases 1-2): ~2-3 h. Tank quant chain (Phase 3): ~8-14 h
wall, mostly unattended. Local fp32 export + full gate (Phase 4): ~3-4 h,
overlaps Phase 3. Kit (Phase 5): ~1.5 h. Bundle + docs (Phase 6): ~2 h.
Ship (Phase 7): ~1-2 h. **Wall-clock with overlap: ~1.5-2 working days.**

## Self-review (done at planning time)

- Spec coverage: all device-free work → GQA fix (P1, P3, P4), SIGSEGV padding
  (P2, P6), gates incl. the missing `lint_gqa_ops` (P3.6, P6.4), one-batch
  single-test package (P6.2's §1 one-command smoke + in-bundle perf path +
  fallbacks), ship (P7) ✓. "Valid inference result in a single test" = padded
  blobs pre-applied + expected caption shipped + load-sim-gated bins ✓.
- Consistency: `-gqa` work-dir names vs unchanged `.bin`/graph names appear
  identically in P3.2/3.3/3.6/6.3; `PAD_BYTES=4096` and 3,149,824 agree across
  preprocess/kit/lint/DEVICE_TEST; `_p` suffix appears in export (3.3),
  convert (3.6 via DSP_SUFFIX), and gate feeds (4.2 defaults).
- Placeholders: the four spots where v2's exact invocation is the authority
  (rename_deepstack in-place behaviour, seam tensor name, parity_vl_text_split
  args, build_test_kit flags) each carry a concrete expected value PLUS a
  recovery command (`~/.bash_history`, `--help`, the captions dir) and a
  downstream gate that catches a wrong guess — none is a bare TBD.
