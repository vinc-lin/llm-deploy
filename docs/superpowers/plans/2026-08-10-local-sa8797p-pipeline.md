# Local (Device-Free) SA8797P Build & Validation Environment — Implementation Plan

> **EXECUTION STATUS (2026-08-10):** Tasks 0–7 and 10 COMPLETE (envs, SDK,
> baseline, export, parity PASSED, W8A16 quant, full build chain, bundles).
> Task 8 (QKV surgery) COMPLETE — 28/28 layers, converter+ctx-bin accepted at
> vtcm16. Task 9 partial (no x86 HTP simulator in Community SDK; Genie qnn-cpu
> engine identified as follow-up). Task 11 (1.7B) + fixed-clip rebuilds running
> via `scripts/build/rebuild_all.sh`. Results: `docs/LOCAL_ENV.md`.

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Rebuild the entire SA8797P W8A16 build-side pipeline (export → AIMET quant → qairt-converter → ctx-bin) locally on this WSL2 machine, with numerical validation at every stage, plus develop the QKV-fusion ONNX surgery — so deployable artifacts are ready the moment device access returns.

**Architecture:** Everything runs on x86_64 (Ryzen 5900X / RTX 4060 Ti 8GB / WSL2). The QAIRT x86 tools do offline HTP prepare for hexagon-v81, so ctx-bins built here are real deployable artifacts. Validation without the device uses: torch reference → onnxruntime parity → AIMET quantsim simulated-quant outputs → QNN CPU backend / x86 HTP simulator (if shipped in SDK).

**Tech Stack:** QAIRT 2.48.40.260702 (exact match to remote host — confirmed publicly downloadable), aimet-torch (PyPI), torch+CUDA (WSL2 passthrough confirmed), transformers, onnx==1.19.0, uv-managed venvs (py3.10 + py3.12), Qwen/Qwen3-0.6B (+1.7B stretch).

**Constraints (from SA8797P_Deployment_Status_Summary.md):**
- No device, no jump-host, internet only. All knowledge of the remote pipeline comes from the status summary doc; the original scripts are NOT available and must be reconstructed.
- What can NEVER be validated locally: tok/s, bandwidth, VTCM/unsigned-PD limits, perf profiles, GVM behavior. Local success criterion = *artifact + numerics correctness*, not performance.

**Known-good facts to bake in (from the summary doc):**
- W8A16: per-channel symmetric INT8 weights, FP16 activations; embed_tokens, final norm, lm_head, K/V-proj outputs stay FP16; RMSNorm 16-bit; `clip_weights_to_7f7f`; `quant_scheme=post_training_tf_enhanced`, `default_param_bw=8`, `default_output_bw=16`.
- Export: split cos/sin RoPE as graph inputs; no fused `nn.RMSNorm` (opset 17); attention mask width = CL+AR; ONNX must be 1.19.0.
- Converter: `--target_backend HTP` (uppercase), `--quantization_overrides`, `--float_bitwidth 16`, `-d` for every input; NO `--float_fallback`/`--target_soc_model`/`--input_list`.
- ctx-bin: `--model libQnnModelDlc.so --dlc_path prefill.dlc,decode.dlc`; `--binary_file` WITHOUT `.bin`; CWD must contain `htp_config.json`/`perf_config.json`; build config `O:3, vtcm_mb:16 (NOT 24!), num_cores:1, hvx_threads:4, pd_session:unsigned, extended_udma:true, rpc_polling_time:9999`.
- Bundle: flat layout, 7 ARM .so + genie-t2t-run + tokenizer.json + dialog JSON + htp_backend_ext_config.json.
- QKV surgery spec: `qkv_proj → Dequant → Split(Q,K,V) → Quantize(Q only)` — Q re-quantized INT16, K/V stay FP16.

---

## Task 0: Preflight (DONE 2026-08-10 during planning)

- [x] Confirm QAIRT 2.48.40.260702 publicly downloadable (ranged GET → 206)
- [x] Confirm aimet-torch on PyPI (1.31.2 … 2.36.0 available)
- [x] Confirm Qwen/Qwen3-0.6B + 1.7B public on HF
- [x] Confirm uv present, CUDA passthrough present, 820 GB free on ext4
- [x] Start background downloads: SDK zip → `downloads/qairt-2.48.40.260702.zip`; model → `models/Qwen3-0.6B`

## Task 1: Python environments

**Files:** Create: `envs/` (uv venvs), `scripts/env.sh`

- [ ] **Step 1: py3.10 env `qwen3-deploy`** (mirrors remote env name)

```bash
cd /home/vinc/code/llm-deploy
uv venv --python 3.10 envs/qwen3-deploy
uv pip install --python envs/qwen3-deploy/bin/python \
  "numpy<2" "onnx==1.19.0" onnxruntime torch torchvision \
  transformers accelerate safetensors sentencepiece
```

Note: install aimet-torch LAST and pinned separately (Step 3) — it constrains torch; if the resolver downgrades torch, accept aimet's pin (CUDA wheels bundle their own runtime; no nvcc needed).

- [ ] **Step 2: Verify torch CUDA works in WSL2**

Run: `envs/qwen3-deploy/bin/python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"`
Expected: `True NVIDIA GeForce RTX 4060 Ti`

- [ ] **Step 3: Install aimet-torch; API smoke test**

```bash
uv pip install --python envs/qwen3-deploy/bin/python aimet-torch
envs/qwen3-deploy/bin/python - <<'EOF'
import torch, aimet_torch
from aimet_torch.quantsim import QuantizationSimModel
m = torch.nn.Sequential(torch.nn.Linear(8, 8))
sim = QuantizationSimModel(m, dummy_input=torch.randn(1, 8),
                           default_param_bw=8, default_output_bw=16)
print("AIMET OK:", aimet_torch.__version__ if hasattr(aimet_torch,'__version__') else "2.x")
EOF
```

Expected: prints AIMET OK. If the v1-style kwargs are rejected by aimet-torch 2.36, downgrade until they work (2.x kept the v1 `QuantizationSimModel` façade; last resort `aimet-torch==1.35.0`). Record chosen version in `docs/LOCAL_ENV.md`. If onnx==1.19.0 conflicts with aimet's pin, let aimet win inside quant env and keep a separate export step env — but ONNX 1.19.0 is REQUIRED for export (≥1.20 breaks AIMET export per summary doc §1.3).

- [ ] **Step 4: py3.12 env `qairt-py312` for qairt-converter** (numpy 2.x fine here)

```bash
uv venv --python 3.12 envs/qairt-py312
uv pip install --python envs/qairt-py312/bin/python numpy packaging pyyaml
# After SDK unpack: install ${QAIRT_SDK}/bin/requirements per SDK docs (check-python-dependency)
```

- [ ] **Step 5: Write `scripts/env.sh`**

```bash
#!/usr/bin/env bash
export LLMDEPLOY_ROOT=/home/vinc/code/llm-deploy
export QAIRT_SDK=$LLMDEPLOY_ROOT/sdk/qairt/2.48.40.260702
export PATH=$QAIRT_SDK/bin/x86_64-linux-clang:$PATH
export LD_LIBRARY_PATH=$QAIRT_SDK/lib/x86_64-linux-clang:$LD_LIBRARY_PATH
export PYTHONPATH=$QAIRT_SDK/lib/python:$PYTHONPATH
```

## Task 2: SDK unpack, inventory, tool smoke tests

**Files:** Create: `sdk/`, `docs/SDK_INVENTORY.md`

- [ ] **Step 1: Unzip when download completes; verify size 2,387,723,706 bytes first**

```bash
cd /home/vinc/code/llm-deploy
stat -c%s downloads/qairt-2.48.40.260702.zip   # must equal 2387723706
unzip -q downloads/qairt-2.48.40.260702.zip -d sdk/
# expected root: sdk/qairt/2.48.40.260702/
```

- [ ] **Step 2: Inventory — confirm/record presence of:**
  - `bin/x86_64-linux-clang/`: qairt-converter, qnn-context-binary-generator, qnn-net-run, qnn-context-binary-utility, genie-t2t-run (x86 Genie — if present, big win for local smoke tests)
  - `lib/x86_64-linux-clang/`: libQnnHtp.so (offline prepare + simulator), libQnnCpu.so, libQnnSystem.so, libQnnHtpPrepare.so, libQnnModelDlc.so, libGenie.so (x86)
  - `lib/aarch64-android/`: the 7 device .so from summary doc §1.3
  - `lib/hexagon-v81/unsigned/`: libQnnHtpV81Skel.so
  - Genie docs + examples (`docs/Genie*`, `examples/Genie*`) — needed for dialog-JSON schema and expected graph I/O tensor naming
  - Write findings to `docs/SDK_INVENTORY.md`

- [ ] **Step 3: Tool smoke tests (x86)**

```bash
source scripts/env.sh
qairt-converter --help | head -30          # runs under envs/qairt-py312 python
qnn-context-binary-generator --version
qnn-net-run --version
```

Expected: all print usage/version without missing-lib errors.

- [ ] **Step 4: Extract Genie LLM graph I/O naming conventions from SDK docs/examples into `docs/NOTES-genie-io.md`** — this pins exact expected input/output tensor names (input_ids / attention_mask / position_ids_cos / position_ids_sin / past_* layouts), dialog JSON schema (`pos-id-dim`, `kv-dim`, `rope-theta` keys per summary §2.1), and htp_backend_ext_config.json schema. The export wrapper (Task 4) must match these names.

## Task 3: Torch reference baseline

**Files:** Create: `scripts/reference/hf_baseline.py`, `work/reference/`

- [ ] **Step 1: Baseline generation + saved logits**

```python
#!/usr/bin/env python
"""Reference outputs from HF Qwen3-0.6B for parity checking."""
import json, torch
from pathlib import Path
from transformers import AutoModelForCausalLM, AutoTokenizer

ROOT = Path(__file__).resolve().parents[2]
MODEL = ROOT / "models/Qwen3-0.6B"
OUT = ROOT / "work/reference"; OUT.mkdir(parents=True, exist_ok=True)

PROMPTS = [
    "The capital of France is",
    "def fibonacci(n):",
    "解释一下什么是注意力机制。",
    "1+2+3+...+100 =",
]

tok = AutoTokenizer.from_pretrained(MODEL)
model = AutoModelForCausalLM.from_pretrained(MODEL, torch_dtype=torch.float32).eval()

results = {}
for i, p in enumerate(PROMPTS):
    ids = tok(p, return_tensors="pt").input_ids
    with torch.no_grad():
        logits = model(ids).logits
    torch.save({"input_ids": ids, "logits": logits}, OUT / f"ref_{i}.pt")
    gen = model.generate(ids, max_new_tokens=32, do_sample=False)
    results[p] = tok.decode(gen[0], skip_special_tokens=True)
json.dump(results, open(OUT / "greedy_texts.json", "w"), ensure_ascii=False, indent=2)
print(json.dumps(results, ensure_ascii=False, indent=2))
```

- [ ] **Step 2: Run it** — `envs/qwen3-deploy/bin/python scripts/reference/hf_baseline.py`. Expected: coherent completions; `work/reference/ref_*.pt` exist. FP32 on CPU is fine (0.6B); use GPU if it fits.

## Task 4: ONNX export (prefill + decode graphs)

**Files:** Create: `scripts/export/export_qwen3.py`, `scripts/export/modeling_export.py`

Design (from summary doc + Genie conventions; final I/O names come from Task 2 Step 4):

- Static shapes. Prefill: `input_ids [1, CL_P=128]`; Decode: `input_ids [1, 1]` + past KV `[1, 8 kv-heads, CL_CTX, 128]` per layer.
- RoPE cos/sin are **graph inputs** (`position_ids_cos`, `position_ids_sin`, shape `[1, seq, 64]` — pos-id-dim 64 per summary §2.1), not computed in-graph.
- Attention mask input, additive float, width CL+AR.
- RMSNorm decomposed to primitive ops (`x / sqrt(mean(x²)+eps) * w`) — never `nn.RMSNorm`; opset 17.
- Q/K per-head RMSNorm (Qwen3 q_norm/k_norm) preserved.
- `--fuse-gate-up`: single `gate_up_proj` MatMul then Split (summary §2.3 — validated remotely at ONNX level).
- `--fuse-qkv`: single `qkv_proj` MatMul then Split — surgery for quantizers happens post-export in Task 8.
- Export with `torch.onnx.export`, opset 17, then `onnx.checker` + shape inference.

- [ ] **Step 1: Write `modeling_export.py`** — an export-friendly forward re-implemented on top of the HF Qwen3 weights (embed → 28 × [ln → attn(QKV proj, q/k-norm, rope-from-inputs, GQA attention with explicit additive mask, KV concat with past) → residual → ln → MLP(gate/up/down, SiLU)] → final norm → lm_head), parameterized by (CL_P, CL_CTX, fuse flags). All shapes static; no data-dependent control flow.
- [ ] **Step 2: Write `export_qwen3.py`** CLI: `--model models/Qwen3-0.6B --out work/onnx/qwen3-0.6b --cl-prefill 128 --ctx 1024 [--fuse-gate-up] [--fuse-qkv]` → emits `prefill.onnx`, `decode.onnx` (+ external-data files).
- [ ] **Step 3: onnx.checker + shape-inference pass both graphs.** Expected: clean.
- [ ] **Step 4: Commit** (repo will be git-inited in Task 0 of execution if not already).

## Task 5: ONNX ↔ torch parity harness

**Files:** Create: `scripts/validate/parity_onnx.py`

- [ ] **Step 1:** onnxruntime (CPU EP) drives prefill graph with ref prompt tokens + host-computed cos/sin + causal mask; compare logits vs `work/reference/ref_*.pt`. Threshold: max|Δ| < 1e-3 (FP32) on final-position logits; argmax must match.
- [ ] **Step 2:** Decode-graph chain test: prefill once, then N=8 decode steps feeding KV; compare each step's argmax vs torch `generate` greedy tokens. Expected: identical token sequence.
- [ ] **Step 3:** Repeat both with `--fuse-gate-up` export. Expected: bit-identical or <1e-5 drift (summary §2.3 says bit-exact in FP16).

## Task 6: AIMET W8A16 quantization (reconstruct `quantize_aimet.py`)

**Files:** Create: `scripts/quant/quantize_aimet.py`, `scripts/quant/filter_aimet_w8a16.py`, `configs/htp_quantsim_config_v81_per_channel_linear.json`

- [ ] **Step 1: Recreate the per-channel HTP quantsim config** — start from aimet's shipped `htp_quantsim_config` template; enable `"per_channel_quantization": "True"` for Linear/MatMul params; ops_to_quantize per HTP v81.
- [ ] **Step 2: Quantsim script**, per summary §2.2: `default_param_bw=8, default_output_bw=16, quant_scheme=post_training_tf_enhanced`; calibration = 10 mixed zh/en/code/math prompts through prefill wrapper; `clip_weights_to_7f7f(sim)` (reimplement: clamp INT8-encoded weights to ±0x7f7f-safe range, i.e. clip encodings so qmax ≤ 0x7f7f pattern — implement as clipping weight quantizer encodings min/max to avoid saturating INT8 packed pairs); force RMSNorm output quantizers to 16-bit; DISABLE quantizers on: embed_tokens (else HTP `Gather`-on-INT16 error 0xc26), final norm, lm_head, and all K/V-proj outputs (cross-graph FP16 requirement). For `--fuse-gate-up`: disable `gate_up_proj` output quantizer (FP16 internal, requantize at down_proj).
- [ ] **Step 3: Export encodings** (`sim.export` → `model.encodings` + FP-weight ONNX), run `filter_aimet_w8a16.py` to strip any residual embed/norm/lm_head entries.
- [ ] **Step 4: Sanity metric:** quantsim-simulated logits vs FP32 reference — argmax agreement on ref prompts ≥ 95% of positions; report max KL. (Local proxy for "coherent output" claim in §2.1.)

## Task 7: Convert + ctx-bin (deployable artifacts)

**Files:** Create: `configs/htp_config.json`, `configs/perf_config.json`, `configs/htp_backend_ext_config.json`, `scripts/build/convert.sh`, `scripts/build/ctxbin.sh`

- [ ] **Step 1: `convert.sh`** (runs envs/qairt-py312):

```bash
source scripts/env.sh
envs/qairt-py312/bin/python $QAIRT_SDK/bin/x86_64-linux-clang/qairt-converter \
  --input_network work/onnx/qwen3-0.6b/prefill.onnx \
  --output_path   work/dlc/qwen3-0.6b/prefill.dlc \
  --quantization_overrides work/quant/model_filtered.encodings \
  --float_bitwidth 16 --target_backend HTP \
  -d input_ids 1,128 -d attention_mask 1,1,128,1152 \
  -d position_ids_cos 1,128,64 -d position_ids_sin 1,128,64
# (exact -d list finalized from actual graph inputs; decode.dlc analogous with AR=1 + past_* dims)
```

- [ ] **Step 2: `configs/htp_config.json`** — v81 device, `"O": 3, "vtcm_mb": 16, "num_cores": 1, "hvx_threads": 4, "sparse_weights_compression": 1, "rpc_polling_time": 9999, "pd_session": "unsigned", "extended_udma": true` (schema validated against SDK docs/examples in Task 2).
- [ ] **Step 3: `ctxbin.sh`** — cd into configs dir (CWD requirement!), then:

```bash
cd configs && qnn-context-binary-generator \
  --model $QAIRT_SDK/lib/x86_64-linux-clang/libQnnModelDlc.so \
  --dlc_path ../work/dlc/qwen3-0.6b/prefill.dlc,../work/dlc/qwen3-0.6b/decode.dlc \
  --backend $QAIRT_SDK/lib/x86_64-linux-clang/libQnnHtp.so \
  --output_dir ../work/ctxbin/qwen3-0.6b-w8a16 \
  --binary_file qwen3_06b_w8a16_ctx        # NO .bin suffix
  --config_file htp_config.json
```

Expected: multi-graph ctx-bin ~1.1–1.5 GB (summary §2.3: AIMET-piped encodings → ~1.1 GB). Verify with `qnn-context-binary-utility --context_binary ... --json_file ...` (graph names, I/O dtypes: FP16 acts, INT8 weights).

- [ ] **Step 4:** Repeat for `--fuse-gate-up` export → second ctx-bin (this is §4.1's pending rebuild-at-vtcm-16, done locally).

## Task 8: QKV-fusion ONNX surgery (§4.1 main unlock — fully local work)

**Files:** Create: `scripts/export/qkv_surgery.py`, `scripts/validate/parity_qkv.py`

- [ ] **Step 1:** Post-export graph transform on the fused-QKV ONNX: locate each `qkv_proj` MatMul output → insert `DequantizeLinear → Split(Q,K,V) → QuantizeLinear(Q only, INT16 per-tensor)`; K/V branches stay FP16. Encodings file updated so Q-split output carries the INT16 activation encoding and qkv_proj output has none.
- [ ] **Step 2:** Parity: fused+surgered graph vs non-fused quantsim outputs — argmax agreement as Task 6 Step 4.
- [ ] **Step 3:** Convert + ctx-bin the surgered model (Task 7 scripts). Success criterion: converter and context-binary-generator accept the graph at `vtcm_mb:16` — locally provable, since err `0x138d` (vtcm=24 rejection) was a runtime issue but the quantizer conflict (error 5005 / garbage) was a graph-level property.

## Task 9: Local functional execution (no device)

- [ ] **Step 1:** `qnn-net-run` with `libQnnCpu.so` (FP32) on converted model, input from ref prompt → logits diff vs torch. Expected: argmax match.
- [ ] **Step 2:** If SDK ships x86 HTP simulator path in `libQnnHtp.so`: run quantized graph on simulator (SLOW — single prompt, few tokens only) → confirms HTP-kernel-level numerics. Document availability either way in `docs/SDK_INVENTORY.md`.
- [ ] **Step 3:** If x86 `genie-t2t-run` + x86 `libGenie.so` exist: assemble an x86 bundle (dialog JSON adapted from summary §2.1 values: pos-id-dim 64, kv-dim 128, rope-theta 1000000.0, graph-switching, mmap) and smoke-test T2T end-to-end on CPU backend. This validates the dialog JSON + tokenizer + ctx-bin plumbing that previously could only be tested on-device.

## Task 10: Device bundle assembly (push-ready)

**Files:** Create: `bundles/qwen3_06b_w8a16_local/` + tarball

- [ ] Flat dir: ctx-bin + 7 ARM64 .so (from `lib/aarch64-android/` + `lib/hexagon-v81/unsigned/`) + `genie-t2t-run` (aarch64-android) + tokenizer.json + dialog JSON + htp_backend_ext_config.json → `qwen3_06b_w8a16_local_bundle.tar.gz`. Everything per summary §1.3 bundle layout. Ready for `adb push` the day access returns.

## Task 11: 1.7B pipeline repeat (stretch, disk/VRAM permitting)

- [ ] Rerun Tasks 3–7 with `models/Qwen3-1.7B`; AIMET calibration may need CPU offload on 8 GB VRAM.

## Task 12: Documentation

**Files:** Create: `docs/LOCAL_ENV.md`; Modify: `SA8797P_Deployment_Status_Summary.md` (both copies)

- [ ] `LOCAL_ENV.md`: machine specs, env versions chosen (record actual aimet/torch pins), what was validated locally vs still device-pending, divergences from remote env, how to hand artifacts back to the remote flow.
- [ ] Add a "Local mirror environment (2026-08-10)" section to the status summary.

---

## Risks / fallbacks

| Risk | Mitigation |
|---|---|
| aimet-torch 2.x API drift vs doc's v1-style calls | Version ladder 2.36 → 2.x older → 1.35; smoke test in Task 1 Step 3 gates this early |
| Genie graph-I/O naming unknown (remote scripts unavailable) | Task 2 Step 4 extracts naming from SDK Genie docs/examples BEFORE export wrapper is written |
| x86 HTP simulator or x86 Genie absent from Community SDK | CPU-backend + quantsim parity still cover numerics; note gap in LOCAL_ENV.md |
| 8 GB VRAM too small for AIMET calibration | Calibrate on CPU (slow but 0.6B feasible) or layer-wise |
| Local ctx-bin ≠ remote ctx-bin behavior | Same SDK version 2.48.40.260702 exact-match; compare against doc's recorded sizes (~1.1–1.5 GB) as weak checksum |
| `clip_weights_to_7f7f` semantics uncertain (reconstructed from name) | Implement as encoding-clip; verify no INT8 saturation warnings; flag in LOCAL_ENV.md for review |
