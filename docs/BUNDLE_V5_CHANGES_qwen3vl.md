# v5 — what this bundle is for

**Read this, then `SESSION_RUNBOOK.md`.** `OPERATOR_GUIDE.md` remains the
reference for install, metric definitions and triage.

v5 is a **diagnostic** bundle. It ships the same ctx-bins as v3/v4 — nothing was
rebuilt — and it is not expected to produce a working caption. Its deliverable
is a verdict on *which stage* breaks the text tower.

## 1. Where things stand

| | Status |
|---|---|
| Image path | **Fixed and confirmed on device (v4)** — pipeline ran 3×, no SIGSEGV, 6 photographs executed. Not re-tested here |
| Text tower | **Garbage on device**, unchanged by v4's `bos-token` fix, while scoring 20/20 token-identical against HF on the host |

Everything that could be eliminated from the host has been. In particular the
"AIMET QDQ double-quantization" theory is **falsified**, and acting on it would
have cost a multi-hour rebuild:

* **0** `QuantizeLinear`/`DequantizeLinear` ops in either ONNX — nothing is
  QDQ-baked, so nothing can be double-quantized.
* The **working 0.6B** (44.707 tok/s on device) is built by the *same*
  `quantize_aimet.py` path as the 4B. `qairt-quantizer` appears nowhere in this
  project, so it cannot be what makes the 0.6B work.
* The 4B DLC already carries `sFxp_8` per-channel `axis-quant` weights with
  `uFxp_16` activations — i.e. the per-channel symmetric INT8 the proposed fix
  was meant to produce.
* The embedding LUT is **bit-exact** against the checkpoint at the runtime's
  own byte offsets (`worst 0.000e+00`), vision/pad markers included.
* The shard-0→shard-1 boundary is `FLOAT_16` on **both** sides — there is no
  encoding to mismatch.

That leaves exactly one untested hop: **ONNX → DLC → ctx-bin → Genie's feed.**
No gate in this project executes the shipped `.bin`; they all validate ONNX.

## 2. What v5 adds

**Probe A — the text-graph probe** (`run_text_probe.sh`, device, ~10 min).
Runs the shipped text ctx-bins under `qnn-net-run` with **no Genie involved**,
on inputs we control, against references computed from the *same per-shard ONNX
the DLCs were converted from*. Two cases:

* `decode1tok` — one token, empty cache. A clean read on pure numerics. It says
  nothing about rope: with a single token attending only to itself, RoPE
  rotates q and k identically and cancels. (Measured — an earlier "position 7"
  case produced byte-identical output to position 0.)
* `prefill4tok` — four tokens through the prefill graphs, where rope and
  cross-token attention are live, and which are the graphs the real 273-token
  prompt actually uses.

Each case runs shard 1 **twice** — chained on the device's own shard-0 output,
and isolated on the host reference boundary — because no single end-to-end
number can distinguish "shard 0 corrupted the boundary" from "shard 1 is
broken".

**Probe B — the feed differential** (already run on the host, results in
`feed_variants.json`). Ranks which feed mistakes could produce garbage *at all*.
The leading candidate is `emb_fp32_as_fp16` (worst cos **−0.87**), which
degenerates into repetition — the shape of what the device actually printed —
and is the same dtype-misinterpretation class as the v3 image crash. Rope theta
and embedding scale are **eliminated**. See `SESSION_RUNBOOK.md` §3b.

**Probe C — the timing decomposition.** v4 reported "~11.1 s" and no breakdown,
so a 4B two-shard W8A16 tower still has no init/TTFT/decode numbers. These are
valid even while the output is garbage — it is the same compute.

## 3. What is deliberately absent

**Genie's own debug-tensor dump.** It was planned and dropped on evidence:
`debug-tensors`/`debug-path` are compiled into the shipped `libGenie.so`, but
`Engine.cpp` validates the public config against a strict whitelist and throws
`Unknown QnnHtp config key` on anything outside it — and the engine level is
equally strict. Shipping that config would have produced a load failure and
burned a device slot, the same trap that broke the v3 fallback. Probe B
replaces it at zero device cost.

**Do not add config keys to enable debugging.** In this SDK there is no such
route.

## 4. Files added since v4

```
run_text_probe.sh          the probe A device runner (POSIX sh)
netrun_htp_config.json     qnn-net-run backend-extension config
probe_cases.txt            case list the runner iterates
cases.json                 case metadata
feed_variants.json         probe B results (already computed)
decode1tok/                inputs + fp32 references   (0.7 MB)
prefill4tok/               inputs + fp32 references   (6.7 MB)
```

Everything else is byte-identical to v4, including all three ctx-bins and the
LUT (md5s in `V4_CHANGES.md` §4) — copy them from the existing deployment
rather than re-downloading ~6 GB.

## 5. What each outcome means

| Probe A verdict | Meaning | Next build |
|---|---|---|
| all runs correct | ctx-bin + converter **exonerated** | chase Genie's feed, using probe B's ranking |
| shard0 wrong | fault in shard 0's ctx-bin | rebuild shard 0 |
| shard0 right, shard1-isolated wrong | fault isolated to shard 1 (owns `lm_head`) | rebuild shard 1 |
| both right alone, chained wrong | the boundary hand-off corrupts the hidden state | inspect the seam, not the weights |
| decode clean, prefill broken | needs rope/cross-token attention to appear | prefill graphs specifically — and that alone would explain the device garbage |

Each row is a different next build. That is the point of the session.
