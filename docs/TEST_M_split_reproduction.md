# Test M — does the 2-ctx-bin split break decode at 0.6B?

**Status:** ready to run · **Opened:** 2026-08-22 · **Needs:** one bundle, ~10 min
on device. Built device-free; nothing was re-quantized.

Registry: `DEVICE_TEST_INDEX.md`. Run **after** Test L, or alongside it.

---

## 0. What this is

The split is the **last structural difference** between the 0.6B that works and
the 4B that does not. This build is the 4B's shape at 0.6B scale:

| build | ctx-bins | embedding feed | decode |
|---|---|---|---|
| `gqafix_cl512_ladekv` | one | in-graph from token ids | ✅ correct |
| `qwen3_06b_lutprobe` | one | **LUT → `inputs_embeds`** | Test L |
| **`qwen3_06b_lutsplit`** ← this | **two** | **LUT → `inputs_embeds`** | **?** |
| Qwen3-VL-4B | two | LUT → `inputs_embeds` | ❌ decode step 1 wrong |

It is a genuine structural clone, not an approximation — read back from the bins:

| | 2-shard 0.6B | VL-4B |
|---|---|---|
| shard 0 input | `inputs_embeds` **uFxp_16** `[1,1,AR,1024]` | `inputs_embeds` **uFxp_16** `[1,1,AR,2560]` |
| shard 1 input | `last_hidden_states` **FLOAT_16** `[1,AR,1024]` | `last_hidden_states` **FLOAT_16** `[1,AR,2560]` |
| graphs | `prefill_0/decode_0` + `prefill_1/decode_1` | same names |
| prefill | past-KV `[1,128,640]` | past-KV `[1,128,2176]` |

Same handoff tensor, same dtype, same rank, same graph names, same past-KV
prefill topology. What differs is scale (28 layers / 1024 vs 36 / 2560) and that
the 0.6B has no MRoPE and no image path.

**Built from the LUT probe's own AIMET exports**, cut at the layer-13/14 seam —
so calibration is identical to the single-bin probe and the split really is the
only variable. The grafted `inputs_embeds` encoding came out
`scale=8.30007343404e-06, offset=-38356`, byte-identical to the single-bin
probe's.

### Host gates, all passing

| gate | result |
|---|---|
| `lint_embedding_dtype.py` on shard 0 | **PASS** — `UFIXED_POINT_16`, not the FLOAT_16 that voided Test K |
| `lint_gqa_ops.py` (per shard, 14 layers) | **PASS** on all four DLCs |
| weight pooling | **100%** on both shards, private const **0 B** (0.21 + 0.50 GB = the 0.6B weight set carried exactly once) |
| backend config readback | `O=3`, `vtcmSize=16`, `numHvxThreads=4` on all four graphs |
| **split chain vs the unsplit answers** | **3/3** — `prefill_0 → prefill_1` on the host gives `220`, `12095`, `220`, the same as the whole graph and as HF |

That last one matters: the split graphs are correct on the host, so a device
failure implicates **the runtime's handling of the split**, not the cut.

---

## 1. Run it

```sh
adb push qwen3_06b_lutsplit /data/local/tmp/lutsplit
adb shell
cd /data/local/tmp/lutsplit && chmod +x genie-t2t-run && export LD_LIBRARY_PATH=.

md5sum qwen3-06b-lutsplit_1_of_2.bin | cut -c1-32   # 1f4dcd44dcbf4b8b9975c982e6585ee6
md5sum qwen3-06b-lutsplit_2_of_2.bin | cut -c1-32   # 11cabce4eb2d0b95d57137c69ddb776a

./genie-t2t-run -c genie_dialog_qwen3_06b_lutsplit.json \
    -p "What is 2+2? Answer with one number." \
    --profile m_short.json 2>&1 | tee m_short.txt

./genie-t2t-run -c genie_dialog_qwen3_06b_lutsplit.json \
    -p "Describe, in three or four sentences, what makes mountain weather change quickly." \
    --profile m_long.json 2>&1 | tee m_long.txt
```

---

## 2. Reading it

Score against the reference, **not** against intuition. HF Qwen3-0.6B fp32,
greedy, these exact raw prompts:

| prompt | correct output |
|---|---|
| short | `' 2+2=4. 2+2=4. 2+2=4. …'` — first ids `[220,17,10,17,28,19]` |
| long | `' Also, explain why it is important to have a good understanding of the weather…'` |

> ⚠ **The repetition is correct.** It is what HF fp32 does on a raw prompt and
> what the shipping 0.6B does on device. Test K inverted a verdict by reading
> that as failure. Do not score "coherent vs garbage".

| what you see | meaning | next |
|---|---|---|
| **matches the reference** | the split does **not** break decode at 0.6B | the split is exonerated at this scale, and the 4B's fault needs something the 0.6B lacks — scale, 36 layers, 2560 hidden, or MRoPE. Note the 4B's first token is *right* too, so look at what only it has |
| **first token right, then diverges** | ✅ **the 4B's exact signature, reproduced at 0.6B** | root cause localised to the split. Bisect it on the **host** for free: the chunk ONNX files are in `work/onnx/qwen3-06b-lutsplit/` and already chain correctly, so the divergence is in how the runtime moves `last_hidden_states` between bins |
| **wrong from token 0** | a prefill/feed problem, not the split | check the shard-0 md5 first; then compare against Test L on the same board |
| **fails to load** | a split-contract problem — a real finding | capture the exact error and `adb logcat -d`. `Failed to create the Genie Node (-1)` plus one ShapeError per shard is the documented AR==CL trap, which this build should not have |

**Run Test L's probe in the same session if you can.** One bin vs two bins,
back to back, same board, same prompts, is a far stronger result than either
alone.

---

## 3. What this build does NOT control for

Stated plainly so a clean result is not over-read:

* **Scale** — 28 layers / 1024 hidden against 36 / 2560. A defect that only
  appears above some size would not show here.
* **MRoPE and the image path** — the 0.6B has neither. Both are inert for a
  text-only 4B run (`nsp-model.cpp:3803` gates MRoPE on `m_visionParam`), so this
  is a fair comparison *for the text defect*.
* **Engine knobs.** The dialog config's `QnnHtp` block is byte-identical to the
  single-bin LUT probe's, deliberately, so Test M reads against Test L. The 4B's
  block differs (`poll: false`, `allow-async-init: false`, `mmap-budget: 0`,
  no `enable-graph-switching`). If Test M comes back clean, re-running it with
  the 4B's knobs is the cheap next variable — config-only, no rebuild.

---

## 4. What to send back

1. `m_short.txt` and `m_long.txt` **verbatim**, plus both `md5sum`s.
2. The two profiles.
3. If you also ran Test L this session, say so — the pair is the measurement.

Rebuild recipe, if ever needed: `scripts/build/lutsplit_06b_build.sh`.
