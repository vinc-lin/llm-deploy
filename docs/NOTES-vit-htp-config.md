# Why the ViT build generates its own HTP backend config

*2026-08-12, Stage 1 Task 4. Decision record for `scripts/build/vit_build.sh`.*

## Mechanism

`configs/htp_backend_config.json` scopes its tuning block by graph name:

```json
"graphs": [{ "graph_names": ["prefill", "decode", "verify32"], "O": 3, "vtcm_mb": 16, ... }]
```

The ViT ctx-bin has exactly one graph, named `vit` (the graph name comes from
the DLC basename, `$DLC/vit.dlc`). It is not in that list, so **none of the
tuning binds to it** — and nothing complains. The generator prints no warning,
exits 0, and emits a plausible-looking 810 MB binary compiled with backend
defaults. `devices` is *not* graph-scoped, so `dsp_arch: v81` applies either
way; only the per-graph block is lost.

Rather than add `vit` to a config three text builds depend on,
`vit_build.sh` writes its own `htp_backend_config.json` + `htp_config.json`
into `$CTXBIN` and runs the generator from there (`config_file_path` resolves
relative to cwd).

## Evidence: same DLC, two configs

Both runs finalize `work/dlc/qwen3vl-4b-vit-fp16/vit.dlc`; only `--config_file`
differs. Values read back from `qnn-context-binary-utility --json_file`
(`info.graphs[0].info.graphBlobInfo.info`), spill/fill from the generator's
"DDR bandwidth summary".

| | generated config | shared `configs/` config |
|---|---|---|
| `optimizationLevel` | 3 | **0** |
| `vtcmSize` | 16 | **4** |
| `numHvxThreads` | 4 | **0** |
| `spillFillBufferSize` | 4,259,840 | 37,945,344 |
| `spill_bytes` | 4,194,304 | **1,446,117,376** |
| `fill_bytes` | 4,194,304 | 1,628,569,600 |
| `read_total_bytes` | 1,001,660,416 | 3,839,750,144 |
| `contextBlobSize` | 849,132,592 | 855,768,104 |
| `dspArch` | 81 | 81 (devices block is not graph-scoped) |

345x the DDR spill traffic, from a build whose log is clean.

The control run (its ctx-bin was deleted afterwards; re-derive with):

```bash
source scripts/env.sh
cd "$LLMDEPLOY_ROOT/configs"      # <-- the shared config, i.e. the bug
qnn-context-binary-generator \
    --model "$QAIRT_SDK/lib/x86_64-linux-clang/libQnnModelDlc.so" \
    --dlc_path "$LLMDEPLOY_DATA/work/dlc/qwen3vl-4b-vit-fp16/vit.dlc" \
    --backend "$QAIRT_SDK/lib/x86_64-linux-clang/libQnnHtp.so" \
    --output_dir /tmp/ctrl --binary_file ctrl_ctx \
    --config_file htp_config.json
qnn-context-binary-utility --context_binary /tmp/ctrl/ctrl_ctx.bin \
    --json_file /tmp/ctrl/ctrl_info.json
```

Note `spill_bytes` / `fill_bytes` appear **only** in the generator's stdout, not
in `info.json` — capture the log if you re-run this.

## Precedent: this already cost a device cycle

Not a hypothetical. `docs/BUILD_GUIDE.md` §5.4b (line 301) records the identical failure on
real hardware: `verify32` was omitted from `graph_names`, silently compiled with
defaults, and device logs showed 4 MB VTCM + 24 MB spill instead of 16 MB + 0.
That was the root cause of the LADE SIGSEGV in
`reports/qwen3-0.6b-w8a16-ladekv-test-report.md`.

## Guard

Because the failure is silent, step 3 of `vit_build.sh` reads the settings back
out of the finalized binary and **fails the build** unless
`graphName == vit`, `optimizationLevel == 3`, `vtcmSize == 16`,
`numHvxThreads == 4`. The `graphName` assert matters independently: it is the
string the config keys on, so if `qairt-converter` ever derives graph names
differently, the config stops binding and everything else silently reverts to
defaults.

## Carried over, and deliberately dropped

Carried from the text config unchanged: `O: 3`, `vtcm_mb: 16`,
`hvx_threads: 4`, `fp16_relaxed_precision: 0`, `dsp_arch: v81`,
`pd_session: unsigned`, `soc_model: 0`, `perf_profile: burst`,
`rpc_control_latency: 100`, `rpc_polling_time: 9999`.

Dropped, with reasons:

- `context.weight_sharing_enabled` — meaningless for a single-graph ctx-bin.
- `memory.extended_udma`, `graph_configs_extra.sparse_weights_compression` —
  text-model tuning, unverified for an FP16 ViT.

## Follow-up: `soc_model` is 0

`soc_model: 0` (unspecified) was inherited from the text config. The SDK's
htp-target-table gives SA8797 → `dsp_arch v81`, `soc_id 72`
(`QNN_SOC_MODEL_SA8797 = 72` in `QnnTypes.h`), and
`docs/QAIRT-Docs/QNN/general/htp/htp_auto_optimization.html` states:

> When preparing a graph using O=3, specifying the correct device "soc_id"
> matching the target to use could turn on additional [algorithms] which may
> further improve inference performance.

We build at O=3, so this is real performance left on the table — for the text
builds too, not just the ViT. Left unchanged for now: it is unverified, it would
touch the shared config, and it is out of Stage 1 Task 4's scope. Worth a
measured A/B (`soc_model: 72`) before the first device run.
