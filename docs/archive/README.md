# Archive

Superseded documents, kept for provenance. **Nothing here is current truth.**
Current truth is `docs/REFERENCE.md`; the current 0.6B plan is
`docs/PLAN_0.6B_max_tps.md`.

These files are retained rather than deleted because they record *why* decisions
were made, and several were falsified by measurement in ways worth remembering —
this project has repeatedly re-derived a wrong conclusion after the document that
recorded it was thrown away. Read them as history, and never as a source for a
number.

## The `MAX_TPS_QWEN3_0.6B` ladder (V1–V4)

Four successive plans for maximising 0.6B decode throughput, replaced by the
single unversioned `docs/PLAN_0.6B_max_tps.md`. The ladder itself was the
problem: each version restated the previous one's analysis, so a correction to
the analysis had to be applied in four places and usually wasn't.

| File | Date | What it was | Why it is archived |
|---|---|---|---|
| `MAX_TPS_QWEN3_0.6B_V1.md` | ~2026-08-11 | The original 10.8 tok/s LADE recipe | Its headline predates the 2026-08-13 measurement showing basic AR-1 at 11.72 on the same device. LADE is now parked entirely (`REFERENCE.md` §6.8) |
| `MAX_TPS_QWEN3_0.6B_V2.md` | 2026-08-13 | Measured baselines, ctx-bin forensics, and **the compute model** | Superseded as a plan. **But its §1.2 was right**: it predicted the post-fix step at `88.5M ÷ 4 HVX @ ~1 GHz ≈ 22.1 ms` and the device measured **22.37 ms**. That out-of-sample hit is the strongest evidence for the compute model (`REFERENCE.md` §8.11); it was written off at the time because the prevailing view was "100% DDR-bound" |
| `MAX_TPS_QWEN3_0.6B_V3.md` | 2026-08-14 | The gqafix trunk and its decision tree | The trunk shipped and was measured; the decision tree is resolved. Its §7 byte projection ("883 MB → ~18.1 tok/s") is the byte model's documented out-of-sample **failure** — reality was 44.7 |
| `MAX_TPS_QWEN3_0.6B_V4.md` | 2026-08-16, **rev 2** | Byte-floor descent plan, revised twice | The original draft anchored on a pre-fix `read_total_bytes` and led with byte reduction. **Rev 2 corrected itself** — the 11.72 blend, the zero-byte finding, the P1 retraction — and is the deepest analysis in the project: the byte-vs-compute degeneracy at the post-fix operating point, and the discriminator-pair design. Archived only because the *ladder* is retired, not because it is wrong. `docs/PLAN_0.6B_max_tps.md` is the acting plan; read this for the reasoning behind it |

**The through-line worth remembering:** V2 contained the correct model and was
overruled by consensus; V3 and V4 then built increasingly detailed plans on the
consensus. The cost was two planning cycles and a nearly-executed build campaign
aimed at the wrong lever. When a document's prediction lands out-of-sample,
promote it over the prevailing narrative.

## Superseded artifact and session documents

| File | Why archived |
|---|---|
| `BUNDLE_README_fuseqkvgu_ladekv.md` | Pre-GQA-fix bundle projected at ~12 tok/s against a measured 44.707 baseline, built for parked LADE. **Documents a config pair (`type:"lade"` + `max-num-tokens`) that SIGSEGVs** — never copy a config from it |
| `PROFILING_PACKAGE_README.md` | Pre-fix profiling package. Its inputs are the defective ones that lost the 2026-08-15 cycle profile, and its ctx-bin differs from the gqafix one by a single filename token |
| `DEVICE_MEASUREMENT_REQUEST_2026-08-13.md` | Work order, fully delivered as `DEVICE_MEASUREMENT_REPORT_2026-08-13.md` |
| `DEVICE_TEAM_EXCHANGE_2026-08-14.md` | Outbound letter for a drop that has shipped and been measured |
| `SA8797P_Deployment_Status_Summary.md` | The project's ancestor document (2026-08-09, remote team, different machine). Its hardware section was replaced by the HTP doc, and it carried six corrections. See the migration note below |

`kit/archive/2026-08-15/` holds the device-session kit for the same reason: its
pre-commitments fired, and its decision table is where the retracted byte
arithmetic originated.

## Facts migrated out before archiving

So the archive is never load-bearing, these were moved into `REFERENCE.md` first:

- `type:"lade"` + `max-num-tokens` → SIGSEGV, and the linter that blocks it → §3.4
- `libQnnHtpQemu.so` → `Request feature arch with value 81 unsupported` → §4.1
- `groupContext.share_resources`, `--preserve_io_datatype`, QAIRT 2.43 quantizer dead ends → §4.1
- The four-tier perf-profile ladder in absolute tok/s (the basis for correction #30) → §1
- `adb push` >500 MB USB disconnects → §9
- The `−100` vs `−1000` mask-constant caveat → §3.5
- The DLC-mtime staleness gate → §5
- The bertcache 444 MB private weight copy → §6.11
- The no-weight-sharing ~3.2 GB counterfactual → §8.2

**Still only in `SA8797P_Deployment_Status_Summary.md`**, worth harvesting if
anyone touches these areas: the QNX-native escape recipe (2.43 QNX binaries,
skel → `/mnt/etc/images/dsp/`); that `vtcm_mb` values 17–23 were never probed;
that `--quantization_overrides` packs better than hand-rolled per-tensor
encodings (1.1 vs 1.5 GB); the Qwen3-1.7B W8A16 artifacts that were never
device-tested; and untried levers — `fp16_relaxed_precision` + O-level A/B,
AdaRound/CLE, W8A8 retried on the zero-spill fused graph.

Explicitly **discarded, not migrated** (wrong, superseded, or unreproducible):
its 1369 MB/token and 10.7 GB/s figures, "decode is 100% DDR-bound", the
2-of-4-NSP hardware model, and its tok/s projection table.
