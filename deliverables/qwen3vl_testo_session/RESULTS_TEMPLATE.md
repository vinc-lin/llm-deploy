# Test O session — results

**Date:** ____________  **Board / build:** ____________  **Operator:** ________

Procedure: `DEVICE_SESSION_PROTOCOL_O.md`. Fill in as you go.

---

## 0. Preconditions

| check | required | what you saw |
|---|---|---|
| shard 0 md5 | `f031e3a7563bf16f2d5ca98a71b357f6` | |
| shard 1 md5 | `0f1c86e89752b499eec09e9e10a73014` | |
| 4 `testn/*.tok` files present | yes | |
| `testo/` config count | 10 | |
| free space | ≥ 20 MB | |
| fingerprint / date | — | |

---

## O1 — KV dumps

| dump | kv-cache bytes (expected) | actual | `dialog.json` (paste whole) |
|---|---:|---:|---|
| `state_p20` | 3,097,168 | | |
| `state_p21` | 3,244,624 | | |
| `state_w18` | 2,802,256 | | |
| `state_w19` | 2,949,712 | | |

`testo_state_dumps.tar.gz` sent to build side at (time): ________
State dirs kept on device:  YES / NO

## O2 — knob sweep

| run | change | verdict (from the script) |
|---|---|---|
| `o2a_ctrl` | none — **must FAIL** | |
| `o2b_gswitch` | +enable-graph-switching | |
| `o2c_poll` | +poll | |
| `o2d_async` | +allow-async-init | |
| `o2e_mmapb` | mmap-budget 25 | |
| `o2f_nommap` | use-mmap false | |
| `o2g_all` | all working-0.6B knobs | |

If any PASS — the weather confirmation, verbatim:

```


```

## O3 — Test M deconfounded

Reference: correct output is `' 2+2=4. 2+2=4. …'` — **repetition IS correct**.

| run | output (first 120 chars, verbatim) | CORRECT / 4B-SIGNATURE / UNCHANGED |
|---|---|---|
| `o3a_4bknobs` | | |
| `o3b_nogswitch` | | |

## O4 — restore probe

```
[o4_restore.txt verbatim -- including any error; the error IS data]


```

## O5 — logcat

`o5_logcat.txt` attached whole:  YES / NO

## O6 — end-to-end (only with a passing O2 config)

Config used: ______________________________

| gate | result |
|---|---|
| text: `4` then stop | |
| sample image caption (paste) | |
| 6 photos — one-line judgement each | |
| timing cold / warm (init, TTFT, tok/s) | |

---

## Anything else

Skipped stages, retries, surprises. Observation vs interpretation — label which
is which.

```


```
