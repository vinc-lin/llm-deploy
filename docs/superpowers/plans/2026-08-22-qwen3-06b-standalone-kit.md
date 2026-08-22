# Qwen3-0.6B Standalone Build Kit — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build `kit-06b/` — a self-contained tree that reproduces the device-measured 44.707 tok/s Qwen3-0.6B SA8797P bundle and its ranked speed-variant arms, without depending on anything else in this repo.

**Architecture:** Four stages with clean interfaces — `quantize.sh` (ONNX + one encodings file) → `convert.sh` (3 DLCs) → `ctxbin.sh` (1 weight-shared ctx-bin) → `bundle.sh` (flat device tarball). Every stage runs standalone against the previous stage's output directory. A shared Python library (`gates/ctxbin_info.py`) reads finalized ctx-bins, and every gate is a separate runnable script. Tasks are ordered cheapest-verification-first: the ctx-bin stage is validated in ~5 min against DLCs already on disk before the ~1 h quantization path is touched.

**Tech Stack:** bash, Python 3.10 (system, has pytest 9.0.2), QAIRT SDK 2.48.40.260702 (`qairt-converter`, `qnn-context-binary-generator`, `qnn-context-binary-utility`, `qairt-dlc-info`), AIMET (in the `qwen3-deploy` uv env), pytest + bats + shellcheck for tests.

**Design spec:** `docs/superpowers/specs/2026-08-22-qwen3-06b-standalone-kit-design.md`

---

## Ground truth — verified on disk 2026-08-22

Every number below was read off the real artifacts in `/home/vinc/llm-local/`.
Tests assert against these. Do not substitute values from memory or from docs.

**Reference ctx-bin** — `work/ctxbin/qwen3-0.6b-w8a16-gqafix-ladekv/`:

| | |
|---|---|
| `qwen3-0.6b-w8a16-gqafix-ladekv_ctx.bin` | md5 `9c6024ad5b141137fbe22f3a4972eb96`, **1,086,570,496** B |
| graphs (name → `attention_mask` dims) | `prefill` → `[1,128,1152]`, `decode` → `[1,1,1152]`, `verify32` → `[1,32,1152]` |
| tuning (all 3 graphs) | `numHvxThreads` 4, `vtcmSize` 16, `optimizationLevel` 3, `spillFillBufferSize` 0 |
| weights | `sharedWeightsSize` 1,067,499,520 (identical on every graph); `constSize` prefill 0 / decode 256 / verify32 0 |
| decode DDR | `read_total_bytes` 961,130,496 · `write_total_bytes` 419,840 |

**Source DLCs** for that bin:

| Path (under `/home/vinc/llm-local/work/dlc/`) | md5 | size |
|---|---|---|
| `qwen3-0.6b-w8a16-gqafix-ladekv/prefill.dlc` | `45840b748efd3e5047b8ba7fb7fb8394` | 1,074,230,392 |
| `qwen3-0.6b-w8a16-gqafix/decode.dlc` | `f1820a378968c36b9020ae664dca2f14` | 1,074,066,208 |
| `qwen3-0.6b-w8a16-gqafix/verify32.dlc` | `a34007ad59cba5eee70cf3757c96978f` | 1,074,230,408 |

⚠ The ship prefill comes from the **`-ladekv` DLC dir**, not the base dir. The
base dir's `prefill.dlc` (md5 `924113949b674a6133e97954863fa464`) is the
bertcache graph and must never enter the ship bin. Both are named `prefill.dlc`
because the graph name is baked in from the filename.

**Class A variant bins** (same DLCs, config-only difference):

| Arm | size | Δ vs ship | md5 |
|---|---:|---:|---|
| `hvx8` | 1,088,847,872 | **+2,277,376** | `b533db214384cbd9401ef045ba259c75` |
| `socmodel72` | 1,086,820,352 | +249,856 | — |
| `udma` | 1,086,783,488 | +212,992 | — |

**info.json structure trap — the two blob structs are different shapes:**

```
graphs[i].info.graphBlobInfo.info   → numHvxThreads, vtcmSize, optimizationLevel, spillFillBufferSize   (V1)
graphs[i].info.graphBlobInfoV2      → sharedWeightsSize, constSize                                       (V2)
```

V1 wraps its payload in a further `.info`; **V2 does not.** V1 has no weight
fields at all; V2 has no tuning fields at all. A gate that uses one path for
both silently reads `None` and passes. Key spellings are exact: `vtcmSize` (not
`vtcmSizeInMB`), `graphBlobInfoV2` (not `graphBlobInfo2`).

**Tooling available:** system `python3` 3.10.12 with `pytest` 9.0.2 (the
`qwen3-deploy` env has **no** pytest — run tests with system python3),
`shellcheck`, `bats`.

---

## File Structure

```
kit-06b/
├── README.md                    # Task 11
├── env.sh                       # Task 1  — paths, disk_guard; sourced by every script
├── setup/
│   ├── check_sdk.sh             # Task 1  — SDK version + tool/lib presence
│   ├── make_envs.sh             # Task 10 — the two Python envs
│   └── fetch_model.sh           # Task 10 — HF checkpoint + tokenizer
├── build/
│   ├── quantize.sh              # Task 8  — calibration + 3 adopted exports
│   ├── convert.sh               # Task 7  — 3 DLCs against ONE encodings file
│   └── ctxbin.sh                # Task 4  — config generator + generation + gate
├── variants/
│   ├── arms.tsv                 # Task 5  — the ledger (data, not code)
│   └── build_arm.sh             # Task 5 (class A), Task 9 (class B)
├── bundle/
│   ├── bundle.sh                # Task 6
│   └── configs/
│       ├── genie_dialog_basic.json   # Task 6 — the 44.707 config, first-class file
│       └── genie_dialog_lade.json    # Task 6 — shipped for optionality only
├── gates/
│   ├── ctxbin_info.py           # Task 2  — shared introspection library
│   ├── check_ctxbin.py          # Task 3  — assertions CLI
│   ├── check_dialogs.py         # Task 6  — dialog contract linter
│   ├── check_gqa_ops.py         # Task 7  — 0 replication ops in every graph
│   └── check_bytes.py           # Task 7  — DDR byte accounting from build logs
├── build.sh                     # Task 9  — top-level: ship | variants
└── tests/
    ├── test_ctxbin_info.py      # Task 2
    ├── test_check_ctxbin.py     # Task 3
    ├── test_check_dialogs.py    # Task 6
    ├── test_arms.py             # Task 5
    ├── test_env.bats            # Task 1
    └── fixtures/                # Task 2 — trimmed info.json copies
```

Responsibility split: `gates/ctxbin_info.py` is the only place that knows
info.json's shape; every gate consumes it. `build/*.sh` orchestrate SDK tools and
never parse JSON inline. `bundle/configs/*.json` are data files, never heredocs.

---

## Task 1: Kit skeleton, portable env.sh, SDK check

**Files:**
- Create: `kit-06b/env.sh`
- Create: `kit-06b/setup/check_sdk.sh`
- Test: `kit-06b/tests/test_env.bats`

- [ ] **Step 1: Write the failing test**

Create `kit-06b/tests/test_env.bats`:

```bash
#!/usr/bin/env bats

setup() {
  KIT="$(cd "$(dirname "$BATS_TEST_FILENAME")/.." && pwd)"
  export KIT
}

@test "env.sh derives KIT_ROOT from its own location" {
  run bash -c "source '$KIT/env.sh' >/dev/null 2>&1; echo \$KIT_ROOT"
  [ "$status" -eq 0 ]
  [ "$output" = "$KIT" ]
}

@test "env.sh honours a pre-set KIT_DATA" {
  run bash -c "KIT_DATA=/tmp/kitdata source '$KIT/env.sh' >/dev/null 2>&1; echo \$KIT_DATA"
  [ "$output" = "/tmp/kitdata" ]
}

@test "disk_guard passes when the requirement is trivially small" {
  run bash -c "source '$KIT/env.sh' >/dev/null 2>&1; disk_guard 1; echo OK"
  [ "$status" -eq 0 ]
  [[ "$output" == *OK* ]]
}

@test "disk_guard aborts when the requirement is absurdly large" {
  run bash -c "source '$KIT/env.sh' >/dev/null 2>&1; disk_guard 999999999"
  [ "$status" -ne 0 ]
  [[ "$output" == *"free space"* ]]
}

@test "disk_guard guards C: when /mnt/c exists, else KIT_DATA" {
  run bash -c "source '$KIT/env.sh' >/dev/null 2>&1; disk_guard_target"
  [ "$status" -eq 0 ]
  if [ -d /mnt/c ]; then [ "$output" = "/mnt/c" ]; else [ "$output" = "$KIT_DATA" ]; fi
}
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `bats kit-06b/tests/test_env.bats`
Expected: FAIL — every test errors, `kit-06b/env.sh: No such file or directory`.

- [ ] **Step 3: Write env.sh**

Create `kit-06b/env.sh`:

```bash
#!/usr/bin/env bash
# Qwen3-0.6B standalone kit environment. Source me first in every shell.
# Self-deriving: a checkout anywhere works unchanged. Everything is overridable
# from the environment so the kit runs on a machine that looks nothing like the
# one it was written on.

export KIT_ROOT=${KIT_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)}
export KIT_DATA=${KIT_DATA:-$HOME/llm-local}

export QAIRT_SDK=${QAIRT_SDK:-$KIT_DATA/sdk/qairt/2.48.40.260702}
export PY_DEPLOY=${PY_DEPLOY:-$KIT_DATA/envs/qwen3-deploy/bin/python}
export PY_QAIRT=${PY_QAIRT:-$KIT_DATA/envs/qairt-py312/bin/python}
export MODEL=${MODEL:-$KIT_DATA/models/Qwen3-0.6B}

# Tests and gates run on the system interpreter: the deploy env has no pytest.
export PY_TEST=${PY_TEST:-python3}

# cuda where present, cpu otherwise. Anything above 0.6B needs cpu on an 8 GB
# card, and a no-GPU host has no choice.
if [ -z "${QUANT_DEVICE:-}" ]; then
    if "$PY_DEPLOY" -c "import torch,sys; sys.exit(0 if torch.cuda.is_available() else 1)" 2>/dev/null
    then export QUANT_DEVICE=cuda; else export QUANT_DEVICE=cpu; fi
fi

if [ -d "$QAIRT_SDK" ]; then
    export PATH=$QAIRT_SDK/bin/x86_64-linux-clang:$PATH
    export LD_LIBRARY_PATH=$QAIRT_SDK/lib/x86_64-linux-clang${KIT_SYSLIBS:+:$KIT_SYSLIBS}${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}
    export PYTHONPATH=$QAIRT_SDK/lib/python${PYTHONPATH:+:$PYTHONPATH}
fi

# Which volume the writes actually land on. On WSL, $KIT_DATA sits inside an
# ext4.vhdx backed by Windows C:, and the guest's own df reports the vhdx's
# virtual free space, not the host's. A failed vhdx grow does NOT surface as
# ENOSPC: the host write fails and the kernel delivers SIGBUS to every process
# touching an mmap'd page, which killed PID 1 and hard-crashed the VM three
# times on 2026-08-12. Off WSL there is no indirection, so guard the data volume.
disk_guard_target() {
    if [ -d /mnt/c ]; then echo /mnt/c; else echo "$KIT_DATA"; fi
}

# Call before any multi-GB step, sized to THAT step: 6 GB is the converter
# floor, an export writes ~8.6 GB and should ask 20. A flat 6 GB check passes
# and then still runs the volume dry mid-step.
disk_guard() {
    local need_gb=${1:-6} free_gb target
    target=$(disk_guard_target)
    free_gb=$(df --output=avail -BG "$target" 2>/dev/null | tail -1 | tr -dc 0-9)
    if [ -z "$free_gb" ]; then
        echo "ABORT: disk_guard cannot read free space on $target" >&2; return 1
    fi
    if (( free_gb < need_gb )); then
        echo "ABORT: $target free space ${free_gb}GB < ${need_gb}GB" >&2; return 1
    fi
}
```

- [ ] **Step 4: Run the test to verify it passes**

Run: `bats kit-06b/tests/test_env.bats`
Expected: PASS — `5 tests, 0 failures`.

- [ ] **Step 5: Write setup/check_sdk.sh**

Quantization behaviour is SDK-specific, so a version mismatch invalidates every
number in the README. Create `kit-06b/setup/check_sdk.sh`:

```bash
#!/usr/bin/env bash
# Verify the SDK is the exact version this kit's measured numbers came from,
# and that every tool and library the build touches is present.
set -euo pipefail
source "$(dirname "$0")/../env.sh"

WANT_SDK=2.48.40.260702
fail=0
note() { echo "  $1"; }
bad()  { echo "  MISSING: $1"; fail=1; }

echo "== SDK =="
[ -d "$QAIRT_SDK" ] || { echo "ABORT: QAIRT_SDK not found: $QAIRT_SDK" >&2; exit 1; }
note "path: $QAIRT_SDK"
case "$QAIRT_SDK" in
  *"$WANT_SDK"*) note "version: $WANT_SDK OK" ;;
  *) echo "  WARN: expected $WANT_SDK; the kit's measured numbers are SDK-specific" ;;
esac

echo "== x86 build tools =="
for t in qairt-converter qnn-context-binary-generator \
         qnn-context-binary-utility qairt-dlc-info; do
    [ -x "$QAIRT_SDK/bin/x86_64-linux-clang/$t" ] && note "$t" || bad "bin/x86_64-linux-clang/$t"
done

echo "== x86 backend libraries =="
for l in libQnnModelDlc.so libQnnHtp.so; do
    [ -f "$QAIRT_SDK/lib/x86_64-linux-clang/$l" ] && note "$l" || bad "lib/x86_64-linux-clang/$l"
done

echo "== device payload (bundled, not run here) =="
for l in libGenie.so libQnnHtp.so libQnnSystem.so libQnnHtpPrepare.so \
         libQnnHtpNetRunExtensions.so libQnnHtpV81Stub.so; do
    [ -f "$QAIRT_SDK/lib/aarch64-android/$l" ] && note "$l" || bad "lib/aarch64-android/$l"
done
[ -f "$QAIRT_SDK/lib/hexagon-v81/unsigned/libQnnHtpV81Skel.so" ] \
    && note "libQnnHtpV81Skel.so" || bad "lib/hexagon-v81/unsigned/libQnnHtpV81Skel.so"
[ -f "$QAIRT_SDK/bin/aarch64-android/genie-t2t-run" ] \
    && note "genie-t2t-run" || bad "bin/aarch64-android/genie-t2t-run"

echo "== interpreters =="
[ -x "$PY_DEPLOY" ] && note "PY_DEPLOY $($PY_DEPLOY -V 2>&1)" || bad "PY_DEPLOY ($PY_DEPLOY)"
[ -x "$PY_QAIRT" ]  && note "PY_QAIRT $($PY_QAIRT -V 2>&1)"  || bad "PY_QAIRT ($PY_QAIRT)"
note "QUANT_DEVICE=$QUANT_DEVICE"

echo "== disk =="
note "guarding $(disk_guard_target)"
disk_guard 20 && note "≥20GB free" || fail=1

(( fail == 0 )) && echo "SDK CHECK PASSED" || { echo "SDK CHECK FAILED" >&2; exit 1; }
```

- [ ] **Step 6: Run the SDK check and shellcheck**

Run:
```bash
chmod +x kit-06b/setup/check_sdk.sh
bash kit-06b/setup/check_sdk.sh
shellcheck kit-06b/env.sh kit-06b/setup/check_sdk.sh
```
Expected: `SDK CHECK PASSED`, and shellcheck silent (exit 0).

- [ ] **Step 7: Commit**

```bash
git add kit-06b/env.sh kit-06b/setup/check_sdk.sh kit-06b/tests/test_env.bats
git commit -m "kit: portable env.sh and SDK check

env.sh self-derives KIT_ROOT and takes KIT_DATA/QAIRT_SDK/PY_* from the
environment, so the kit runs off this box. disk_guard is retained and still
guards Windows C: when /mnt/c exists -- a failed vhdx grow delivers SIGBUS
rather than ENOSPC, which hard-crashed the VM three times on 2026-08-12."
```

---

## Task 2: `gates/ctxbin_info.py` — ctx-bin introspection library

The single place that knows info.json's shape. Every gate consumes it.

**Files:**
- Create: `kit-06b/gates/ctxbin_info.py`
- Create: `kit-06b/tests/fixtures/make_fixtures.sh`
- Test: `kit-06b/tests/test_ctxbin_info.py`

- [ ] **Step 1: Build the test fixtures**

The real info.json is 400 KB; trim it to the fields the gates read. Create
`kit-06b/tests/fixtures/make_fixtures.sh`:

```bash
#!/usr/bin/env bash
# Regenerate the trimmed info.json fixtures from real ctx-bins on this box.
# Committed outputs are the source of truth for tests; this script exists so a
# future SDK's output can be re-captured rather than hand-edited.
set -euo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
SRC=${1:-$HOME/llm-local/work/ctxbin}

trim() {  # trim <in.json> <out.json>
python3 - "$1" "$2" <<'PYEOF'
import json, sys
d = json.load(open(sys.argv[1]))
out = {"info": {"graphs": []}}
for g in d["info"]["graphs"]:
    gi = g["info"]
    keep = {
        "graphName": gi["graphName"],
        "graphInputs": [t for t in gi.get("graphInputs", [])
                        if (t.get("info", t)).get("name") == "attention_mask"],
        "graphBlobInfo": gi.get("graphBlobInfo"),
        "graphBlobInfoV2": gi.get("graphBlobInfoV2"),
    }
    out["info"]["graphs"].append({"info": keep})
json.dump(out, open(sys.argv[2], "w"), indent=2)
PYEOF
}

trim "$SRC/qwen3-0.6b-w8a16-gqafix-ladekv/info.json"      "$HERE/ship_info.json"
trim "$SRC/qwen3-0.6b-w8a16-gqafix-hvx8-ladekv/info.json" "$HERE/hvx8_info.json"
echo "fixtures written to $HERE"
```

Run:
```bash
chmod +x kit-06b/tests/fixtures/make_fixtures.sh
bash kit-06b/tests/fixtures/make_fixtures.sh
python3 -c "import json;d=json.load(open('kit-06b/tests/fixtures/ship_info.json'));print([g['info']['graphName'] for g in d['info']['graphs']])"
```
Expected: `['prefill', 'decode', 'verify32']`

- [ ] **Step 2: Write the failing test**

Create `kit-06b/tests/test_ctxbin_info.py`:

```python
"""Tests for the ctx-bin introspection library.

Values are read off real artifacts (see the plan's ground-truth table), not
from documentation. If the SDK changes shape these tests fail loudly, which is
the point -- info.json key names are not guessable and this repo has been
bitten three times by silently reading a field that never existed.
"""
import json
import pathlib
import sys

import pytest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "gates"))
import ctxbin_info as ci  # noqa: E402

FIX = pathlib.Path(__file__).parent / "fixtures"
SHIP = FIX / "ship_info.json"
HVX8 = FIX / "hvx8_info.json"


@pytest.fixture
def ship():
    return ci.load(SHIP)


def test_graph_names_in_order(ship):
    assert ci.graph_names(ship) == ["prefill", "decode", "verify32"]


def test_ar_cl_per_graph(ship):
    assert ci.ar_cl(ship) == {
        "prefill": (128, 1152),
        "decode": (1, 1152),
        "verify32": (32, 1152),
    }


def test_tuning_reads_the_v1_struct(ship):
    # graphBlobInfo.info -- V1 wraps its payload in a further ".info"
    assert ci.tuning(ship) == {
        "prefill": {"numHvxThreads": 4, "vtcmSize": 16,
                    "optimizationLevel": 3, "spillFillBufferSize": 0},
        "decode": {"numHvxThreads": 4, "vtcmSize": 16,
                   "optimizationLevel": 3, "spillFillBufferSize": 0},
        "verify32": {"numHvxThreads": 4, "vtcmSize": 16,
                     "optimizationLevel": 3, "spillFillBufferSize": 0},
    }


def test_hvx8_fixture_differs_only_in_thread_count():
    got = ci.tuning(ci.load(HVX8))
    assert {g["numHvxThreads"] for g in got.values()} == {8}
    assert {g["vtcmSize"] for g in got.values()} == {16}


def test_weights_read_the_v2_struct(ship):
    # graphBlobInfoV2 -- NOT nested under ".info", unlike V1
    w = ci.weights(ship)
    assert w["shared_pool"] == 1_067_499_520
    assert w["private"] == {"prefill": 0, "decode": 256, "verify32": 0}
    assert w["private_total"] == 256


def test_pooled_fraction_is_essentially_one(ship):
    assert ci.pooled_fraction(ship) > 0.9999


def test_shared_pool_rejects_disagreeing_graphs(tmp_path):
    # The pool is one context-wide number printed identically on every graph.
    # If graphs disagree, the file is not what we think it is -- never sum it.
    d = json.loads(SHIP.read_text())
    d["info"]["graphs"][1]["info"]["graphBlobInfoV2"]["sharedWeightsSize"] = 123
    p = tmp_path / "bad.json"
    p.write_text(json.dumps(d))
    with pytest.raises(ValueError, match="sharedWeightsSize disagrees"):
        ci.weights(ci.load(p))


def test_missing_v2_struct_raises_rather_than_returning_none(tmp_path):
    d = json.loads(SHIP.read_text())
    for g in d["info"]["graphs"]:
        del g["info"]["graphBlobInfoV2"]
    p = tmp_path / "v1only.json"
    p.write_text(json.dumps(d))
    with pytest.raises(ValueError, match="graphBlobInfoV2"):
        ci.weights(ci.load(p))
```

- [ ] **Step 3: Run the test to verify it fails**

Run: `python3 -m pytest kit-06b/tests/test_ctxbin_info.py -v`
Expected: FAIL — collection error, `ModuleNotFoundError: No module named 'ctxbin_info'`.

- [ ] **Step 4: Write the implementation**

Create `kit-06b/gates/ctxbin_info.py`:

```python
"""Read a finalized QNN context binary's info.json.

Produce the JSON with:
    qnn-context-binary-utility --context_binary X.bin --json_file X.info.json

THE TWO BLOB STRUCTS ARE DIFFERENT SHAPES. This is the whole reason this module
exists in one place:

    graphs[i].info.graphBlobInfo.info  -> numHvxThreads, vtcmSize,
                                          optimizationLevel, spillFillBufferSize
    graphs[i].info.graphBlobInfoV2     -> sharedWeightsSize, constSize

V1 wraps its payload in a further ".info"; V2 does not. V1 carries no weight
fields at all and V2 carries no tuning fields at all, so a reader that uses one
path for both silently gets None and every assertion built on it passes
vacuously. Key spellings are exact: vtcmSize (not vtcmSizeInMB), graphBlobInfoV2
(not graphBlobInfo2).
"""
from __future__ import annotations

import json
import pathlib
from typing import Any

TUNING_KEYS = ("numHvxThreads", "vtcmSize", "optimizationLevel",
               "spillFillBufferSize")


def load(path: str | pathlib.Path) -> dict:
    """Parse an info.json into the raw dict the other helpers consume."""
    return json.loads(pathlib.Path(path).read_text())


def _graphs(info: dict) -> list[dict]:
    try:
        return [g["info"] for g in info["info"]["graphs"]]
    except (KeyError, TypeError) as exc:
        raise ValueError(f"not a context-binary info.json: {exc}") from exc


def graph_names(info: dict) -> list[str]:
    """Graph names in file order. These are baked in at conversion time from
    each DLC's --output_path basename and cannot be changed by renaming."""
    return [g["graphName"] for g in _graphs(info)]


def ar_cl(info: dict) -> dict[str, tuple[int, int]]:
    """(AR, CL) per graph, from attention_mask's last two dimensions.

    Genie selects graphs by numeric best-fit on this pair, so two graphs sharing
    one pair are indistinguishable to it.
    """
    out: dict[str, tuple[int, int]] = {}
    for g in _graphs(info):
        masks = [t for t in g.get("graphInputs", [])
                 if (t.get("info", t)).get("name") == "attention_mask"]
        if not masks:
            raise ValueError(f"graph {g['graphName']} has no attention_mask input")
        dims = (masks[0].get("info", masks[0])).get("dimensions", [])
        if len(dims) < 2:
            raise ValueError(f"graph {g['graphName']} attention_mask dims={dims}")
        out[g["graphName"]] = (int(dims[-2]), int(dims[-1]))
    return out


def tuning(info: dict) -> dict[str, dict[str, Any]]:
    """Compiled tuning values per graph, from the V1 struct.

    These are baked in at ctx-bin build time. hvx_threads in particular is NOT
    read at runtime -- editing it in a deployed backend config does nothing
    (measured: -0.1%).
    """
    out: dict[str, dict[str, Any]] = {}
    for g in _graphs(info):
        blob = (g.get("graphBlobInfo") or {}).get("info")
        if blob is None:
            raise ValueError(
                f"graph {g['graphName']}: no graphBlobInfo.info -- cannot read "
                "compiled tuning values")
        out[g["graphName"]] = {k: blob.get(k) for k in TUNING_KEYS}
    return out


def weights(info: dict) -> dict[str, Any]:
    """Weight-sharing accounting, from the V2 struct.

    sharedWeightsSize is the context's single pool, printed identically on every
    graph -- never sum it. constSize is that graph's PRIVATE copy (the SDK
    header's own map calls it "Non-Shared (Const)").
    """
    pools: set[int] = set()
    private: dict[str, int] = {}
    for g in _graphs(info):
        blob = g.get("graphBlobInfoV2")
        if blob is None:
            raise ValueError(
                f"graph {g['graphName']}: no graphBlobInfoV2 -- the V1 struct "
                "has no weight fields, so weight sharing cannot be verified")
        pools.add(int(blob["sharedWeightsSize"]))
        private[g["graphName"]] = int(blob["constSize"])
    if len(pools) != 1:
        raise ValueError(
            f"sharedWeightsSize disagrees across graphs: {sorted(pools)} -- it is "
            "one context-wide pool and must be identical on every graph")
    pool = pools.pop()
    return {"shared_pool": pool, "private": private,
            "private_total": sum(private.values())}


def pooled_fraction(info: dict) -> float:
    """Share of total weight bytes living in the shared pool.

    Gate on THIS, not on constSize == 0. A hard-zero gate rejects good bins:
    --quant-head moves ~144 MB into a private decode block by design, and a
    bertcache graph forces a private ~444 MB copy.
    """
    w = weights(info)
    total = w["shared_pool"] + w["private_total"]
    return w["shared_pool"] / total if total else 0.0
```

- [ ] **Step 5: Run the test to verify it passes**

Run: `python3 -m pytest kit-06b/tests/test_ctxbin_info.py -v`
Expected: PASS — `8 passed`.

- [ ] **Step 6: Commit**

```bash
git add kit-06b/gates/ctxbin_info.py kit-06b/tests/test_ctxbin_info.py \
        kit-06b/tests/fixtures/
git commit -m "kit: ctx-bin introspection library

One place that knows info.json's shape. The two blob structs differ: V1
(graphBlobInfo.info) carries tuning, V2 (graphBlobInfoV2, not nested) carries
weights, and neither carries the other's fields -- a reader using one path for
both silently returns None and every assertion on it passes vacuously.

weights() refuses a file whose graphs disagree on sharedWeightsSize, and
pooled_fraction() is the weight-sharing metric rather than constSize == 0,
which would reject good bins."
```

---

## Task 3: `gates/check_ctxbin.py` — the assertions CLI

**Files:**
- Create: `kit-06b/gates/check_ctxbin.py`
- Test: `kit-06b/tests/test_check_ctxbin.py`

- [ ] **Step 1: Write the failing test**

Create `kit-06b/tests/test_check_ctxbin.py`:

```python
"""Tests for the ctx-bin assertions CLI."""
import json
import pathlib
import subprocess
import sys

import pytest

GATES = pathlib.Path(__file__).resolve().parents[1] / "gates"
FIX = pathlib.Path(__file__).parent / "fixtures"
SHIP = FIX / "ship_info.json"
HVX8 = FIX / "hvx8_info.json"


def run(*args):
    return subprocess.run(
        [sys.executable, str(GATES / "check_ctxbin.py"), *map(str, args)],
        capture_output=True, text=True)


def test_ship_bin_passes_every_check():
    r = run(SHIP, "--graphs", "prefill,decode,verify32", "--expect-hvx", "4")
    assert r.returncode == 0, r.stdout + r.stderr
    assert "ALL CHECKS PASSED" in r.stdout


def test_wrong_graph_names_fail():
    r = run(SHIP, "--graphs", "prefill,decode,verify16", "--expect-hvx", "4")
    assert r.returncode != 0
    assert "graph names" in (r.stdout + r.stderr)


def test_hvx_mismatch_fails_because_a_silent_one_reads_as_a_measurement():
    r = run(SHIP, "--graphs", "prefill,decode,verify32", "--expect-hvx", "8")
    assert r.returncode != 0
    assert "numHvxThreads" in (r.stdout + r.stderr)


def test_hvx8_fixture_passes_with_expect_hvx_8():
    r = run(HVX8, "--graphs", "prefill,decode,verify32", "--expect-hvx", "8")
    assert r.returncode == 0, r.stdout + r.stderr


def test_duplicate_ar_cl_fails(tmp_path):
    d = json.loads(SHIP.read_text())
    # make verify32 claim decode's (AR, CL) -- Genie could not tell them apart
    for g in d["info"]["graphs"]:
        if g["info"]["graphName"] == "verify32":
            g["info"]["graphInputs"][0]["info"]["dimensions"] = [1, 1, 1152]
    p = tmp_path / "dup.json"
    p.write_text(json.dumps(d))
    r = run(p, "--graphs", "prefill,decode,verify32", "--expect-hvx", "4")
    assert r.returncode != 0
    assert "AR, CL" in (r.stdout + r.stderr)


def test_a_bertcache_graph_fails_the_purity_check(tmp_path):
    # AR == CL means the graph registers ctx_size == AR and is never selected
    # for prompts longer than AR. A bin carrying one produces a PHASE BLEND,
    # not a decode rate -- that is where the phantom 11.72 tok/s came from.
    d = json.loads(SHIP.read_text())
    for g in d["info"]["graphs"]:
        if g["info"]["graphName"] == "prefill":
            g["info"]["graphInputs"][0]["info"]["dimensions"] = [1, 128, 128]
    p = tmp_path / "bertcache.json"
    p.write_text(json.dumps(d))
    r = run(p, "--graphs", "prefill,decode,verify32", "--expect-hvx", "4")
    assert r.returncode != 0
    assert "bertcache" in (r.stdout + r.stderr).lower()


def test_low_pooled_fraction_fails(tmp_path):
    d = json.loads(SHIP.read_text())
    for g in d["info"]["graphs"]:
        g["info"]["graphBlobInfoV2"]["constSize"] = 755_000_000
    p = tmp_path / "unshared.json"
    p.write_text(json.dumps(d))
    r = run(p, "--graphs", "prefill,decode,verify32", "--expect-hvx", "4")
    assert r.returncode != 0
    assert "pooled" in (r.stdout + r.stderr).lower()


def test_quant_head_exception_permits_a_private_block(tmp_path):
    # --quant-head moves ~144 MB private on decode BY DESIGN. Gating on
    # constSize == 0 would reject a good bin.
    d = json.loads(SHIP.read_text())
    for g in d["info"]["graphs"]:
        if g["info"]["graphName"] == "decode":
            g["info"]["graphBlobInfoV2"]["constSize"] = 144_000_000
    p = tmp_path / "qh.json"
    p.write_text(json.dumps(d))
    assert run(p, "--graphs", "prefill,decode,verify32", "--expect-hvx", "4").returncode != 0
    r = run(p, "--graphs", "prefill,decode,verify32", "--expect-hvx", "4",
            "--min-pooled", "0.85")
    assert r.returncode == 0, r.stdout + r.stderr
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `python3 -m pytest kit-06b/tests/test_check_ctxbin.py -v`
Expected: FAIL — all 8 tests fail; `check_ctxbin.py` does not exist so every
`returncode` is 2 with `can't open file`.

- [ ] **Step 3: Write the implementation**

Create `kit-06b/gates/check_ctxbin.py`:

```python
#!/usr/bin/env python3
"""Assert a finalized ctx-bin matches what was asked for.

Run against the info.json emitted by qnn-context-binary-utility. Every check
here has cost this project a real failure at least once:

  graph names   a mismatch against the backend config's graph_names does not
                error -- that graph silently takes backend defaults (4 MB VTCM,
                24 MB spill), or SIGSEGVs on the first lade speculation step.
  (AR, CL)      Genie dispatches by numeric best-fit; two graphs sharing a pair
                are indistinguishable to it.
  tuning        a config key that failed to bind reads as "the knob under test
                did nothing" -- the most expensive possible failure, because it
                looks like a measurement rather than an error.
  pooled        weight sharing is silently optional; an unshared bin still runs,
                just ~700 MB bigger per graph.
"""
from __future__ import annotations

import argparse
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import ctxbin_info as ci


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("info_json", type=pathlib.Path)
    ap.add_argument("--graphs", required=True,
                    help="comma-separated names expected in the bin")
    ap.add_argument("--expect-hvx", type=int, required=True,
                    help="compiled numHvxThreads (build-time; runtime cannot change it)")
    ap.add_argument("--expect-vtcm", type=int, default=16,
                    help="compiled vtcmSize in MB; 16 is the ceiling (24 is "
                         "accepted offline and rejected at runtime, 0x138d)")
    ap.add_argument("--expect-o", type=int, default=3)
    ap.add_argument("--min-pooled", type=float, default=0.99,
                    help="minimum shared-pool fraction. Lower it deliberately "
                         "for known-private designs: --quant-head moves ~144 MB "
                         "private on decode, a bertcache graph ~444 MB")
    ap.add_argument("--allow-bertcache", action="store_true",
                    help="permit a graph with AR == CL. Off by default: such a "
                         "graph is never selected for prompts longer than AR, "
                         "and a bin carrying one reports a phase blend rather "
                         "than a decode rate")
    args = ap.parse_args()

    info = ci.load(args.info_json)
    want = args.graphs.split(",")
    failures: list[str] = []

    got = ci.graph_names(info)
    print(f"graphs in bin: {got}")
    if sorted(got) != sorted(want):
        failures.append(
            f"graph names {got} != configured {want} -- the unmatched graph "
            "would silently take backend defaults")

    pairs = ci.ar_cl(info)
    print(f"(AR, CL) per graph: {pairs}")
    seen: dict[tuple[int, int], str] = {}
    for name, pair in pairs.items():
        if pair in seen:
            failures.append(
                f"{name} and {seen[pair]} share (AR, CL)={pair} -- Genie picks "
                "by numeric best-fit and cannot distinguish them")
        seen[pair] = name
        ar, cl = pair
        if ar == cl and not args.allow_bertcache:
            failures.append(
                f"{name} has AR == CL == {ar} -- a bertcache graph. It registers "
                "ctx_size == AR and is never selected for prompts longer than "
                "AR, so the whole prompt silently goes through AR=1 decode. A "
                "bin carrying one yields a PHASE BLEND, not a decode rate")

    tune = ci.tuning(info)
    print(f"compiled tuning: {tune}")
    expect = {"numHvxThreads": args.expect_hvx, "vtcmSize": args.expect_vtcm,
              "optimizationLevel": args.expect_o}
    for name, vals in tune.items():
        for key, exp in expect.items():
            if vals.get(key) != exp:
                failures.append(
                    f"{name}: {key} compiled {vals.get(key)}, config asked {exp} "
                    "-- any A/B built on this binary measures nothing")

    w = ci.weights(info)
    frac = ci.pooled_fraction(info)
    print(f"shared pool: {w['shared_pool']:,} B (one context-wide pool, not summed)")
    print(f"private per graph: {w['private']}  total {w['private_total']:,} B")
    print(f"pooled fraction: {frac:.6f} (floor {args.min_pooled})")
    if frac < args.min_pooled:
        failures.append(
            f"pooled fraction {frac:.4f} < {args.min_pooled} -- weight sharing "
            "is degraded; each graph is re-carrying its own copy")

    if failures:
        print("\nFAILED:", file=sys.stderr)
        for f in failures:
            print(f"  - {f}", file=sys.stderr)
        return 1
    print("\nALL CHECKS PASSED")
    return 0


if __name__ == "__main__":
    sys.exit(main())
```

- [ ] **Step 4: Run the test to verify it passes**

Run: `python3 -m pytest kit-06b/tests/test_check_ctxbin.py -v`
Expected: PASS — `8 passed`.

- [ ] **Step 5: Run it against the real reference bin**

Run:
```bash
python3 kit-06b/gates/check_ctxbin.py \
    ~/llm-local/work/ctxbin/qwen3-0.6b-w8a16-gqafix-ladekv/info.json \
    --graphs prefill,decode,verify32 --expect-hvx 4
```
Expected: `ALL CHECKS PASSED`, with `shared pool: 1,067,499,520 B`,
`private ... total 256 B`, `pooled fraction: 1.000000`.

- [ ] **Step 6: Commit**

```bash
git add kit-06b/gates/check_ctxbin.py kit-06b/tests/test_check_ctxbin.py
git commit -m "kit: ctx-bin assertions CLI

Gates graph names, (AR, CL) uniqueness, compiled tuning bind-back, and pooled
weight fraction. Pooled fraction rather than constSize == 0, with --min-pooled
so the known-private designs (--quant-head ~144 MB, bertcache ~444 MB) can be
admitted explicitly instead of forcing the gate off."
```

---

## Task 4: `build/ctxbin.sh` — the keystone

Generates the backend config from the graph names, builds the bin, gates it.
**This task's integration test reproduces the shipped md5 in ~5 minutes**, from
DLCs already on disk — proving the generated-config path is equivalent before
any of the expensive upstream work is touched.

**Files:**
- Create: `kit-06b/build/ctxbin.sh`

- [ ] **Step 1: Write the script**

Create `kit-06b/build/ctxbin.sh`:

```bash
#!/usr/bin/env bash
# Generate one weight-shared ctx-bin from an explicit DLC list, then gate it.
#
# Usage:
#   ctxbin.sh <out_dir> <bin_name_no_suffix> <dlc_csv> <graph_names_csv> [overrides_json]
#
# <overrides_json> is merged into the "graphs" entry (e.g. '{"hvx_threads":8}').
# Two escape hatches: a "__context" key merges into the "context" section and a
# "__devices" key into the "devices" entry.
#
# THE TRAP THIS AVOIDS
# --------------------
# A graph's name is baked in at conversion time from the --output_path basename,
# dots included -- converting to decode.dlc.new yields graph "decode_dlc", and
# renaming the file afterwards does NOT change it. graph_names in the backend
# config must match the names INSIDE the bin exactly. A mismatch does not error:
# that graph silently gets backend defaults (4 MB VTCM, 24 MB spill), or under
# lade a null-pointer SIGSEGV on the first speculation step. So the config is
# built FROM the names you pass, and the built bin is then verified against them.
set -euo pipefail
source "$(dirname "$0")/../env.sh"

OUT_DIR=$(realpath -m "${1:?out dir}")
BIN_NAME=${2:?binary name (no .bin)}
DLC_CSV=${3:?comma-separated dlc paths}
GRAPHS_CSV=${4:?comma-separated graph names}
OVERRIDES=${5:-'{}'}

IFS=',' read -r -a _dlcs <<< "$DLC_CSV"
for d in "${_dlcs[@]}"; do
    [[ -f $d ]] || { echo "ABORT: missing DLC: $d" >&2; exit 1; }
done

mkdir -p "$OUT_DIR"
CFGDIR=$(mktemp -d)
trap 'rm -rf "$CFGDIR"' EXIT

python3 - "$CFGDIR" "$GRAPHS_CSV" "$OVERRIDES" <<'PYEOF'
import json, sys
cfgdir, graphs_csv, overrides = sys.argv[1], sys.argv[2], json.loads(sys.argv[3])
ctx_over = overrides.pop("__context", {})
dev_over = overrides.pop("__devices", {})

# vtcm_mb 16 is the ceiling: 24 compiles offline and is rejected at runtime
# (0x138d). hvx_threads 4 is what the 44.707 tok/s bin was built with -- the
# part has 8, and raising it is the hvx8 arm, not a default.
graph = {"graph_names": graphs_csv.split(","), "O": 3, "vtcm_mb": 16,
         "hvx_threads": 4}
graph.update(overrides)

# soc_model 0 is generic. The SDK maps SA8797 to soc_id 72 and Qualcomm document
# extra O=3 algorithms behind naming it; that is the socmodel72 arm, never A/B'd.
device = {"dsp_arch": "v81", "soc_model": 0, "pd_session": "unsigned",
          "cores": [{"core_id": 0, "perf_profile": "burst",
                     "rpc_control_latency": 100, "rpc_polling_time": 9999}]}
device.update(dev_over)

# Deliberately NOT carried over from the old shared config, all three proven
# inert by ctrl == ship byte-identity:
#   memory.extended_udma            -- "memory" is extra="forbid" with exactly
#                                      one field, so it never applied. The real
#                                      key lives in "context" (the udma arm).
#   sparse_weights_compression      -- measured, 0 bytes saved, model not sparse.
#   fp16_relaxed_precision: 0       -- equals the default.
backend = {"graphs": [graph], "devices": [device],
           "context": dict({"weight_sharing_enabled": True}, **ctx_over)}
json.dump(backend, open(f"{cfgdir}/htp_backend_config.json", "w"), indent=2)
json.dump({"backend_extensions": {
    "shared_library_path": "libQnnHtpNetRunExtensions.so",
    "config_file_path": "htp_backend_config.json"}},
    open(f"{cfgdir}/htp_config.json", "w"), indent=2)
print("graph_names:", graph["graph_names"])
print("graph opts :", {k: v for k, v in graph.items() if k != "graph_names"})
print("context    :", backend["context"])
print("devices    :", {k: v for k, v in device.items() if k != "cores"})
PYEOF

disk_guard 6
echo "== generating ctx-bin =="
( cd "$CFGDIR" && qnn-context-binary-generator \
    --model "$QAIRT_SDK/lib/x86_64-linux-clang/libQnnModelDlc.so" \
    --dlc_path "$DLC_CSV" \
    --backend "$QAIRT_SDK/lib/x86_64-linux-clang/libQnnHtp.so" \
    --output_dir "$OUT_DIR" --binary_file "$BIN_NAME" \
    --config_file htp_config.json ) | tee "$OUT_DIR/$BIN_NAME.build.log"

qnn-context-binary-utility --context_binary "$OUT_DIR/$BIN_NAME.bin" \
    --json_file "$OUT_DIR/$BIN_NAME.info.json"

# The gate reads the requested hvx_threads back out of the FINALIZED binary.
EXPECT_HVX=$(python3 -c "import json,sys; print(json.loads(sys.argv[1]).get('hvx_threads', 4))" "$OVERRIDES")
python3 "$KIT_ROOT/gates/check_ctxbin.py" "$OUT_DIR/$BIN_NAME.info.json" \
    --graphs "$GRAPHS_CSV" --expect-hvx "$EXPECT_HVX" \
    ${MIN_POOLED:+--min-pooled "$MIN_POOLED"}

echo "== DDR byte accounting (record with the log name and date) =="
grep -E "read_total_bytes|write_total_bytes" "$OUT_DIR/$BIN_NAME.build.log" || \
    echo "  (no DDR summary in log)"

echo "md5: $(md5sum "$OUT_DIR/$BIN_NAME.bin" | cut -d' ' -f1)"
ls -l "$OUT_DIR/$BIN_NAME.bin"
echo "CTXBIN READY: $OUT_DIR/$BIN_NAME.bin"
```

- [ ] **Step 2: shellcheck it**

Run: `shellcheck kit-06b/build/ctxbin.sh`
Expected: exit 0, no output.

- [ ] **Step 3: THE KEYSTONE TEST — reproduce the shipped md5**

This is the acceptance criterion for the whole restructuring. Takes ~5 min and
needs ~6 GB free.

```bash
chmod +x kit-06b/build/ctxbin.sh
D=$HOME/llm-local
bash kit-06b/build/ctxbin.sh /tmp/kit-keystone qwen3-0.6b-w8a16-gqafix-ladekv_ctx \
  "$D/work/dlc/qwen3-0.6b-w8a16-gqafix-ladekv/prefill.dlc,$D/work/dlc/qwen3-0.6b-w8a16-gqafix/decode.dlc,$D/work/dlc/qwen3-0.6b-w8a16-gqafix/verify32.dlc" \
  prefill,decode,verify32
```

Expected, all of:
- `ALL CHECKS PASSED`
- `md5: 9c6024ad5b141137fbe22f3a4972eb96`
- size `1086570496`
- decode `read_total_bytes` 961,130,496 and `write_total_bytes` 419,840 in the log

**If the md5 differs, STOP.** The generated config is not equivalent to the
checked-in one and the premise of Task 4 is wrong. Diff `/tmp/kit-keystone/*.info.json`
against `~/llm-local/work/ctxbin/qwen3-0.6b-w8a16-gqafix-ladekv/info.json` and
reconcile before continuing — do not proceed to Task 5.

- [ ] **Step 4: Record the keystone result**

Run:
```bash
md5sum /tmp/kit-keystone/qwen3-0.6b-w8a16-gqafix-ladekv_ctx.bin \
  > kit-06b/tests/keystone_md5.txt
cat kit-06b/tests/keystone_md5.txt
```
Expected: the md5 is `9c6024ad5b141137fbe22f3a4972eb96`.

- [ ] **Step 5: Commit**

```bash
git add kit-06b/build/ctxbin.sh kit-06b/tests/keystone_md5.txt
git commit -m "kit: ctx-bin stage, generated backend config

Builds the config from the graph names passed in, so graph_names cannot drift
from the names baked into the DLC filenames -- a mismatch does not error, it
silently hands that graph backend defaults.

Verified: this path reproduces the shipped gqafix_ladekv bin byte for byte
(md5 9c6024ad5b141137fbe22f3a4972eb96, 1,086,570,496 B) from the three DLCs on
disk, which also confirms the three dropped config keys were inert."
```

---

## Task 5: Class A variant arms

**Files:**
- Create: `kit-06b/variants/arms.tsv`
- Create: `kit-06b/variants/build_arm.sh`
- Test: `kit-06b/tests/test_arms.py`

- [ ] **Step 1: Write the failing test first**

Write `kit-06b/tests/test_arms.py` exactly as given in Step 3 below, then run it
before `arms.tsv` exists:

Run: `python3 -m pytest kit-06b/tests/test_arms.py -v`
Expected: FAIL — 6 errors, `FileNotFoundError: .../variants/arms.tsv`.

- [ ] **Step 2: Write the ledger**

Create `kit-06b/variants/arms.tsv` (tab-separated; `byte_pct`/`compute_pct` are
**predictions**, never results):

```
arm	class	overrides	byte_pct	compute_pct	artifact_delta	note
ship	A	{}	0.0	0.0	0	the measured 44.707 tok/s baseline
hvx8	A	{"hvx_threads": 8}	0.0	unknown	2277376	priority 1 -- byte model predicts 0% BY CONSTRUCTION, so a win is unambiguous
socmodel72	A	{"__devices": {"soc_model": 72, "soc_id": 72}}	0.0	unknown	249856	free knob, never A/B'd
udma	A	{"__context": {"extended_udma": true}}	0.0	unknown	212992	key lives in context, NOT memory -- it never applied before
dlbc	A	{"dlbc": 1}	0.0	unknown	0	activation compression, not the weight stream; artifact unchanged -- rank last
wpack	A	{"weights_packing": 1}	0.0	unknown	0	never tried; artifact unchanged -- rank last
cl512	B	{}	10.1	26.0	-1589248	needs re-export: context 512 -> ctx-bin CL 640, decode past 639. Caps context at 512
qh	B	{}	17.9	3.6	-8384512	needs re-export with --quant-head; confounded, may be ~0% on device
fuseqkvgu	B	{}	9.1	11.0	unknown	needs re-export with --fuse-qkv --fuse-gate-up; NEVER BUILT on the gqafix base
```

⚠ Never set `dlbc_weights` — it is not weight-sharing-compatible on SDK ≥ 2.36,
and every 0.6B bin depends on a healthy shared pool.

- [ ] **Step 3: The test written in Step 1, for reference**

`kit-06b/tests/test_arms.py`:

```python
"""The arms ledger is data the build reads, so its shape is worth a test."""
import csv
import json
import pathlib

import pytest

ARMS = pathlib.Path(__file__).resolve().parents[1] / "variants" / "arms.tsv"


@pytest.fixture
def arms():
    with ARMS.open() as fh:
        return {r["arm"]: r for r in csv.DictReader(fh, delimiter="\t")}


def test_every_arm_has_parseable_overrides(arms):
    for name, row in arms.items():
        json.loads(row["overrides"]), f"{name} overrides is not JSON"


def test_class_a_arms_carry_a_ctxbin_only_override(arms):
    a = {n: r for n, r in arms.items() if r["class"] == "A" and n != "ship"}
    assert set(a) == {"hvx8", "socmodel72", "udma", "dlbc", "wpack"}
    for name, row in a.items():
        assert json.loads(row["overrides"]), f"{name} has no override"


def test_class_b_arms_need_reexport_so_carry_no_ctxbin_override(arms):
    for name, row in arms.items():
        if row["class"] == "B":
            assert json.loads(row["overrides"]) == {}, name


def test_hvx8_predicts_zero_bytes_by_construction(arms):
    # This is what makes hvx8 a clean null test for the byte model.
    assert float(arms["hvx8"]["byte_pct"]) == 0.0
    assert arms["hvx8"]["artifact_delta"] == "2277376"


def test_no_arm_enables_dlbc_weights(arms):
    # dlbc_weights is not weight-sharing-compatible on SDK >= 2.36.
    for name, row in arms.items():
        assert "dlbc_weights" not in row["overrides"], name


def test_unproven_arms_are_flagged(arms):
    for name in ("dlbc", "wpack"):
        assert arms[name]["artifact_delta"] == "0"
        assert "rank last" in arms[name]["note"]
```

- [ ] **Step 4: Run the test to verify it now passes**

Run: `python3 -m pytest kit-06b/tests/test_arms.py -v`
Expected: PASS — `6 passed`. Do not write `build_arm.sh` until this is green;
it reads the ledger's column order directly.

- [ ] **Step 5: Write build_arm.sh**

Create `kit-06b/variants/build_arm.sh`:

```bash
#!/usr/bin/env bash
# Build one variant arm by name from variants/arms.tsv.
#
# Usage: build_arm.sh <arm> [ship_bin_dir]
#
# Class A arms are ctx-bin-only: same DLCs, new config, ~5 min. Class B arms
# need their own export and conversion (~1 h) and are handled by build.sh.
#
# Every arm is compared against the ship bin: an arm whose artifact did not
# change did not necessarily do nothing, but it has earned nothing offline
# either, and an arm whose knob failed to bind would otherwise read as "the
# thing under test had no effect" -- a false measurement, not an error.
set -euo pipefail
source "$(dirname "$0")/../env.sh"

ARM=${1:?arm name}
SHIP_DIR=${2:-$KIT_DATA/kit-out/ship}
ARMS=$KIT_ROOT/variants/arms.tsv

row=$(awk -F'\t' -v a="$ARM" 'NR>1 && $1==a' "$ARMS")
[[ -n $row ]] || { echo "ABORT: unknown arm '$ARM'. Known:" >&2
                   awk -F'\t' 'NR>1{print "  "$1"  ("$2")"}' "$ARMS" >&2; exit 1; }

CLASS=$(cut -f2 <<< "$row")
OVERRIDES=$(cut -f3 <<< "$row")
EXPECT_DELTA=$(cut -f6 <<< "$row")

if [[ $CLASS != A ]]; then
    echo "ABORT: '$ARM' is class $CLASS -- it needs its own export/conversion." >&2
    echo "       Use: ./build.sh variants $ARM" >&2
    exit 1
fi

SHIP_BIN=$SHIP_DIR/qwen3-0.6b-w8a16-gqafix-ladekv_ctx.bin
DLCDIR=$KIT_DATA/kit-out/dlc
OUT=$KIT_DATA/kit-out/variants/$ARM
NAME="qwen3-0.6b-w8a16-gqafix-${ARM}-ladekv_ctx"

for f in "$DLCDIR/prefill.dlc" "$DLCDIR/decode.dlc" "$DLCDIR/verify32.dlc"; do
    [[ -f $f ]] || { echo "ABORT: missing $f -- run ./build.sh ship first" >&2; exit 1; }
done

echo "== arm '$ARM' (class A) overrides: $OVERRIDES =="
bash "$KIT_ROOT/build/ctxbin.sh" "$OUT" "$NAME" \
    "$DLCDIR/prefill.dlc,$DLCDIR/decode.dlc,$DLCDIR/verify32.dlc" \
    prefill,decode,verify32 "$OVERRIDES"

if [[ -f $SHIP_BIN ]]; then
    ship_sz=$(stat -c %s "$SHIP_BIN"); arm_sz=$(stat -c %s "$OUT/$NAME.bin")
    delta=$((arm_sz - ship_sz))
    echo "== artifact delta vs ship: $delta B (ledger predicts $EXPECT_DELTA) =="
    if [[ $EXPECT_DELTA != unknown && $delta -ne $EXPECT_DELTA ]]; then
        echo "FAIL: artifact delta $delta != expected $EXPECT_DELTA" >&2
        echo "      The knob bound differently than when the ledger was written." >&2
        exit 1
    fi
    if [[ $delta -eq 0 ]]; then
        echo "NOTE: byte-identical to ship. That does NOT prove a no-op -- some"
        echo "      keys are runtime hints -- but it earns nothing offline."
    fi
else
    echo "NOTE: no ship bin at $SHIP_BIN; skipping the delta comparison"
fi
echo "ARM READY: $OUT/$NAME.bin"
```

- [ ] **Step 6: Run tests and shellcheck**

Run:
```bash
python3 -m pytest kit-06b/tests/test_arms.py -v
shellcheck kit-06b/variants/build_arm.sh
```
Expected: `6 passed`; shellcheck silent.

- [ ] **Step 7: Integration-test the hvx8 arm against the real reference**

Uses the DLCs on disk directly, so it runs before the kit's own DLC stage exists.

```bash
D=$HOME/llm-local
bash kit-06b/build/ctxbin.sh /tmp/kit-hvx8 qwen3-0.6b-w8a16-gqafix-hvx8-ladekv_ctx \
  "$D/work/dlc/qwen3-0.6b-w8a16-gqafix-ladekv/prefill.dlc,$D/work/dlc/qwen3-0.6b-w8a16-gqafix/decode.dlc,$D/work/dlc/qwen3-0.6b-w8a16-gqafix/verify32.dlc" \
  prefill,decode,verify32 '{"hvx_threads": 8}'
```

Expected:
- `ALL CHECKS PASSED` with `numHvxThreads: 8` on all three graphs
- `md5: b533db214384cbd9401ef045ba259c75`
- size `1088847872` (= ship + 2,277,376)
- decode `read_total_bytes` still 961,130,496 — the arm moves **zero** bytes,
  which is exactly what makes it a clean null test for the byte model

- [ ] **Step 8: Commit**

```bash
git add kit-06b/variants/ kit-06b/tests/test_arms.py
git commit -m "kit: variant arms ledger and class-A builder

arms.tsv carries each arm's two predictions as data, labelled predicted --
only 44.707 is measured. build_arm.sh refuses a class-B arm rather than
silently building a class-A bin under its name, and fails when the artifact
delta disagrees with the ledger, which is how a knob that failed to bind gets
caught before it reads as a measurement.

Verified: hvx8 reproduces b533db214384cbd9401ef045ba259c75 at 1,088,847,872 B
(+2,277,376) with numHvxThreads 8 and byte-identical DDR figures."
```

---

## Task 6: Bundle stage with both dialogs, linted after

**Files:**
- Create: `kit-06b/bundle/configs/genie_dialog_basic.json`
- Create: `kit-06b/bundle/configs/genie_dialog_lade.json`
- Create: `kit-06b/gates/check_dialogs.py`
- Create: `kit-06b/bundle/bundle.sh`
- Test: `kit-06b/tests/test_check_dialogs.py`

- [ ] **Step 1: Write the basic dialog — the 44.707 config, as a real file**

Create `kit-06b/bundle/configs/genie_dialog_basic.json`:

```json
{
  "dialog": {
    "version": 1,
    "type": "basic",
    "context": {
      "version": 1,
      "size": 1024,
      "n-vocab": 151936,
      "bos-token": -1,
      "eos-token": [151645, 151643]
    },
    "sampler": {
      "version": 1,
      "seed": 42,
      "temp": 0.0,
      "top-k": 1,
      "top-p": 1.0
    },
    "tokenizer": { "version": 1, "path": "tokenizer.json" },
    "engine": {
      "version": 1,
      "n-threads": 3,
      "backend": {
        "version": 1,
        "type": "QnnHtp",
        "QnnHtp": {
          "version": 1,
          "use-mmap": true,
          "spill-fill-bufsize": 0,
          "mmap-budget": 25,
          "poll": true,
          "cpu-mask": "0xe0",
          "kv-dim": 128,
          "allow-async-init": true,
          "enable-graph-switching": true
        },
        "extensions": "htp_backend_ext_config.json"
      },
      "model": {
        "version": 1,
        "type": "binary",
        "binary": { "version": 1, "ctx-bins": ["PLACEHOLDER_SET_BY_BUNDLE.bin"] },
        "positional-encoding": { "type": "rope", "rope-dim": 64, "rope-theta": 1000000 }
      }
    }
  }
}
```

⚠ `rope-theta` sits inside `positional-encoding`. Declaring `pos-id-dim` in the
backend block alongside `positional-encoding` is a hard load-time schema error,
not a warning — declare one, never both.

- [ ] **Step 2: Write the lade dialog**

Shipped for optionality only — LADE is parked as a 30% regression. Create
`kit-06b/bundle/configs/genie_dialog_lade.json` as a copy of the basic file with
`"type": "lade"` and a `lade` block, and **no `max-num-tokens` anywhere**:

```bash
python3 - <<'PYEOF'
import json, pathlib
p = pathlib.Path("kit-06b/bundle/configs")
d = json.loads((p / "genie_dialog_basic.json").read_text())
d["dialog"]["type"] = "lade"
# (ngram-1)*(window+gcap) must be <= the verify graph's AR (32), or Genie routes
# verification batches to a graph that cannot serve them: 2*(8+8) = 32 exactly.
d["dialog"]["lade"] = {"version": 1, "window": 8, "ngram": 3, "gcap": 8}
(p / "genie_dialog_lade.json").write_text(json.dumps(d, indent=2) + "\n")
print("wrote", p / "genie_dialog_lade.json")
PYEOF
```

- [ ] **Step 3: Write the failing dialog-linter test**

Create `kit-06b/tests/test_check_dialogs.py`:

```python
"""Tests for the dialog contract linter.

A "lade" dialog carrying "max-num-tokens" SIGSEGVs on device (exit 139). That
pair shipped in three bundles because the dialogs were hand-added AFTER the
bundling script's own lint ran.
"""
import json
import pathlib
import shutil
import subprocess
import sys

import pytest

GATES = pathlib.Path(__file__).resolve().parents[1] / "gates"
CONFIGS = pathlib.Path(__file__).resolve().parents[1] / "bundle" / "configs"


def run(d):
    return subprocess.run(
        [sys.executable, str(GATES / "check_dialogs.py"), str(d)],
        capture_output=True, text=True)


@pytest.fixture
def bundle(tmp_path):
    (tmp_path / "ctx.bin").write_bytes(b"\0")
    for name in ("genie_dialog_basic.json", "genie_dialog.json"):
        d = json.loads((CONFIGS / "genie_dialog_basic.json").read_text())
        if name == "genie_dialog.json":
            d = json.loads((CONFIGS / "genie_dialog_lade.json").read_text())
        d["dialog"]["engine"]["model"]["binary"]["ctx-bins"] = ["ctx.bin"]
        (tmp_path / name).write_text(json.dumps(d, indent=2))
    return tmp_path


def test_a_good_bundle_passes(bundle):
    r = run(bundle)
    assert r.returncode == 0, r.stdout + r.stderr


def test_lade_plus_max_num_tokens_fails(bundle):
    p = bundle / "genie_dialog.json"
    d = json.loads(p.read_text())
    d["dialog"]["max-num-tokens"] = 1024
    p.write_text(json.dumps(d))
    r = run(bundle)
    assert r.returncode != 0
    assert "max-num-tokens" in (r.stdout + r.stderr)


def test_lade_window_exceeding_verify_ar_fails(bundle):
    p = bundle / "genie_dialog.json"
    d = json.loads(p.read_text())
    d["dialog"]["lade"]["window"] = 16   # 2*(16+8) = 48 > 32
    p.write_text(json.dumps(d))
    r = run(bundle)
    assert r.returncode != 0
    assert "verify" in (r.stdout + r.stderr).lower()


def test_dangling_ctx_bin_reference_fails(bundle):
    p = bundle / "genie_dialog_basic.json"
    d = json.loads(p.read_text())
    d["dialog"]["engine"]["model"]["binary"]["ctx-bins"] = ["nope.bin"]
    p.write_text(json.dumps(d))
    r = run(bundle)
    assert r.returncode != 0
    assert "nope.bin" in (r.stdout + r.stderr)


def test_pos_id_dim_alongside_positional_encoding_fails(bundle):
    p = bundle / "genie_dialog_basic.json"
    d = json.loads(p.read_text())
    d["dialog"]["engine"]["backend"]["QnnHtp"]["pos-id-dim"] = 64
    p.write_text(json.dumps(d))
    r = run(bundle)
    assert r.returncode != 0
    assert "pos-id-dim" in (r.stdout + r.stderr)


def test_missing_basic_dialog_fails(bundle):
    (bundle / "genie_dialog_basic.json").unlink()
    r = run(bundle)
    assert r.returncode != 0
    assert "genie_dialog_basic.json" in (r.stdout + r.stderr)
```

- [ ] **Step 4: Run the test to verify it fails**

Run: `python3 -m pytest kit-06b/tests/test_check_dialogs.py -v`
Expected: FAIL — 6 failures, `check_dialogs.py` missing.

- [ ] **Step 5: Write the linter**

Create `kit-06b/gates/check_dialogs.py`:

```python
#!/usr/bin/env python3
"""Lint every Genie dialog JSON in a bundle directory.

MUST run after ALL dialogs are written. The historical failure is exactly the
opposite order: bundle.sh linted, then a second dialog was hand-added by a
heredoc in the build guide, so the pair that SIGSEGVs on device (a "lade" dialog
carrying "max-num-tokens", exit 139) reached three shipped bundles unchecked.
"""
from __future__ import annotations

import json
import pathlib
import sys

VERIFY_AR = 32  # the AR of the verify graph in the ladekv topology


def check(bundle: pathlib.Path) -> list[str]:
    problems: list[str] = []
    dialogs = sorted(bundle.glob("genie_dialog*.json"))
    if not dialogs:
        return [f"{bundle}: no genie_dialog*.json found"]

    names = {p.name for p in dialogs}
    if "genie_dialog_basic.json" not in names:
        problems.append(
            "genie_dialog_basic.json is missing -- it is the 44.707 tok/s "
            "configuration and the primary A/B arm")

    for p in dialogs:
        try:
            d = json.loads(p.read_text())["dialog"]
        except Exception as exc:                       # noqa: BLE001
            problems.append(f"{p.name}: unreadable ({exc})")
            continue

        dtype = d.get("type")
        if dtype == "lade":
            if "max-num-tokens" in d:
                problems.append(
                    f"{p.name}: type 'lade' with 'max-num-tokens' SIGSEGVs on "
                    "device (exit 139). Remove max-num-tokens")
            lade = d.get("lade", {})
            window, ngram, gcap = (lade.get("window", 0), lade.get("ngram", 0),
                                   lade.get("gcap", 0))
            batch = (ngram - 1) * (window + gcap)
            if batch > VERIFY_AR:
                problems.append(
                    f"{p.name}: (ngram-1)*(window+gcap) = {batch} > verify AR "
                    f"{VERIFY_AR}; Genie would route verification batches to a "
                    "graph that cannot serve them")

        engine = d.get("engine", {})
        htp = engine.get("backend", {}).get("QnnHtp", {})
        model = engine.get("model", {})
        if "pos-id-dim" in htp and "positional-encoding" in model:
            problems.append(
                f"{p.name}: 'pos-id-dim' alongside 'positional-encoding' is a "
                "hard load-time schema error. Declare one, not both")

        for b in model.get("binary", {}).get("ctx-bins", []):
            if b.startswith("PLACEHOLDER"):
                problems.append(f"{p.name}: ctx-bins still holds {b}")
            elif not (bundle / b).is_file():
                problems.append(f"{p.name}: ctx-bins references {b}, not in bundle")
    return problems


def main() -> int:
    if len(sys.argv) != 2:
        print(__doc__)
        print("usage: check_dialogs.py <bundle_dir>", file=sys.stderr)
        return 2
    bundle = pathlib.Path(sys.argv[1])
    problems = check(bundle)
    if problems:
        print("DIALOG LINT FAILED:", file=sys.stderr)
        for p in problems:
            print(f"  - {p}", file=sys.stderr)
        return 1
    print(f"DIALOG LINT PASSED ({bundle})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
```

- [ ] **Step 6: Run the test to verify it passes**

Run: `python3 -m pytest kit-06b/tests/test_check_dialogs.py -v`
Expected: PASS — `6 passed`.

- [ ] **Step 7: Write bundle.sh**

Create `kit-06b/bundle/bundle.sh`:

```bash
#!/usr/bin/env bash
# Assemble a push-ready flat device bundle: 7 .so + genie-t2t-run + ctx-bin +
# tokenizer + BOTH dialog configs + backend ext config, in ONE directory, tarred.
#
# Usage: bundle.sh <bundle_name> <ctxbin_path> [tokenizer_json]
#
# Both dialogs are written HERE, from committed files, and the linter runs AFTER
# both exist. The old flow linted first and had the second dialog hand-added by
# a heredoc afterwards, which is how a lade+max-num-tokens pair (device exit
# 139) reached three shipped bundles.
set -euo pipefail
source "$(dirname "$0")/../env.sh"

NAME=${1:?bundle name}
CTXBIN=${2:?path to ctx-bin .bin}
TOKENIZER=${3:-$MODEL/tokenizer.json}

[[ -f $CTXBIN ]] || { echo "ABORT: no ctx-bin at $CTXBIN" >&2; exit 1; }
[[ -f $TOKENIZER ]] || { echo "ABORT: no tokenizer at $TOKENIZER" >&2; exit 1; }

OUT=$KIT_DATA/kit-out/bundles/$NAME
rm -rf "$OUT"          # stale binaries from a previous run must not leak in
mkdir -p "$OUT"

A=$QAIRT_SDK/lib/aarch64-android
H=$QAIRT_SDK/lib/hexagon-v81/unsigned

disk_guard 6
cp "$A/libGenie.so" "$A/libQnnHtp.so" "$A/libQnnSystem.so" \
   "$A/libQnnHtpPrepare.so" "$A/libQnnHtpNetRunExtensions.so" \
   "$A/libQnnHtpV81Stub.so" "$H/libQnnHtpV81Skel.so" "$OUT/"
cp "$QAIRT_SDK/bin/aarch64-android/genie-t2t-run" "$OUT/"
cp "$CTXBIN" "$OUT/"
cp "$TOKENIZER" "$OUT/tokenizer.json"

# The device-side backend extension config. hvx_threads is absent on purpose:
# it is baked into the ctx-bin at build time and setting it here does nothing
# (measured: -0.1%).
cat > "$OUT/htp_backend_ext_config.json" <<'EOF'
{
  "devices": [
    {
      "dsp_arch": "v81",
      "soc_model": 0,
      "pd_session": "unsigned",
      "cores": [
        {
          "core_id": 0,
          "perf_profile": "burst",
          "rpc_control_latency": 100,
          "rpc_polling_time": 9999
        }
      ]
    }
  ]
}
EOF

BIN_NAME=$(basename "$CTXBIN")
python3 - "$KIT_ROOT/bundle/configs" "$OUT" "$BIN_NAME" <<'PYEOF'
import json, pathlib, sys
src, out, bin_name = pathlib.Path(sys.argv[1]), pathlib.Path(sys.argv[2]), sys.argv[3]
# genie_dialog.json is what genie-t2t-run defaults to; basic is the ship config,
# so basic gets that slot and lade ships beside it for the A/B.
for srcname, dstname in (("genie_dialog_basic.json", "genie_dialog.json"),
                         ("genie_dialog_basic.json", "genie_dialog_basic.json"),
                         ("genie_dialog_lade.json", "genie_dialog_lade.json")):
    d = json.loads((src / srcname).read_text())
    d["dialog"]["engine"]["model"]["binary"]["ctx-bins"] = [bin_name]
    (out / dstname).write_text(json.dumps(d, indent=2) + "\n")
    print("wrote", dstname, "<-", srcname)
PYEOF

# MANDATORY, and only meaningful here -- after every dialog exists.
python3 "$KIT_ROOT/gates/check_dialogs.py" "$OUT"

ls -lh "$OUT"
tar -C "$KIT_DATA/kit-out/bundles" -czf "$KIT_DATA/kit-out/bundles/$NAME.tar.gz" "$NAME"
ls -lh "$KIT_DATA/kit-out/bundles/$NAME.tar.gz"
cat <<EOF
BUNDLE READY: $KIT_DATA/kit-out/bundles/$NAME.tar.gz
On device: adb push $NAME.tar.gz /data/local/tmp/ && \\
           adb shell 'cd /data/local/tmp && tar xzf $NAME.tar.gz'
Run:       cd /data/local/tmp/$NAME && \\
           LD_LIBRARY_PATH=. ./genie-t2t-run -c genie_dialog_basic.json -p '<prompt>'
EOF
```

- [ ] **Step 8: Integration-test the bundle against the real ctx-bin**

```bash
chmod +x kit-06b/bundle/bundle.sh
bash kit-06b/bundle/bundle.sh kit_test_bundle \
    /tmp/kit-keystone/qwen3-0.6b-w8a16-gqafix-ladekv_ctx.bin
```
Expected: `DIALOG LINT PASSED`, then `BUNDLE READY`. Verify the layout:
```bash
ls ~/llm-local/kit-out/bundles/kit_test_bundle | sort
```
Expected exactly 13 entries — 7 `.so`, `genie-t2t-run`,
`qwen3-0.6b-w8a16-gqafix-ladekv_ctx.bin`, `tokenizer.json`,
`htp_backend_ext_config.json`, `genie_dialog.json`, `genie_dialog_basic.json`,
`genie_dialog_lade.json` (14 with the lade file — count them, do not guess).

- [ ] **Step 9: shellcheck and commit**

```bash
shellcheck kit-06b/bundle/bundle.sh
git add kit-06b/bundle/ kit-06b/gates/check_dialogs.py kit-06b/tests/test_check_dialogs.py
git commit -m "kit: bundle stage, both dialogs as committed files

genie_dialog_basic.json -- the 44.707 tok/s config -- is a real file rather
than a heredoc in a build guide, and every dialog is written before the linter
runs. The old order linted first and had the second dialog hand-added after,
which is how a lade dialog carrying max-num-tokens (device exit 139) reached
three shipped bundles."
```

---

## Task 7: DLC conversion stage and its two gates

**Files:**
- Create: `kit-06b/build/convert.sh`
- Create: `kit-06b/gates/check_gqa_ops.py`
- Create: `kit-06b/gates/check_bytes.py`

- [ ] **Step 1: Write the GQA gate**

Create `kit-06b/gates/check_gqa_ops.py`:

```python
#!/usr/bin/env python3
"""Assert a DLC contains ZERO KV-replication ops.

Without --grouped-gqa the export emits 56 Eltwise_Binary ops with
"operation": 13 that replicate KV heads. They were 74.7% of decode DSP cycles
and cost 6.54x throughput -- 6.836 tok/s against 44.707.

Run this on EVERY graph, not just decode. lade_build/ladekv-equivalent stages
re-export verify32 and the past-KV prefill, so the usual failure is old
attention in those two while decode looks correct.
"""
from __future__ import annotations

import argparse
import json
import os
import pathlib
import subprocess
import sys

REPLICATION_OP = 13


def replication_ops(dlc: pathlib.Path, sdk: pathlib.Path) -> int:
    tool = sdk / "bin/x86_64-linux-clang/qairt-dlc-info"
    out = subprocess.run([str(tool), "-i", str(dlc), "--json"],
                         capture_output=True, text=True)
    if out.returncode != 0:
        raise SystemExit(f"qairt-dlc-info failed on {dlc}:\n{out.stderr}")
    try:
        d = json.loads(out.stdout)
    except json.JSONDecodeError:
        # Older builds print a table; fall back to counting the op signature.
        return out.stdout.count('"operation": 13')
    n = 0
    for node in json.dumps(d).split("{"):
        if "Eltwise_Binary" in node and f'"operation": {REPLICATION_OP}' in node:
            n += 1
    return n


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("dlcs", nargs="+", type=pathlib.Path)
    ap.add_argument("--sdk", type=pathlib.Path, default=os.environ.get("QAIRT_SDK"))
    args = ap.parse_args()
    if not args.sdk:
        raise SystemExit("QAIRT_SDK is unset and --sdk was not given; "
                         "source env.sh first")

    bad = []
    for dlc in args.dlcs:
        n = replication_ops(dlc, args.sdk)
        print(f"{dlc.name}: {n} replication ops")
        if n:
            bad.append(f"{dlc.name} has {n} KV-replication ops -- --grouped-gqa "
                       "was missed on this graph's export")
    if bad:
        print("\nGQA GATE FAILED:", file=sys.stderr)
        for b in bad:
            print(f"  - {b}", file=sys.stderr)
        return 1
    print("GQA GATE PASSED: 0 replication ops in every graph")
    return 0


if __name__ == "__main__":
    sys.exit(main())
```

- [ ] **Step 2: Verify the GQA gate against the real ship DLCs**

Run:
```bash
source kit-06b/env.sh
python3 kit-06b/gates/check_gqa_ops.py \
  ~/llm-local/work/dlc/qwen3-0.6b-w8a16-gqafix-ladekv/prefill.dlc \
  ~/llm-local/work/dlc/qwen3-0.6b-w8a16-gqafix/decode.dlc \
  ~/llm-local/work/dlc/qwen3-0.6b-w8a16-gqafix/verify32.dlc
```
Expected: `0 replication ops` for all three, then `GQA GATE PASSED`.

If `qairt-dlc-info --json` is not a supported flag on this SDK, run
`qairt-dlc-info -i <dlc> --help` to find the JSON/dump option and adjust
`replication_ops()` to use it. Do not weaken the gate to a text grep that could
match zero for the wrong reason — confirm the count is non-zero on a pre-fix
DLC first if one is available.

- [ ] **Step 3: Write the byte-accounting gate**

Create `kit-06b/gates/check_bytes.py`:

```python
#!/usr/bin/env python3
"""Record and check the converter's DDR byte accounting.

Source: the '====== DDR bandwidth summary ======' block a build log carries. It
is emitted even for builds that cannot run on device, which is what makes it a
device-free gate.

NEVER quote a read_total_bytes without its log name and date. The figure
961,130,496 is the POST-GQA-fix decode graph; the identical-looking pre-fix
number from ctxbin-ws.log (2026-08-10) has already put two documents wrong.
"""
from __future__ import annotations

import argparse
import datetime
import pathlib
import re
import sys

PAT = re.compile(r"(read_total_bytes|write_total_bytes)\D+(\d+)")


def parse(log: pathlib.Path) -> list[tuple[str, int]]:
    return [(m.group(1), int(m.group(2))) for m in PAT.finditer(log.read_text())]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("log", type=pathlib.Path)
    ap.add_argument("--expect-read", type=int, action="append", default=[],
                    help="a read_total_bytes value that must appear")
    ap.add_argument("--expect-write", type=int, action="append", default=[])
    args = ap.parse_args()

    found = parse(args.log)
    if not found:
        print(f"no DDR summary in {args.log}", file=sys.stderr)
        return 1

    reads = [v for k, v in found if k == "read_total_bytes"]
    writes = [v for k, v in found if k == "write_total_bytes"]
    stamp = datetime.date.fromtimestamp(args.log.stat().st_mtime)
    print(f"source: {args.log.name}  ({stamp})   <-- quote BOTH with any number")
    for k, v in found:
        print(f"  {k}: {v:,}")

    missing = [e for e in args.expect_read if e not in reads]
    missing += [e for e in args.expect_write if e not in writes]
    if missing:
        print(f"\nBYTE GATE FAILED: expected {missing} not present -- a variant "
              "whose bytes did not move did not do what it claims",
              file=sys.stderr)
        return 1
    print("BYTE GATE PASSED")
    return 0


if __name__ == "__main__":
    sys.exit(main())
```

- [ ] **Step 4: Verify the byte gate against the keystone build log**

Run:
```bash
python3 kit-06b/gates/check_bytes.py \
    /tmp/kit-keystone/qwen3-0.6b-w8a16-gqafix-ladekv_ctx.build.log \
    --expect-read 961130496 --expect-write 419840
```
Expected: `BYTE GATE PASSED`, with the log name and date printed above the
figures.

- [ ] **Step 5: Write convert.sh**

Create `kit-06b/build/convert.sh`:

```bash
#!/usr/bin/env bash
# Convert the three renamed ONNX graphs to DLCs against ONE encodings file.
#
# Usage: convert.sh <quant_dir> <out_dlc_dir> [cl] [ctx] [ar_prefill] [ar_verify]
#
# TWO CONTRACTS, both fatal when broken:
#
# 1. All three DLCs MUST convert against the SAME encodings file. Mixed
#    encodings are a fatal Genie load error -- KV quant params must be
#    byte-identical across graphs sharing one context.
# 2. A graph's name is baked in from the --output_path BASENAME, dots included.
#    Convert straight to the final filename; renaming afterwards does nothing.
set -euo pipefail
source "$(dirname "$0")/../env.sh"

QUANT=${1:?quant dir (output of quantize.sh)}
DLC=${2:?output dlc dir}
CL=${3:-128}
CTX=${4:-1024}
AR_PREFILL=${5:-128}
AR_VERIFY=${6:-32}

CONVERTER="$QAIRT_SDK/bin/x86_64-linux-clang/qairt-converter"
ENC=$QUANT/calib/model_filtered_renamed.encodings
TOTAL=$((CTX + CL))                      # 1152
PAST_DECODE=$((TOTAL - 1))               # 1151
PAST_PREFILL=$((TOTAL - AR_PREFILL))     # 1024
PAST_VERIFY=$((TOTAL - AR_VERIFY))       # 1120
LAYERS=28                                # Qwen3-0.6B

[[ -f $ENC ]] || { echo "ABORT: missing encodings $ENC -- run quantize.sh first" >&2; exit 1; }
mkdir -p "$DLC"

# past_key is [1, kv_heads, head_dim, past]; past_value is [1, kv_heads, past, head_dim]
past_dims() {
    local past=$1 i
    for ((i = 0; i < LAYERS; i++)); do
        printf -- '-d\npast_key_%d_in\n1,8,128,%d\n-d\npast_value_%d_in\n1,8,%d,128\n' \
               "$i" "$past" "$i" "$past"
    done
}

convert_pastkv() {   # convert_pastkv <onnx> <out.dlc> <ar> <past>
    local onnx=$1 out=$2 ar=$3 past=$4
    mapfile -t dims < <(past_dims "$past")
    disk_guard 6
    $PY_QAIRT "$CONVERTER" --input_network "$onnx" --output_path "$out" \
        --quantization_overrides "$ENC" --float_bitwidth 16 --target_backend HTP \
        -d input_ids "1,$ar" -d attention_mask "1,$ar,$TOTAL" \
        -d position_ids_cos "1,$ar,64" -d position_ids_sin "1,$ar,64" \
        "${dims[@]}"
}

echo "== [1/3] decode  (AR=1,  past=$PAST_DECODE) =="
convert_pastkv "$QUANT/decode/model_renamed.onnx"   "$DLC/decode.dlc"   1 "$PAST_DECODE"

echo "== [2/3] verify$AR_VERIFY (AR=$AR_VERIFY, past=$PAST_VERIFY) =="
convert_pastkv "$QUANT/verify/model_renamed.onnx"   "$DLC/verify$AR_VERIFY.dlc" \
               "$AR_VERIFY" "$PAST_VERIFY"

echo "== [3/3] prefill (AR=$AR_PREFILL, past=$PAST_PREFILL) =="
convert_pastkv "$QUANT/prefill/model_renamed.onnx"  "$DLC/prefill.dlc" \
               "$AR_PREFILL" "$PAST_PREFILL"

echo "== prefill contract: all-position logits AND past-KV inputs =="
INFO=$DLC/prefill_info.txt
"$QAIRT_SDK/bin/x86_64-linux-clang/qairt-dlc-info" -i "$DLC/prefill.dlc" > "$INFO" 2>/dev/null
grep -qE "^\| logits +\| 1,$AR_PREFILL,151936" "$INFO" || {
    echo "FAIL: prefill logits shape is not [1,$AR_PREFILL,151936]." >&2
    echo "      A prefill emitting last-token-only logits ships silent garbage." >&2
    exit 1; }
grep -qE "^\| past_key_0_in" "$INFO" || {
    echo "FAIL: prefill has no past-KV inputs -- this is a bertcache graph." >&2
    echo "      It registers ctx_size == AR and is never selected for prompts" >&2
    echo "      longer than AR; the whole prompt would go through AR=1 decode." >&2
    exit 1; }
echo "  OK: [1,$AR_PREFILL,151936] logits + past-KV inputs present"

echo "== GQA gate: 0 replication ops in EVERY graph =="
python3 "$KIT_ROOT/gates/check_gqa_ops.py" \
    "$DLC/prefill.dlc" "$DLC/decode.dlc" "$DLC/verify$AR_VERIFY.dlc"

md5sum "$DLC"/*.dlc | tee "$DLC/dlc.md5"
echo "CONVERT COMPLETE: $DLC"
```

- [ ] **Step 6: shellcheck and commit**

```bash
shellcheck kit-06b/build/convert.sh
git add kit-06b/build/convert.sh kit-06b/gates/check_gqa_ops.py kit-06b/gates/check_bytes.py
git commit -m "kit: DLC conversion stage with GQA and prefill-contract gates

All three graphs convert against ONE encodings file -- mixed encodings are a
fatal Genie load error. Conversion writes straight to the final filename
because the graph name is baked in from the basename.

The prefill gate rejects both failure modes that ship silent garbage:
last-token-only logits, and a no-past-KV bertcache graph that registers
ctx_size == AR and is never selected for prompts longer than AR.

check_bytes.py prints the log name and date beside every figure -- quoting a
read_total_bytes without them has already put two documents wrong."
```

---

## Task 8: Quantization stage — grouped GQA made structural

The long pole (~1 h). It is the only stage that reaches for AIMET, and the only
one where `--grouped-gqa` could historically be forgotten.

**Files:**
- Create: `kit-06b/build/quantize.sh`
- Copy + adapt: `kit-06b/quant/` (from `scripts/quant/` and `scripts/export/`)

- [ ] **Step 1: Vendor the Python quantization sources**

These are large, working, and out of scope to rewrite. Copy them, then trim.

```bash
mkdir -p kit-06b/quant kit-06b/validate
cp scripts/quant/quantize_aimet.py      kit-06b/quant/
cp scripts/quant/filter_aimet_w8a16.py  kit-06b/quant/
cp scripts/quant/rename_aimet_io.py     kit-06b/quant/
cp scripts/export/modeling_export.py    kit-06b/quant/
cp scripts/export/export_qwen3.py       kit-06b/quant/
cp scripts/validate/parity_ladekv_read.py kit-06b/validate/
git add kit-06b/quant kit-06b/validate
git commit -m "kit: vendor the quantization and parity sources unchanged

Copied verbatim from scripts/ so the diff that follows shows only what the kit
changes. Qwen3-VL paths are removed in the next commit."
```

- [ ] **Step 2: Strip the Qwen3-VL paths from the vendored copies**

Run:
```bash
grep -n "vl_text\|deepstack\|n-deepstack\|vl-calib\|vl_calib" kit-06b/quant/*.py | head -40
```
Remove every `--vl-text`, `--n-deepstack`, `--vl-calib` argument and the code
branches they gate, in `kit-06b/quant/quantize_aimet.py` and
`kit-06b/quant/modeling_export.py`. Leave everything else byte-identical.

Verify nothing else broke:
```bash
"$PY_DEPLOY" -c "import sys; sys.path.insert(0,'kit-06b/quant'); import quantize_aimet; print('import OK')"
"$PY_DEPLOY" kit-06b/quant/quantize_aimet.py --help | head -30
```
Expected: `import OK`, and a help text with no `--vl-*` flags but with
`--grouped-gqa`, `--quant-head`, `--fuse-qkv`, `--fuse-gate-up`,
`--export-decode`, `--decode-ar`, `--eval`.

- [ ] **Step 3: Commit the trim**

```bash
git add kit-06b/quant
git commit -m "kit: drop the Qwen3-VL branches from the vendored quantizer

This is a 0.6B kit; the multimodal chain is a separate tower pair with its own
contracts and belongs in its own kit."
```

- [ ] **Step 4: Write quantize.sh**

Create `kit-06b/build/quantize.sh`:

```bash
#!/usr/bin/env bash
# Stage 1: one calibration run plus three adopted exports.
#
# Usage: quantize.sh <out_dir> [cl] [ctx] [ar_prefill] [ar_verify]
#
# GROUPED GQA IS STRUCTURAL HERE, NOT A FLAG.
# ------------------------------------------
# Every quantize_aimet.py call below hard-codes --grouped-gqa. In the old
# four-script chain it was a positional pass-through on one script and a
# FUSE_FLAGS environment variable on two others, and omitting it on the latter
# silently shipped pre-fix attention in verify32 and the past-KV prefill while
# decode looked correct -- 6.836 tok/s instead of 44.707. There is no way to
# express "without grouped GQA" through this script, by design.
#
# WHY FOUR RUNS AND NOT THREE
# ---------------------------
# The CL=128 calibration run is NOT one of the shipped graphs. It is the
# encodings donor: every other export adopts its scales via --export-decode, so
# all three shipping graphs carry byte-identical KV quant params. Its own ONNX
# is a bertcache prefill and is deliberately never converted.
set -euo pipefail
source "$(dirname "$0")/../env.sh"

OUT=${1:?output quant dir}
CL=${2:-128}
CTX=${3:-1024}
AR_PREFILL=${4:-128}
AR_VERIFY=${5:-32}

Q=$KIT_ROOT/quant
CALIB=$OUT/calib
EXTRA=(--grouped-gqa)     # structural. Do not make this configurable.

mkdir -p "$OUT"

disk_guard 20
echo "== [1/6] calibration quantsim, CL=$CL (the encodings donor) =="
# --eval rides along on this run rather than costing a second ~20 min pass.
"$PY_DEPLOY" "$Q/quantize_aimet.py" --model "$MODEL" --cl-prefill "$CL" \
    --out "$CALIB" --device "$QUANT_DEVICE" "${EXTRA[@]}" --eval \
    2>&1 | tee "$OUT/calib.log"

echo "== quant quality gate: reference is 3/4 last-token argmax agreement =="
python3 - "$OUT/calib.log" <<'PYEOF'
import re, sys
m = re.search(r"(\d+)\s*/\s*4", open(sys.argv[1]).read())
if not m:
    sys.exit("FAIL: --eval printed no N/4 agreement score; quality is ungated. "
             "Check quantize_aimet.py's --eval output format before proceeding.")
score = int(m.group(1))
print(f"argmax agreement: {score}/4 (floor 3/4)")
if score < 3:
    sys.exit(f"FAIL: {score}/4 < 3/4 -- this quantization is not shippable")
PYEOF

echo "== [2/6] encodings filter =="
"$PY_DEPLOY" "$Q/filter_aimet_w8a16.py" "$CALIB/model.encodings"

echo "== [3/6] canonical I/O rename (calibration graph) =="
"$PY_DEPLOY" "$Q/rename_aimet_io.py" --model "$CALIB/model.onnx" \
    --encodings "$CALIB/model_filtered.encodings" --layers 28

export_adopted() {   # export_adopted <subdir> <ar>
    local sub=$1 ar=$2
    disk_guard 20
    "$PY_DEPLOY" "$Q/quantize_aimet.py" --model "$MODEL" --cl-prefill "$CL" \
        --ctx "$CTX" --decode-ar "$ar" --export-decode "$CALIB" \
        --out "$OUT/$sub" --device "$QUANT_DEVICE" "${EXTRA[@]}"
    disk_guard 6
    "$PY_DEPLOY" "$Q/rename_aimet_io.py" --model "$OUT/$sub/model.onnx" \
        --encodings "$CALIB/model_filtered.encodings" --layers 28 --with-past
}

echo "== [4/6] decode export (AR=1), encodings adopted =="
export_adopted decode 1

echo "== [5/6] verify export (AR=$AR_VERIFY), encodings adopted =="
export_adopted verify "$AR_VERIFY"

echo "== [6/6] past-KV prefill export (AR=$AR_PREFILL), encodings adopted =="
export_adopted prefill "$AR_PREFILL"

echo "== FP parity: qualla feed pattern incl. chunked prompts (AR=$AR_PREFILL) =="
"$PY_DEPLOY" "$KIT_ROOT/validate/parity_ladekv_read.py" \
    --model "$MODEL" --onnx "$OUT/prefill/model_renamed.onnx" \
    --ar "$AR_PREFILL" --ctx "$CTX"

echo "QUANTIZE COMPLETE: $OUT"
echo "  encodings (single source of truth): $CALIB/model_filtered_renamed.encodings"
```

- [ ] **Step 5: shellcheck and confirm grouped GQA cannot be disabled**

```bash
shellcheck kit-06b/build/quantize.sh
grep -c -- "--grouped-gqa" kit-06b/build/quantize.sh
grep -rn "FUSE_FLAGS" kit-06b/ || echo "no FUSE_FLAGS anywhere -- correct"
```
Expected: shellcheck silent; the grep count is `1` (set once in `EXTRA`, used by
every call); `no FUSE_FLAGS anywhere`.

- [ ] **Step 6: Commit**

```bash
git add kit-06b/build/quantize.sh
git commit -m "kit: quantization stage, grouped GQA structural

--grouped-gqa is hard-coded and there is no way to express its absence. In the
old chain it was a positional pass-through on one script and a FUSE_FLAGS env
var on two others; omitting it on the latter silently shipped pre-fix attention
in verify32 and the past-KV prefill while decode looked correct -- 6.836 tok/s
instead of 44.707.

Four AIMET runs, three shipped graphs: the CL=128 calibration run is the
encodings donor, not a graph, and its bertcache ONNX is never converted."
```

---

## Task 9: `build.sh` — top level, plus class-B arms

**Files:**
- Create: `kit-06b/build.sh`
- Modify: `kit-06b/variants/build_arm.sh` (class B support)

- [ ] **Step 1: Write build.sh**

Create `kit-06b/build.sh`:

```bash
#!/usr/bin/env bash
# Qwen3-0.6B SA8797P build kit.
#
#   ./build.sh ship                 the measured 44.707 tok/s bundle
#   ./build.sh variants [arm...]    speed-experiment arms (default: all class A)
#   ./build.sh check                run every gate over the existing ship output
#
# Building is device-free. This kit CANNOT measure tok/s -- it builds
# candidates; a device team measures them.
set -euo pipefail
source "$(dirname "$0")/env.sh"

CMD=${1:-ship}; shift || true
O=$KIT_DATA/kit-out
SHIP_NAME=qwen3-0.6b-w8a16-gqafix-ladekv_ctx
SHIP_BIN=$O/ship/$SHIP_NAME.bin
KNOWN_MD5=9c6024ad5b141137fbe22f3a4972eb96

ship() {
    bash "$KIT_ROOT/setup/check_sdk.sh"
    bash "$KIT_ROOT/build/quantize.sh" "$O/quant"
    bash "$KIT_ROOT/build/convert.sh"  "$O/quant" "$O/dlc"
    bash "$KIT_ROOT/build/ctxbin.sh"   "$O/ship" "$SHIP_NAME" \
        "$O/dlc/prefill.dlc,$O/dlc/decode.dlc,$O/dlc/verify32.dlc" \
        prefill,decode,verify32
    python3 "$KIT_ROOT/gates/check_bytes.py" "$O/ship/$SHIP_NAME.build.log" \
        --expect-read 961130496 --expect-write 419840
    bash "$KIT_ROOT/bundle/bundle.sh" qwen3_06b_w8a16_gqafix_ladekv "$SHIP_BIN"

    local got; got=$(md5sum "$SHIP_BIN" | cut -d' ' -f1)
    echo
    echo "ship ctx-bin md5: $got"
    if [[ $got == "$KNOWN_MD5" ]]; then
        echo "MATCHES the device-measured 44.707 tok/s bin ($KNOWN_MD5)."
    else
        echo "⚠️  DOES NOT MATCH the reference bin $KNOWN_MD5."
        echo "    ctx-bin generation is deterministic, so this build is NOT the"
        echo "    binary that was measured at 44.707 tok/s. Do not quote that"
        echo "    number for it. Diff $O/ship/$SHIP_NAME.info.json against the"
        echo "    reference before shipping."
    fi
}

variants() {
    local arms=("$@")
    if [[ ${#arms[@]} -eq 0 ]]; then
        mapfile -t arms < <(awk -F'\t' 'NR>1 && $2=="A" && $1!="ship"{print $1}' \
                            "$KIT_ROOT/variants/arms.tsv")
    fi
    echo "arms: ${arms[*]}"
    for a in "${arms[@]}"; do
        echo; echo "########## $a ##########"
        bash "$KIT_ROOT/variants/build_arm.sh" "$a" "$O/ship"
    done
}

check() {
    [[ -f $SHIP_BIN ]] || { echo "ABORT: no ship bin at $SHIP_BIN" >&2; exit 1; }
    python3 "$KIT_ROOT/gates/check_ctxbin.py" "$O/ship/$SHIP_NAME.info.json" \
        --graphs prefill,decode,verify32 --expect-hvx 4
    python3 "$KIT_ROOT/gates/check_gqa_ops.py" \
        "$O/dlc/prefill.dlc" "$O/dlc/decode.dlc" "$O/dlc/verify32.dlc"
    python3 "$KIT_ROOT/gates/check_bytes.py" "$O/ship/$SHIP_NAME.build.log" \
        --expect-read 961130496 --expect-write 419840
    python3 "$KIT_ROOT/gates/check_dialogs.py" \
        "$O/bundles/qwen3_06b_w8a16_gqafix_ladekv"
    echo "ALL GATES PASSED"
}

case "$CMD" in
    ship)     ship ;;
    variants) variants "$@" ;;
    check)    check ;;
    *) echo "usage: build.sh {ship|variants [arm...]|check}" >&2; exit 2 ;;
esac
```

- [ ] **Step 2: Add class-B support to build_arm.sh**

In `kit-06b/variants/build_arm.sh`, replace the block that begins
`if [[ $CLASS != A ]]; then` with:

```bash
O=$KIT_DATA/kit-out
if [[ $CLASS == B ]]; then
    # Class B needs its own export + conversion (~1 h): the graph geometry or
    # the wrapper structure differs, so the ctx-bin knob path cannot reach it.
    QDIR=$O/variants/$ARM/quant
    DLCB=$O/variants/$ARM/dlc
    case $ARM in
      cl512)
        # context 512 -> ctx-bin CL 640, decode past 639. Its saving is
        # activation traffic, which -- unlike the W8 head -- cannot be
        # re-materialized away at context-prepare time.
        bash "$KIT_ROOT/build/quantize.sh" "$QDIR" 128 512 128 32
        bash "$KIT_ROOT/build/convert.sh"  "$QDIR" "$DLCB" 128 512 128 32 ;;
      qh)
        # --quant-head needs the filter told to spare lm_head's weight encoding,
        # or the filter strips it and you silently get an FP16 head.
        QUANT_EXTRA="--quant-head" bash "$KIT_ROOT/build/quantize.sh" "$QDIR"
        bash "$KIT_ROOT/build/convert.sh" "$QDIR" "$DLCB"
        "$QAIRT_SDK/bin/x86_64-linux-clang/qairt-dlc-info" -i "$DLCB/decode.dlc" \
            | grep "lm_head.weight" | grep -q "sFxp_8" || {
            echo "FAIL: lm_head is not sFxp_8 -- the filter stripped the head" >&2
            echo "      encoding and this build has a silent FP16 head." >&2
            exit 1; } ;;
      fuseqkvgu)
        QUANT_EXTRA="--fuse-qkv --fuse-gate-up" \
            bash "$KIT_ROOT/build/quantize.sh" "$QDIR"
        bash "$KIT_ROOT/build/convert.sh" "$QDIR" "$DLCB" ;;
      *) echo "ABORT: no class-B recipe for '$ARM'" >&2; exit 1 ;;
    esac
    bash "$KIT_ROOT/build/ctxbin.sh" "$O/variants/$ARM" \
        "qwen3-0.6b-w8a16-gqafix-${ARM}-ladekv_ctx" \
        "$DLCB/prefill.dlc,$DLCB/decode.dlc,$DLCB/verify32.dlc" \
        prefill,decode,verify32 "$OVERRIDES"
    echo "ARM READY (class B): $O/variants/$ARM"
    exit 0
fi
```

Then add `QUANT_EXTRA` support to `kit-06b/build/quantize.sh` — change the
`EXTRA` line to:

```bash
# --grouped-gqa is structural and always present. QUANT_EXTRA appends VARIANT
# flags (--quant-head, --fuse-qkv, --fuse-gate-up); it can never remove one.
EXTRA=(--grouped-gqa)
[[ -n ${QUANT_EXTRA:-} ]] && read -r -a _extra <<< "$QUANT_EXTRA" && EXTRA+=("${_extra[@]}")
```

and, because `--quant-head` needs the filter told to spare the head encoding,
change the filter call to:

```bash
echo "== [2/6] encodings filter =="
HEAD_FLAG=()
[[ " ${EXTRA[*]} " == *" --quant-head "* ]] && HEAD_FLAG=(--keep-head-weight)
"$PY_DEPLOY" "$Q/filter_aimet_w8a16.py" "$CALIB/model.encodings" "${HEAD_FLAG[@]}"
```

- [ ] **Step 3: shellcheck everything**

Run: `shellcheck kit-06b/build.sh kit-06b/variants/build_arm.sh kit-06b/build/quantize.sh`
Expected: exit 0.

- [ ] **Step 4: Dry-run the dispatcher**

Run:
```bash
bash kit-06b/build.sh 2>&1 | head -5 || true
bash kit-06b/build.sh nosuchcmd; echo "exit=$?"
```
Expected: `ship` starts with the SDK check; `nosuchcmd` prints the usage line and
`exit=2`.

- [ ] **Step 5: Commit**

```bash
git add kit-06b/build.sh kit-06b/variants/build_arm.sh kit-06b/build/quantize.sh
git commit -m "kit: top-level build.sh and class-B arms

ship reports whether its ctx-bin md5 matches the device-measured reference and
says plainly not to quote 44.707 for a bin that does not.

QUANT_EXTRA can only ADD variant flags -- --grouped-gqa stays structural. The
qh arm verifies lm_head is sFxp_8 after conversion, because --quant-head
without --keep-head-weight on the filter silently yields an FP16 head."
```

---

## Task 10: setup/ — envs and model fetch

**Files:**
- Create: `kit-06b/setup/make_envs.sh`
- Create: `kit-06b/setup/fetch_model.sh`

- [ ] **Step 1: Write make_envs.sh**

Create `kit-06b/setup/make_envs.sh`:

```bash
#!/usr/bin/env bash
# Create the two Python environments the kit needs.
#
#   qwen3-deploy  py3.10  torch + AIMET + onnx     (export and quantization)
#   qairt-py312   py3.12  the SDK's converter only
#
# Two because the converter's pinned dependencies conflict with AIMET's. Do not
# merge them.
set -euo pipefail
source "$(dirname "$0")/../env.sh"

command -v uv >/dev/null || { echo "ABORT: uv not found. https://docs.astral.sh/uv/" >&2; exit 1; }
mkdir -p "$KIT_DATA/envs"

if [ ! -x "$PY_DEPLOY" ]; then
    echo "== creating qwen3-deploy (py3.10) =="
    uv venv --python 3.10 "$KIT_DATA/envs/qwen3-deploy"
    VIRTUAL_ENV=$KIT_DATA/envs/qwen3-deploy uv pip install \
        torch --index-url https://download.pytorch.org/whl/cu121
    VIRTUAL_ENV=$KIT_DATA/envs/qwen3-deploy uv pip install \
        transformers onnx onnxruntime numpy 'aimet-onnx'
else
    echo "== qwen3-deploy exists: $PY_DEPLOY =="
fi

if [ ! -x "$PY_QAIRT" ]; then
    echo "== creating qairt-py312 (py3.12) =="
    uv venv --python 3.12 "$KIT_DATA/envs/qairt-py312"
    VIRTUAL_ENV=$KIT_DATA/envs/qairt-py312 uv pip install \
        -r "$QAIRT_SDK/lib/python/requirements.txt" 2>/dev/null || \
        echo "  NOTE: no SDK requirements.txt; install per the SDK's own docs"
else
    echo "== qairt-py312 exists: $PY_QAIRT =="
fi

echo "== verifying =="
"$PY_DEPLOY" -c "import torch, onnx; print('deploy env OK, cuda:', torch.cuda.is_available())"
"$PY_QAIRT" -c "print('qairt env OK')"
echo "ENVS READY. If AIMET needs a workaround on your platform, apply it now and re-run."
```

- [ ] **Step 2: Write fetch_model.sh**

Create `kit-06b/setup/fetch_model.sh`:

```bash
#!/usr/bin/env bash
# Fetch the Qwen3-0.6B checkpoint. Needs network reachability to Hugging Face;
# set HF_ENDPOINT or HTTPS_PROXY first if your host needs one.
set -euo pipefail
source "$(dirname "$0")/../env.sh"

REPO=${HF_MODEL_REPO:-Qwen/Qwen3-0.6B}
if [ -f "$MODEL/tokenizer.json" ] && [ -f "$MODEL/config.json" ]; then
    echo "model already present: $MODEL"; exit 0
fi
mkdir -p "$(dirname "$MODEL")"
disk_guard 10

if command -v hf >/dev/null; then
    hf download "$REPO" --local-dir "$MODEL"
elif command -v huggingface-cli >/dev/null; then
    huggingface-cli download "$REPO" --local-dir "$MODEL"
else
    "$PY_DEPLOY" - "$REPO" "$MODEL" <<'PYEOF'
import sys
from huggingface_hub import snapshot_download
snapshot_download(repo_id=sys.argv[1], local_dir=sys.argv[2])
PYEOF
fi

for f in config.json tokenizer.json; do
    [ -f "$MODEL/$f" ] || { echo "ABORT: $f missing after download" >&2; exit 1; }
done
echo "MODEL READY: $MODEL"
```

- [ ] **Step 3: shellcheck and commit**

```bash
shellcheck kit-06b/setup/make_envs.sh kit-06b/setup/fetch_model.sh
chmod +x kit-06b/setup/*.sh kit-06b/build/*.sh kit-06b/variants/*.sh \
         kit-06b/bundle/*.sh kit-06b/build.sh
git add kit-06b/setup/ kit-06b/build kit-06b/variants kit-06b/bundle kit-06b/build.sh
git commit -m "kit: environment and model bootstrap

Two envs because the SDK converter's pins conflict with AIMET's. fetch_model.sh
degrades from hf to huggingface-cli to the Python API, and notes that a host
needing a proxy must set it first -- some build hosts cannot reach HF at all."
```

---

## Task 11: README and final end-to-end verification

**Files:**
- Create: `kit-06b/README.md`

- [ ] **Step 1: Write the README**

Create `kit-06b/README.md` covering, in this order:

1. **What this builds** — the flat device bundle layout and the 3-graph topology
   table (`prefill` AR=128/CL=1152, `decode` AR=1/CL=1152, `verify32` AR=32/CL=1152).
2. **Quick start** — `setup/check_sdk.sh` → `setup/make_envs.sh` →
   `setup/fetch_model.sh` → `./build.sh ship` → `./build.sh variants`.
3. **The decision set** — the §2 table from the design spec, with the losing
   alternative and its measured cost in each row.
4. **Measured vs predicted** — a table with exactly one measured row (44.707 ±
   0.030 tok/s, basic mode, 2026-08-15, TTFT 103 ms) and every variant row
   labelled **predicted**, copied from `variants/arms.tsv`.
5. **On exceeding 50 tok/s** — reproduce §5.1 of the design spec verbatim,
   including: this kit cannot measure; `hvx8` is the best free shot because every
   shipping bin uses 4 of 8 HVX units; `cl512` lands ~56 (compute model) or ~49
   (byte model); the regime question is unresolved and the two models are 1.4%
   apart at the current operating point.
6. **Measurement protocol** — the device rep spread on ONE binary was
   23.4 / 44.5 / 29.3, so: 5 reps per arm, report the median and every raw value,
   record thermal state before and after, fixed cool-down between arms, never
   compare arms from different thermal regimes, and **an A/B whose delta falls
   inside the rep spread decides nothing** — re-run it, do not interpret it.
7. **Gates** — the table from the design spec §6, each with its command.
8. **Do not** — LADE tuning (parked, −30%), W4A16 (no INT4 kernels on v81),
   `dlbc_weights` (breaks weight sharing on SDK ≥2.36), multi-core (Genie 5005),
   `hvx_threads` in the device-side config (build-time only, measured −0.1%),
   quoting a `read_total_bytes` without its log name and date.

- [ ] **Step 2: Full end-to-end run**

This is the real acceptance test. ~1–2 h, needs ~20 GB free.

```bash
cd kit-06b && ./build.sh ship 2>&1 | tee /tmp/kit-ship.log
```

Expected, all of:
- `SDK CHECK PASSED`
- `parity_ladekv_read.py` reports all prompts OK
- `GQA GATE PASSED: 0 replication ops in every graph`
- prefill contract `OK: [1,128,151936] logits + past-KV inputs present`
- `ALL CHECKS PASSED` from `check_ctxbin.py` with pooled fraction ≥ 0.9999
- `BYTE GATE PASSED` — decode read 961,130,496 / write 419,840
- `DIALOG LINT PASSED`
- **`ship ctx-bin md5: 9c6024ad5b141137fbe22f3a4972eb96`** and
  `MATCHES the device-measured 44.707 tok/s bin`

If the md5 does not match, the restructuring changed something that reaches the
device. Diff `~/llm-local/kit-out/ship/*.info.json` against
`~/llm-local/work/ctxbin/qwen3-0.6b-w8a16-gqafix-ladekv/info.json`, and diff the
DLC md5s in `~/llm-local/kit-out/dlc/dlc.md5` against the ground-truth table at
the top of this plan to localise which stage diverged.

- [ ] **Step 3: Build the class-A arms**

```bash
cd kit-06b && ./build.sh variants 2>&1 | tee /tmp/kit-variants.log
```

Expected: `hvx8` at +2,277,376 B with `numHvxThreads: 8`; `socmodel72` at
+249,856; `udma` at +212,992; `dlbc` and `wpack` byte-identical with the
"earns nothing offline" note. Every arm's decode `read_total_bytes` stays
961,130,496 — Class A moves zero bytes by construction.

- [ ] **Step 4: Confirm independence from this repo**

The kit must not reach outside itself.

```bash
grep -rn "LLMDEPLOY_ROOT\|LLMDEPLOY_DATA\|scripts/\|configs/\|/mnt/x" kit-06b/ \
  --include=*.sh --include=*.py --include=*.md | grep -v "^kit-06b/README.md" \
  || echo "no references to the parent repo -- correct"
```
Expected: `no references to the parent repo -- correct`. Fix any hit before
committing.

Then prove it by copying the kit out and running the cheap path:
```bash
cp -r kit-06b /tmp/kit-standalone
cd /tmp/kit-standalone && bash setup/check_sdk.sh && \
  python3 -m pytest tests/ -q && bats tests/test_env.bats
```
Expected: `SDK CHECK PASSED`, all pytest tests pass, all bats tests pass.

- [ ] **Step 5: Run the whole test suite once more from the repo copy**

```bash
cd /mnt/x/code/llm-deploy/.claude/worktrees/tlm
python3 -m pytest kit-06b/tests/ -v
bats kit-06b/tests/test_env.bats
shellcheck kit-06b/*.sh kit-06b/*/*.sh
```
Expected: all pytest tests pass, all bats tests pass, shellcheck exit 0.

- [ ] **Step 6: Commit**

```bash
git add kit-06b/README.md
git commit -m "kit: README

Separates the one measured number (44.707 tok/s, basic mode, 2026-08-15) from
every predicted one, carries the B0 measurement protocol -- the rep spread on a
single binary was 23.4/44.5/29.3, so a delta inside that spread decides nothing
-- and states plainly that this kit builds candidates and cannot measure them.

Verified end to end: ./build.sh ship reproduces ctx-bin md5
9c6024ad5b141137fbe22f3a4972eb96, and the kit runs with the parent repo absent."
```

---

## Appendix: what this kit deliberately does not do

Carried from the design spec §10, and worth re-reading before anyone extends it:

- **Measure tok/s.** Both build hosts are device-free. The kit builds candidates.
- **Qwen3-VL / multimodal** — a separate two-tower chain with its own contracts.
- **LADE tuning or learned draft heads** — parked by measurement; post-fix
  break-even is 2.30 accepted tokens/call against ~1.6 measured.
- **W4A16 / INT4** — v81 ships zero INT4 matmul kernels; the converter folds
  s4→f16.
- **Multi-core / multi-process** — Genie returns 5005.
- **KV INT8 via Genie config** — the flag exists only in the CPU backend.
- **`coord.py` dedup, HF upload machinery, device-report docs** — repo-local.
- **Re-deriving the decision set.** If a future measurement overturns one, change
  the README table and `arms.tsv` together, and say which measurement did it.
